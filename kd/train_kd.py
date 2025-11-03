import argparse
import math
import os
from datetime import datetime
import logging
from transformers.trainer import get_scheduler
import torch
from customized_sft_dataset import SFTDataset
from openrlhf.models import Actor
# from openrlhf.trainer import KDTrainer
from customized_kd_trainer import MY_KDTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
# from openrlhf.utils.deepspeed import DeepspeedStrategy
from torch import distributed as dist
# from torch.utils.data import DataLoader
# from openrlhf.utils.distributed_sampler import DistributedSampler
import numpy as np
import random
import transformers

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

seed_everything(42)




logging.basicConfig(
    level=logging.INFO,  # 设置日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('kd_training')
def train(args):
    cache_dir = "/mnt/workspace/wangyifei/projects/huggingcache"
    dataset_dir = "/mnt/workspace/wangyifei/projects/self_training/datasets"
    if args.multi_gold_doc:
        assert args.dataset_name in ["musique"]
    else:
        assert args.dataset_name in ["nq","tqa","webq"]
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    logger.info(f"Loading student model from {os.path.join(cache_dir, args.pretrain)}")
    model = Actor(
        pretrain_or_model = os.path.join(cache_dir, args.pretrain),
        # pretrain_or_model="/mnt/workspace/wangyifei/projects/self_training/kl_checkpoints/Qwen1.5-7B-Chat_20total_docs_filter_5random_2strengthen_400_kl_0.8",
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )
    logger.info(f"Loading teacher model from {os.path.join(cache_dir, args.teacher_model)}")
    # load teacher model for inference
    teacher_model = Actor(
        os.path.join(cache_dir, args.teacher_model),
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=args.teacher_offload),
    )
    if args.teacher_offload:
        teacher_model._offload = True

    
    # configure tokenizer
    logger.info(f"Loading tokenizer from {os.path.join(cache_dir, args.pretrain)}")
    tokenizer = get_tokenizer(os.path.join(cache_dir, args.pretrain), model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)

    strategy.print(model)

    
    # configure optimizer
    logger.info(f"Creating optimizer....")
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    logger.info(f"Prepare for data and dataset....")
    # prepare for data and dataset
    logger.info(f"args.dataset_probs: {args.dataset_probs}")
    logger.info(f"args.dataset: {args.dataset}")
    dataset_dir = os.path.join(dataset_dir, args.dataset_name)
    train_data = blending_datasets(
        os.path.join(dataset_dir, args.dataset),
        args.dataset_probs,
        strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    
    # eval_dataset_name = args.dataset.split("_unfilter")[0] + "_dev"
    # logger.info(f"Prepare for eval data and dataset....")
    # logger.info(f"eval dataset name: {eval_dataset_name}")
    # eval_data = blending_datasets(
    #     os.path.join(dataset_dir, eval_dataset_name),
    #     args.dataset_probs,
    #     strategy,
    #     args.seed,
    #     return_eval=False,
    #     max_count=args.max_samples,
    #     train_split=args.train_split,
    #     eval_split=args.eval_split,
    # )
    
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    train_data = train_data.shuffle(seed=args.seed)
    # eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multi_gold_doc=args.multi_gold_doc,
        num_processors=1,
    )

    # eval_dataset = SFTDataset(
    #     eval_data,
    #     tokenizer,
    #     args.max_len,
    #     strategy,
    #     pretrain_mode=args.pretrain_mode,
    #     input_template=args.input_template,
    # )

    train_dataloader = strategy.setup_dataloader(
        replay_buffer = train_dataset, 
        batch_size = args.micro_train_batch_size, 
        pin_memory = True, 
        shuffle = True, 
        collate_fn = train_dataset.collate_fn,
        # use_block_split=True ,
        # block_size= args.micro_train_batch_size
    )

    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # prepare models
    ((model, optim, scheduler), teacher_model) = strategy.prepare((model, optim, scheduler), teacher_model)

    # load checkpoint
    args.kd_coef = [float(x) for x in args.kd_coef.split(',')]
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")
    if args.output_key == "oracle_answer":
        checkpoint_name = args.dataset + f"_kd{str(args.kd_coef[0])}_lm{str(args.kd_coef[1])}_rank{str(args.kd_coef[2])}_adaptive{str(args.kd_coef[3])}_{args.perserve}"
        # args.save_path = os.path.join(args.save_path, args.dataset + f"_kd{str(args.kd_coef[0])}_lm{str(args.kd_coef[1])}_rank{str(args.kd_coef[2])}_adaptive{str(args.kd_coef[3])}")
        strategy.args.wandb_run_name = args.dataset + f"_kd{str(args.kd_coef[0])}_lm{str(args.kd_coef[1])}_rank{str(args.kd_coef[2])}_adaptive{str(args.kd_coef[3])}_{args.perserve}"
    else:
        checkpoint_name = args.dataset + f"_sft"
        # args.save_path = os.path.join(args.save_path, args.dataset + f"_sft")
        strategy.args.wandb_run_name = args.dataset + f"_sft"
        assert args.kd_coef[0] == 0.0
    

    # configure Trainer
    attention_dillution_coef = args.perserve
    trainer = MY_KDTrainer(
        model=model,
        teacher_model=teacher_model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=None,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        multi_gold_doc=args.multi_gold_doc,
        attention_dillution_coef=round(attention_dillution_coef, 1),
        # **({"attention_dillution_coef": round(attention_dillution_coef, 1)} if args.output_key == "oracle_answer" else {}),

    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)
    # save model checkpoint after fitting on only rank0
    save_path = os.path.join(args.save_path, checkpoint_name)
    os.makedirs(save_path, exist_ok=True)
    strategy.save_model(model, tokenizer, save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoints
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_kd")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # KD
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--teacher_model", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--kd_coef", type=str)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--teacher_offload", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="output", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_sft")
    # parser.add_argument(
    #     "--wandb_run_name",
    #     type=str,
    #     default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    # )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="temp")
    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)
    # kd trainer parameters
    parser.add_argument("--perserve", type=float,default=1.0)
    parser.add_argument("--multi_gold_doc",action="store_true",default=False)

    args = parser.parse_args()

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        # patch_hub()

    train(args)
