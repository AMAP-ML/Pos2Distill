from openrlhf.trainer import KDTrainer
from typing import Optional, Tuple
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import os
from abc import ABC
import torch

from torch.optim import Optimizer
from tqdm import tqdm
from openrlhf.models import GPTLMLoss, KDLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
import logging
from typing import Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loss_design import AdaptiveKLWeightedKLLoss,MY_KDLoss,MY_rankLoss,MY_topkLoss,MY_GPTLMLoss,AdaptiveKLWeightedKLLoss_multi
logger = logging.getLogger('customized kd trainer')

class MY_KDTrainer(KDTrainer):
    def __init__(self, *args, **kwargs):
        attention_dillution_coef = kwargs.pop("attention_dillution_coef", 1.0)
        multi_gold_doc = kwargs.pop("multi_gold_doc", False)
        super().__init__(*args, **kwargs)
        self.kd_loss = MY_KDLoss()
        self.rank_loss = MY_topkLoss()
        self.loss_fn = MY_GPTLMLoss()
        self.adaptive_kd_loss = AdaptiveKLWeightedKLLoss(batch_norm=True,attention_dillution_coef = attention_dillution_coef) if not multi_gold_doc else AdaptiveKLWeightedKLLoss_multi(batch_norm=True,attention_dillution_coef = attention_dillution_coef)
        # self.loss_fn = MY_rankLoss()

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        loss_sum = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )
            # train
            self.model.train()
            self.teacher_model.eval()
            
            for teacher_prompt_id_lens,teacher_inputs,teacher_attention_masks, student_prompt_id_lens,student_inputs,student_attention_masks,infos in self.train_dataloader:
                student_inputs = student_inputs.squeeze(1).to(torch.cuda.current_device())
                student_attention_mask = student_attention_masks.squeeze(1).to(torch.cuda.current_device())
                student_output = self.model(student_inputs, attention_mask=student_attention_mask, return_output=True)
                student_labels = torch.where(
                    student_attention_mask.bool(),
                    student_inputs, 
                    self.loss_fn.IGNORE_INDEX,
                )
                teacher_inputs = teacher_inputs.squeeze(1).to(torch.cuda.current_device())
                teacher_attention_mask = teacher_attention_masks.squeeze(1).to(torch.cuda.current_device())
                teacher_labels = torch.where(
                    teacher_attention_mask.bool(),
                    teacher_inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                if not self.pretrain_mode:
                    for label, source_len in zip(student_labels, student_prompt_id_lens):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                    
                    for label, source_len in zip(teacher_labels, teacher_prompt_id_lens):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                if args.kd_coef[1] > 0:
                    gpt_loss = self.loss_fn(student_output.logits, student_labels)
                else:
                    gpt_loss = torch.tensor(0.0).to(student_output.logits.device)
                    
                with torch.no_grad():
                    teacher_logits = self.teacher_model(teacher_inputs, attention_mask=teacher_attention_mask, return_output=True)[
                        "logits"
                    ]
                if args.kd_coef[0] > 0:
                    distil_loss = self.kd_loss(student_output.logits, teacher_logits, student_labels,teacher_labels)
                else:
                    distil_loss = torch.tensor(0.0).to(teacher_logits.device)

                if args.kd_coef[2] > 0:
                    rank_loss = self.rank_loss(student_output.logits, teacher_logits, student_labels,teacher_labels)
                else:
                    rank_loss = torch.tensor(0.0).to(teacher_logits.device)

                if args.kd_coef[3] > 0:
                    adaptive_kd_loss= self.adaptive_kd_loss(student_output.logits, teacher_logits, student_labels,teacher_labels,infos["gold_idx"])
                else:
                    adaptive_kd_loss = torch.tensor(0.0).to(teacher_logits.device)
                #均衡系数

                loss =  distil_loss * self.args.kd_coef[0] + gpt_loss* args.kd_coef[1] + rank_loss * args.kd_coef[2] + adaptive_kd_loss* self.args.kd_coef[3]

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_sum += loss.item()
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "distil_loss": distil_loss.item(),
                    "rank_loss": rank_loss.item(),
                    "adaptive_kd_loss": adaptive_kd_loss.item(),
                    # "lr": self.scheduler.get_last_lr()[0],
                }
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        # if global_step % args.eval_steps == 0:
        #     # do eval when len(dataloader) > 0, avoid zero division in eval.
        #     if len(self.eval_dataloader) > 0:
        #         self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        # if global_step % args.save_steps == 0:
        #     tag = f"global_step{global_step}"
        #     self.strategy.save_ckpt(
        #         self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
        #     )

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        self.teacher_model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            for teacher_prompt_id_lens,teacher_inputs,teacher_attention_masks, student_prompt_id_lens,student_inputs,student_attention_masks,infos in eval_dataloader:
                student_inputs = student_inputs.squeeze(1).to(torch.cuda.current_device())
                student_attention_mask = student_attention_masks.squeeze(1).to(torch.cuda.current_device())
                student_output = self.model(student_inputs, attention_mask=student_attention_mask, return_output=True)
                student_labels = torch.where(
                    student_attention_mask.bool(),
                    student_inputs, 
                    self.loss_fn.IGNORE_INDEX,
                )
                teacher_inputs = teacher_inputs.squeeze(1).to(torch.cuda.current_device())
                teacher_attention_mask = teacher_attention_masks.squeeze(1).to(torch.cuda.current_device())
                teacher_labels = torch.where(
                    teacher_attention_mask.bool(),
                    teacher_inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                if not self.pretrain_mode:
                    for label, source_len in zip(student_labels, student_prompt_id_lens):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                    
                    for label, source_len in zip(teacher_labels, teacher_prompt_id_lens):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                if args.kd_coef[1] > 0:
                    gpt_loss = self.loss_fn(student_output.logits, student_labels)
                else:
                    gpt_loss = torch.tensor(0.0).to(student_output.logits.device)
                    
                with torch.no_grad():
                    teacher_logits = self.teacher_model(teacher_inputs, attention_mask=teacher_attention_mask, return_output=True)[
                        "logits"
                    ]
                if args.kd_coef[0] > 0:
                    distil_loss = self.kd_loss(student_output.logits, teacher_logits, student_labels,teacher_labels)
                else:
                    distil_loss = torch.tensor(0.0).to(teacher_logits.device)

                if args.kd_coef[2] > 0:
                    rank_loss = self.rank_loss(student_output.logits, teacher_logits, student_labels,teacher_labels)
                else:
                    rank_loss = torch.tensor(0.0).to(teacher_logits.device)

                if args.kd_coef[3] > 0:
                    adaptive_kd_loss = self.adaptive_kd_loss(student_output.logits, teacher_logits, student_labels,teacher_labels,infos["gold_idx"])
                else:
                    adaptive_kd_loss = torch.tensor(0.0).to(teacher_logits.device)
                #均衡系数

                loss =  distil_loss * self.args.kd_coef[0] + gpt_loss* args.kd_coef[1] + rank_loss * args.kd_coef[2] + adaptive_kd_loss* self.args.kd_coef[3]

            


                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)

        self.model.train()  # reset model state
