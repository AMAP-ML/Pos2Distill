import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import numpy as np
import logging
import pathlib
import random
import sys
from copy import deepcopy
import torch
from tqdm import tqdm
from transformers import AutoTokenizer,LlamaTokenizer
from xopen import xopen
from vllm import LLM, SamplingParams
import pandas as pd
import random
import itertools
from utils import *
logger = logging.getLogger(__name__)




def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
def get_prompt_at_gold_index(input_path,gold_doc,model_name,total_doc,tokenizer=None):
    examples = []
    prompts = []
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            gold_document = input_example["gold_document"]
            distractors = input_example["distractors"]
            random.shuffle(distractors)
            # Get the prediction for the input example
            
            shuffled_distractors = distractors[:]
            shuffled_distractors.insert(gold_doc, gold_document)

            ins = deepcopy(input_example)
            ins["documents"] = shuffled_distractors if total_doc >0 else []
            assert len(ins["documents"])==total_doc
            prompt = get_template_prompt(ins,model_name,tokenizer)["prompt"]
            prompts.append(prompt)
            examples.append(ins)
    return prompts,examples 

def write_responses(output_path,examples, responses,prompts):
    with xopen(output_path, "w") as f:
        for (example,response,prompt) in zip(examples,responses,prompts):
            example["oracle_answer"] = response
            example["oracle_prompt"] = prompt
            f.write(json.dumps(example) + "\n")
    average_metric_value = evaluate_qa_data(input_path=output_path)
    print(f"acc when gold doc at 0: {average_metric_value}")
    return average_metric_value

def main(
    input_path,
    model_name,
    temperature,
    top_p,
    num_gpus,
    max_new_tokens,
    max_prompt_length,
    output_path,
    gold_doc,
    total_doc,
    cache_dir
): 
    os.makedirs(output_path, exist_ok=True)
    output_path_gold_index = os.path.join(output_path,f"{model_name}_{total_doc}docs_{gold_doc}.jsonl.gz")
    model_path = os.path.join(cache_dir,model_name)
    model = LLM(
            model=os.path.join(cache_dir,model_name),
            tensor_parallel_size=num_gpus,
            # load_format="safetensors",
            # load_format="pt",
            max_num_batched_tokens=max_prompt_length,
        )
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_new_tokens,top_p=top_p)
    output_path_gold_index = os.path.join(output_path,f"{model_name}_{total_doc}docs_{gold_doc}.jsonl.gz")
    if "mistral" in model_name.lower():
        prompts,examples = get_prompt_at_gold_index(input_path,total_doc,model_name,total_doc,model.get_tokenizer())
    else:
        prompts,examples = get_prompt_at_gold_index(input_path,0,model_name,total_doc,model.get_tokenizer())
    raw_responses = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in raw_responses] 
    write_responses(output_path_gold_index,examples,responses,prompts)
    return    
        

    

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
        choices=[
            "Mistral-7B-Instruct-v0.3",   
            "Qwen1.5-7B-Chat",
            "Meta-Llama-3-8B-Instruct",
            "Qwen2.5-7B-Instruct"
        ]
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.6)
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument("--cache_dir", help="Path to huggingface cache to use.")
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-prompt-length",
        help="Maximum number of tokens in the prompt. Longer prompts will be skipped.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--top_p",
        help="top_p",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--gold_doc",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--total_doc",
        type=int,
    )

    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model,
        args.temperature,
        args.top_p,
        args.num_gpus,
        args.max_new_tokens,
        args.max_prompt_length,
        args.output_path,
        args.gold_doc,
        args.total_doc,
        args.cache_dir
    )
    logger.info("finished running %s", sys.argv[0])
