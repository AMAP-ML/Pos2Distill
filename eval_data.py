import os
import argparse
import json
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
import numpy as np
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from gen_advantageous_pos_for_R1 import seed_everything,get_prompt_at_gold_index
cache_dir="/mnt/workspace/wangyifei/projects/huggingcache" # use your cache dir of model weights
seed = random.randint(0, 2**32 - 1)
seed_everything(seed)
def write_responses(output_path,examples, responses,prompts):
    with xopen(output_path, "w") as f:
        for (example,response,prompt) in zip(examples,responses,prompts):
            is_correct  = int(best_subspan_em(response.split("\n")[0],example["answers"]))
            example["current_answer"] = response
            example["current_prompt"] = prompt
            f.write(json.dumps(example) + "\n")
    average_metric_value = evaluate_qa_data(input_path=output_path)
    print(f"average_metric_value: {average_metric_value}")
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
    sample_num,
    gold_doc,
    total_doc,
    dataset_name

): 

    # os.makedirs(f"{output_path}/{total_doc}",exist_ok=True)
    os.makedirs(output_path,exist_ok=True)
    logger.info(f"model name {model_name}")
    if model_name.lower().endswith("instruct") or model_name.lower().endswith("chat") or model_name.lower().endswith("v0.3") or model_name.lower().endswith("7b") or model_name.lower().endswith("lite") or model_name.lower().endswith("hf"):
        model_path = os.path.join(cache_dir,model_name)
    else:
        model_path = os.path.join(f"checkpoints/nq",model_name)
    model = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus,
            # load_format="safetensors",
            # load_format="pt",
            trust_remote_code=True,
            max_num_batched_tokens=max_prompt_length,
        )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens,top_p=top_p,seed=seed)

    results_average = {"Model_name": model_name}
    results_std = {"Model_name": model_name}
    for i in range(0,total_doc+1,5):
        output_path_gold_index = os.path.join(output_path, f"{model_name}_{i}.jsonl.gz")
        average_iterations = 1
        average_metric_values = []
        for j in range(average_iterations):
            prompts, examples = get_prompt_at_gold_index(input_path, i, model_name, total_doc, model.get_tokenizer())
            prompts = prompts[-sample_num:]
            examples = examples[-sample_num:]
            if i==0:
            # compute average tokens
                stats = []
                for prompt in prompts:
                    tok=model.get_tokenizer()
                    tok.pad_token = tok.eos_token
                    prompt_len = tok(prompt,padding=True,add_special_tokens=False,return_tensors="pt")["input_ids"].size()[1]
                    stats.append(prompt_len)
                logger.info(f"tokens stats: max={max(stats)} min={min(stats)} avg={np.mean(stats)}")
            sampling_params.seed+=1
            raw_responses = model.generate(prompts, sampling_params)
            responses = [output.outputs[0].text.strip() for output in raw_responses] 


            average_metric_value = write_responses(output_path_gold_index, examples, responses,prompts)
            average_metric_values.append(average_metric_value)
        # 记录均值和标准差
        results_average[i] = np.mean(average_metric_values)
        results_std[i] = np.std(average_metric_values)
    # 转换为 DataFrame
    df_avg = pd.DataFrame([results_average])
    file_path = f"{dataset_name}_{total_doc}.xlsx"
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path, engine='openpyxl')
        combined_df = pd.concat([existing_df, df_avg], ignore_index=True)
    else:
        combined_df = pd.concat([df_avg], ignore_index=True)
    combined_df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"Results saved to {file_path}")
    return    
        

    

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.6)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
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
        default=4096,
    )
    parser.add_argument(
        "--sample_num",
        help="sample size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--top_p",
        help="top_p",
        type=float,
        default=1.0,
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
    parser.add_argument(
        "--dataset_name",
        type=str,
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
        args.sample_num,
        args.gold_doc,
        args.total_doc,
        args.dataset_name

    )
    logger.info("finished running %s", sys.argv[0])
