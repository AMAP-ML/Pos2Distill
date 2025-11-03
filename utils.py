#!/usr/bin/env python3
import string
import re
from metrics import compute_exact,best_subspan_em
from typing import List
import regex
import argparse
import json
import logging
import statistics
import sys
from copy import deepcopy
import os
from tqdm import tqdm
from xopen import xopen
from pydantic.dataclasses import dataclass
from typing import List, Optional, Tuple, Type, TypeVar
logger = logging.getLogger(__name__)
def read_xopen(file_path):
    data = []
    with xopen(file_path) as fin:
        for line in fin:
            example = json.loads(line)
            data.append(example)
    return data
def write_xopen(file_path,data):
    with xopen(file_path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
                
def get_template_prompt(ins,model_name,tokenizer):
    # assert tokenizer.chat_template is not None
    context = ''.join([f"- Title: {doc['title']}\n{doc['text']}\n" for doc in ins["documents"]])
    task_instruction = "Please write a high-quantify answer for the given question using only the provided search documents (some of which might be irrelevant)."
    prompt_message = f"{task_instruction}\n{context}\nQuestion: {ins['question']}\n"
    system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    messages = [
        {"role": "user", "content": prompt_message},
        ]

    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    ins["prompt"] = prompt
    return ins


def get_metrics_for_example(example,METRICS):
    gold_answers = example["answers"]
    model_answer = example["current_answer"]
    # NOTE: we take everything up to the first newline, since otherwise models could hack
    model_answer = model_answer.strip("\n").split("\n")[0].strip()
    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)



def evaluate_qa_data(input_path,output_path=None,sample_num = None):
    METRICS = [(best_subspan_em, "best_subspan_em"),]
    all_examples = []
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            all_examples.append(input_example)
    if sample_num:
        all_examples = all_examples[:sample_num]
    # Compute normal metrics in parallel, if applicable
    logger.info("Computing metrics")
    all_example_metrics = []
    for example in tqdm(all_examples):
        all_example_metrics.append(get_metrics_for_example(example,METRICS))
    # Average metrics across examples
    for (_, metric_name) in METRICS:
        average_metric_value = statistics.mean(
            example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
        )
        print(f"{metric_name}: {average_metric_value}")
        logger.info(f"{metric_name}: {average_metric_value}")

    # summary_path = os.path.join(os.path.dirname(input_path),"A_metrics_summary.txt")
    # with xopen(summary_path,"a") as f:
    #     f.write(f"{input_path.split('/')[-1].split('.jsonl.gz')[0]}\n{metric_name}: {average_metric_value}\n\n")
    if output_path:
        with xopen(output_path, "w") as f:
            for (example_metrics, example) in all_example_metrics:
                example_with_metrics = deepcopy(example)
                for metric_name, metric_value in example_metrics.items():
                    example_with_metrics[f"metric_{metric_name}"] = metric_value
                f.write(json.dumps(example_with_metrics) + "\n")
    return average_metric_value