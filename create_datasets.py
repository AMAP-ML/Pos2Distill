from utils import best_subspan_em,read_xopen,write_xopen
from xopen import xopen
import json
from random import sample
import random
from copy import deepcopy
from collections import defaultdict
from datasets import Dataset
from collections import Counter
import itertools
import os
import numpy as np
import re
from metrics import compute_exact
from argparse import ArgumentParser
random.seed(42)
def best_subspan_em_musique(r,answers):
    match = re.search(r'the answer is[:：]?\s*(.*)', r.lower())
    if match is None:
        return int(0)
    a_pred = match.group(1)
    qa_em_score = 0
    for a_gold in answers:
        qa_em_score = max(qa_em_score, compute_exact(a_gold,a_pred))
    return int(qa_em_score)

def chunk_list(lst, chunk_size):
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]



def get_chunked_answer(model_answer):
    # chunked_answer = model_answer.strip("\n").strip(" ").split(".")[0] + "."
    chunked_answer = model_answer
    print(chunked_answer)
    return chunked_answer

def process_example(example,position_sample_num,strengthen=2,total_doc=20,train_counter=Counter()):
    sample_examples = []
    gold_doc = example["gold_document"]
    sample_pool = list(range(1,total_doc)) if strengthen!=0 else list(range(total_doc+1))
    random_pos = sample(sample_pool,position_sample_num)
    print(f"iteration1: {random_pos}")

        


    # 加上位置 0 的加强数据
    sample_positions = random_pos + [0] * strengthen
    for gold_idx in sample_positions:
        new_example = deepcopy(example)
        new_example["distractors"] = random.sample(example["distractors"],len(example["distractors"]))
        new_example["distractors"].insert(gold_idx,gold_doc)
        new_example["documents"]=deepcopy(new_example["distractors"])
        new_example["distractors"]=example["distractors"]
        new_example["gold_idx"] = gold_idx
        assert len(new_example["documents"])==total_doc
        assert new_example["documents"] != new_example["distractors"]
        new_example["oracle_answer"]=get_chunked_answer(example["oracle_answer"])
        sample_examples.append(new_example)
        train_counter[gold_idx] += 1
    
    return sample_examples,train_counter


# dict_keys(['question', 'answers', 'ctxs', 'nq_annotated_gold', 'gold_document', 'distractors', 'documents', 'prompt', 'model_answer'])
def generate_dataset(model_name,position_sample_num,total_doc=20,strengthen=2,train_size=200,position=0,filter=True,dataset_name="nq"):
    datasets_path = f"datasets/{dataset_name}"
    file_path = f"raw_data/{dataset_name}"
    file_name = f"{model_name}_{total_doc}docs_{position}.jsonl.gz"
    file_path = os.path.join(file_path,file_name)
    position_statistics = []
    examples = read_xopen(file_path)
    train_datasets = defaultdict(list)
    print(f"example nums: {len(examples)}")

    
    # process train_examples
    cnt = 0 
    train_counter = Counter({i: 0 for i in range(0,total_doc+1)})

    for idx,example in enumerate(examples):
        if cnt == train_size:
            break
        is_correct = best_subspan_em(get_chunked_answer(example["oracle_answer"]),example["answers"])
        # is_correct = best_subspan_em_musique(get_chunked_answer(example["oracle_answer"]),example["answers"])
        print(f"is_correct: {is_correct}")
        if filter:
            if int(is_correct)==0:
                continue
        if "predicted_label" not in example:
            example["predicted_label"] = "correct"
        if "predicted" not in example:
            example["predicted"] = example["answers"][0]
        if example["predicted_label"] == "incorrect":
            continue
        if not example["predicted"]:
            continue
        cnt+=1
        sample_examples,train_counter = process_example(example,position_sample_num,strengthen,total_doc,train_counter)
        assert len(sample_examples) == (position_sample_num + strengthen)
        train_datasets["data"].extend(sample_examples)
        # train_datasets["position_statistics"].append(random_pos)  
        # train_positions = list(itertools.chain.from_iterable(train_datasets["position_statistics"]))
           
    print(f"Train filter generate datapoints number: {len(train_datasets['data'])}")
    train_last_idx = idx
    assert len(train_datasets["data"]) == train_size*(position_sample_num + strengthen)
    filter_name = "filter" if filter else "unfilter"
    train_datasets_path = os.path.join(datasets_path,f"{model_name}_{total_doc}total_docs_{filter_name}_{position_sample_num}random_{strengthen}strengthen_{train_size}")
    train = Dataset.from_list(train_datasets["data"])
    train.save_to_disk(train_datasets_path)
    # save position_statistics
    print(f"{model_name}_{file_name}_{position_sample_num}_{strengthen}: {train_counter}")


    position_statistics = {f"train_{file_name}":train_counter}
    os.makedirs(datasets_path+"/position_statistics",exist_ok=True)
    with open(datasets_path+f"/position_statistics/{model_name}.json", "w") as f:
        json.dump(position_statistics, f,indent=1)
    return  

    
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Mistral-7B-Instruct-v0.3")
    parser.add_argument("--position_sample_num", type=int, default=4)
    parser.add_argument("--example_num", type=int, default=250)
    parser.add_argument("--dataset_name", type=str, default="nq")
    args = parser.parse_args()
    generate_dataset(args.model_name,position_sample_num=args.position_sample_num,total_doc=20,strengthen=1,train_size=args.example_num,position=0,filter=True,dataset_name=args.dataset_name)

