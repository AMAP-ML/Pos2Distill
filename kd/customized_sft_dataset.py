from typing import Callable
import pdb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from openrlhf.datasets.utils import zero_pad_sequences
import logging
import torch.distributed as dist
logger = logging.getLogger('customized sft dataset')
def get_teacher_student_prompt(data,multi_gold_doc=False):
    # logger.info(f"Get teacher student prompt")
    question = data["question"]
    distractors = data["distractors"]
    gold_doc = data["gold_document"]

    # key = "idx" if multi_gold_doc else "id"
    # distractor_indexs = [x[key] for x in distractors]
    # gold_indexs = [x[key] for x in gold_doc]
    
    if not multi_gold_doc:
        distractors.insert(0, gold_doc)
    else:
        distractors[len(distractors):len(distractors)] = gold_doc
    # complex_indexs = [x[key] for x in distractors]
    # assert complex_indexs[-len(gold_indexs):] == gold_indexs
    # print(f"complex_indexs: {complex_indexs}")
    # print(f"distractor_indexs: {distractor_indexs}")
    # print(f"gold_indexs: {gold_indexs}")

       

    teacher_documents = distractors
    student_documents = data["documents"]

    student_context = ''.join([f"- Title: {doc['title']}\n{doc['text']}\n" for doc in student_documents])
    teacher_context = ''.join([f"- Title: {doc['title']}\n{doc['text']}\n" for doc in teacher_documents])
    if not multi_gold_doc:
        task_instruction = "Please write a high-quantify answer for the given question using only the provided search documents (some of which might be irrelevant)."
    else:
        task_instruction = """Let’s first identify the relevant information from the long context and list it. Then, carry out step-by-step reasoning based on that information, and \
        finally, provide the answer. The final answer must end with “The answer is:”."""
    student_prompt = f"{task_instruction}\n{student_context}\nQuestion: {question}\n"
    teacher_prompt = f"{task_instruction}\n{teacher_context}\nQuestion: {question}\n"

    return student_prompt, teacher_prompt



def preprocess_data(
    data, input_template=None, input_key="input", output_key=None, apply_chat_template=None, multiturn=False,multi_gold_doc=False
):
    if apply_chat_template:
        # logger.info(f"Apply chat template")
        if output_key:
            # question = data[input_key]
            # oracle_answer = data[output_key]
            student_prompt, teacher_prompt = get_teacher_student_prompt(data,multi_gold_doc)
            if not multi_gold_doc:
                response = data[output_key].split("\n")[0]
            else:
                response = data[output_key]
           

            if isinstance(student_prompt, str) and isinstance(teacher_prompt, str) and isinstance(response, str):
                student_prompt_message = [{"role": "user", "content": student_prompt}]
                teacher_prompt_message = [{"role": "user", "content": teacher_prompt}]
                response_message = [{"role": "assistant", "content": response}]

            student_template_prompt = apply_chat_template(student_prompt_message, tokenize=False, add_generation_prompt=True)
            teacher_template_prompt = apply_chat_template(teacher_prompt_message, tokenize=False, add_generation_prompt=True)
            student_response = apply_chat_template(student_prompt_message + response_message, tokenize=False)[len(student_template_prompt):]
            teacher_response = apply_chat_template(teacher_prompt_message + response_message, tokenize=False)[len(teacher_template_prompt):]
            assert student_response == teacher_response
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""
    return student_template_prompt, teacher_template_prompt, student_response


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors= 48,  # Specify the number of processors you want to use
        multiple_of=1,
        multiturn=False,
        multi_gold_doc=False
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiple_of = multiple_of
        self.multiturn = multiturn
        self.multi_gold_doc = multi_gold_doc

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets

        processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )
        processed_dataset = processed_dataset.filter(lambda x: x["teacher_prompt"] is not None and x["student_prompt"] is not None and x["response"] is not None)

        # Store the processed data in class attributes
        self.student_prompts = processed_dataset["student_prompt"]
        self.teacher_prompts = processed_dataset["teacher_prompt"]
        self.responses = processed_dataset["response"]
        self.student_prompts_ids_lens = processed_dataset["student_prompt_ids_len"]
        self.teacher_prompts_ids_lens = processed_dataset["teacher_prompt_ids_len"]
        self.response_ranges = processed_dataset["response_ranges"] if self.multiturn else None
        self.gold_idx = processed_dataset["gold_idx"]

    def process_data(self, data):
        if self.multiturn and self.output_key:
            data[self.input_key].append(data[self.output_key])
            data[self.output_key] = None

        if self.multiturn:
            assert (
                not self.output_key or not data[self.output_key]
            ), "You should put the whole trajactory into data[input_key] and do not set output_key"
            input_key = self.input_key
            apply_chat_template = self.apply_chat_template
            response_ranges = []
            for idx, message in enumerate(data[input_key]):
                if message["role"] == "assistant":
                    prompt = apply_chat_template(data[input_key][:idx], tokenize=False, add_generation_prompt=True)
                    response = apply_chat_template(data[input_key][: idx + 1], tokenize=False)[len(prompt) :]

                    start_idx = (
                        self.tokenizer(
                            prompt,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                    )

                    end_idx = (
                        start_idx
                        + self.tokenizer(
                            response,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                        - 1
                    )
                    response_ranges.append((start_idx, end_idx))  # left close right open

        student_template_prompt, teacher_template_prompt, student_response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            multiturn=self.multiturn,
            multi_gold_doc=self.multi_gold_doc
        )

        if not self.pretrain_mode:
            student_prompt_token = self.tokenizer(
                student_template_prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            student_prompt_ids_len = student_prompt_token["attention_mask"].int().sum().item()

            teacher_prompt_token = self.tokenizer(
                teacher_template_prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            teacher_prompt_ids_len = teacher_prompt_token["attention_mask"].int().sum().item()

            # filter the sample whose length is greater than max_length (2 for answer length)
            if not student_template_prompt or not student_response or student_prompt_ids_len >= self.max_length - 2:
                student_template_prompt = None
        else:
            prompt_ids_len = 0

        return {
            "teacher_prompt": teacher_template_prompt,
            "student_prompt":student_template_prompt,
            "response": student_response,
            "student_prompt_ids_len": student_prompt_ids_len,
            "teacher_prompt_ids_len": teacher_prompt_ids_len,
            "response_ranges": response_ranges if self.multiturn else None,
            "gold_idx":data["gold_idx"]
        }

    def __len__(self):
        length = len(self.teacher_prompts)
        return length

    def __getitem__(self, idx):
        teacher_prompt = self.teacher_prompts[idx]
        student_prompt = self.student_prompts[idx]
        response = self.responses[idx]
        teacher_prompt_ids_len = self.teacher_prompts_ids_lens[idx]
        student_prompt_ids_len = self.student_prompts_ids_lens[idx]

        if not self.pretrain_mode:
            teacher_text = (teacher_prompt + response).rstrip("\n")
            student_text = (student_prompt + response).rstrip("\n")

            if not teacher_text.endswith(self.tokenizer.eos_token):
                teacher_text += " " + self.tokenizer.eos_token
            if not student_text.endswith(self.tokenizer.eos_token):
                student_text += " " + self.tokenizer.eos_token
        else:
            text = prompt

        teacher_input_token = self.tokenizer(
            teacher_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        student_input_token = self.tokenizer(
            student_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        if not self.pretrain_mode:
            # to avoid EOS_token truncation
            teacher_input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            teacher_input_token["attention_mask"][0][-1] = True

            student_input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            student_input_token["attention_mask"][0][-1] = True
        info = {
            "teacher_input": teacher_prompt,
            "student_input": student_prompt,
            "output": response,
            "teacher_input_length": teacher_input_token["attention_mask"].int().sum().item(),
            "student_input_length": student_input_token["attention_mask"].int().sum().item(),
            "response_ranges": self.response_ranges[idx] if self.multiturn else None,
            "gold_idx":self.gold_idx[idx]
        }
        # print(info)
        return teacher_input_token["input_ids"], teacher_input_token["attention_mask"], teacher_prompt_ids_len, \
             student_input_token["input_ids"], student_input_token["attention_mask"],student_prompt_ids_len, info

    def collate_fn(self, item_list):
        # logger.info(f"Collate fn ....")
        teacher_prompt_id_lens = []
        student_prompt_id_lens = []

        teacher_input_ids = []
        student_input_ids = []

        teacher_attention_masks = []
        student_attention_masks = []
        infos = {"teacher_prompt": [], "output": [],\
            "student_prompt": [],"gold_idx":[]}

        for  teacher_input_id, teacher_attention_mask, teacher_prompt_ids_len, \
            student_input_id, student_attention_mask, student_prompt_ids_len, info in item_list:

            teacher_prompt_id_lens.append(teacher_prompt_ids_len)
            student_prompt_id_lens.append(student_prompt_ids_len)
            teacher_input_ids.append(teacher_input_id)
            student_input_ids.append(student_input_id)
            teacher_attention_masks.append(teacher_attention_mask)
            student_attention_masks.append(student_attention_mask)
            infos["teacher_prompt"].append(info["teacher_input"])
            infos["student_prompt"].append(info["student_input"])
            infos["output"].append(info["output"])
            infos["gold_idx"].append(info["gold_idx"])


        teacher_input_ids = zero_pad_sequences(teacher_input_ids, "right", self.tokenizer.pad_token_id)
        teacher_attention_masks = zero_pad_sequences(teacher_attention_masks, "right")

        student_input_ids = zero_pad_sequences(student_input_ids, "right", self.tokenizer.pad_token_id)
        student_attention_masks = zero_pad_sequences(student_attention_masks, "right")
        return teacher_prompt_id_lens,teacher_input_ids,teacher_attention_masks, \
            student_prompt_id_lens,student_input_ids,student_attention_masks,infos

    def packing_collate_fn(self, item_list):
        packed_input_ids = []
        packed_attention_masks = []     
        prompt_ids_lens = []
        infos = {"input_length": [], "response_ranges": [] if self.multiturn else None}
        index = 1
        for prompt_ids_len, input_id, attention_mask, info in item_list:
            packed_input_ids.append(input_id.flatten())
            packed_attention_masks.append(torch.full_like(input_id.flatten(), index))
            prompt_ids_lens.append(prompt_ids_len)
            infos["input_length"].append(info["input_length"])
            if self.multiturn:
                if len(infos["response_ranges"]) >= 1:
                    for i in range(len(info["response_ranges"])):
                        info["response_ranges"][i][0] += infos["response_ranges"][-1][-1][
                            1
                        ]  # end_index of the last response of the last item
                        info["response_ranges"][i][1] += infos["response_ranges"][-1][-1][1]
                infos["response_ranges"].append(info["response_ranges"])
            index += 1

        packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(packed_attention_masks, dim=0).unsqueeze(0)

        if (
            self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0
        ):  # not divisible by multiple_of; here we align for grouping
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return prompt_ids_lens, packed_input_ids, packed_attention_masks, infos
