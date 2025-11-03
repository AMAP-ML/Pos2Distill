total_doc=20
num_gpus=8
# export CUDA_VISIBLE_DEVICES=3
dataset_name="nq"

INPUT_PATH="nq-open-${total_doc}_total_documents.jsonl.gz"
model_name=Mistral-7B-Instruct-v0.3
dataset_name=nq
python gen_advantageous_pos_for_R1.py \
        --input-path "$INPUT_PATH" \
        --model $model_name \
        --output-path raw_data/$dataset_name \
        --max-prompt-length 32768 \
        --max-new-tokens 100 \
        --num_gpus "$num_gpus" \
        --total_doc "$total_doc" \
        --cache_dir $cache_dir \
        --gold_doc 0


model_name=Mistral-7B-Instruct-v0.3
dataset_name=nq
python create_datasets.py \
        --model_name $model_name \
        --position_sample_num 4 \
        --example_num 400 \
        --dataset_name $dataset_name