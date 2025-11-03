total_docs=20
num_gpus=8
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7

random_num=4
strengthen=1
INPUT_PATH="nq-open-${total_docs}_total_documents.jsonl.gz"
# model_name=Mistral-7B-Instruct-v0.3_20total_docs_filter_4random_1strengthen_300_kd0.0_lm0.0_rank0.0_adaptive1.0_1.0
# model_name=Mistral-7B-Instruct-v0.3_20total_docs_filter_4random_1strengthen_400_kd0.0_lm0.0_rank0.0_adaptive1.0_1.0
# /mnt/workspace/wangyifei/miniconda3/envs/openrlhf/bin/python eval_data.py \
#         --input-path "$INPUT_PATH" \
#         --model $model_name \
#         --output-path evaluate \
#         --sample_num 500 \
#         --max-prompt-length 32768 \
#         --max-new-tokens 100 \
#         --num_gpus "$num_gpus" \
#         --total_doc "$total_docs" \

model_name=Mistral-7B-Instruct-v0.3
INPUT_PATH="webq_dev.jsonl.gz"
total_docs=20
num_gpus=8
python eval_data.py \
        --input-path "$INPUT_PATH" \
        --model $model_name \
        --output-path evaluate \
        --sample_num 500 \
        --max-prompt-length 32768 \
        --max-new-tokens 100 \
        --num_gpus "$num_gpus" \
        --total_doc "$total_docs" \

