############################ RUNNING CONFIG ############################ 
# export LD_PRELOAD="/usr/lib64/libjemalloc.so.2"
# sudo mount -o size=20480M -o nr_inodes=1000000 -o noatime,nodiratime -o remount /dev/shm
# if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
#     . "/opt/conda/etc/profile.d/conda.sh"
# else
#     export PATH="/opt/conda/bin:$PATH"
# fi
# conda activate /mnt/workspace/wangyifei/miniconda3/envs/openrlhf
# workdir="/mnt/workspace/wangyifei/projects/self_training"
# cd $workdir
############################ RUNNING CONFIG ############################ 




export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}
export NCCL_DEBUG=ERROR
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=PIX
export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
#Meta-Llama-3-8B-Instruct Qwen1.5-7B-Chat 0.1 0.2 0.3 0.4 0.0 0.6 0.7 0.8 0.9 1.0
# /mnt/workspace/wangyifei/projects/self_training/kd/eval_kl.sh



strengthen=1
total_docs=20
dataset_name="nq"
for i in  0 ; do
   for model_name in Mistral-7B-Instruct-v0.3;  do
      for num in  400; do
            for kd_coef in "0.0,0.0,0.0,1.0" ; do
               for K in  4 ; do
                  deepspeed --master_port 6666 \
                        kd/train_kd.py \
                     --max_len 32000 \
                     --dataset ${model_name}_${total_docs}total_docs_filter_${K}random_${strengthen}strengthen_${num}\
                     --input_key question \
                     --output_key  oracle_answer \
                     --train_batch_size 32 \
                     --micro_train_batch_size  4 \
                     --max_samples 500000 \
                     --pretrain $model_name \
                     --teacher_model $model_name \
                     --save_path checkpoints/${dataset_name}\
                     --save_steps -1 \
                     --logging_steps 1 \
                     --eval_steps -1 \
                     --zero_stage 3 \
                     --max_epochs 1 \
                     --l2 0.01 \
                     --bf16 \
                     --flash_attn \
                     --kd_coef $kd_coef \
                     --learning_rate 3e-6 \
                     --teacher_offload \
                     --apply_chat_template \
                     --gradient_checkpointing  \
                     --perserve 1.0 \
                     --dataset_name nq \
                     --use_tensorboard tensorBoard
                     # --use_wandb "c0c629e7ba14b453e7da5da4ff86f33816c5cc6a" # > training_kl2.log 2>&1
               done
            done
      done
   done
done
