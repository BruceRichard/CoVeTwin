#!/bin/bash
# Non-SLURM Qwen2.5-VL-7B full-parameter training on eight RTX 4090 GPUs.
# Run from qwen-vl-finetune/: bash scripts/run_sft_7b_local.sh
set -e

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=8

# Force CUDA 11.8 for the cu118 PyTorch environment and CUDA extension builds.
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_PATH=${CUDA_HOME}
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

# Use the Hugging Face mirror for checkpoint downloads.
export HF_ENDPOINT=https://hf-mirror.com

# Reduce CUDA allocator fragmentation.
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUM_GPUS=8
export NNODES=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((RANDOM % 101 + 20000))
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed=./scripts/zero3.json           # Use zero3_offload.json if memory is insufficient.
llm=Qwen/Qwen2.5-VL-7B-Instruct          # The first run downloads about 16 GB.

lr=2e-5
batch_size=1
grad_accum_steps=8
entry_file=qwenvl/train/train_qwen.py

datasets=physxmobility_v2
output_dir=./output_7b_mobility_v2codec
run_name=qwen2vl_7b_physxmobility_v2codec

# Resolution and sequence-length settings.
# max_pixels=262144
# min_pixels=65536
# model_max_length=8192
max_pixels=65536
min_pixels=16384
model_max_length=2048

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 30 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to none"

torchrun --nproc_per_node=${NUM_GPUS} \
         --nnodes=${NNODES} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
