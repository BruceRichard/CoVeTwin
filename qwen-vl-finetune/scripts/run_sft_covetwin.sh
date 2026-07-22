#!/usr/bin/env bash
# Fine-tune Qwen2.5-VL on the two-stage CoVeTwin conversations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${QWEN_ROOT}/.." && pwd)"
cd "${QWEN_ROOT}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NUM_GPUS="${NUM_GPUS:-8}"
export NNODES="${NNODES:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29517}"
export COVETWIN_ANNOTATION_PATH="${COVETWIN_ANNOTATION_PATH:-${REPO_ROOT}/dataset/covetwin_training/conversations.json}"
export COVETWIN_IMAGE_ROOT="${COVETWIN_IMAGE_ROOT:-${REPO_ROOT}/dataset_toolkits/renders_all}"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_covetwin_7b}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./scripts/zero3.json}"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --nnodes="${NNODES}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  qwenvl/train/train_qwen.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --model_name_or_path "${MODEL}" \
  --dataset_use covetwin \
  --data_flatten True \
  --tune_mm_vision False \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --bf16 \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${EPOCHS:-30}" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-8}" \
  --max_pixels "${MAX_PIXELS:-262144}" \
  --min_pixels "${MIN_PIXELS:-65536}" \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-300}" \
  --save_total_limit 1 \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length "${MODEL_MAX_LENGTH:-8192}" \
  --gradient_checkpointing True \
  --dataloader_num_workers "${DATALOADER_WORKERS:-8}" \
  --run_name covetwin_qwen2_5_vl_7b \
  --report_to none
