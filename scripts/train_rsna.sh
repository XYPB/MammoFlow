#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export MODEL_NAME="${MODEL_NAME:-stabilityai/stable-diffusion-3.5-medium}"
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-config/1gpu_no_ds_config.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/sd35-mv-sbs-fft-rsna-better-contrast-emd-1e-1-cosine}"
export WANDB_NAME="${WANDB_NAME:-sd35-mv-sbs-fft-rsna-better-contrast-emd-1e-1-cosine}"
export REPORT_TO="${REPORT_TO:-wandb}"

accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" train_sd35_lora_mv.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --dataset_name="rsna-mammo" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=40000 \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --short_caption \
  --better_caption \
  --same_side \
  --seed=42 \
  --full_finetune \
  --compute_alignment_emd \
  --alignment_emd_lambda=0.1 \
  --emd_loss_weight_schedule="cosine" \
  --contrast_enhance \
  --num_validation_images=32 \
  --report_to="$REPORT_TO" \
  --checkpointing_steps=5000 \
  --resume_from_checkpoint="latest" \
  --dataloader_num_workers=4 \
  --output_dir="$OUTPUT_DIR"
