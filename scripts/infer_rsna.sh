#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_NAME="${MODEL_NAME:-stabilityai/stable-diffusion-3.5-medium}"
RUN_DIR="${RUN_DIR:-runs/sd35-mv-sbs-fft-rsna-better-contrast-emd-1e-1-cosine}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/inference_100_1k}"

python inference_sd35_lora.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --full_finetune \
  --finetuned_transformer_path="$RUN_DIR/transformer" \
  --dataset_name="rsna-mammo" \
  --same_side \
  --short_caption \
  --better_caption \
  --resolution=512 \
  --cancer_ratio=0.5 \
  --num_inference_images="${NUM_INFERENCE_IMAGES:-1000}" \
  --output_dir="$OUTPUT_DIR" \
  --seed=42 \
  --mixed_precision="fp16" \
  --num_inference_steps="${NUM_INFERENCE_STEPS:-100}" \
  --paired_views \
  --batch_size="${BATCH_SIZE:-16}"
