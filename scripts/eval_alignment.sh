#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE_DIR="${IMAGE_DIR:?Set IMAGE_DIR to a generated image directory}"
python compute_alignment_score.py \
  --image_dir "$IMAGE_DIR" \
  --save_intermediate \
  --num_workers "${NUM_WORKERS:-4}"
