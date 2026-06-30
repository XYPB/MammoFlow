# MammoFlow

[![arXiv:2606.28537](https://img.shields.io/badge/arXiv-2606.28537-B31B1B.svg)](https://arxiv.org/abs/2606.28537)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **MammoFlow: Multiview Mammogram Synthesis with Anatomically Consistent Flow Matching** ([arXiv](https://arxiv.org/abs/2606.28537) | [Project Page](https://xypb.github.io/MammoFlow/)). This repository contains the release path for data preprocessing, SD3.5 multiview training, experiment logging, generation inference, alignment evaluation, and downstream classification tasks.

This code is for research use only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## Installation

The released experiments were developed with Python 3.10, PyTorch 2.5.1, CUDA 12.4, Hugging Face Diffusers/Accelerate, and NVIDIA A100/H200-class GPUs. The commands below install both the PyTorch CUDA 12.4 runtime and the CUDA 12.4 compiler toolkit (`nvcc`).

```bash
conda create -n mammoflow python=3.10 pip -y
conda activate mammoflow
conda install --override-channels -c pytorch -c nvidia -c conda-forge \
  pytorch::pytorch=2.5.1 \
  pytorch::torchvision=0.20.1 \
  pytorch::torchaudio=2.5.1 \
  pytorch::pytorch-cuda=12.4 \
  nvidia::cuda-toolkit=12.4.1 \
  nvidia::cuda-command-line-tools=12.4.1 \
  nvidia::cuda-tools=12.4.1 \
  nvidia::cuda-compiler=12.4.1 \
  nvidia::cuda-nvcc=12.4.131
pip install -r requirements.txt
```

The same conda pins are also recorded in `environment.yml`. If your conda installation honors `nodefaults` without consulting Anaconda defaults, you can create the environment from that file while standing in the repository root:

```bash
conda env create -f environment.yml
conda activate mammoflow
```

Verify the CUDA versions after installation:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
nvcc --version  # should report release 12.4
```

Note: the paper-matching PyTorch 2.5.1/CUDA 12.4 stack supports NVIDIA Ampere/Ada/Hopper GPUs. Newer Blackwell or RTX 50-series GPUs may require a newer PyTorch/CUDA stack for local experimentation, which will not exactly match the reported training environment.

Stable Diffusion 3.5 requires access to the upstream model weights. Log in with Hugging Face before training or inference:

```bash
huggingface-cli login
# or
export HF_TOKEN=your_huggingface_token
```

Weights, logs, generated images, and datasets are intentionally ignored by git.

## Data Preparation

MammoFlow does not redistribute CSAW, VinDr-Mammo, RSNA, mammogram images, private metadata, or derived patient data. Download each dataset from its official source and follow its license and data-use terms.

The training code expects paired CC/MLO data with this structure for local-directory datasets:

```text
data/<dataset-name>/
  CC/train/*.png or *.jpg
  CC/val/*.png or *.jpg
  CC/test/*.png or *.jpg
  MLO/train/*.png or *.jpg
  MLO/val/*.png or *.jpg
  MLO/test/*.png or *.jpg
```

Prepare CSAW pairs:

```bash
python prepare_csaw_dataset.py \
  --image_dir /path/to/csaw/images \
  --train_metadata /path/to/metadata_visible_train.jsonl \
  --test_metadata /path/to/metadata_visible_test.jsonl \
  --output_dir data/csaw-paired
```

Prepare VinDr-Mammo pairs:

```bash
python prepare_vindr_dataset.py \
  --annotations_csv /path/to/breast-level_annotations.csv \
  --source_image_dir /path/to/vindr-resized/images \
  --output_dir data/vindr-paired
```

Prepare RSNA pairs:

```bash
python prepare_rsna_dataset.py \
  --source_image_dir /path/to/RSNA_MAMMO_1080_JPG \
  --train_csv /path/to/rsna_mammo_train.csv \
  --test_csv /path/to/rsna_mammo_test.csv \
  --output_dir data/rsna-paired
```

## Training

The paper configuration is SD3.5 medium with same-side multiview side-by-side training, better captions, full transformer finetuning, EMD alignment loss with lambda 0.1, cosine EMD schedule, 40k steps, fp16, and seed 42.

CSAW:

```bash
export DATASET_DIR=/path/to/csaw-paired/train
bash scripts/train_csaw.sh
```

VinDr-Mammo:

```bash
bash scripts/train_vindr.sh
```

RSNA:

```bash
bash scripts/train_rsna.sh
```

Useful environment overrides:

```bash
export MODEL_NAME=stabilityai/stable-diffusion-3.5-medium
export ACCELERATE_CONFIG_FILE=config/1gpu_no_ds_config.yaml
export CUDA_VISIBLE_DEVICES=0
export REPORT_TO=wandb
export OUTPUT_DIR=runs/my_mammoflow_run
```

The default script names match the camera-ready experiments:

```text
sd35-mv-sbs-fft-csaw-better-40k-emd-1e-1-cosine
sd35-mv-sbs-fft-vindr-better-40k-emd-1e-1-cosine
sd35-mv-sbs-fft-rsna-better-contrast-emd-1e-1-cosine
```

## Inference

Run generation from a trained full-finetune transformer checkpoint:

```bash
export RUN_DIR=runs/sd35-mv-sbs-fft-csaw-better-40k-emd-1e-1-cosine
export DATASET_DIR=/path/to/csaw-paired/train
bash scripts/infer_csaw.sh
```

For VinDr-Mammo and RSNA:

```bash
export RUN_DIR=runs/sd35-mv-sbs-fft-vindr-better-40k-emd-1e-1-cosine
bash scripts/infer_vindr.sh

export RUN_DIR=runs/sd35-mv-sbs-fft-rsna-better-contrast-emd-1e-1-cosine
bash scripts/infer_rsna.sh
```

Common overrides:

```bash
export NUM_INFERENCE_IMAGES=1000
export NUM_INFERENCE_STEPS=100
export BATCH_SIZE=16
export OUTPUT_DIR=$RUN_DIR/inference_100_1k
```

## Evaluation and Downstream Tasks

Compute multiview alignment scores for generated images:

```bash
export IMAGE_DIR=runs/sd35-mv-sbs-fft-vindr-better-40k-emd-1e-1-cosine/inference_100_1k
bash scripts/eval_alignment.sh
```

FID can be computed with `pytorch-fid` after arranging generated and reference images into comparable folders:

```bash
python -m pytorch_fid /path/to/real_images /path/to/generated_images
```

Downstream classifiers are provided through:

```bash
python train_classification.py --help
python train_classification_mv.py --help
```

Use `--use_wandb` to enable Weights & Biases logging for downstream classification. Training and generation scripts use the Diffusers/Accelerate logging stack and write checkpoints under `runs/` by default.

## Pretrained Weights

We are working on the pretrained model, and it will be released soon.

## Citation

```bibtex
@inproceedings{mammoflow2026,
  title     = {MammoFlow: Multiview Mammogram Synthesis with Anatomically Consistent Flow Matching},
  author    = {Yuexi Du and Leya Barrientos and Laura Sheiman and John Lewin and Hemant D. Tagare and Nicha C. Dvornek},
  booktitle = {International Conference on Medical Image Computing and Computer Assisted Intervention},
  year      = {2026},
  eprint    = {2606.28537},
  archivePrefix = {arXiv}
}
```

## License

This code is released under the MIT License. See `LICENSE` for details.
