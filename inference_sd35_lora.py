#!/usr/bin/env python
# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stable Diffusion 3.5 Text-to-Image LoRA Inference Script

A script for generating images using fine-tuned Stable Diffusion 3.5 models with LoRA adapters.
Prompts are sampled from the training dataset.

"""

import argparse
import os
import random
import copy
import re
import numpy as np
import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
)

import diffusers
from diffusers import (
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
)
from diffusers.utils.torch_utils import is_compiled_module

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import datasets
from datasets import load_dataset
from dataset.transforms import OtsuCut, RemoveTextLabel
from dataset.embed import EmbedDiffusionDataset
from dataset.vindr import VinDr
from dataset.rsna_mammo import RSNAMammo

from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from tqdm.auto import tqdm


def parse_args():
    """Parse command line arguments for SD3.5 LoRA inference."""
    parser = argparse.ArgumentParser(
        description="Stable Diffusion 3.5 LoRA Inference"
    )

    # ═══════════════════════════════════════════════════════════
    # Model Configuration
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained SD3.5 model or HuggingFace Hub identifier",
    )
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        default=None,
        help="Path to the directory containing the trained LoRA weights",
    )
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Whether the model was fully fine-tuned (no LoRA).",
    )
    parser.add_argument(
        "--finetuned_transformer_path",
        type=str,
        default=None,
        help="Path to the fine-tuned transformer directory (required if --full_finetune is set)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Specific model revision to use (branch name, tag, or commit hash)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model weight variant (e.g., 'fp16')",
    )
    parser.add_argument(
        "--log_steps",
        action="store_true",
        help="Log intermediate steps during inference (for debugging).",
    )

    # ═══════════════════════════════════════════════════════════
    # Dataset Configuration (for prompt sampling)
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace Dataset name for training data",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Local directory containing metadata.jsonl and images folder",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="Column name for image paths in the dataset",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="Column name for captions in the dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache HuggingFace models and datasets",
    )
    parser.add_argument(
        "--short_caption",
        action="store_true",
        help="Whether to use a short caption format.",
    )
    parser.add_argument(
        "--ablate_caption",
        default=None,
        type=str,
        help="Caption contents to include in ablation."
    )
    parser.add_argument(
        "--otsu_cut",
        action="store_true",
        help="Whether to use Otsu's method to cut the breast region.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Training image resolution (used for dataset loading)",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Use center crop instead of random crop for image preprocessing",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Apply random horizontal flip augmentation during training",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Normalize images to [0, 1] range.",
    )
    parser.add_argument(
        "--paired_views",
        action="store_true",
        help="Whether to use paired multi-view images as output.",
    )
    parser.add_argument(
        "--same_side",
        action="store_true",
        help="Whether to use same side multi-view images as output.",
    )
    parser.add_argument(
        "--stack_ch",
        action="store_true",
        help="Whether to stack multi-view images as channels.",
    )
    parser.add_argument(
        "--four_views",
        action="store_true",
        help="Whether to use 4-view multi-view images as output.",
    )
    parser.add_argument(
        "--cancer_ratio",
        type=float,
        default=-1.0,
        help="Ratio of cancer to non-cancer images in the dataset",
    )
    parser.add_argument(
        "--better_caption",
        action="store_true",
        help="Use more detailed captions for training.",
    )
    parser.add_argument(
        "--strict_prompt",
        action="store_true",
        help="Use strict prompt filtering for training data.",
    )
    parser.add_argument(
        "--sampled_strict_prompt",
        action="store_true",
        help="Use subsampled strict prompt filtering for training data.",
    )
    parser.add_argument(
        "--sampled_10pct_strict_prompt",
        action="store_true",
        help="Use subsampled strict prompt filtering for training data.",
    )
    parser.add_argument(
        "--health_only",
        action="store_true",
        help="Whether to use only healthy images for training.",
    )
    parser.add_argument(
        "--cancer_only",
        action="store_true",
        help="Whether to use only cancerous images for training.",
    )
    parser.add_argument(
        "--cc_first",
        action="store_true",
        help="Whether to force paired same-side loading order to CC first then MLO.",
    )

    # ═══════════════════════════════════════════════════════════
    # Inference Configuration
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--num_inference_images",
        type=int,
        default=4,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_outputs",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Guidance scale",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to compile the model with torch.compile (if supported)",
    )

    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Must specify either --dataset_name or --train_data_dir")
    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Cannot specify both --dataset_name and --train_data_dir")

    if args.full_finetune:
        if args.finetuned_transformer_path is None:
            raise ValueError("Must specify --finetuned_transformer_path when --full_finetune is set")
    else:
        if args.lora_weights_path is None:
            raise ValueError("Must specify --lora_weights_path when --full_finetune is not set")

    return args


def extract_info_from_caption(caption):
    pattern = re.compile(
        r"""^This\s+is\s+a\s+high-resolution\s+2D\s+full-field\s+screening\s+
            (?P<view>.+?)\s+view\s+mammogram\s+of\s+the\s+
            (?P<laterality>.+?)\s+breast\s+with\s+
            (?P<cancer>.+?)(?:\.)?$""",
        re.IGNORECASE | re.VERBOSE
    )
    m = pattern.search(caption)
    if m:
        view = m.group("view").strip().lower()
        laterality = m.group("laterality").strip().lower()
        cancer = m.group("cancer").strip().lower()
        return view, laterality, cancer
    else:
        return "unknown", "unknown", "unknown"


def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # Setup logging
    logger = get_logger(__name__)
    
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # Dataset Loading (for prompts)
    # ═══════════════════════════════════════════════════════════
    logger.info("Loading dataset to sample prompts...")
    
    if args.dataset_name in ['embed', 'vindr-mammo', 'rsna-mammo']:
        from types import SimpleNamespace
        
        dataset_args = {
            'pred_density': False,
            'pred_mass': True,
            'pred_calc': False,
            'screen_only': True,
            'data_pct': 1.0,
            'aug_text': False,
            'heavy_aug': False,
            'structural_cap': True,
            'save_mask': False,
            'health_only': args.health_only,
            'cancer_only': args.cancer_only,
            'cc_first': args.cc_first,
        }
        
        dataset_args = SimpleNamespace(**dataset_args)
        if args.dataset_name == 'embed':
            transform = transforms.Compose(
                [
                    RemoveTextLabel(),
                    transforms.Resize(
                        (args.resolution, args.resolution)
                    ),
                    transforms.ToTensor(),
                    transforms.RandomApply([transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))], p=1 if not args.no_normalize else 0),
                ]
            )
            dataset = EmbedDiffusionDataset(dataset_args, split='test', transform=transform)
        elif args.dataset_name == 'vindr-mammo':
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (args.resolution, args.resolution)
                    ),
                    transforms.ToTensor(),
                    transforms.RandomApply([transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))], p=1 if not args.no_normalize else 0),
                ]
            )
            dataset = VinDr(dataset_args, split='test', transform=transform,
                            short_caption=args.short_caption, paired_view=args.paired_views, same_side=args.same_side, four_view=args.four_views)
        elif args.dataset_name == 'rsna-mammo':
            transform = transforms.Compose(
                [
                    OtsuCut() if args.otsu_cut else transforms.Lambda(lambda x: x),
                    transforms.Resize(
                        (args.resolution, args.resolution)
                    ),
                    transforms.ToTensor(),
                    transforms.RandomApply([transforms.Normalize(mean=(0.5,), std=(0.5,))], p=1 if not args.no_normalize else 0),
                ]
            )
            dataset = RSNAMammo(dataset_args, split='test', transform=transform, short_caption=args.short_caption, paired_view=args.paired_views, same_side=args.same_side, four_view=args.four_views)
        
        # For these datasets, we can access prompts directly via __getitem__
        # But we need to know the length to sample indices
        dataset_len = len(dataset)
        
    else:
        # Load dataset from HuggingFace Hub or local directory
        if args.strict_prompt:
            data_files = {
                "train": os.path.join(args.train_data_dir, "metadata_visible.jsonl")
            }
        elif args.health_only:
            data_files = {
                "train": os.path.join(args.train_data_dir, "metadata_healthy.jsonl")
            }
        elif args.cancer_only:
            data_files = {
                "train": os.path.join(args.train_data_dir, "metadata_visible_cancer.jsonl")
            }
        elif args.sampled_strict_prompt:
            data_files = {
                "train": os.path.join(
                    args.train_data_dir, "metadata_visible_resampled.jsonl"
                )
            }
        elif args.sampled_10pct_strict_prompt:
            data_files = {
                "train": os.path.join(
                    args.train_data_dir, "metadata_visible_resampled_10pct.jsonl"
                )
            }
        else:
            data_files = {"train": os.path.join(args.train_data_dir, "metadata.jsonl")}
        hf_dataset = load_dataset("json", data_files=data_files, cache_dir=args.cache_dir)

        dataset = hf_dataset["train"]
        column_names = dataset.column_names
        caption_column = (
            args.caption_column if args.caption_column in column_names else column_names[1]
        )
        dataset_len = len(dataset)

    # Sample prompts
    logger.info(f"Sampling {args.num_inference_images} prompts from dataset of size {dataset_len}...")
    indices = random.choices(range(dataset_len), k=args.num_inference_images)
    
    prompts = []
    if args.cancer_only:
        args.cancer_ratio = 1.0
    elif args.health_only:
        args.cancer_ratio = 0.0
    for idx in indices:
        if args.dataset_name in ['embed', 'vindr-mammo', 'rsna-mammo']:
            item = dataset.__getitem__(idx)
            caption = item['prompts']
        else:
            caption = dataset[idx][caption_column]
            
            if args.short_caption:
                view, laterality, cancer = extract_info_from_caption(caption)
                if random.random() < args.cancer_ratio:
                    cancer = "cancer"
                if args.paired_views:
                    if args.same_side:
                        if args.better_caption:
                            if view == 'cc':
                                other_view = 'mlo'
                            elif view == 'mlo':
                                other_view = 'cc'
                            # force to generate cc view first then mlo view for better caption
                            if args.cc_first:
                                view = 'cc'
                                other_view = 'mlo'
                            # caption = f"two x-ray of {laterality} breast with {cancer}, the first image is {view} view, the other image is {other_view} view."
                            if cancer.startswith("no "):
                                caption = f'Two healthy {laterality} breast mammograms, the first image is {view} view and the second image is the {other_view} view.'
                            else:
                                caption = f'Two {laterality} side breast mammograms with malignant cancer, the first image is {view} view and the second image is the {other_view} view.'
                        else:
                            caption = f"a stacked x-ray of {laterality} breast with {cancer}"
                    else:
                        if args.better_caption:
                            if laterality == 'left':
                                other_side = 'right'
                            elif laterality == 'right':
                                other_side = 'left'
                            if cancer.startswith("no "):
                                caption = f'Two healthy {view} view breast mammograms, the first image is the {laterality} breast and the second image is the {other_side} breast.'
                            else:
                                caption = f'Two {view} view breast mammograms with malignant cancer, the first image is the {laterality} breast and the second image is the {other_side} breast.'
                        else:
                            caption = f"a stacked {view} view x-ray of both side of breast with {cancer}"
                elif args.four_views:
                    caption = f"four views of breast x-ray with {cancer}, left column is {view} view and first row is {laterality} breast."
                else:
                    # caption = f"a {view} view x-ray of {laterality} breast with {cancer}"
                    if cancer == 'no cancer':
                        caption = f'This is a healthy {laterality} breast {view} view mammogram. '
                    else:
                        caption = f'This is a {laterality} breast {view} view mammogram with malignant cancer. '
            elif args.ablate_caption is not None:
                view, laterality, cancer = extract_info_from_caption(caption)
                cur_caption = 'a '
                if 'view' in args.ablate_caption:
                    cur_caption += f'{view} view '
                if 'laterality' in args.ablate_caption:
                    cur_caption += f'{laterality} side '
                cur_caption += 'breast mammogram '
                if 'cancer' in args.ablate_caption:
                    cur_caption += f'with {cancer}'
                caption = cur_caption.strip()
        
        prompts.append(caption)

    logger.info(f"Sampled prompts: {prompts}")

    # ═══════════════════════════════════════════════════════════
    # Model Loading
    # ═══════════════════════════════════════════════════════════
    logger.info("Loading model...")
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    
    if args.full_finetune:
        logger.info(f"Loading fine-tuned transformer from {args.finetuned_transformer_path}...")
        transformer = SD3Transformer2DModel.from_pretrained(
            args.finetuned_transformer_path,
            torch_dtype=weight_dtype,
        )
        pipeline.transformer = transformer
    else:
        # Load LoRA weights
        logger.info(f"Loading LoRA weights from {args.lora_weights_path}...")
        pipeline.load_lora_weights(args.lora_weights_path)
    
    pipeline.to(accelerator.device)

    if args.compile:
        # Compile transformer for faster inference
        pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)
        # Optionally compile VAE decoder too
        pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

    # ═══════════════════════════════════════════════════════════
    # Inference Loop
    # ═══════════════════════════════════════════════════════════
    logger.info("Starting inference...")
    
    # Process in batches
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    
    # generated_images = []
    
    tgt_height = args.resolution
    tgt_width = args.resolution
    if args.paired_views and not args.stack_ch:
        tgt_width *= 2
    if args.four_views:
        tgt_height *= 2
        tgt_width *= 2

    for i in tqdm(range(num_batches)):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        
        with torch.no_grad():
            if args.log_steps:
                assert args.batch_size == 1, "log_steps is only supported for batch_size=1"
                step_latents = []

                def callback_dynamic(pipe, step_index, timestep, callback_kwargs):
                    if "latents" in callback_kwargs:
                        step_latents.append(callback_kwargs["latents"].detach().cpu())
                    return callback_kwargs

                outputs = pipeline(
                    prompt=batch_prompts,
                    num_inference_steps=args.num_inference_steps,
                    height=tgt_height,
                    width=tgt_width,
                    guidance_scale=7.0,
                    callback_on_step_end=callback_dynamic,
                    callback_on_step_end_tensor_inputs=["latents"],
                    max_sequence_length=256,
                )

                # Generate GIF from stored latents
                if step_latents:
                    gif_images = []
                    # Ensure VAE is on the correct device
                    pipeline.vae.to(accelerator.device)
                    
                    for latent in step_latents:
                        latent = latent.to(accelerator.device).to(pipeline.vae.dtype)
                        # SD3 VAE decoding scaling
                        if hasattr(pipeline.vae.config, "shift_factor") and pipeline.vae.config.shift_factor is not None:
                            latent = (latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                        else:
                            latent = latent / pipeline.vae.config.scaling_factor
                        
                        image = pipeline.vae.decode(latent, return_dict=False)[0]
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                        image = (image * 255).round().astype("uint8")
                        gif_images.append(Image.fromarray(image[0]).convert("RGB"))
                    
                    # Add the final image as well
                    gif_images.append(outputs.images[0].convert("RGB"))

                    # Save GIF
                    validation_save_dir = os.path.join(args.output_dir, "validation_images")
                    os.makedirs(validation_save_dir, exist_ok=True)
                    gif_path = os.path.join(validation_save_dir, f"inference_validation_{i}.gif")
                    
                    gif_images[0].save(
                        gif_path,
                        save_all=True,
                        append_images=gif_images[1:],
                        optimize=False,
                        duration=100, # 100ms per frame
                        loop=0
                    )

            else:
                outputs = pipeline(
                    prompt=batch_prompts,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=tgt_height,
                    width=tgt_width,
                    max_sequence_length=256,
                )
            
        for j, image in enumerate(outputs.images):
            image_idx = start_idx + j
            prompt = batch_prompts[j]
            
            # Save image
            safe_prompt = prompt[:50].replace(" ", "_").replace("/", "_")
            if args.paired_views:
                filename1 = f"img_{image_idx}_view1_{safe_prompt}.jpg"
                filename2 = f"img_{image_idx}_view2_{safe_prompt}.jpg"
                if args.stack_ch:
                    image1 = Image.fromarray(np.array(image)[:, :, 0]).convert("L")
                    image2 = Image.fromarray(np.array(image)[:, :, 1]).convert("L")
                else:
                    image1 = Image.fromarray(np.array(image)[:, :args.resolution])
                    image2 = Image.fromarray(np.array(image)[:, args.resolution:])
                save_path1 = os.path.join(args.output_dir, filename1)
                save_path2 = os.path.join(args.output_dir, filename2)
                image1.save(save_path1)
                image2.save(save_path2)
            elif args.four_views:
                filename1 = f"img_{image_idx}_view1_{safe_prompt}.jpg"
                filename2 = f"img_{image_idx}_view2_{safe_prompt}.jpg"
                filename3 = f"img_{image_idx}_view3_{safe_prompt}.jpg"
                filename4 = f"img_{image_idx}_view4_{safe_prompt}.jpg"
                img1 = Image.fromarray(np.array(image)[:args.resolution, :args.resolution]).convert("L")
                img2 = Image.fromarray(np.array(image)[:args.resolution, args.resolution:]).convert("L")
                img3 = Image.fromarray(np.array(image)[args.resolution:, :args.resolution]).convert("L")
                img4 = Image.fromarray(np.array(image)[args.resolution:, args.resolution:]).convert("L")
                save_path1 = os.path.join(args.output_dir, filename1)
                save_path2 = os.path.join(args.output_dir, filename2)
                save_path3 = os.path.join(args.output_dir, filename3)
                save_path4 = os.path.join(args.output_dir, filename4)
                img1.save(save_path1)
                img2.save(save_path2)
                img3.save(save_path3)
                img4.save(save_path4)
            else:
                filename = f"img_{image_idx}_{safe_prompt}.jpg"
                save_path = os.path.join(args.output_dir, filename)
                image = image.convert("L")
                image.save(save_path)
            
            # Save prompt text
            txt_filename = f"img_{image_idx}_{safe_prompt}.txt"
            with open(os.path.join(args.output_dir, txt_filename), "w") as f:
                f.write(prompt)
                
    logger.info(f"Inference complete. Images saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
