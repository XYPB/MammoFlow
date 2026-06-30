import os
import re
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from .utils import get_imgs


@dataclass
class _Sample:
    sample_id: str
    prompt: str
    image_paths: List[str]
    label: int
    data_dir: str


class SyntheticMammo(torch.utils.data.Dataset):
    """
    Synthetic mammography dataset based on folders of images + per-sample prompt files.

    Expected layout per sample (index is part of filename):
      img_{index}_<anything>.txt
      img_{index}_view1_<anything>.jpg/.png
      img_{index}_view2_<anything>.jpg/.png

    Label is inferred from prompt text:
      - 'healthy', 'no visible cancer', 'no cancer' -> 0
      - 'cancer', 'malignant' -> 1
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        short_caption: bool = False,
        paired_view: bool = False,
        same_side: bool = False,
        four_view: bool = False,
        data_ratio: Optional[float] = None,
        cc_first=False,
        **kwargs,
    ):
        self.data_dirs = self._normalize_data_dirs(data_dir)
        self.transform = transform
        self.short_caption = short_caption
        self.paired_view = paired_view
        self.same_side = same_side
        self.four_view = four_view
        self.cc_first = cc_first
        self.n_classes = 2

        self.samples = self._load_samples()

        if data_ratio is not None:
            self.samples = self._filter_samples_by_ratio(self.samples, data_ratio)

        self.labels = [s.label for s in self.samples]
        print("### Sampled split distribution: ", Counter(self.labels))

    def _filter_samples_by_ratio(self, samples: List[_Sample], ratio: float) -> List[_Sample]:
        random.seed(42)  # For reproducibility
        random.shuffle(samples)
        if ratio <= 0 or ratio > 1:
            raise ValueError("data_ratio must be in the range (0, 1]")
        num_samples = int(len(samples) * ratio)
        return samples[:num_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int, no_image: bool = False):
        sample = self.samples[idx]
        prompt = self._build_prompt(sample)

        one_hot_labels = torch.zeros(self.n_classes)
        one_hot_labels[sample.label] = 1

        if no_image:
            return {
                "pixel_values": None,
                "prompts": prompt,
                "label": one_hot_labels,
                "mask": None,
                "image_path": sample.image_paths[0],
            }

        img, foreground_mask = self._load_images(sample)

        return {
            "pixel_values": img,
            "prompts": prompt,
            "label": one_hot_labels,
            "mask": foreground_mask,
            "image_path": sample.image_paths[0],
        }

    def _normalize_data_dirs(self, data_dir: str) -> List[str]:
        if isinstance(data_dir, str):
            dirs = [part for part in data_dir.split(":") if part]
        else:
            dirs = list(data_dir)
        return [os.path.expanduser(d) for d in dirs]

    def _load_samples(self) -> List[_Sample]:
        samples: List[_Sample] = []
        for dir_index, dir_path in enumerate(self.data_dirs):
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"Data directory not found: {dir_path}")

            txt_files = [
                f for f in os.listdir(dir_path) if f.startswith("img_") and f.endswith(".txt")
            ]

            for txt_file in sorted(txt_files):
                base_sample_id = self._parse_sample_id(txt_file)
                if base_sample_id is None:
                    continue

                sample_id = f"{dir_index}-{base_sample_id}"
                prompt_path = os.path.join(dir_path, txt_file)
                with open(prompt_path, "r") as f:
                    prompt = f.read().strip()

                image_paths = self._find_images_for_sample(base_sample_id, dir_path)
                if not image_paths:
                    continue

                label = self._parse_label(prompt)

                samples.append(
                    _Sample(
                        sample_id=sample_id,
                        prompt=prompt,
                        image_paths=image_paths,
                        label=label,
                        data_dir=dir_path,
                    )
                )

        return samples

    def _parse_sample_id(self, filename: str) -> Optional[str]:
        match = re.match(r"img_(\d+)_", filename)
        if not match:
            return None
        return match.group(1)

    def _find_images_for_sample(self, sample_id: str, data_dir: str) -> List[str]:
        prefix = f"img_{sample_id}_"
        image_files = [
            f
            for f in os.listdir(data_dir)
            if f.startswith(prefix) and (f.endswith(".jpg") or f.endswith(".png"))
        ]

        # Prefer ordering by view number if present.
        def sort_key(name: str) -> Tuple[int, str]:
            view_match = re.search(r"view(\d+)", name)
            view_num = int(view_match.group(1)) if view_match else 0
            return (view_num, name)

        image_files.sort(key=sort_key)
        return [os.path.join(data_dir, f) for f in image_files]

    def _parse_label(self, prompt: str) -> int:
        text = prompt.lower()
        if "no visible cancer" in text or "no cancer" in text or "healthy" in text:
            return 0
        if "cancer" in text or "malignant" in text:
            return 1
        return 0

    def _parse_view(self, sample: _Sample) -> Optional[str]:
        prompt = sample.prompt.lower()

        if self.paired_view:
            if "the first image is cc" in prompt or "cc view first" in prompt:
                return "cc"
            else:
                return "mlo"
        else:
            if "cc" in prompt:
                return "cc"
            elif "mlo" in prompt:
                return "mlo"
        return None

    def _build_prompt(self, sample: _Sample) -> str:
        if self.short_caption:
            # Keep a minimal caption while preserving health/cancer signal.
            return "healthy breast mammogram" if sample.label == 0 else "breast mammogram with cancer"
        return sample.prompt.lower().strip()

    def _load_images(self, sample: _Sample):
        if self.four_view:
            if len(sample.image_paths) < 4:
                raise ValueError("four_view=True requires at least 4 images per sample")
            paths = sample.image_paths[:4]
            img1, mask1 = get_imgs(paths[0], scale=None, transform=self.transform, return_mask=True)
            img2, mask2 = get_imgs(paths[1], scale=None, transform=self.transform, return_mask=True)
            img3, mask3 = get_imgs(paths[2], scale=None, transform=self.transform, return_mask=True)
            img4, mask4 = get_imgs(paths[3], scale=None, transform=self.transform, return_mask=True)

            img_top = torch.cat([img1, img2], dim=-1)
            img_bottom = torch.cat([img3, img4], dim=-1)
            img = torch.cat([img_top, img_bottom], dim=-2)

            mask_top = torch.cat([mask1, mask2], dim=-1)
            mask_bottom = torch.cat([mask3, mask4], dim=-1)
            foreground_mask = torch.cat([mask_top, mask_bottom], dim=-2)
            return img, foreground_mask

        if self.paired_view:
            if len(sample.image_paths) < 2:
                raise ValueError("paired_view=True requires at least 2 images per sample")
            img1, mask1 = get_imgs(sample.image_paths[0], scale=None, transform=self.transform, return_mask=True)
            img2, mask2 = get_imgs(sample.image_paths[1], scale=None, transform=self.transform, return_mask=True)
            img1_view = self._parse_view(sample)
            if self.cc_first and img1_view == "mlo":
                img1, img2 = img2, img1
                mask1, mask2 = mask2, mask1
            img = torch.cat([img1, img2], dim=-1)
            foreground_mask = torch.cat([mask1, mask2], dim=-1)
            return img, foreground_mask

        # Single-view default: use the first image.
        img, foreground_mask = get_imgs(
            sample.image_paths[0], scale=None, transform=self.transform, return_mask=True
        )
        return img, foreground_mask
