import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
import json
from collections import Counter
import random
import re
from tqdm import tqdm
from .constants_val import *
from .utils import get_imgs, get_breast_area
from .transforms import TextTransform


# Default CSAW data directory
CSAW_DATA_DIR = os.path.join(DATA_BASE_DIR, "CSAW-CC-resized-1024/train")


class CSAW(torch.utils.data.Dataset):
    """
    CSAW Mammography Dataset.
    
    This dataset loads mammography images from the CSAW dataset with metadata
    stored in JSONL format. Each entry contains a file_name and text prompt.
    
    Example metadata entry:
    {"file_name": "00002_20990909_L_CC_1_resized.png", 
     "text": "This is a high-resolution 2D full-field screening CC view mammogram 
              of the left breast with no visible cancer."}
    
    The classification label is determined from the text prompt:
    - "no visible cancer" or "no cancer" -> label = 0 (healthy)
    - otherwise (cancer present) -> label = 1 (cancer)
    """

    def __init__(self,
                 args,
                 split='train',
                 transform=None,
                 short_caption=False,
                 paired_view=False,
                 same_side=False,
                 four_view=False,
                 cc_first=False,
                 data_dir=None,
                 **kwargs):
        """
        Initialize the CSAW dataset.
        
        Args:
            args: Namespace containing dataset arguments (e.g., data_pct)
            split: 'train' or 'valid'/'test'
            transform: Image transforms to apply
            short_caption: Whether to use short caption format
            paired_view: Whether to load paired view images
            same_side: Whether paired views are from the same side
            four_view: Whether to load all four views
            data_dir: Override default data directory
        """
        if split == 'test':
            split = 'valid'
        assert split in ['train', 'valid']
        
        self.args = args
        self.transform = transform
        self.short_caption = short_caption
        self.paired_view = paired_view
        self.same_side = same_side
        self.four_view = four_view
        self.cc_first = cc_first
        self.n_classes = 2
        self.split = split
        
        # Set data directory
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = CSAW_DATA_DIR
        
        meta_filename = "metadata_visible_train.jsonl" if split == "train" else "metadata_visible_test.jsonl"
        
        # Load metadata from JSONL file
        metadata_path = os.path.join(self.data_dir, meta_filename)
        self.df = self._load_metadata(metadata_path)
        
        # Sample data if data_pct is specified
        if hasattr(args, 'data_pct') and args.data_pct != 1.0 and split == "train":
            random.seed(42)
            sample_size = int(len(self.df) * args.data_pct)
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Filter to only CC and MLO views
        self.df = self.df[self.df['view'].isin(['CC', 'MLO'])].reset_index(drop=True)
        
        self.train_idx = list(range(len(self.df)))
        self.filenames = []
        self.path2label = {}
        self.labels = []
        self.pid2paths = {}
        self.pid2cancer_side = {}
        
        missing_cnt = 0
        for idx in self.train_idx:
            entry = self.df.iloc[idx]
            label = entry['cancer']
            pid = entry['patient_id']
            path = os.path.join(self.data_dir, entry['file_name'])
            
            if not os.path.exists(path):
                missing_cnt += 1
                
            self.labels.append(label)
            self.filenames.append(path)
            self.path2label[path] = label
            
            if pid not in self.pid2paths:
                self.pid2paths[pid] = {}
            
            side = entry['laterality']
            view = entry['view']
            self.pid2paths[pid][f"{side}_{view}"] = path
            
            if pid not in self.pid2cancer_side:
                self.pid2cancer_side[pid] = set()
            if label == 1:
                self.pid2cancer_side[pid].add(side)
        
        if missing_cnt > 0:
            print(f"### Warning: {missing_cnt} images are missing from {self.data_dir}")
        print('### Sampled split distribution: ', Counter(self.labels))
        
        # Filter images without paired/four view images
        if self.paired_view:
            valid_idx = []
            for idx in self.train_idx:
                entry = self.df.iloc[idx]
                pid = entry['patient_id']
                side = entry['laterality']
                view = entry['view']
                if self.same_side:
                    opposite_view = 'MLO' if view == 'CC' else 'CC'
                    paired_key = f"{side}_{opposite_view}"
                else:
                    opposite_side = 'R' if side == 'L' else 'L'
                    paired_key = f"{opposite_side}_{view}"
                if paired_key in self.pid2paths[pid]:
                    valid_idx.append(idx)
                    
        if self.four_view:
            valid_idx = []
            for idx in self.train_idx:
                entry = self.df.iloc[idx]
                pid = entry['patient_id']
                side = entry['laterality']
                view = entry['view']
                opposite_side = 'R' if side == 'L' else 'L'
                opposite_view = 'MLO' if view == 'CC' else 'CC'
                keys = [
                    f"{side}_{view}",
                    f"{opposite_side}_{view}",
                    f"{side}_{opposite_view}",
                    f"{opposite_side}_{opposite_view}"
                ]
                if all([k in self.pid2paths[pid] for k in keys]):
                    valid_idx.append(idx)
                    
        if (self.paired_view or self.four_view) and len(valid_idx) < len(self.train_idx):
            print(f"### Filtered {len(self.train_idx) - len(valid_idx)} samples without paired/four view images.")
            self.df = self.df.iloc[valid_idx].reset_index(drop=True)
            self.filenames = [self.filenames[i] for i in valid_idx]
            self.labels = [self.labels[i] for i in valid_idx]
            self.train_idx = list(range(len(self.df)))
        
        self.prompts = self.get_prompt()

    def _load_metadata(self, metadata_path):
        """
        Load metadata from JSONL file and parse file names to extract 
        patient_id, laterality, view, and cancer label.
        
        Expected file_name format: {patient_id}_{date}_{laterality}_{view}_{num}_resized.png
        Example: 00002_20990909_L_CC_1_resized.png
        """
        data = []
        with open(metadata_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                file_name = entry['file_name']
                text = entry['text']
                
                # Parse file name to extract metadata
                # Format: {patient_id}_{date}_{laterality}_{view}_{num}_resized.png
                base_name = os.path.splitext(file_name)[0]  # Remove .png
                parts = base_name.split('_')
                
                # Extract fields from filename
                patient_id = parts[0] if len(parts) > 0 else ""
                laterality = parts[2] if len(parts) > 2 else ""  # L or R
                view = parts[3] if len(parts) > 3 else ""  # CC or MLO
                
                # Determine cancer label from text
                # "no visible cancer" or "no cancer" -> 0, otherwise -> 1
                cancer = self._parse_cancer_label(text)
                
                data.append({
                    'file_name': file_name,
                    'text': text,
                    'patient_id': patient_id,
                    'laterality': laterality,
                    'view': view,
                    'cancer': cancer
                })
        
        return pd.DataFrame(data)

    def _parse_cancer_label(self, text):
        """
        Parse the text prompt to determine the cancer label.
        
        Args:
            text: The text prompt describing the mammogram
            
        Returns:
            0 if no cancer, 1 if cancer is present
        """
        text_lower = text.lower()
        # Check for negative cancer indicators
        if 'no visible cancer' in text_lower or 'no cancer' in text_lower:
            return 0
        # Check for positive cancer indicators
        if 'cancer' in text_lower or 'malignant' in text_lower:
            return 1
        # Default to no cancer if unclear
        return 0

    def __len__(self):
        return len(self.df)

    def get_prompt(self):
        """Generate prompts for each sample based on metadata."""
        prompts = []
        for idx in range(len(self.df)):
            entry = self.df.iloc[idx]
            pid = entry['patient_id']
            side = entry['laterality']
            view = entry['view']
            label = entry['cancer']
            
            # Convert laterality to full word
            side_word = 'left' if side == 'L' else 'right'
            
            if self.paired_view:
                if self.same_side:
                    first_view = 'CC' if self.cc_first else view
                    second_view = 'MLO' if self.cc_first else 'MLO' if view == 'CC' else 'CC'
                    if label == 1:
                        base_caption = f'Two {side_word} side breast mammograms with malignant cancer, the first image is {first_view} view and the second image is {second_view} view. '
                    else:
                        base_caption = f'Two healthy {side_word} breast mammograms, the first image is {first_view} view and the second image is {second_view} view. '
                else:
                    if label == 1:
                        base_caption = f'Two {view} view breast mammograms with malignant cancer, the first image is the {side_word} breast and the second image is the opposite side. '
                    else:
                        base_caption = f'Two healthy {view} view breast mammograms, the first image is the {side_word} breast and the second image is the opposite side. '
                prompt = base_caption.lower().strip()
                
            elif self.four_view:
                opposite_side = 'right' if side_word == 'left' else 'left'
                opposite_view = 'MLO' if view == 'CC' else 'CC'
                cancer_side = self.pid2cancer_side[pid]
                label = 1 if len(cancer_side) > 0 else 0
                cancer_side_str = ' and '.join(['left' if s == 'L' else 'right' for s in cancer_side])
                if label == 1:
                    base_caption = f'Four breast mammograms with malignant cancer on {cancer_side_str} side, top left image is the {side_word} breast {view} view, top right image is the {opposite_side} breast {view} view, bottom left image is the {side_word} breast {opposite_view} view, bottom right image is the {opposite_side} breast {opposite_view} view.'
                else:
                    base_caption = f'Four healthy breast mammograms of the same patient, top left image is the {side_word} breast {view} view, top right image is the {opposite_side} breast {view} view, bottom left image is the {side_word} breast {opposite_view} view, bottom right image is the {opposite_side} breast {opposite_view} view.'
                prompt = base_caption.lower().strip()
                
            else:
                # Use original text from metadata or generate short caption
                if self.short_caption:
                    if label == 1:
                        base_caption = f'This is a {side_word} breast {view} view mammogram with malignant cancer.'
                    else:
                        base_caption = f'This is a healthy {side_word} breast {view} view mammogram.'
                    prompt = base_caption.lower().strip()
                else:
                    # Use the original text from metadata
                    prompt = entry['text'].lower().strip()
                    
            prompts.append(prompt)
        return prompts

    def __getitem__(self, idx, no_image=False):
        entry = self.df.iloc[idx]
        pid = entry['patient_id']
        label = self.labels[idx]
        path = self.filenames[idx]
        prompt = self.prompts[idx]

        one_hot_labels = torch.zeros(self.n_classes)
        one_hot_labels[label] = 1

        if no_image:
            return {
                "pixel_values": None,
                "prompts": prompt,
                "label": label,
                "mask": None,
                "image_path": path
            }

        img, foreground_mask = get_imgs(path, scale=None, transform=self.transform, return_mask=True)
        
        if self.paired_view:
            if self.same_side:
                side = entry['laterality']
                if self.cc_first:
                    cc_path = self.pid2paths[pid][f"{side}_CC"]
                    mlo_path = self.pid2paths[pid][f"{side}_MLO"]
                    path, paired_path = cc_path, mlo_path
                    img, foreground_mask = get_imgs(path, scale=None, transform=self.transform, return_mask=True)
                else:
                    opposite_view = 'MLO' if entry['view'] == 'CC' else 'CC'
                    paired_path = self.pid2paths[pid][f"{side}_{opposite_view}"]
            else:
                view = entry['view']
                opposite_side = 'R' if entry['laterality'] == 'L' else 'L'
                paired_path = self.pid2paths[pid][f"{opposite_side}_{view}"]
            paired_img, paired_foreground_mask = get_imgs(paired_path, scale=None, transform=self.transform, return_mask=True)

            img = torch.cat([img, paired_img], dim=-1)  # (C, H, 2 * W)
            foreground_mask = torch.cat([foreground_mask, paired_foreground_mask], dim=-1)  # (1, H, 2 * W)
            
        elif self.four_view:
            side = entry['laterality']
            view = entry['view']
            opposite_side = 'R' if side == 'L' else 'L'
            opposite_view = 'MLO' if view == 'CC' else 'CC'
            path1 = self.pid2paths[pid][f"{side}_{view}"]
            path2 = self.pid2paths[pid][f"{opposite_side}_{view}"]
            path3 = self.pid2paths[pid][f"{side}_{opposite_view}"]
            path4 = self.pid2paths[pid][f"{opposite_side}_{opposite_view}"]
            img1, mask1 = get_imgs(path1, scale=None, transform=self.transform, return_mask=True)
            img2, mask2 = get_imgs(path2, scale=None, transform=self.transform, return_mask=True)
            img3, mask3 = get_imgs(path3, scale=None, transform=self.transform, return_mask=True)
            img4, mask4 = get_imgs(path4, scale=None, transform=self.transform, return_mask=True)
            img_top = torch.cat([img1, img2], dim=-1)  # (C, H, 2 * W)
            img_bottom = torch.cat([img3, img4], dim=-1)  # (C, H, 2 * W)
            img = torch.cat([img_top, img_bottom], dim=-2)  # (C, 2 * H, 2 * W)
            foreground_mask_top = torch.cat([mask1, mask2], dim=-1)  # (1, H, 2 * W)
            foreground_mask_bottom = torch.cat([mask3, mask4], dim=-1)  # (1, H, 2 * W)
            foreground_mask = torch.cat([foreground_mask_top, foreground_mask_bottom], dim=-2)  # (1, 2 * H, 2 * W)

        return {
            "pixel_values": img,
            "prompts": prompt,
            "label": one_hot_labels,
            "mask": foreground_mask,
            "image_path": path
        }
