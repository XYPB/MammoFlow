import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
from collections import Counter 
import random
from tqdm import tqdm
from .constants_val import *
from .utils import get_imgs, get_breast_area
from .transforms import TextTransform


class RSNAMammo(torch.utils.data.Dataset):

    def __init__(self,
                 args,
                 split='train', 
                 transform=None, 
                 short_caption=False,
                 paired_view=False,
                 same_side=False,
                 four_view=False,
                 cc_first=False,
                 stack_ch=False,
                 **kwargs):
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
        self.stack_ch = stack_ch
        self.n_classes = 2

        if split == 'train':
            self.df = pd.read_csv(RSNA_MAMMO_TRAIN_CSV)
        else:
            self.df = pd.read_csv(RSNA_MAMMO_TEST_CSV)

        if args.data_pct != 1.0 and split == "train":
            random.seed(42)
            self.df = self.df.sample(frac=args.data_pct)
        self.df = self.df[self.df['view'].isin(['CC', 'MLO'])]

        if self.args.health_only:
            self.df = self.df[self.df['cancer'] == 0].reset_index(drop=True)
        if self.args.cancer_only:
            self.df = self.df[self.df['cancer'] == 1].reset_index(drop=True)

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
            pid, iid = entry['patient_id'], entry['image_id']
            pid = str(pid)
            path = os.path.join(RSNA_MAMMO_JPEG_DIR, f"{pid}/{iid}_resized.jpg")
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
        print('### Sampled split distribution: ', Counter(self.labels))
        
        # filter images without paired/four view images
        if self.paired_view:
            valid_idx = []
            for idx in self.train_idx:
                entry = self.df.iloc[idx]
                pid = str(entry['patient_id'])
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
                pid = str(entry['patient_id'])
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
                if all([ k in self.pid2paths[pid] for k in keys ]):
                    valid_idx.append(idx)
        if (self.paired_view or self.four_view) and len(valid_idx) < len(self.train_idx):
            print(f"### Filtered {len(self.train_idx) - len(valid_idx)} samples without paired/four view images.")
            self.df = self.df.iloc[valid_idx].reset_index(drop=True)
            self.filenames = [ self.filenames[i] for i in valid_idx ]
            self.labels = [ self.labels[i] for i in valid_idx ]
            self.train_idx = list(range(len(self.df)))
        
        self.prompts = self.get_prompt()

    def __len__(self):
        return len(self.df)

    def get_prompt(self):
        
        prompts = []
        for idx in range(len(self.df)):
            entry = self.df.iloc[idx]
            pid = str(entry['patient_id'])
            side = entry['laterality']
            view = entry['view']
            label = entry['cancer']
            birads = entry['BIRADS']
            density = entry['density']
            implant = entry['implant']
            
            side = 'left' if side == 'L' else 'right'
            
            if self.paired_view:
                if self.same_side:
                    first_view = 'CC' if self.cc_first else view
                    second_view = 'MLO' if self.cc_first else 'MLO' if view == 'CC' else 'CC'
                    if label == 1:
                        base_caption = f'Two {side} side breast mammograms with malignant cancer, the first image is {first_view} view and the second image is {second_view} view. '
                    else:
                        base_caption = f'Two healthy {side} breast mammograms, the first image is {first_view} view and the second image is {second_view} view. '
                else:
                    if label == 1:
                        base_caption = f'Two {view} view breast mammograms with malignant cancer, the first image is the {side} breast and the second image is the opposite side. '
                    else:
                        base_caption = f'Two healthy {view} view breast mammograms, the first image is the {side} breast and the second image is the opposite side. '
                prompt = base_caption.lower().strip()
            elif self.four_view:
                opposite_side = 'right' if side == 'left' else 'left'
                opposite_view = 'MLO' if view == 'CC' else 'CC'
                cancer_side = self.pid2cancer_side[pid]
                label = 1 if len(cancer_side) > 0 else 0
                cancer_side_str = ' and '.join([ 'left' if s=='L' else 'right' for s in cancer_side])
                if label == 1:
                    base_caption = f'Four breast mammograms with malignant cancer on {cancer_side_str} side, top left image is the {side} breast {view} view, top right image is the {opposite_side} breast {view} view, bottom left image is the {side} breast {opposite_view} view, bottom right image is the {opposite_side} breast {opposite_view} view.'
                else:
                    base_caption = f'Four healthy breast mammograms of the same patient, top left image is the {side} breast {view} view, top right image is the {opposite_side} breast {view} view, bottom left image is the {side} breast {opposite_view} view, bottom right image is the {opposite_side} breast {opposite_view} view.'
                prompt = base_caption.lower().strip()
            else:
                if label == 1:
                    base_caption = f'This is a {side} breast {view} view mammogram with malignant cancer. '
                else:
                    base_caption = f'This is a healthy {side} breast {view} view mammogram. '
                
                if not np.isnan(birads):
                    if birads == 0:
                        birads_desc = 'BIRADS 0: Additional follow-up is needed.'
                    elif birads == 1:
                        birads_desc = 'BIRADS 1: Negative finding.'
                    elif birads == 2:
                        birads_desc = 'BIRADS 2: Normal.'
                else:
                    birads_desc = ''
        
                if not isinstance(density, str) and  not np.isnan(density):
                    if density == 'A':
                        density_desc = 'The breast tissue is almost entirely fatty.'
                    elif density == 'B':
                        density_desc = 'There are scattered areas of fibroglandular density.'
                    elif density == 'C':
                        density_desc = 'The breast tissue is heterogeneously dense.'
                    elif density == 'D':
                        density_desc = 'The breast tissue is extremely dense.'
                else:
                    density_desc = ''
                
                if implant == 1:
                    implant_desc = 'The patient has breast implants.'
                else:
                    implant_desc = 'The patient does not have breast implants.'
            
                if self.short_caption:
                    prompt = base_caption
                else:
                    prompt = base_caption + birads_desc + ' ' + density_desc + ' ' + implant_desc
            prompts.append(prompt.lower().strip())
        return prompts


    def __getitem__(self, idx, no_image=False):
        entry = self.df.iloc[idx]
        pid = str(entry['patient_id'])
        label = self.labels[idx]
        path = self.filenames[idx]
        prompt = self.prompts[idx]

        one_hot_labels = torch.zeros(self.n_classes)
        one_hot_labels[label] = 1

        if no_image:
            return {
                "pixel_values": None,
                "prompts": prompt,
                "label": one_hot_labels,
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

            if self.stack_ch:
                ch_axis = 0 if len(img.shape) == 3 else 1
                img = img.mean(dim=ch_axis, keepdim=True)  # (1, H, W)
                paired_img = paired_img.mean(dim=ch_axis, keepdim=True)  # (1, H, W)
                zero_channel = torch.zeros_like(img).to(img.device)  # (1, H, W)
                img = torch.cat([img, paired_img, zero_channel], dim=ch_axis)  # (3, H, W)
                foreground_mask = torch.cat([foreground_mask, paired_foreground_mask, zero_channel.clone()], dim=1)  # (3, H, W)
            else:
                img = torch.cat([img, paired_img], dim=-1)  # (C, H, 2 * W)
                foreground_mask = torch.cat([foreground_mask, paired_foreground_mask], dim=-1)  # (H, 2 * W)
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
            img_top = torch.cat([img1, img2], dim=-1)  # (1, C, H, 2 * W)
            img_bottom = torch.cat([img3, img4], dim=-1)  # (1, C, H, 2 * W)
            img = torch.cat([img_top, img_bottom], dim=-2)  # (1, C, 2 * H, 2 * W)
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
