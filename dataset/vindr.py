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



class VinDr(torch.utils.data.Dataset):

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
        super().__init__()
        self.df = pd.read_csv(VINDR_CSV_DIR)
        self.data_path = VINDR_IMAGE_DIR
        self.args = args
        self.split = split
        self.short_caption = short_caption
        self.paired_view = paired_view
        self.same_side = same_side
        self.four_view = four_view
        self.cc_first = cc_first
        self.stack_ch = stack_ch
        self.transform = transform
    
        if args.pred_density:
            self.n_classes = 4
        elif args.pred_mass:
            self.n_classes = 2
        elif args.pred_calc:
            self.n_classes = 2
        else:
            self.n_classes = 5

        if split == 'test':
            self.df = self.df[self.df['split'] == 'test']
        else:
            self.df = self.df[self.df['split'] == 'training']
        self.findings_df = pd.read_csv(VINDR_DET_CSV_DIR)

        if self.args.data_pct != 1.0 and split == "train":
            random.seed(42)
            self.df = self.df.sample(frac=self.args.data_pct)
        self.df = self.df[self.df['view_position'].isin(['CC', 'MLO'])]

        if self.args.health_only:
            print("### Using only healthy samples (BI-RADS 1).")
            self.df = self.df[self.df['breast_birads'] == 'BI-RADS 1']
        elif self.args.cancer_only:
            print("### Using only deterministic cancerous samples (with Mass).")
            cancer_img_ids = []
            for idx, entry in self.df.iterrows():
                image_id = entry['image_id']
                findings = self.findings_df[self.findings_df['image_id'] == image_id]['finding_categories']
                findings_list = findings.to_list()
                findings_str = ' '.join(findings_list)
                if 'Mass' in findings_str:
                    cancer_img_ids.append(image_id)
            self.df = self.df[self.df['image_id'].isin(cancer_img_ids)]

        self.train_idx = list(range(len(self.df)))
        self.filenames = []
        self.labels = []
        self.density = []
        self.mass = []
        self.calc = []
        self.birads = []
        self.view = []
        self.side = []
        self.path2label = {}
        self.sid2paths = {}
        self.sid2cancer_side = {}

        for idx in self.train_idx:
            entry = self.df.iloc[idx]
            density = entry['breast_density'].split(' ')[-1]
            density = VINDR_DENSITY_LETTER2DIGIT[density]
            self.density.append(density)
            
            image_id = entry['image_id']
            findings = self.findings_df[self.findings_df['image_id'] == image_id]['finding_categories']
            findings_list = findings.to_list()
            findings_str = ' '.join(findings_list)
            mass = 2 if 'Mass' in findings_str else 1
            calc = 2 if 'Suspicious Calcification' in findings_str else 1
            self.mass.append(mass)
            self.calc.append(calc)
            
            birads = int(entry['breast_birads'].split(' ')[-1])
            self.birads.append(birads)
            
            view = entry['view_position']
            self.view.append(view)
            side = entry['laterality']
            self.side.append(side)
            
            if args.pred_density:
                label = density
            elif args.pred_mass:
                label = mass
            elif args.pred_calc:
                label = calc
            else:
                # BIRADS 1 ~ 5
                label = birads
            sid = entry['study_id']
            imid = entry['image_id']
            dicom_path = os.path.join(self.data_path, f'{sid}/{imid}.dicom')
            self.filenames.append(dicom_path)
            self.labels.append(label - 1)
            self.path2label[dicom_path] = label - 1
            
            if sid not in self.sid2paths:
                self.sid2paths[sid] = {}
            self.sid2paths[sid][f"{side}_{view}"] = dicom_path
            
            if sid not in self.sid2cancer_side:
                self.sid2cancer_side[sid] = []
            if mass == 2:
                self.sid2cancer_side[sid].append(side)
            
        print('### Sampled split distribution: ', Counter(self.labels))
        
        if self.paired_view:
            valid_idx = []
            for idx in self.train_idx:
                entry = self.df.iloc[idx]
                sid = entry['study_id']
                side = entry['laterality']
                view = entry['view_position']
                if self.same_side:
                    opposite_view = 'MLO' if view == 'CC' else 'CC'
                    paired_key = f'{side}_{opposite_view}'
                else:
                    opposite_side = 'R' if side == 'L' else 'L'
                    paired_key = f'{opposite_side}_{view}'
                if paired_key in self.sid2paths[sid]:
                    valid_idx.append(idx)
        if self.four_view:
            valid_idx = []
            for idx in self.train_idx:
                entry = self.df.iloc[idx]
                sid = entry['study_id']
                side = entry['laterality']
                view = entry['view_position']
                opposite_side = 'R' if side == 'L' else 'L'
                opposite_view = 'MLO' if view == 'CC' else 'CC'
                keys = [
                    f"{side}_{view}",
                    f"{opposite_side}_{view}",
                    f"{side}_{opposite_view}",
                    f"{opposite_side}_{opposite_view}"
                ]
                if all([ k in self.sid2paths[sid] for k in keys ]):
                    valid_idx.append(idx)
        if (self.paired_view or self.four_view) and len(valid_idx) < len(self.train_idx):
            print(f"### Filtered {len(self.train_idx) - len(valid_idx)} samples without paired/four view images.")
            self.df = self.df.iloc[valid_idx].reset_index(drop=True)
            self.filenames = [ self.filenames[i] for i in valid_idx ]
            self.labels = [ self.labels[i] for i in valid_idx ]
            self.train_idx = list(range(len(self.df)))
        
        self.sents = self.get_prompt()
        
        if args.aug_text:
            if args.heavy_aug:
                self.text_transform = TextTransform(
                    is_train=(split == 'train'),
                    remove_stop_word_prob=0.5,
                    synonym_replacement_prob=0.3,
                    random_swap_prob=0.3,
                    random_deletion_prob=0.3,
                    random_sent_swap_prob=0.3,
                    random_back_translation_prob=0.5,
                )
            else:
                self.text_transform = TextTransform(
                    is_train=(split == 'train'),
                )
        else:
            self.text_transform = None

    def __len__(self):
        return len(self.df)
    
    def get_roi_description(self, study_id):
        """
        Generate a natural language description of ROI location and size based on coordinates.
        
        Args:
            study_id: The study ID to look up in findings_df
            
        Returns:
            str: A natural language description of the ROI location and size
        """
        # Filter findings_df for the given study_id
        study_findings = self.findings_df[self.findings_df['study_id'] == study_id]
        
        # Check if there are any findings for this study_id
        if study_findings.empty:
            return "No region of interest is found in the mammography."
        
        # Check if there are any findings with valid ROI coordinates
        valid_roi = study_findings.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax'])
        if valid_roi.empty:
            return "No region of interest is found in the mammography."
        
        # If multiple entries, pick a random one
        if len(valid_roi) > 1:
            roi_data = valid_roi.sample(n=1).iloc[0]
        else:
            roi_data = valid_roi.iloc[0]
        
        # Extract ROI coordinates and image dimensions
        xmin = roi_data['xmin']
        ymin = roi_data['ymin']
        xmax = roi_data['xmax']
        ymax = roi_data['ymax']
        img_height = roi_data['height']
        img_width = roi_data['width']
        
        # Calculate ROI dimensions
        roi_width = xmax - xmin
        roi_height = ymax - ymin
        
        # Calculate ROI center
        roi_center_x = (xmin + xmax) / 2
        roi_center_y = (ymin + ymax) / 2
        
        # Determine horizontal position
        if roi_center_x < img_width / 2:
            h_position = "left"
        else:
            h_position = "right"
        
        # Determine vertical position
        if roi_center_y < img_height / 2:
            v_position = "upper"
        else:
            v_position = "lower"
        
        # Determine ROI size description
        roi_area = roi_width * roi_height
        img_area = img_width * img_height
        roi_ratio = roi_area / img_area
        
        if roi_ratio < 0.01:
            size_desc = "very small"
        elif roi_ratio < 0.03:
            size_desc = "small"
        elif roi_ratio < 0.1:
            size_desc = "medium-sized"
        elif roi_ratio < 0.25:
            size_desc = "large"
        else:
            size_desc = "very large"
        
        # Generate description
        position = f"{v_position} {h_position}"
        finding_type = roi_data.get('finding_categories', 'abnormality')
        if isinstance(finding_type, list) and len(finding_type) > 0:
            finding_type = finding_type[0].lower()
        elif isinstance(finding_type, str) and finding_type.startswith("['") and finding_type.endswith("']"):
            finding_type = finding_type[2:-2].lower()
        else:
            finding_type = "abnormality"
            
        description = f"There is a {size_desc} {finding_type} in the {position} area of the mammography."
        return description
    
    def get_prompt(self):
        """
        Returns a list of natural-language descriptions for each mammography in the dataset.
        Each description includes:
        - Side and view information
        - BI-RADS score and its description
        - Breast density description
        - ROI description from get_roi_description function
        """
        prompts = []
        
        for idx in tqdm(self.train_idx):
            entry = self.df.iloc[idx]
            study_id = entry['study_id']
            
            # Get side and view information
            side = entry['laterality']  # L or R 
            view = entry['view_position']  # CC or MLO
            cancer_side = self.sid2cancer_side.get(study_id, [])
            cancer = 1 if side in cancer_side else 0
            
            # Format side information using descriptive terms
            side_desc = "left" if side == "L" else "right"
            
            # Get BI-RADS score and its description
            birads = int(entry['breast_birads'].split(' ')[-1])
            birads_letter = VINDR_BIRADS_DIGIT2LETTER[birads-1]  # Convert to 0-indexed
            birads_desc = EMBED_BIRADS_DESC[birads_letter]
            
            # Get density information and description
            density_letter = entry['breast_density'].split(' ')[-1]
            density_digit = VINDR_DENSITY_LETTER2DIGIT[density_letter]
            density_desc = EMBED_DENSITY_DESC[density_digit]
            
            if self.paired_view:
                if self.same_side:
                    first_view = 'CC' if self.cc_first else view
                    second_view = 'MLO' if self.cc_first else 'MLO' if view == 'CC' else 'CC'
                    if cancer == 1:
                        base_caption = f'Two {side_desc} side breast mammograms with malignant cancer, the first image is {first_view} view and the second image is {second_view} view. '
                    else:
                        base_caption = f'Two healthy {side_desc} breast mammograms, the first image is {first_view} view and the second image is {second_view} view. '
                else:
                    if cancer == 1:
                        base_caption = f'Two {view} view breast mammograms with malignant cancer, the first image is the {side} breast and the second image is the opposite side. '
                    else:
                        base_caption = f'Two healthy {view} view breast mammograms, the first image is the {side} breast and the second image is the opposite side. '
                prompt = base_caption.lower().strip()
            elif self.four_view:
                opposite_side = 'right' if side == 'left' else 'left'
                opposite_view = 'MLO' if view == 'CC' else 'CC'
                cancer_side_str = ' and '.join([ 'left' if s=='L' else 'right' for s in cancer_side])
                if cancer == 1:
                    base_caption = f'Four breast mammograms with malignant cancer on {cancer_side_str} side, top left image is the {side} breast {view} view, top right image is the {opposite_side} breast {view} view, bottom left image is the {side} breast {opposite_view} view, bottom right image is the {opposite_side} breast {opposite_view} view.'
                else:
                    base_caption = f'Four healthy breast mammograms of the same patient, top left image is the {side} breast {view} view, top right image is the {opposite_side} breast {view} view, bottom left image is the {side} breast {opposite_view} view, bottom right image is the {opposite_side} breast {opposite_view} view.'
                prompt = base_caption.lower().strip()
            else:
                if cancer == 1:
                    base_caption = f'This is a {side} breast {view} view mammogram with malignant cancer. '
                else:
                    base_caption = f'This is a healthy {side} breast {view} view mammogram. '
                # Construct the prompt
                if not self.short_caption:
                    prompt = base_caption + f"BI-RADS assessment category is {birads}: {birads_desc}. "
                    prompt = base_caption + f"The breast composition is {density_desc}. "
            
                    # Add ROI description
                    roi_desc = self.get_roi_description(study_id)
                    prompt += roi_desc
                else:
                    prompt = base_caption
            
            prompts.append(prompt.lower().strip())
            
        return prompts

    def __getitem__(self, idx, no_image=False):
        entry = self.df.iloc[idx]
        sid = entry['study_id']
        imid = entry['image_id']
        label = self.labels[idx]
        dicom_path = os.path.join(self.data_path, f'{sid}/{imid}.dicom')

        one_hot_labels = torch.zeros(self.n_classes)
        one_hot_labels[label] = 1
        
        sent = self.sents[idx]

        img_path = dicom_path.replace('vindr-1.0.0', 'vindr-1.0.0-resized-1024')
        img_path = img_path.replace('.dicom', '_resized.png')
        assert os.path.exists(img_path)
        
        if no_image:
            return {
                "pixel_values": None, 
                "prompts": sent,
                "label": one_hot_labels,
                "mask": None,
                "image_path": img_path,
            }
        
        img, foreground_mask = get_imgs(img_path, scale=None, transform=self.transform, return_mask=True)
        if self.paired_view:
            if self.same_side:
                side = entry['laterality']
                if self.cc_first:
                    cc_path = self.sid2paths[sid][f'{side}_CC']
                    mlo_path = self.sid2paths[sid][f'{side}_MLO']
                    img_path, paired_path = cc_path, mlo_path
                    img_path = img_path.replace('vindr-1.0.0', 'vindr-1.0.0-resized-1024')
                    img_path = img_path.replace('.dicom', '_resized.png')
                    img, foreground_mask = get_imgs(img_path, scale=None, transform=self.transform, return_mask=True)
                else:
                    opposite_view = 'MLO' if entry['view_position'] == 'CC' else 'CC'
                    paired_path = self.sid2paths[sid][f'{side}_{opposite_view}']
            else:
                view = entry['view_position']
                opposite_side = 'R' if entry['laterality'] == 'L' else 'L'
                paired_path = self.sid2paths[sid][f"{opposite_side}_{view}"]
            paired_path = paired_path.replace('vindr-1.0.0', 'vindr-1.0.0-resized-1024')
            paired_path = paired_path.replace('.dicom', '_resized.png')
            assert os.path.exists(paired_path)
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
            view = entry['view_position']
            opposite_side = 'R' if side == 'L' else 'L'
            opposite_view = 'MLO' if view == 'CC' else 'CC'
            path1 = self.sid2paths[sid][f"{side}_{view}"]
            path2 = self.sid2paths[sid][f"{opposite_side}_{view}"]
            path3 = self.sid2paths[sid][f"{side}_{opposite_view}"]
            path4 = self.sid2paths[sid][f"{opposite_side}_{opposite_view}"]
            path_list = []
            for path in [path1, path2, path3, path4]:
                path = path.replace('vindr-1.0.0', 'vindr-1.0.0-resized-1024')
                path = path.replace('.dicom', '_resized.png')
                assert os.path.exists(path)
                path_list.append(path)
            img1, mask1 = get_imgs(path_list[0], scale=None, transform=self.transform, return_mask=True)
            img2, mask2 = get_imgs(path_list[1], scale=None, transform=self.transform, return_mask=True)
            img3, mask3 = get_imgs(path_list[2], scale=None, transform=self.transform, return_mask=True)
            img4, mask4 = get_imgs(path_list[3], scale=None, transform=self.transform, return_mask=True)
            img_top = torch.cat([img1, img2], dim=-1)  # (1, C, H, 2 * W)
            img_bottom = torch.cat([img3, img4], dim=-1)  # (1, C, H, 2 * W)
            img = torch.cat([img_top, img_bottom], dim=-2)  # (1, C, 2 * H, 2 * W)
            foreground_mask_top = torch.cat([mask1, mask2], dim=-1)  # (1, H, 2 * W)
            foreground_mask_bottom = torch.cat([mask3, mask4], dim=-1)  # (1, H, 2 * W)
            foreground_mask = torch.cat([foreground_mask_top, foreground_mask_bottom], dim=-2)  # (1, 2 * H, 2 * W)

        if self.args.save_mask:
            # save the mask as image to tmp
            mask_numpy = foreground_mask[0].numpy() * 255
            mask_numpy = mask_numpy.astype(np.uint8)
            mask_img = Image.fromarray(mask_numpy)
            mask_img.save(f"./tmp/cur_mask_{idx}.png")

        if not self.short_caption:
            # inject breast size information
            breast_ratio = get_breast_area(img)
            breast_size = "small" if breast_ratio < 0.10 else "large" if breast_ratio > 0.5 else "medium"
            sent = sent.replace("view mammogram", f"view {breast_size} size mammogram")

        if self.text_transform is not None:
            sent = self.text_transform(sent)

        return {
            "pixel_values": img, 
            "prompts": sent,
            "label": one_hot_labels,
            "mask": foreground_mask,
            "image_path": img_path,
        }