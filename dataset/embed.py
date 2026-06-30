import os
import pickle
import re
import time

import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image
from .constants_val import *
from .utils import get_imgs, get_brief_prompt
from tqdm import tqdm
from .transforms import TextTransform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


SEP_TOKEN = '<|separatetext|>'
BOS_TOKEN = '<|startoftext|>'
EOS_TOKEN = '<|endoftext|>'


class EmbedDiffusionDataset(data.Dataset):
    def __init__(self, args, split, transform=None, paired_view=False, same_side=False, cc_first=False, **kwargs):
        self.args = args
        self.split = split
        self.transform = transform
        self.paired_view = paired_view
        self.same_side = same_side
        self.cc_first = cc_first

        if split == "train":
            self.df = pd.read_csv(EMBED_TRAIN_META_CSV)
        elif split == "valid":
            self.df = pd.read_csv(EMBED_VALID_META_CSV)
        elif split == "test":
            self.df = pd.read_csv(EMBED_TEST_META_CSV)
            self.cls_prompt = True
        else:
            raise ValueError(f"split {split} not supported")
        self.df_anno = pd.read_csv(EMBED_ANNO_CSV_REDUCED)
        self.df_anno_full = pd.read_csv(EMBED_ANNO_CSV)
        if self.args.pred_density:
            if split == 'train':
                density_file = EMBED_TRAIN_PATH2DENSITY
            elif split == 'valid':
                density_file = EMBED_VALID_PATH2DENSITY
            elif split == 'test':
                density_file = EMBED_TEST_PATH2DENSITY
            else:
                raise ValueError(f"split {split} not supported")
            assert os.path.exists(density_file)
            self.path2density = pickle.load(open(density_file, "rb"))

        self.df = self.df[self.df[EMBED_IMAGE_TYPE_COL].isin(["2D"])]
        self.df[EMBED_PATH_COL] = self.df[EMBED_PATH_COL].apply(EMBED_PATH_TRANS_FUNC)

        if args.screen_only:
            screen_idx = self.df[EMBED_PROCEDURE_COL].apply(
                lambda x: x.lower().find("screen") > 0
            )
            self.df = self.df[screen_idx]
            # Clean up the magnification view and none CC/MLO view
            self.df = self.df[self.df["spot_mag"] != 0]
            self.df = self.df[self.df[EMBED_VIEW_COL].isin(["CC", "MLO"])]

        self.filenames, self.path2sent, self.path2label = self.load_text_data(split)

        # Build sid2paths mapping for paired_view support
        self.sid2paths = {}
        self.path2info = {}  # path -> (sid, side, view)
        for _, row in self.df.iterrows():
            sid = row[EMBED_SID_COL]
            side = row[EMBED_SIDE_COL]
            view = row[EMBED_VIEW_COL]
            path = row[EMBED_PATH_COL]
            path = path.replace("mammo_sd", "PEMedCLIP")
            if sid not in self.sid2paths:
                self.sid2paths[sid] = {}
            self.sid2paths[sid][f"{side}_{view}"] = path
            self.path2info[path] = (sid, side, view)

        if self.paired_view:
            valid_filenames = []
            for p in self.filenames:
                # print(p)
                # print(p in self.path2info)
                if p not in self.path2info:
                    continue
                sid, side, view = self.path2info[p]
                if sid not in self.sid2paths:
                    continue
                if self.same_side:
                    opposite_view = 'MLO' if view == 'CC' else 'CC'
                    paired_key = f'{side}_{opposite_view}'
                else:
                    opposite_side = 'R' if side == 'L' else 'L'
                    paired_key = f'{opposite_side}_{view}'
                if paired_key in self.sid2paths[sid]:
                    valid_filenames.append(p)
            print(f"### Filtered {len(self.filenames) - len(valid_filenames)} samples without paired view images.")
            self.filenames = valid_filenames

        if args.data_pct < 1.0:
            self.filenames = self.filenames[:int(len(self.filenames) * args.data_pct)]
        
        if args.aug_text:
            if args.heavy_aug:
                self.text_transform = TextTransform(
                    is_train=(split == 'train'),
                    bos_token=BOS_TOKEN,
                    eos_token=EOS_TOKEN,
                    stop_token=SEP_TOKEN,
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
                    bos_token=BOS_TOKEN,
                    eos_token=EOS_TOKEN,
                    stop_token=SEP_TOKEN,
                )
        else:
            self.text_transform = None

    def load_text_data(self, split):
        base_filename = f"{split}_captions.pickle"
        if self.args.structural_cap:
            base_filename = base_filename.replace(".pickle", "_structural.pickle")
        else:
            base_filename = base_filename.replace(".pickle", "_simple.pickle")
        base_filename = base_filename.replace(".pickle", "_raw.pickle")
        if self.args.screen_only:
            base_filename = base_filename.replace(".pickle", "_screen.pickle")
        filepath = os.path.join(EMBED_DATA_DIR, base_filename)

        print(f"### Loading captions from {filepath}...")
        assert os.path.exists(filepath)
        st = time.time()
        with open(filepath, "rb") as f:
            path2sent = pickle.load(f)
        print(f"### Loaded captions in {time.time() - st:.2} seconds")

        # Some of the paths in the dataframe are not in the captions
        filenames = []
        path2label = {}

        print("### extract label from captions...")
        for p, sentences in tqdm(path2sent.items()):
            # Extract BI-RAS label from the last sentence
            # print(sentences)
            sent = sentences[0].lower().replace("-", "")
            sent = sent.replace("bi rads", "birads")
            assert "birads" in sent
            # print(sent)
            if self.args.structural_cap:
                birads = re.findall(r"\bbirads\s\bcategory\s(\d+)", sent)[0]
            else:
                birads = re.findall(r"\bbirads\s\bscore\s(\d+)", sent)[0]
            # skip birads 3 - 6 considering only screening image with 0, 1, 2
            if self.args.screen_only and int(birads) > 2:
                continue
            if self.args.pred_density:
                if p not in self.path2density.keys():
                    print(f"### {p} not in density map")
                    continue
                # Ignore male images
                label = self.path2density[p] - 1
                if label == 4:
                    continue
                path2label[p] = label
                filenames.append(p)
            else:
                path2label[p] = int(birads)
                filenames.append(p)
            filenames.append(p)
        print(np.unique(list(path2label.values()), return_counts=True))
        return filenames, path2sent, path2label

    def __len__(self):
        return len(self.filenames)
    
    def get_birads_one_hot_label(self, index, get_full=False):
        num_classes = 3 if self.args.screen_only else len(EMBED_LETTER_TO_BIRADS)
        multi_hot_label = torch.zeros(num_classes)
        key = self.filenames[index]
        asses = self.path2label[key]
        multi_hot_label[asses] = 1
        return multi_hot_label
    
    def get_density_one_hot_label(self, index, get_full=False):
        multi_hot_label = torch.zeros(len(EMBED_DENSITY_DESC) - 1)
        key = self.filenames[index]
        density = self.path2label[key]
        multi_hot_label[density] = 1
        return multi_hot_label

    def _get_paired_prompt(self, orig_key):
        """Generate a prompt appropriate for paired view images."""
        if orig_key not in self.path2info:
            return "two breast mammograms."
        sid, side, view = self.path2info[orig_key]
        side_desc = EMBED_SIDES_DESC.get(side, side.lower())

        # Determine cancer status from BI-RADS label
        label_val = self.path2label.get(orig_key, 0)
        if self.args.pred_density:
            cancer = False
        else:
            # BI-RADS >= 4 (S or M or K) is considered cancer
            cancer = label_val >= 4

        if self.same_side:
            first_view = 'CC' if self.cc_first else view
            opposite_view = 'MLO' if self.cc_first else 'MLO' if view == 'CC' else 'CC'
            if cancer:
                prompt = f'two {side_desc} side breast mammograms with malignant cancer, the first image is {first_view} view and the second image is {opposite_view} view. '
            else:
                prompt = f'two healthy {side_desc} breast mammograms, the first image is {first_view} view and the second image is {opposite_view} view. '
        else:
            opposite_side_desc = 'right' if side == 'L' else 'left'
            if cancer:
                prompt = f'two {view} view breast mammograms with malignant cancer, the first image is the {side_desc} breast and the second image is the {opposite_side_desc} breast. '
            else:
                prompt = f'two healthy {view} view breast mammograms, the first image is the {side_desc} breast and the second image is the {opposite_side_desc} breast. '
        return prompt.lower().strip()

    def __getitem__(self, idx, no_image=False):
        orig_key = self.filenames[idx]
        key = GET_JPEG_PATH_FUNC(orig_key.replace("PEMedCLIP", "mammo_sd"))
        
        if self.paired_view:
            sent = self._get_paired_prompt(orig_key)
        else:
            sent = self.path2sent[orig_key]
            sent = sent[0].replace(BOS_TOKEN, "").replace(EOS_TOKEN, "").replace(SEP_TOKEN, ".")
            sent = get_brief_prompt(sent)

        if self.args.pred_density:
            label = self.get_density_one_hot_label(idx)
        else:
            label = self.get_birads_one_hot_label(idx)
        
        if self.text_transform is not None:
            sent = self.text_transform(sent)
        
        if no_image:
            return {
                "pixel_values": None, 
                "prompts": sent,
                "label": label,
                "mask": None,
                "image_path": key,
            }

        img, foreground_mask = get_imgs(key, None, self.transform, return_orig_img=False, return_mask=True)

        if self.paired_view:
            sid, side, view = self.path2info[orig_key]
            if self.same_side:
                if self.cc_first:
                    first_orig_key = self.sid2paths[sid][f'{side}_CC']
                    paired_orig_key = self.sid2paths[sid][f'{side}_MLO']
                    key = GET_JPEG_PATH_FUNC(first_orig_key.replace("PEMedCLIP", "mammo_sd"))
                    img, foreground_mask = get_imgs(key, None, self.transform, return_orig_img=False, return_mask=True)
                else:
                    opposite_view = 'MLO' if view == 'CC' else 'CC'
                    paired_orig_key = self.sid2paths[sid][f'{side}_{opposite_view}']
            else:
                opposite_side = 'R' if side == 'L' else 'L'
                paired_orig_key = self.sid2paths[sid][f'{opposite_side}_{view}']
            paired_key = GET_JPEG_PATH_FUNC(paired_orig_key.replace("PEMedCLIP", "mammo_sd"))
            paired_img, paired_foreground_mask = get_imgs(
                paired_key, None, self.transform, return_orig_img=False, return_mask=True
            )
            img = torch.cat([img, paired_img], dim=-1)  # (C, H, 2 * W)
            foreground_mask = torch.cat([foreground_mask, paired_foreground_mask], dim=-1)

        if self.args.save_mask:
            # save the mask as image to tmp
            mask_numpy = foreground_mask[0].numpy() * 255
            mask_numpy = mask_numpy.astype(np.uint8)
            mask_img = Image.fromarray(mask_numpy)
            mask_img.save(f"./tmp/cur_mask_{idx}.png")

        return {
            "pixel_values": img, 
            "prompts": sent,
            "label": label,
            "mask": foreground_mask,
            "image_path": key,
        }



