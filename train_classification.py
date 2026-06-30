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
Breast Image Classification Training Script

A comprehensive implementation for training CNN and Transformer models on breast mammography datasets.
Supports multiple datasets (EMBED, RSNA-Mammo, VinDr) and various loss functions for handling
imbalanced classification scenarios.

Key Features:
- Multiple dataset support with automatic loading
- CNN architectures: ResNet, ResNeXt, ConvNeXt, EfficientNet, DenseNet
- Transformer architectures: ViT, Swin, DeiT
- Loss functions: CE, BCE, Weighted CE, Focal Loss, ASL Loss
- Comprehensive metrics: Accuracy, AUC-ROC, Recall, Precision, F1
- WandB logging and checkpointing
- Mixed precision training support
"""

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
import timm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from tqdm import tqdm

# Import local modules
from dataset.embed import EmbedDiffusionDataset
from dataset.vindr import VinDr
from dataset.rsna_mammo import RSNAMammo
from dataset.csaw import CSAW
from dataset.synthetic_data import SyntheticMammo
from dataset.transforms import ContrastEnhance
from dataset.concat_dataset import ConcatDataset
from models.asl import AsymmetricLoss, AsymmetricLossOptimized, ASLSingleLabel
from models.balanced_sampler import BalancedBatchSampler, create_balanced_sampler

# Optional WandB support
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


# ═══════════════════════════════════════════════════════════
# Command Line Arguments Parser
# ═══════════════════════════════════════════════════════════

def parse_args():
    """Parse command line arguments for classification training."""
    parser = argparse.ArgumentParser(
        description="Train classification models on breast mammography datasets"
    )
    
    # ═══════════════════════════════════════════════════════════
    # Dataset Configuration
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="rsna-mammo",
        help="Dataset(s) to use for training. Use colon ':' to concatenate multiple datasets. "
             "Supported: embed, rsna-mammo, vindr-mammo, csaw, synthetic. "
             "Use 'synthetic' for synthetic mammography data (requires --synthetic_data_dir).",
    )
    parser.add_argument(
        "--synthetic_data_dir",
        type=str,
        default=None,
        help="Path to directory containing synthetic mammography data. "
             "Required when using 'synthetic' dataset. Can also be used alongside other datasets "
             "to add synthetic data as supplementary training data.",
    )
    parser.add_argument(
        "--data_pct",
        type=float,
        default=1.0,
        help="Percentage of training data to use (for debugging or ablation studies).",
    )
    parser.add_argument(
        "--screen_only",
        action="store_true",
        help="Use only screening images (EMBED dataset).",
    )
    parser.add_argument(
        "--pred_density",
        action="store_true",
        help="Predict breast density instead of BIRADS/cancer (multi-class: 4 classes).",
    )
    parser.add_argument(
        "--pred_mass",
        action="store_true",
        help="Predict mass presence (VinDr dataset, binary classification).",
    )
    parser.add_argument(
        "--pred_calc",
        action="store_true",
        help="Predict calcification presence (VinDr dataset, binary classification).",
    )
    
    # ═══════════════════════════════════════════════════════════
    # Model Configuration
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        help="Model architecture to use. Supports: resnet18, resnet34, resnet50, resnet101, resnet152, "
             "resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, wide_resnet50_2, wide_resnet101_2, "
             "densenet121, densenet169, densenet201, densenet161, "
             "efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, "
             "efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_v2_s, efficientnet_v2_m, "
             "efficientnet_v2_l, convnext_tiny, convnext_small, convnext_base, convnext_large, "
             "vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14, "
             "swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b, "
             "deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224, deit_base_patch16_384, "
             "dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14, "
             "dinov2_vits14_reg, dinov2_vitb14_reg, dinov2_vitl14_reg, dinov2_vitg14_reg",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet pretrained weights for model initialization.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of output classes. If None, automatically inferred from dataset.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate for the classifier head.",
    )
    
    # ═══════════════════════════════════════════════════════════
    # Training Configuration
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./runs/classification",
        help="Directory to save model checkpoints and logs.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this training run (used for WandB and output directory).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Input image resolution (images will be resized to resolution x resolution).",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size per GPU/device during training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Batch size per GPU/device during evaluation.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization) coefficient.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["step", "cosine", "plateau", "none"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs for learning rate scheduler.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps before optimizer update.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping.",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision (FP16) training.",
    )
    
    # ═══════════════════════════════════════════════════════════
    # Loss Function Configuration
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ce",
        choices=["ce", "bce", "weighted_ce", "focal", "asl", "asl_single"],
        help="Loss function to use. Options: ce (cross-entropy), bce (binary cross-entropy), "
             "weighted_ce (class-weighted CE), focal (focal loss), asl (asymmetric loss multi-label), "
             "asl_single (asymmetric loss single-label).",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Alpha parameter for focal loss (balancing factor).",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss (focusing parameter).",
    )
    parser.add_argument(
        "--asl_gamma_neg",
        type=float,
        default=4.0,
        help="Negative gamma parameter for ASL loss.",
    )
    parser.add_argument(
        "--asl_gamma_pos",
        type=float,
        default=1.0,
        help="Positive gamma parameter for ASL loss.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for cross-entropy loss.",
    )
    
    # ═══════════════════════════════════════════════════════════
    # Data Augmentation Configuration
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--contrast_enhance",
        action="store_true",
        help="Apply contrast enhancement to images.",
    )
    parser.add_argument(
        "--ce_mode",
        type=str,
        default="histeq",
        choices=["histeq", "clahe", "minmax"],
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Apply random horizontal flip augmentation.",
    )
    parser.add_argument(
        "--random_rotation",
        type=int,
        default=0,
        help="Random rotation degree range (e.g., 15 for ±15 degrees).",
    )
    parser.add_argument(
        "--random_crop",
        action="store_true",
        help="Use random crop instead of center crop.",
    )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="Apply color jitter augmentation.",
    )
    
    # ═══════════════════════════════════════════════════════════
    # Balanced Sampling Configuration (for imbalanced datasets)
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--min_positive_per_batch",
        type=int,
        default=0,
        help="Minimum number of positive (minority class) samples per batch. "
             "Set to 0 to disable balanced sampling and use standard random sampling. "
             "Set to 1 for at least 1 positive sample per batch, 2 for at least 2, etc. "
             "This helps prevent gradient spikes from batches with only negative samples. "
             "Only applicable for binary classification.",
    )
    parser.add_argument(
        "--drop_last",
        action="store_true",
        help="Drop the last incomplete batch when using balanced sampling.",
    )
    
    # ═══════════════════════════════════════════════════════════
    # Evaluation and Logging Configuration
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--eval_every_n_epochs",
        type=int,
        default=1,
        help="Evaluate model every N epochs.",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=5,
        help="Save checkpoint every N epochs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="breast-classification",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (username or team name).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    
    args = parser.parse_args()
    return args


# ═══════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# ═══════════════════════════════════════════════════════════
# Model Creation Functions
# ═══════════════════════════════════════════════════════════

def create_model(args, num_classes):
    """
    Create a classification model based on the specified architecture.
    
    Args:
        args: Command line arguments
        num_classes: Number of output classes
        
    Returns:
        model: PyTorch model
    """
    model_name = args.model_name.lower()
    
    print(f"Creating model: {model_name} with {num_classes} classes")
    
    # Use the new 'weights' parameter instead of deprecated 'pretrained'
    # weights="DEFAULT" loads the best available weights, weights=None loads random weights
    weights = "DEFAULT" if args.pretrained else None
    
    # ═══════════════════════════════════════════════════════════
    # ResNet Models
    # ═══════════════════════════════════════════════════════════
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet152":
        model = models.resnet152(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # ═══════════════════════════════════════════════════════════
    # ResNeXt Models
    # ═══════════════════════════════════════════════════════════
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnext101_32x8d":
        model = models.resnext101_32x8d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnext101_64x4d":
        model = models.resnext101_64x4d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # ═══════════════════════════════════════════════════════════
    # Wide ResNet Models
    # ═══════════════════════════════════════════════════════════
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "wide_resnet101_2":
        model = models.wide_resnet101_2(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # ═══════════════════════════════════════════════════════════
    # DenseNet Models
    # ═══════════════════════════════════════════════════════════
    elif model_name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "densenet169":
        model = models.densenet169(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "densenet201":
        model = models.densenet201(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "densenet161":
        model = models.densenet161(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # ═══════════════════════════════════════════════════════════
    # EfficientNet Models
    # ═══════════════════════════════════════════════════════════
    elif model_name in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", 
                        "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
                        "efficientnet_b6", "efficientnet_b7"]:
        model_fn = getattr(models, model_name)
        model = model_fn(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name in ["efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"]:
        model_fn = getattr(models, model_name)
        model = model_fn(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # ═══════════════════════════════════════════════════════════
    # ConvNeXt Models
    # ═══════════════════════════════════════════════════════════
    elif model_name in ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]:
        model_fn = getattr(models, model_name)
        model = model_fn(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    
    # ═══════════════════════════════════════════════════════════
    # Vision Transformer Models
    # ═══════════════════════════════════════════════════════════
    elif model_name in ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"]:
        model_fn = getattr(models, model_name)
        model = model_fn(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # ═══════════════════════════════════════════════════════════
    # Swin Transformer Models
    # ═══════════════════════════════════════════════════════════
    elif model_name in ["swin_t", "swin_s", "swin_b", "swin_v2_t", "swin_v2_s", "swin_v2_b"]:
        model_fn = getattr(models, model_name)
        model = model_fn(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
    
    # ═══════════════════════════════════════════════════════════
    # TIMM Models (DeiT and others)
    # ═══════════════════════════════════════════════════════════
    elif model_name.startswith("deit_"):
        try:
            model = timm.create_model(model_name, pretrained=args.pretrained, num_classes=num_classes)
        except Exception as e:
            print(f"Error loading TIMM model {model_name}: {e}")
            print("Falling back to resnet50")
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # ═══════════════════════════════════════════════════════════
    # DINOv2 Models (via torch.hub)
    # ═══════════════════════════════════════════════════════════
    elif model_name in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
                        "dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg"]:
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=args.pretrained)
        feat_dim = model.embed_dim
        model.head = nn.Linear(feat_dim, num_classes)
        # # Override forward to use cls token + classification head
        # _original_forward = model.forward
        # def _dinov2_forward(x, _orig=_original_forward, _head=model.head):
        #     features = _orig(x, is_training=False)
        #     print(features)
        #     if isinstance(features, dict):
        #         cls_token = features["x_norm_clstoken"]
        #     else:
        #         cls_token = features
        #     return _head(cls_token)
        # model.forward = _dinov2_forward
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Add dropout if specified
    if args.dropout > 0.0:
        # This is a simplified approach - for better control, modify the classifier head
        print(f"Note: Dropout {args.dropout} specified but not implemented for all models")
    
    # No need to convert the first layer since we use RGB input
    # # Convert to grayscale input if needed (mammograms are typically grayscale)
    # # Modify the first convolutional layer to accept 1 channel instead of 3
    # if hasattr(model, 'conv1'):
    #     # For ResNet-like models
    #     original_conv = model.conv1
    #     model.conv1 = nn.Conv2d(
    #         1, original_conv.out_channels,
    #         kernel_size=original_conv.kernel_size,
    #         stride=original_conv.stride,
    #         padding=original_conv.padding,
    #         bias=False
    #     )
    #     # Initialize with mean of RGB weights
    #     if args.pretrained:
    #         with torch.no_grad():
    #             model.conv1.weight = nn.Parameter(
    #                 original_conv.weight.mean(dim=1, keepdim=True)
    #             )
    # elif hasattr(model, 'features') and hasattr(model.features, '0'):
    #     # For DenseNet and some other models
    #     if isinstance(model.features[0], nn.Conv2d):
    #         original_conv = model.features[0]
    #         model.features[0] = nn.Conv2d(
    #             1, original_conv.out_channels,
    #             kernel_size=original_conv.kernel_size,
    #             stride=original_conv.stride,
    #             padding=original_conv.padding,
    #             bias=False
    #         )
    #         if args.pretrained:
    #             with torch.no_grad():
    #                 model.features[0].weight = nn.Parameter(
    #                     original_conv.weight.mean(dim=1, keepdim=True)
    #                 )
    # elif hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
    #     # For Vision Transformers
    #     original_conv = model.patch_embed.proj
    #     if isinstance(original_conv, nn.Conv2d):
    #         model.patch_embed.proj = nn.Conv2d(
    #             1, original_conv.out_channels,
    #             kernel_size=original_conv.kernel_size,
    #             stride=original_conv.stride,
    #             padding=original_conv.padding
    #         )
    #         if args.pretrained:
    #             with torch.no_grad():
    #                 model.patch_embed.proj.weight = nn.Parameter(
    #                     original_conv.weight.mean(dim=1, keepdim=True)
    #                 )
    
    return model


# ═══════════════════════════════════════════════════════════
# Loss Function Classes
# ═══════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions (logits) of shape (batch_size, num_classes)
            targets: ground truth labels of shape (batch_size,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(args, class_weights=None):
    """
    Create loss function based on configuration.
    
    Args:
        args: Command line arguments
        class_weights: Class weights for weighted loss (optional)
        
    Returns:
        loss_fn: Loss function
    """
    loss_type = args.loss_type.lower()
    
    if loss_type == "ce":
        # Standard Cross-Entropy Loss
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            loss_fn = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=args.label_smoothing
            )
        else:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"Using Cross-Entropy Loss (label_smoothing={args.label_smoothing})")
    
    elif loss_type == "bce":
        # Binary Cross-Entropy Loss (for multi-label or binary classification)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        print("Using Binary Cross-Entropy Loss")
    
    elif loss_type == "weighted_ce":
        # Weighted Cross-Entropy Loss
        if class_weights is None:
            print("Warning: weighted_ce specified but no class weights provided. Using standard CE.")
            loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            loss_fn = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=args.label_smoothing
            )
        print(f"Using Weighted Cross-Entropy Loss with weights: {class_weights}")
    
    elif loss_type == "focal":
        # Focal Loss
        loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    
    elif loss_type == "asl":
        # Asymmetric Loss (multi-label)
        loss_fn = AsymmetricLossOptimized(
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos
        )
        print(f"Using Asymmetric Loss (gamma_neg={args.asl_gamma_neg}, gamma_pos={args.asl_gamma_pos})")
    
    elif loss_type == "asl_single":
        # Asymmetric Loss (single-label)
        loss_fn = ASLSingleLabel(
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos
        )
        print(f"Using Asymmetric Loss Single-Label (gamma_neg={args.asl_gamma_neg}, gamma_pos={args.asl_gamma_pos})")
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss_fn


# ═══════════════════════════════════════════════════════════
# Dataset Loading Functions
# ═══════════════════════════════════════════════════════════

def get_transforms(args, is_training=True):
    """
    Create image transformation pipeline.
    
    Args:
        args: Command line arguments
        is_training: Whether transforms are for training (with augmentation) or evaluation
        
    Returns:
        transform: Composed transforms
    """
    transform_list = []
    
    if is_training:
        # Training augmentations
        if args.contrast_enhance:
            transform_list.append(ContrastEnhance(method=args.ce_mode))
        
        if args.random_crop:
            transform_list.append(transforms.RandomResizedCrop(args.resolution, scale=(0.8, 1.0)))
        else:
            transform_list.append(transforms.Resize((args.resolution, args.resolution)))
        
        if args.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if args.random_rotation > 0:
            transform_list.append(transforms.RandomRotation(degrees=args.random_rotation))
        
        if args.color_jitter:
            transform_list.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
            )
    else:
        # Evaluation transforms (no augmentation)
        if args.contrast_enhance:
            transform_list.append(ContrastEnhance(method=args.ce_mode))
        transform_list.append(transforms.Resize((args.resolution, args.resolution)))
    
    # Convert to tensor and normalize
    transform_list.append(transforms.ToTensor())
    # Note: For grayscale images, normalization is typically done with single-channel mean/std
    # transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    
    return transforms.Compose(transform_list)


def load_datasets(args):
    """
    Load training and validation datasets based on configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_classes: Number of classes in the dataset
    """
    # Create dataset arguments namespace
    dataset_args = SimpleNamespace(
        pred_density=args.pred_density,
        pred_mass=args.pred_mass,
        pred_calc=args.pred_calc,
        screen_only=args.screen_only,
        data_pct=args.data_pct,
        aug_text=False,
        heavy_aug=False,
        structural_cap=True,
        save_mask=False,
        health_only=False,
        cancer_only=False
    )
    
    # Parse dataset names (support multiple datasets separated by colon)
    dataset_names = args.dataset_name.split(":")
    
    train_datasets = []
    val_datasets = []
    num_classes = None
    
    # Get transforms
    train_transform = get_transforms(args, is_training=True)
    val_transform = get_transforms(args, is_training=False)
    
    for dataset_name in dataset_names:
        dataset_name = dataset_name.strip()
        print(f"\nLoading dataset: {dataset_name}")
        
        if dataset_name == "embed":
            # EMBED Dataset
            train_ds = EmbedDiffusionDataset(
                dataset_args, split="train", transform=train_transform
            )
            val_ds = EmbedDiffusionDataset(
                dataset_args, split="valid", transform=val_transform
            )
            
            # Determine number of classes
            if args.pred_density:
                dataset_num_classes = 4  # Breast density classes
            elif args.screen_only:
                dataset_num_classes = 3  # BIRADS 0, 1, 2
            else:
                dataset_num_classes = 7  # BIRADS 0-6
            
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            print(f"EMBED: Loaded {len(train_ds)} training, {len(val_ds)} validation samples")
        
        elif dataset_name == "vindr-mammo":
            # VinDr Mammography Dataset
            train_ds = VinDr(dataset_args, split="train", transform=train_transform)
            val_ds = VinDr(dataset_args, split="test", transform=val_transform)
            
            # Determine number of classes
            if args.pred_density:
                dataset_num_classes = 4  # Breast density classes
            elif args.pred_mass:
                dataset_num_classes = 2  # Mass presence
            elif args.pred_calc:
                dataset_num_classes = 2  # Calcification presence
            else:
                dataset_num_classes = 5  # BIRADS 1-5
            
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            print(f"VinDr: Loaded {len(train_ds)} training, {len(val_ds)} validation samples")
        
        elif dataset_name == "rsna-mammo":
            # RSNA Breast Cancer Detection Dataset
            train_ds = RSNAMammo(dataset_args, split="train", transform=train_transform)
            val_ds = RSNAMammo(dataset_args, split="test", transform=val_transform)
            
            dataset_num_classes = 2  # Binary: cancer or no cancer
            
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            print(f"RSNA: Loaded {len(train_ds)} training, {len(val_ds)} validation samples")
        elif dataset_name == "csaw":
            train_ds = CSAW(dataset_args, split="train", transform=train_transform)
            val_ds = CSAW(dataset_args, split="test", transform=val_transform)

            dataset_num_classes = 2  # Binary: cancer or no cancer

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            print(f"CSAW: Loaded {len(train_ds)} training, {len(val_ds)} validation samples")
        elif dataset_name == "synthetic":
            # Synthetic Mammography Dataset (training only, no validation split)
            if args.synthetic_data_dir is None:
                raise ValueError(
                    "--synthetic_data_dir is required when using 'synthetic' dataset."
                )
            train_ds = SyntheticMammo(
                data_dir=args.synthetic_data_dir,
                transform=train_transform,
            )

            dataset_num_classes = 2  # Binary: cancer or no cancer

            train_datasets.append(train_ds)
            # No validation set for synthetic data
            print(f"Synthetic: Loaded {len(train_ds)} training samples (no validation split)")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Set or verify number of classes
        if num_classes is None:
            num_classes = dataset_num_classes
        else:
            if num_classes != dataset_num_classes:
                print(f"Warning: Inconsistent number of classes across datasets!")
                print(f"Using the first dataset's number of classes: {num_classes}")
    
    # Concatenate datasets if multiple
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        print(f"\nCombined datasets: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    else:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]
    
    # Override num_classes if specified in args
    if args.num_classes is not None:
        num_classes = args.num_classes
        print(f"Overriding number of classes to: {num_classes}")
    
    return train_dataset, val_dataset, num_classes


def compute_class_weights(dataset, num_classes):
    """
    Compute class weights for handling imbalanced datasets.
    
    Args:
        dataset: PyTorch dataset
        num_classes: Number of classes
        
    Returns:
        class_weights: List of class weights
    """
    print("\nComputing class weights for imbalanced dataset...")
    
    # Count samples per class
    class_counts = np.zeros(num_classes)
    
    # Sample a subset if dataset is very large
    sample_size = min(len(dataset), 10000)
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in tqdm(indices, desc="Analyzing class distribution"):
        try:
            try:
                sample = dataset.__getitem__(idx, no_image=True)
            except TypeError:
                sample = dataset.__getitem__(idx)
            # Handle different label formats
            if 'label' in sample:
                label = sample['label']
            else:
                label = sample[1]  # Assume (image, label) tuple
            
            # Convert one-hot to class index if needed
            if torch.is_tensor(label):
                if len(label.shape) > 0 and label.shape[0] == num_classes:
                    label = torch.argmax(label).item()
                else:
                    label = label.item()
            
            if isinstance(label, (int, np.integer)):
                class_counts[label] += 1
        except Exception as e:
            continue
    
    # Scale counts to full dataset
    scale_factor = len(dataset) / sample_size
    class_counts = class_counts * scale_factor
    
    # Compute weights (inverse frequency)
    total_samples = np.sum(class_counts)
    class_weights = []
    
    for i, count in enumerate(class_counts):
        if count > 0:
            weight = total_samples / (num_classes * count)
            class_weights.append(weight)
        else:
            class_weights.append(1.0)
    
    print(f"Class distribution: {class_counts.astype(int)}")
    print(f"Class weights: {[f'{w:.4f}' for w in class_weights]}")
    
    return class_weights


# ═══════════════════════════════════════════════════════════
# Training and Evaluation Functions
# ═══════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, args, epoch, scaler=None):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        args: Command line arguments
        epoch: Current epoch number
        scaler: GradScaler for mixed precision training
        
    Returns:
        avg_loss: Average training loss
        metrics: Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Extract images and labels from batch
        if isinstance(batch, dict):
            images = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
        else:
            images, labels = batch[0].to(device), batch[1].to(device)
        
        # Convert one-hot labels to class indices if needed
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            if args.loss_type in ['bce', 'asl']:
                # Keep one-hot for multi-label losses
                pass
            else:
                # Convert to class indices for single-label losses
                labels = torch.argmax(labels, dim=1)
        
        # Forward pass with mixed precision
        if args.mixed_precision and scaler is not None:
            with torch.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss = loss / args.gradient_accumulation_steps
        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        if args.mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            if args.mixed_precision and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss += loss.item() * args.gradient_accumulation_steps
        
        # Get predictions
        if args.loss_type in ['bce', 'asl']:
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
        else:
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
        
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_probs.append(probs.detach().cpu())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}'})
    
    # Compute average loss
    avg_loss = total_loss / len(dataloader)
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels, all_probs, args)
    metrics['loss'] = avg_loss
    
    return avg_loss, metrics


def evaluate(model, dataloader, loss_fn, device, args, split="Val"):
    """
    Evaluate the model.
    
    Args:
        model: PyTorch model
        dataloader: Evaluation data loader
        loss_fn: Loss function
        device: Device to evaluate on
        args: Command line arguments
        split: Split name for logging (e.g., "Val", "Test")
        
    Returns:
        avg_loss: Average evaluation loss
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"{split}")
        
        for batch in progress_bar:
            # Extract images and labels from batch
            if isinstance(batch, dict):
                images = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)
            else:
                images, labels = batch[0].to(device), batch[1].to(device)
            
            # Convert one-hot labels to class indices if needed
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                if args.loss_type in ['bce', 'asl']:
                    # Keep one-hot for multi-label losses
                    pass
                else:
                    # Convert to class indices for single-label losses
                    labels = torch.argmax(labels, dim=1)
            
            # Forward pass
            if args.mixed_precision:
                with torch.amp.autocast():
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            if args.loss_type in ['bce', 'asl']:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            else:
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute average loss
    avg_loss = total_loss / len(dataloader)
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels, all_probs, args)
    metrics['loss'] = avg_loss
    
    return avg_loss, metrics


def compute_metrics(preds, labels, probs, args):
    """
    Compute evaluation metrics.
    
    Args:
        preds: Predictions
        labels: Ground truth labels
        probs: Prediction probabilities
        args: Command line arguments
        
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}
    
    # Convert to numpy
    preds_np = preds.numpy()
    labels_np = labels.numpy()
    probs_np = probs.numpy()
    
    # Handle multi-label vs single-label
    if args.loss_type in ['bce', 'asl']:
        # Multi-label classification
        # Average across all labels
        accuracy = accuracy_score(labels_np, preds_np)
        metrics['accuracy'] = accuracy
        
        # Compute AUC-ROC for each class
        try:
            auc_scores = []
            for i in range(labels_np.shape[1]):
                if len(np.unique(labels_np[:, i])) > 1:
                    auc = roc_auc_score(labels_np[:, i], probs_np[:, i])
                    auc_scores.append(auc)
            metrics['auc_roc'] = np.mean(auc_scores) if auc_scores else 0.0
        except Exception as e:
            metrics['auc_roc'] = 0.0
        
        # Other metrics
        metrics['balanced_accuracy'] = balanced_accuracy_score(labels_np, preds_np)
        metrics['precision'] = precision_score(labels_np, preds_np, average='macro', zero_division=0)
        metrics['recall'] = recall_score(labels_np, preds_np, average='macro', zero_division=0)
        metrics['f1'] = f1_score(labels_np, preds_np, average='macro', zero_division=0)
    
    else:
        # Single-label classification
        accuracy = accuracy_score(labels_np, preds_np)
        metrics['accuracy'] = accuracy
        metrics['balanced_accuracy'] = balanced_accuracy_score(labels_np, preds_np)
        
        # Compute AUC-ROC
        try:
            num_classes = probs_np.shape[1] if len(probs_np.shape) > 1 else 2
            if num_classes == 2:
                # Binary classification
                auc = roc_auc_score(labels_np, probs_np[:, 1] if len(probs_np.shape) > 1 else probs_np)
            else:
                # Multi-class classification
                auc = roc_auc_score(labels_np, probs_np, multi_class='ovr', average='macro')
            metrics['auc_roc'] = auc
        except Exception as e:
            print(f"Warning: Could not compute AUC-ROC: {e}")
            metrics['auc_roc'] = 0.0
        
        # Other metrics
        metrics['precision'] = precision_score(labels_np, preds_np, average='macro', zero_division=0)
        metrics['recall'] = recall_score(labels_np, preds_np, average='macro', zero_division=0)
        metrics['f1'] = f1_score(labels_np, preds_np, average='macro', zero_division=0)
    
    # Compute confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(labels_np, preds_np)
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, args, filename="checkpoint.pth"):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Evaluation metrics
        args: Command line arguments
        filename: Checkpoint filename
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
        'args': vars(args),
    }
    
    filepath = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        checkpoint_path: Path to checkpoint file
        
    Returns:
        start_epoch: Epoch to resume from
        best_metrics: Best metrics from checkpoint
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_metrics = checkpoint.get('metrics', {})
    
    print(f"Resumed from epoch {start_epoch}")
    return start_epoch, best_metrics


# ═══════════════════════════════════════════════════════════
# Main Training Function
# ═══════════════════════════════════════════════════════════

def main():
    """Main training function."""
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    if args.run_name is not None:
        args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Breast Image Classification Training")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    
    # Initialize WandB
    if args.use_wandb and _WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
        print("WandB logging enabled")
    
    # Get device
    device = get_device()
    
    # Load datasets
    print("\n" + "="*70)
    print("Loading Datasets")
    print("="*70)
    train_dataset, val_dataset, num_classes = load_datasets(args)
    print(f"\nTotal training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Create data loaders
    # Use balanced batch sampler for imbalanced binary classification if specified
    if args.min_positive_per_batch > 0 and num_classes == 2:
        print(f"\nUsing balanced batch sampler with min {args.min_positive_per_batch} positive samples per batch")
        balanced_sampler = BalancedBatchSampler(
            dataset=train_dataset,
            batch_size=args.train_batch_size,
            min_positive_per_batch=args.min_positive_per_batch,
            drop_last=args.drop_last,
            num_classes=num_classes,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=balanced_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        if args.min_positive_per_batch > 0 and num_classes != 2:
            print(f"Warning: Balanced sampling is only supported for binary classification (num_classes=2). "
                  f"Current num_classes={num_classes}. Using standard random sampling.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Compute class weights for imbalanced data
    class_weights = None
    if args.loss_type in ["weighted_ce"]:
        class_weights = compute_class_weights(train_dataset, num_classes)
    
    # Create model
    print("\n" + "="*70)
    print("Creating Model")
    print("="*70)
    model = create_model(args, num_classes)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    loss_fn = get_loss_function(args, class_weights)
    if class_weights is not None:
        loss_fn = loss_fn.to(device)
    
    # Create optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    print(f"Optimizer: {args.optimizer}")
    
    # Create learning rate scheduler
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )
    elif args.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    
    if scheduler is not None:
        print(f"Learning rate scheduler: {args.scheduler}")
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    if args.mixed_precision:
        print("Mixed precision training enabled")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_auc = 0.0
    if args.resume_from is not None:
        start_epoch, best_metrics = load_checkpoint(model, optimizer, scheduler, args.resume_from)
        best_auc = best_metrics.get('auc_roc', 0.0)
    
    # Training loop
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train for one epoch
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, args, epoch, scaler
        )
        
        # Print training metrics
        print(f"\nEpoch {epoch+1}/{args.num_epochs} - Training Results:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {train_metrics['balanced_accuracy']:.4f}")
        print(f"  AUC-ROC: {train_metrics['auc_roc']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall: {train_metrics['recall']:.4f}")
        print(f"  F1 Score: {train_metrics['f1']:.4f}")
        
        # Log to WandB
        if args.use_wandb and _WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/accuracy': train_metrics['accuracy'],
                'train/balanced_accuracy': train_metrics['balanced_accuracy'],
                'train/auc_roc': train_metrics['auc_roc'],
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'train/f1': train_metrics['f1'],
                'learning_rate': optimizer.param_groups[0]['lr'],
            })
        
        # Evaluate on validation set
        if (epoch + 1) % args.eval_every_n_epochs == 0:
            val_loss, val_metrics = evaluate(
                model, val_loader, loss_fn, device, args, split="Val"
            )
            
            # Print validation metrics
            print(f"\nEpoch {epoch+1}/{args.num_epochs} - Validation Results:")
            print(f"  Loss: {val_loss:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
            print(f"  AUC-ROC: {val_metrics['auc_roc']:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f}")
            print(f"  Recall: {val_metrics['recall']:.4f}")
            print(f"  F1 Score: {val_metrics['f1']:.4f}")
            
            # Log to WandB
            if args.use_wandb and _WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch + 1,
                    'val/loss': val_loss,
                    'val/accuracy': val_metrics['accuracy'],
                    'val/balanced_accuracy': val_metrics['balanced_accuracy'],
                    'val/auc_roc': val_metrics['auc_roc'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall'],
                    'val/f1': val_metrics['f1'],
                })
            
            # Save best model based on AUC-ROC
            if val_metrics['auc_roc'] > best_auc:
                best_auc = val_metrics['auc_roc']
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics, args,
                    filename="best_model.pth"
                )
                print(f"  New best model! AUC-ROC: {best_auc:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every_n_epochs == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_metrics, args,
                filename=f"checkpoint_epoch_{epoch+1}.pth"
            )
        
        # Update learning rate
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss if (epoch + 1) % args.eval_every_n_epochs == 0 else train_loss)
            else:
                scheduler.step()
    
    # Final evaluation
    print("\n" + "="*70)
    print("Final Evaluation")
    print("="*70)
    
    # Load best model
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from: {best_model_path}")
    
    # Evaluate on validation set
    val_loss, val_metrics = evaluate(
        model, val_loader, loss_fn, device, args, split="Final Val"
    )
    
    print(f"\nFinal Validation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
    print(f"  AUC-ROC: {val_metrics['auc_roc']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1 Score: {val_metrics['f1']:.4f}")
    
    # Log confusion matrix for final evaluation
    print(f"\nFinal Validation Confusion Matrix:")
    cm = val_metrics.get('confusion_matrix')
    if cm is not None:
        print(cm)
    
    # Log final metrics to WandB
    if args.use_wandb and _WANDB_AVAILABLE:
        wandb.log({
            'final/val_loss': val_loss,
            'final/val_accuracy': val_metrics['accuracy'],
            'final/val_balanced_accuracy': val_metrics['balanced_accuracy'],
            'final/val_auc_roc': val_metrics['auc_roc'],
            'final/val_precision': val_metrics['precision'],
            'final/val_recall': val_metrics['recall'],
            'final/val_f1': val_metrics['f1'],
        })
        if cm is not None:
            try:
                wandb.log({"final/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=list(range(cm.shape[0])), preds=list(range(cm.shape[0])),
                    class_names=[str(i) for i in range(cm.shape[0])])})
            except Exception:
                pass
        wandb.finish()
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best validation AUC-ROC: {best_auc:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
