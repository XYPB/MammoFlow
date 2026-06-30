import os
import warnings
from collections import OrderedDict
from typing import Union
from enum import Enum
import numpy as np
import torch
import models.dino_transformer as vits
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MY_API_TOKEN = os.environ.get("HF_TOKEN")


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def pad_tensor(x, h, w):
    """
    Pad a tensor to the desired height and width with zeros.
    
    Args:
        x (torch.Tensor): Input tensor with shape (B, C, H, W)
        h (int): Desired height
        w (int): Desired width
        
    Returns:
        torch.Tensor: Padded tensor with shape (B, C, h, w)
    """
    b, c, height, width = x.shape
    if height == h and width == w:
        return x
    
    # Calculate padding
    pad_h = max(0, h - height)
    pad_w = max(0, w - width)
    
    # Compute padding for each side
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Use functional padding to ensure gradients are preserved
    padded_x = torch.nn.functional.pad(
        x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0
    )
    
    return padded_x


class Weights(Enum):
    LVD142M = "LVD142M"


def _parse_dinov2_model_name(dino_model_name):
    # dinov2_vitb14_reg_lc
    items = dino_model_name.split("_")
    num_register_tokens = 4 if items[-1] == 'reg' else 0
    model_size = items[1][3]
    patch_size = int(items[1][4:])
    if model_size == 's':
        arch_name = 'vit_small'
        if patch_size == 14:
            if  num_register_tokens > 0:
                pretrained = os.path.expanduser('~/.cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth')
            else:
                pretrained = os.path.expanduser('~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
        else:
            pretrained = None
    elif model_size == 'b':
        arch_name = 'vit_base'
        if patch_size == 14:
            if num_register_tokens > 0:
                pretrained = os.path.expanduser('~/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth')
            else:
                pretrained = os.path.expanduser('~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth')
        else:
            pretrained = None
    elif model_size == 'l':
        arch_name = 'vit_large'
        if patch_size == 14 and num_register_tokens > 0:
            pretrained = os.path.expanduser('~/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth')
        else:
            pretrained = None
    else:
        arch_name = 'vit_giant2'
        warnings.warn('Using the large model w/o pretraining.')
        pretrained = None
    return arch_name, pretrained, num_register_tokens, patch_size


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: str = None,
    weights: Union[Weights, str] = Weights.LVD142M,
    grad_ckpt: bool = False,
    **kwargs,
):
    

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
        grad_ckpt=grad_ckpt,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu")
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise e
    return model


def images_to_video(images, output_path, fps=30, quality=9):
    """
    Convert a sequence of PIL images into a single MP4 video.
    
    Args:
        images (List[PIL.Image.Image]): List of PIL images.
        output_path (str): Path where the output MP4 file will be saved.
        fps (int, optional): Frames per second. Defaults to 30.
        quality (int, optional): Video quality (0-10, higher is better). Defaults to 8.
        
    Returns:
        str: Path to the created video file.
    """
    import numpy as np
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    
    # Convert PIL images to numpy arrays
    frames = [np.array(img) for img in images]
    
    # Create video clip
    clip = ImageSequenceClip(frames, fps=fps)
    
    # Write to file
    clip.write_videofile(output_path, codec="libx264", fps=fps, 
                         bitrate=f"{quality}M", audio=False, logger=None)
    
    return output_path

def background_masking(img, mask):
    # Blur the mask and apply it to the image
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (7, 7), 2.0)
    mask = mask / 255.0
    mask[mask > 1e-5] = 1.0
    mask = np.clip(mask, 0, 1)
    mask = mask.squeeze()[:, :, None]
    img = (img * mask).astype(np.uint8)
    return img