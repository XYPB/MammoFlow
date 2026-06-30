import os
import copy
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import diffusers
from diffusers import StableDiffusion3Pipeline
from diffusers.image_processor import VaeImageProcessor

from models.mammo_aligner import mammo_ap_alignment_compute
from train_sd35_lora_mv import import_model_class, tokenize_prompt, encode_prompt

# -----------------------------
# Fill these paths
# -----------------------------
BASE_MODEL_PATH = "stabilityai/stable-diffusion-3.5-medium"            # e.g. /path/to/stable-diffusion-3.5-medium
CHECKPOINT_PATH = "runs/sd35-mv-sbs-fft-csaw-better-40k-emd-1e-1-cosine/transformer"            # e.g. /path/to/output_dir (LoRA) OR /path/to/output_dir/transformer (full)
CHECKPOINT_TYPE = "full_transformer"        # "lora" or "full_transformer"
CC_IMAGE_PATH = "./data/CSAW-CC-resized-1024/train/00817_20990909_L_CC_4_resized.png"              # e.g. /path/to/cc.png
MLO_IMAGE_PATH = "./data/CSAW-CC-resized-1024/train/00817_20990909_L_MLO_4_resized.png"             # e.g. /path/to/mlo.png
OUTPUT_DIR = "./tmp/intermediate_denoising"                 # e.g. /path/to/save/one_step_outputs

PROMPT = "Two healthy left breast mammograms, the first image is cc view and the second image is the opposite view"                     # fill with your report/prompt text
TIMESTEPS_TO_RUN = [900, 700, 500, 300, 100]
RESOLUTION = 512
SMOOTH_SIGMA = 5.0
BACKGROUND_THRESHOLD = 5
SEED = 42
PRECONDITION_OUTPUTS = True
CC_FIRST = True                  # True: combined as [CC | MLO], False: [MLO | CC]

def get_sigmas(noise_scheduler_copy, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def tensor_to_pil_gray(t):
    arr = (t.squeeze().detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("L")


if __name__ == "__main__":
    assert BASE_MODEL_PATH and CHECKPOINT_PATH and CC_IMAGE_PATH and MLO_IMAGE_PATH and OUTPUT_DIR and PROMPT, "Please fill all blank path/prompt fields first."
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Device / dtype
    if torch.cuda.is_available():
        device = torch.device("cuda")
        weight_dtype = torch.float16
    else:
        device = torch.device("cpu")
        weight_dtype = torch.float32

    # Load SD3.5 base pipeline first
    pipe = StableDiffusion3Pipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )

    if CHECKPOINT_TYPE == "lora":
        pipe.load_lora_weights(CHECKPOINT_PATH)
    elif CHECKPOINT_TYPE == "full_transformer":
        # CHECKPOINT_PATH should point to the saved transformer folder.
        from diffusers import SD3Transformer2DModel

        full_transformer = SD3Transformer2DModel.from_pretrained(
            CHECKPOINT_PATH,
            torch_dtype=weight_dtype,
        )
        pipe.transformer.load_state_dict(full_transformer.state_dict())
        del full_transformer
    else:
        raise ValueError("CHECKPOINT_TYPE must be 'lora' or 'full_transformer'.")

    pipe = pipe.to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()
    pipe.text_encoder_3.eval()

    # Build scheduler copy + image processor like training
    noise_scheduler_copy = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
        BASE_MODEL_PATH,
        subfolder="scheduler",
    )
    image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae.config.scaling_factor)

    # Recreate training-style tokenizers/encoders setup
    from transformers import CLIPTokenizer, T5TokenizerFast

    tokenizer_one = CLIPTokenizer.from_pretrained(BASE_MODEL_PATH, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(BASE_MODEL_PATH, subfolder="tokenizer_2")
    tokenizer_three = T5TokenizerFast.from_pretrained(BASE_MODEL_PATH, subfolder="tokenizer_3")

    text_encoder_cls_one = import_model_class(BASE_MODEL_PATH, revision=None, subfolder="text_encoder")
    text_encoder_cls_two = import_model_class(BASE_MODEL_PATH, revision=None, subfolder="text_encoder_2")
    text_encoder_cls_three = import_model_class(BASE_MODEL_PATH, revision=None, subfolder="text_encoder_3")

    text_encoder_one = text_encoder_cls_one.from_pretrained(BASE_MODEL_PATH, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
    text_encoder_two = text_encoder_cls_two.from_pretrained(BASE_MODEL_PATH, subfolder="text_encoder_2", torch_dtype=weight_dtype).to(device)
    text_encoder_three = text_encoder_cls_three.from_pretrained(BASE_MODEL_PATH, subfolder="text_encoder_3", torch_dtype=weight_dtype).to(device)

    if CHECKPOINT_TYPE == "lora":
        # Use LoRA-injected text encoders from pipeline for consistency
        text_encoder_one = pipe.text_encoder
        text_encoder_two = pipe.text_encoder_2
        text_encoder_three = pipe.text_encoder_3

    print("Model + checkpoint loaded.")

    # Prepare paired GT image tensor: [CC | MLO]
    prep = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    cc_img = Image.open(CC_IMAGE_PATH).convert("RGB")
    mlo_img = Image.open(MLO_IMAGE_PATH).convert("RGB")

    cc_tensor = prep(cc_img)   # (3, H, W), normalized to [-1, 1]
    mlo_tensor = prep(mlo_img) # (3, H, W), normalized to [-1, 1]

    if CC_FIRST:
        paired_tensor = torch.cat([cc_tensor, mlo_tensor], dim=2).unsqueeze(0)  # (1, 3, H, 2W)
    else:
        paired_tensor = torch.cat([mlo_tensor, cc_tensor], dim=2).unsqueeze(0)

    paired_tensor = paired_tensor.to(device)

    # Compute operation_dict from GT images (0~1 range), as in training
    with torch.no_grad():
        gt_01 = image_processor.denormalize(paired_tensor)
        if CC_FIRST:
            gt_cc = gt_01[:, :, :, :RESOLUTION]
            gt_mlo = gt_01[:, :, :, RESOLUTION:]
        else:
            gt_cc = gt_01[:, :, :, RESOLUTION:]
            gt_mlo = gt_01[:, :, :, :RESOLUTION]

        _, _, _, operation_dict = mammo_ap_alignment_compute(
            gt_mlo,
            gt_cc,
            smooth_sigma=SMOOTH_SIGMA,
        )

    # Encode prompt embeddings exactly like training script
    with torch.no_grad():
        prompt_embeds, pooled_embeds = encode_prompt(
            [text_encoder_one, text_encoder_two, text_encoder_three],
            [tokenizer_one, tokenizer_two, tokenizer_three],
            [PROMPT],
            max_sequence_length=256,
            device=device,
            weight_dtype=weight_dtype,
        )

    # Encode GT image to latent space
    with torch.no_grad():
        pixel_values = paired_tensor.to(dtype=pipe.vae.dtype)
        latents = pipe.vae.encode(pixel_values).latent_dist.sample()
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        latents = latents.to(dtype=weight_dtype)

    # One-step denoise for each t
    gen = torch.Generator(device=device).manual_seed(SEED)

    for t in TIMESTEPS_TO_RUN:
        with torch.no_grad():
            # timesteps = torch.tensor([t], device=device, dtype=noise_scheduler_copy.timesteps.dtype)
            timesteps = noise_scheduler_copy.timesteps[t].to(
                    device=latents.device
                ).unsqueeze(0)
            # print(timesteps, timesteps.shape)
            timestep_int = int(timesteps.squeeze().item())

            # Add noise at timestep t (training-style)
            noise = torch.randn(latents.shape, generator=gen, device=device, dtype=latents.dtype)
            sigmas = get_sigmas(noise_scheduler_copy, timesteps, device, n_dim=latents.ndim, dtype=latents.dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            # Single denoising forward step through transformer
            model_pred_raw = pipe.transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

            if PRECONDITION_OUTPUTS:
                model_pred = model_pred_raw * (-sigmas) + noisy_latents
            else:
                model_pred = model_pred_raw

            # Reconstruct full image first, then postprocess (as requested)
            reconstruct_latents = model_pred / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            reconstructed_images = pipe.vae.decode(reconstruct_latents).sample  # normalized image tensor
            reconstructed_pil = image_processor.postprocess(reconstructed_images.detach().cpu(), output_type="pil")[0]

            # Apply GT-derived alignment operation dict to reconstructed image
            recon_01 = image_processor.denormalize(reconstructed_images)
            if CC_FIRST:
                recon_cc = recon_01[:, :, :, :RESOLUTION]
                recon_mlo = recon_01[:, :, :, RESOLUTION:]
            else:
                recon_cc = recon_01[:, :, :, RESOLUTION:]
                recon_mlo = recon_01[:, :, :, :RESOLUTION]

            aligned_mlo, aligned_cc, _, _ = mammo_ap_alignment_compute(
                recon_mlo,
                recon_cc,
                operation_dict=operation_dict,
                background_threshold=BACKGROUND_THRESHOLD,
                criterion=None,
                smooth_sigma=SMOOTH_SIGMA,
            )

            aligned_mlo_pil = tensor_to_pil_gray(aligned_mlo)
            aligned_cc_pil = tensor_to_pil_gray(aligned_cc)
            aligned_pair = Image.new("L", (aligned_mlo_pil.width + aligned_cc_pil.width, aligned_mlo_pil.height))
            aligned_pair.paste(aligned_mlo_pil, (0, 0))
            aligned_pair.paste(aligned_cc_pil, (aligned_mlo_pil.width, 0))

            # Save outputs
            reconstructed_pil.save(os.path.join(OUTPUT_DIR, f"denoised_full_t{timestep_int}.png"))
            aligned_pair.save(os.path.join(OUTPUT_DIR, f"denoised_aligned_t{timestep_int}.png"))

    print(f"Saved one-step denoised outputs to: {OUTPUT_DIR}")