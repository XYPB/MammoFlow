import os
import torch
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from dataset.transforms import OtsuCut, RemoveTextLabel
from dataset.embed import EmbedDiffusionDataset
from multiprocessing import Pool, cpu_count

class Args:
    def __init__(self):
        self.pred_density = False
        self.screen_only = True
        self.data_pct = 1
        self.aug_text = False
        self.heavy_aug = False
        self.structural_cap = True
        self.save_mask = False

def process_indices(args_tuple):
    indices, dataset, output_dir = args_tuple
    for i in indices:
        data = dataset.__getitem__(i)
        image = data["image"].numpy().astype('uint8')
        prompt = data["prompt"]
        output_path = os.path.join(output_dir, f"image_{i}.jpg")
        prompt_path = os.path.join(output_dir, f"image_{i}.txt")
        Image.fromarray(image).convert('L').save(output_path)
        with open(prompt_path, "w") as f:
            f.write(prompt)

if __name__ == "__main__":
    args = Args()
    transform = transforms.Compose(
            [
                RemoveTextLabel(),
                transforms.Resize(
                    (1024, 1024)
                ),  # Adjust based on model requirements
            ]
        )
    dataset = EmbedDiffusionDataset(args, split="train", transform=transform)
    output_dir = "~/palmer_scratch/EMBED_FLUX_DATASET"
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    num_workers = 8
    total_len = len(dataset)
    # total_len = 128
    indices = list(range(total_len))
    chunk_size = (total_len + num_workers - 1) // num_workers
    chunks = [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_workers)]
    # Prepare arguments for each worker
    worker_args = [(chunk, deepcopy(dataset), output_dir) for chunk in chunks]
    with Pool(num_workers) as pool:
        list(pool.map(process_indices, worker_args))