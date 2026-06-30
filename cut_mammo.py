from dataset.transforms import OtsuCut, ContrastEnhance
from glob import glob
import os
from PIL import Image
import sys
from tqdm import tqdm

def cut_mammo(input_dir, output_dir, contrast_enhance=False, contrast_method="histeq"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')) + glob(os.path.join(input_dir, '*.jpeg'))
    if len(image_paths) == 0:
        image_paths = glob(os.path.join(input_dir, '**.png')) + glob(os.path.join(input_dir, '**.jpg')) + glob(os.path.join(input_dir, '**.jpeg'))

    otsu_cut = OtsuCut(False, True)
    contrast_enhance_transform = ContrastEnhance(method=contrast_method)
    for img_path in tqdm(image_paths):
        img = Image.open(img_path).convert('RGB')
        cut_img = otsu_cut(img)
        if contrast_enhance:
            cut_img = contrast_enhance_transform(cut_img)
        # base_name = os.path.basename(img_path)
        # save_path = os.path.join(output_dir, base_name)
        save_path = img_path.replace(input_dir, output_dir)
        if os.path.exists(save_path):
            # print(f"Warning: Output file already exists, skipping: {save_path}")
            continue
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cut_img.convert('L').resize((512, 512)).save(save_path)
        # print(f'Saved cut image to {save_path}')


if __name__ == '__main__':
    if len(sys.argv) > 3:
        print("Usage: python cut_mammo.py <input_dir> enhance(optional)")
    contrast_enhance = (len(sys.argv) == 3)
    if contrast_enhance:
        contrast_method = sys.argv[2]
    else:
        contrast_method = "histeq"  # default method


    input_directory = sys.argv[1]
    if input_directory.endswith('/'):
        input_directory = input_directory[:-1]
    output_directory = input_directory + '_cut' + ('_enhance' if contrast_enhance else '')
    output_directory = output_directory.replace('1024', '512')

    cut_mammo(input_directory, output_directory, contrast_enhance, contrast_method=contrast_method)