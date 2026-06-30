import os
import json
import copy
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from tqdm import tqdm
from torchvision import transforms
from matplotlib import pylab as pylab
from models.mammo_aligner import mammo_ap_alignment_compute
from models.align_utils import display_distribution_res
from models.differentiable_affine import ProjectedEMDLoss
from dataset.transforms import ContrastEnhance
import pickle
from multiprocessing import Pool
from functools import partial
from scipy import stats


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='synthetic_results', help='Directory containing synthetic images and prompts')
parser.add_argument('--compare_dir', type=str, default=None, help='Directory containing real images and prompts for comparison')
parser.add_argument('--save_intermediate', action='store_true', help='Whether to save intermediate results')
parser.add_argument('--contrast_enhance', action='store_true', help='Whether to apply contrast enhancement')
parser.add_argument("--ce_mode",type=str,default="histeq",choices=["histeq", "clahe", "minmax"],)
parser.add_argument('--resolution', type=int, default=512, help='Resolution to resize images to for alignment')
parser.add_argument('--recompute', action='store_true', help='Whether to recompute alignment scores even if they exist')
parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for multiprocessing')
parser.add_argument('--shuffle_view', action='store_true', help='Shuffle the order of the paired image')
parser.add_argument('--compare_shuffle', action='store_true', help='Whether to shuffle the paired images in the compare_dir for ablation study')
args = parser.parse_args()

def load_gt_images(image_dir):
    if 'csaw' in image_dir.lower():
        gt_image_paths_json = './data/csaw_test_1k_paired_cc_mlo.json'
        if '_2' in image_dir:
            gt_image_paths_json = './data/csaw_test_1k_paired_cc_mlo_2.json'
    elif 'rsna' in image_dir.lower():
        gt_image_paths_json = './data/rsna_test_1k_paired_cc_mlo.json'
        if '_2' in image_dir:
            gt_image_paths_json = './data/rsna_test_1k_paired_cc_mlo_2.json'
    elif 'vindr' in image_dir.lower():
        gt_image_paths_json = './data/vindr_test_1k_paired_cc_mlo.json'
        if '_2' in image_dir:
            gt_image_paths_json = './data/vindr_test_1k_paired_cc_mlo_2.json'
    else:
        raise ValueError(f"Unknown dataset in {image_dir}. Cannot determine GT image paths.")
    
    gt_image_paris = json.load(open(gt_image_paths_json, 'r'))
    paired_images = []
    for img1, img2 in gt_image_paris:
        # fixed CC, MLO order
        if 'mlo' in img1.lower():
            cc_img, mlo_img = img2, img1
        else:
            cc_img, mlo_img = img1, img2
        cc_img_path = os.path.join(image_dir, cc_img.replace('/', '_'))
        mlo_img_path = os.path.join(image_dir, mlo_img.replace('/', '_'))
        assert os.path.exists(cc_img_path), f"CC image not found: {cc_img_path}"
        assert os.path.exists(mlo_img_path), f"MLO image not found: {mlo_img_path}"
        paired_images.append((cc_img_path, mlo_img_path))
    
    return paired_images

def load_synthetic_images(image_dir):
    synthetic_paired_images = sorted(glob(os.path.join(image_dir, "img_*.jpg")))
    if len(synthetic_paired_images) == 0:
        synthetic_paired_images = sorted(glob(os.path.join(image_dir, "img_*.png")))
    prompt_list = sorted(glob(os.path.join(image_dir, "*.txt")))

    # CC first and MLO second
    paired_images = []
    for i in range(len(prompt_list)):
        prompt = open(prompt_list[i], 'r').read().strip()
        view = prompt.split('view')[0].split('the first image is')[-1].strip().lower()
        image1_path = synthetic_paired_images[2 * i]
        image2_path = synthetic_paired_images[2 * i + 1]
        if 'cc' in view:
            paired_images.append((image1_path, image2_path))
        else:
            paired_images.append((image2_path, image1_path))
    print(f"Paired images loaded: {len(paired_images)}")
    return paired_images

def load_ca3d_image(image_dir):
    paired_images = []
    cc_images = glob(os.path.join(image_dir, "*_MLO*"))
    for cc_img_path in cc_images:
        mlo_img_path = cc_img_path.replace('_MLO', '_CC')
        assert os.path.exists(cc_img_path), f"CC image not found: {cc_img_path}"
        assert os.path.exists(mlo_img_path), f"MLO image not found for {cc_img_path}"
        paired_images.append((cc_img_path, mlo_img_path))
    return paired_images

def load_image(data_dir):
    if 'inference' in data_dir:
        return load_synthetic_images(data_dir)
    elif 'ca3d' in data_dir.lower():
        return load_ca3d_image(data_dir)
    else:
        return load_gt_images(data_dir)

def _process_single_pair(task, pre_transform, save_intermediate=False, intermediate_dir=None):
    """Worker function to process a single image pair. Designed to be picklable for multiprocessing."""
    img1_path, img2_path = task
    criterion = ProjectedEMDLoss(axis=3, sigma=5.0)

    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')

    img1_tensor = pre_transform(img1).unsqueeze(0)  # Add batch dimension
    img2_tensor = pre_transform(img2).unsqueeze(0)

    # Already ensured CC first
    cc_tensor, mlo_tensor = img1_tensor, img2_tensor

    # Normalize to 0~1
    cc_tensor = (cc_tensor * 0.5 + 0.5).clamp(0, 1)
    mlo_tensor = (mlo_tensor * 0.5 + 0.5).clamp(0, 1)

    aligned_mlo_tensor, aligned_cc_tensor, emd_loss, _ = mammo_ap_alignment_compute(
        mlo_tensor,
        cc_tensor,
        background_threshold=5,
        criterion=criterion,
    )

    aligned_mlo_arr = (aligned_mlo_tensor.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    aligned_cc_arr = (aligned_cc_tensor.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    num_res_dict = display_distribution_res(
        aligned_mlo_arr, aligned_cc_arr, show_img=save_intermediate, save_path=os.path.join(intermediate_dir, f"aligned_{os.path.basename(img1_path)}".replace('.png', '.jpg')) if save_intermediate else None
    )

    return {
        "image_pair": (img1_path, img2_path),
        "emd_loss": emd_loss.item(),
        "emd": num_res_dict["emd"],
        "jsd": num_res_dict["jsd"],
        "mi": num_res_dict["mi"],
        "overlap": num_res_dict["overlap"],
    }


def compute_alignment_scores(paired_images, pre_transform, save_intermediate=False, intermediate_dir=None, num_workers=1):
    tasks = paired_images
    worker_fn = partial(
        _process_single_pair,
        pre_transform=pre_transform,
        save_intermediate=save_intermediate,
        intermediate_dir=intermediate_dir,
    )

    if num_workers <= 1:
        # Single-process fallback
        alignment_scores = []
        for task in tqdm(tasks, total=len(tasks)):
            alignment_scores.append(worker_fn(task))
    else:
        # Multi-process with Pool
        alignment_scores = []
        with Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap(worker_fn, tasks), total=len(tasks)):
                alignment_scores.append(result)

    return alignment_scores


def statistical_comparison(list1, list2, name="EMD Loss", alpha=0.05):
    """Compare two lists of scores using multiple statistical tests.

    Uses non-parametric tests (Mann-Whitney U, Kolmogorov-Smirnov) since
    alignment scores are typically not normally distributed, plus Welch's
    t-test for completeness.

    Args:
        list1: First list of scores.
        list2: Second list of scores.
        name: Name of the metric being compared.
        alpha: Significance level (default 0.05).

    Returns:
        dict with test statistics and p-values.
    """
    list1, list2 = np.array(list1), np.array(list2)

    # Effect size: Cohen's d (pooled std)
    pooled_std = np.sqrt((np.std(list1, ddof=1)**2 + np.std(list2, ddof=1)**2) / 2)
    cohens_d = (np.mean(list1) - np.mean(list2)) / pooled_std

    # Mann-Whitney U test (rank-based, no normality assumption)
    u_stat, mw_p = stats.mannwhitneyu(list1, list2, alternative='two-sided')

    # Two-sample Kolmogorov-Smirnov test (compares full distributions)
    ks_stat, ks_p = stats.ks_2samp(list1, list2)

    # Welch's t-test (does not assume equal variances)
    t_stat, t_p = stats.ttest_ind(list1, list2, equal_var=False)

    print(f"\n--- Statistical Tests for {name} ---")
    print(f"  Mean #1: {np.mean(list1):.6f} ± {np.std(list1):.6f}")
    print(f"  Mean #2: {np.mean(list2):.6f} ± {np.std(list2):.6f}")
    print(f"  Cohen's d:                 {cohens_d:.4f}  ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})")
    print(f"  Mann-Whitney U test:       U={u_stat:.4f}, p={mw_p:.6f}  {'*' if mw_p < alpha else ''}")
    print(f"  Kolmogorov-Smirnov test:   KS={ks_stat:.4f}, p={ks_p:.6f}  {'*' if ks_p < alpha else ''}")
    print(f"  Welch's t-test:            t={t_stat:.4f},  p={t_p:.6f}  {'*' if t_p < alpha else ''}")
    print(f"  (* significant at alpha={alpha})")

    return {
        "mann_whitney": {"U": u_stat, "p": mw_p},
        "ks_test": {"KS": ks_stat, "p": ks_p},
        "welch_t": {"t": t_stat, "p": t_p},
    }


def randomize_paired_images(paired_images):
    img1_list = [p[0] for p in paired_images]
    img2_list = [p[1] for p in paired_images]

    random.seed(42)
    random.shuffle(img2_list)
    randomized_images = list(zip(img1_list, img2_list))
    return randomized_images


if __name__ == "__main__":
    if args.save_intermediate:
        intermediate_dir = os.path.join(args.image_dir, 'alignment_results')
        compare_intermediate_dir = os.path.join(args.compare_dir, 'alignment_results') if args.compare_dir is not None else None
        if args.shuffle_view:
            intermediate_dir = intermediate_dir.replace('alignment_results', 'alignment_results_shuffle')
        if args.compare_shuffle:
            compare_intermediate_dir = intermediate_dir.replace('alignment_results', 'alignment_results_shuffle')
        os.makedirs(intermediate_dir, exist_ok=True)
        os.makedirs(compare_intermediate_dir, exist_ok=True) if compare_intermediate_dir is not None else None
    else:
        intermediate_dir = None
        compare_intermediate_dir =None

    # if model already trained with contrast enhancement, then do not apply contrast enhancement regardless of the flag, otherwise follow the flag
    contrast_flag = False if (args.contrast_enhance and 'contrast' in args.image_dir.lower()) else args.contrast_enhance
    pre_transform = transforms.Compose(
        [
            transforms.RandomApply(
                [ContrastEnhance(method=args.ce_mode)], p=1 if contrast_flag else 0
            ),
            transforms.Resize(
                (args.resolution, args.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),  # Adjust based on model requirements
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    pickle_path = args.image_dir + '_alignment_scores.pkl'
    if args.shuffle_view:
        pickle_path = pickle_path.replace('.pkl', '_shuffle.pkl')
    if contrast_flag:
        pickle_path = pickle_path.replace('.pkl', '_contrast.pkl')

    if os.path.exists(pickle_path) and not args.recompute:
        with open(pickle_path, 'rb') as f:
            alignment_scores = pickle.load(f)
    else:
        paired_images = load_image(args.image_dir)
        if args.shuffle_view:
            paired_images = randomize_paired_images(paired_images)
        assert len(paired_images) > 0, f"No paired images found in {args.image_dir}. Exiting."
        alignment_scores = compute_alignment_scores(paired_images, pre_transform, save_intermediate=args.save_intermediate, intermediate_dir=intermediate_dir, num_workers=args.num_workers)
        with open(pickle_path, 'wb') as f:
            pickle.dump(alignment_scores, f)

    # Print and save the alignment scores
    all_emd_losses = [score["emd_loss"] for score in alignment_scores]
    all_emds = [score["emd"] for score in alignment_scores]
    all_jsds = [score["jsd"] for score in alignment_scores]
    all_mis = [score["mi"] for score in alignment_scores]
    all_overlaps = [score["overlap"] for score in alignment_scores]

    print(f"\n\n################# Alignment Scores for {args.image_dir}:")
    print(f"Average EMD Loss: {np.mean(all_emd_losses):.6f} ± {np.std(all_emd_losses):.6f}")
    print(f"Average EMD: {np.mean(all_emds):.6f} ± {np.std(all_emds):.6f}")
    print(f"Average JSD: {np.mean(all_jsds):.6f} ± {np.std(all_jsds):.6f}")
    print(f"Average MI: {np.mean(all_mis):.6f} ± {np.std(all_mis):.6f}")
    print(f"Average Overlap: {np.mean(all_overlaps):.6f} ± {np.std(all_overlaps):.6f}")


    if args.compare_dir is not None or args.compare_shuffle:
        if args.compare_shuffle:
            pickle_path2 = pickle_path.replace('.pkl', '_shuffle.pkl')
            contrast_flag = False if (args.contrast_enhance and 'contrast' in args.image_dir.lower()) else args.contrast_enhance
        else:
            pickle_path2 = args.compare_dir + '_alignment_scores.pkl'
            contrast_flag = False if (args.contrast_enhance and 'contrast' in args.compare_dir.lower()) else args.contrast_enhance
        if contrast_flag:
            pickle_path2 = pickle_path2.replace('.pkl', '_contrast.pkl')
        if os.path.exists(pickle_path2) and not args.recompute:
            with open(pickle_path2, 'rb') as f:
                alignment_scores2 = pickle.load(f)
        else:
            if args.compare_shuffle:
                paired_images2 = load_image(args.image_dir)
                paired_images2 = randomize_paired_images(paired_images2)
            else:
                paired_images2 = load_image(args.compare_dir)
            pre_transform = transforms.Compose(
                [
                    transforms.RandomApply(
                        [ContrastEnhance()], p=1 if contrast_flag else 0
                    ),
                    transforms.Resize(
                        (args.resolution, args.resolution),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),  # Adjust based on model requirements
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            assert len(paired_images2) > 0, f"No paired images found in {args.compare_dir}. Exiting."
            alignment_scores2 = compute_alignment_scores(paired_images2, pre_transform, save_intermediate=args.save_intermediate, intermediate_dir=compare_intermediate_dir, num_workers=args.num_workers)
            with open(pickle_path2, 'wb') as f:
                pickle.dump(alignment_scores2, f)

        all_emd_losses2 = [score["emd_loss"] for score in alignment_scores2]
        all_emds2 = [score["emd"] for score in alignment_scores2]
        all_jsds2 = [score["jsd"] for score in alignment_scores2]
        all_mis2 = [score["mi"] for score in alignment_scores2]
        all_overlaps2 = [score["overlap"] for score in alignment_scores2]
        print(f"\n\n################# Alignment Scores for {args.compare_dir}:")
        print(f"Average EMD Loss: {np.mean(all_emd_losses2):.6f} ± {np.std(all_emd_losses2):.6f}")
        print(f"Average EMD: {np.mean(all_emds2):.6f} ± {np.std(all_emds2):.6f}")
        print(f"Average JSD: {np.mean(all_jsds2):.6f} ± {np.std(all_jsds2):.6f}")
        print(f"Average MI: {np.mean(all_mis2):.6f} ± {np.std(all_mis2):.6f}")
        print(f"Average Overlap: {np.mean(all_overlaps2):.6f} ± {np.std(all_overlaps2):.6f}")

        # Statistical comparison of alignment metrics
        print("\n################# Statistical Comparison:")
        statistical_comparison(all_emd_losses, all_emd_losses2, name="EMD Loss")
        # statistical_comparison(all_emds, all_emds2, name="EMD")
        # statistical_comparison(all_jsds, all_jsds2, name="JSD")
        # statistical_comparison(all_mis, all_mis2, name="MI")
        # statistical_comparison(all_overlaps, all_overlaps2, name="Overlap")

        # plot histogram of emd_list_opt and emd_list_orig side by side
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(all_emd_losses, bins=30, alpha=0.5, color='blue', label='EMD #1')
        plt.hist(all_emd_losses2, bins=30, alpha=0.5, color='orange', label='EMD #2')
        plt.title('Histogram of EMD #1 and EMD #2')
        plt.xlabel('EMD Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.boxplot([all_emd_losses, all_emd_losses2], tick_labels=['EMD #1', 'EMD #2'])
        plt.title('Boxplot of EMD #1 and EMD #2')
        plt.ylabel('EMD Value')
        # plt.show()

        plt.savefig(os.path.join(intermediate_dir, 'alignment_score_comparison.png'), bbox_inches='tight')



