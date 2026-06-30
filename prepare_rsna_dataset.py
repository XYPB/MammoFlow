import argparse
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def find_paired_images(df):
    pairs = []
    df = df[df["view"].isin(["CC", "MLO"])]
    for (_, _), group in df.groupby(["patient_id", "laterality"]):
        cc_rows = group[group["view"] == "CC"]
        mlo_rows = group[group["view"] == "MLO"]
        if len(cc_rows) == 1 and len(mlo_rows) == 1:
            pairs.append((cc_rows.iloc[0], mlo_rows.iloc[0]))
    return pairs


def create_output_structure(output_dir):
    for view in ["CC", "MLO"]:
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(output_dir, view, split), exist_ok=True)


def copy_paired_images(pairs, split_name, source_image_dir, output_dir):
    for cc_row, mlo_row in tqdm(pairs, desc=f"Copying {split_name}"):
        patient_id = str(cc_row["patient_id"])
        laterality = cc_row["laterality"]
        base_name = f"{patient_id}_{laterality}"
        cc_src = os.path.join(source_image_dir, patient_id, f"{cc_row['image_id']}_resized.jpg")
        mlo_src = os.path.join(source_image_dir, patient_id, f"{mlo_row['image_id']}_resized.jpg")
        cc_dst = os.path.join(output_dir, "CC", split_name, f"{base_name}_CC.jpg")
        mlo_dst = os.path.join(output_dir, "MLO", split_name, f"{base_name}_MLO.jpg")

        if os.path.exists(cc_src):
            shutil.copy2(cc_src, cc_dst)
        else:
            print(f"Warning: source file not found: {cc_src}")
        if os.path.exists(mlo_src):
            shutil.copy2(mlo_src, mlo_dst)
        else:
            print(f"Warning: source file not found: {mlo_src}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare paired RSNA CC/MLO images.")
    parser.add_argument("--source_image_dir", required=True, help="Directory with patient subfolders.")
    parser.add_argument("--train_csv", required=True, help="Training CSV with patient_id/image_id/view/laterality.")
    parser.add_argument("--test_csv", required=True, help="Test CSV with patient_id/image_id/view/laterality.")
    parser.add_argument("--output_dir", required=True, help="Output paired dataset directory.")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation split from training pairs.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    train_pairs = find_paired_images(train_df)
    test_pairs = find_paired_images(test_df)

    if train_pairs:
        train_pairs, val_pairs = train_test_split(
            train_pairs, test_size=args.val_split, random_state=args.seed
        )
    else:
        val_pairs = []

    create_output_structure(args.output_dir)
    copy_paired_images(train_pairs, "train", args.source_image_dir, args.output_dir)
    copy_paired_images(val_pairs, "val", args.source_image_dir, args.output_dir)
    copy_paired_images(test_pairs, "test", args.source_image_dir, args.output_dir)

    print("Dataset preparation complete.")
    print(f"Output directory: {args.output_dir}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")


if __name__ == "__main__":
    main()
