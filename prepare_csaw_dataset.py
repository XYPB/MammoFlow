import argparse
import json
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_metadata(metadata_path):
    rows = []
    with open(metadata_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            file_name = entry["file_name"]
            parts = os.path.splitext(file_name)[0].split("_")
            rows.append(
                {
                    "file_name": file_name,
                    "patient_id": parts[0] if len(parts) > 0 else "",
                    "laterality": parts[2] if len(parts) > 2 else "",
                    "view": parts[3] if len(parts) > 3 else "",
                }
            )
    return pd.DataFrame(rows)


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


def copy_paired_images(pairs, split_name, image_dir, output_dir):
    for cc_row, mlo_row in tqdm(pairs, desc=f"Copying {split_name}"):
        laterality = cc_row["laterality"]
        base_name = f"{cc_row['patient_id']}_{laterality}"
        cc_src = os.path.join(image_dir, cc_row["file_name"])
        mlo_src = os.path.join(image_dir, mlo_row["file_name"])
        cc_dst = os.path.join(output_dir, "CC", split_name, f"{base_name}_CC.png")
        mlo_dst = os.path.join(output_dir, "MLO", split_name, f"{base_name}_MLO.png")

        if os.path.exists(cc_src):
            shutil.copy2(cc_src, cc_dst)
        else:
            print(f"Warning: source file not found: {cc_src}")
        if os.path.exists(mlo_src):
            shutil.copy2(mlo_src, mlo_dst)
        else:
            print(f"Warning: source file not found: {mlo_src}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare paired CSAW CC/MLO images.")
    parser.add_argument("--image_dir", required=True, help="Directory containing CSAW image files.")
    parser.add_argument("--train_metadata", required=True, help="Training metadata JSONL.")
    parser.add_argument("--test_metadata", required=True, help="Test metadata JSONL.")
    parser.add_argument("--output_dir", required=True, help="Output paired dataset directory.")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation split from training pairs.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    train_df = load_metadata(args.train_metadata)
    test_df = load_metadata(args.test_metadata)
    train_pairs = find_paired_images(train_df)
    test_pairs = find_paired_images(test_df)

    if train_pairs:
        train_pairs, val_pairs = train_test_split(
            train_pairs, test_size=args.val_split, random_state=args.seed
        )
    else:
        val_pairs = []

    create_output_structure(args.output_dir)
    copy_paired_images(train_pairs, "train", args.image_dir, args.output_dir)
    copy_paired_images(val_pairs, "val", args.image_dir, args.output_dir)
    copy_paired_images(test_pairs, "test", args.image_dir, args.output_dir)

    print("Dataset preparation complete.")
    print(f"Output directory: {args.output_dir}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")


if __name__ == "__main__":
    main()
