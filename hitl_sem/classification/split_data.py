"""Split the class-foldered images into an init set and a test set.

The init set keeps the folder hierarchy (that hierarchy *is* the label tree read
by `initialize`); the test set is a flat folder, with labels recorded in a ground
truth CSV.

Two modes:

    python -m hitl_sem.classification.split_data --from-csv
        Reproduce the published split from classification_ground_truth_mapping.csv.
        Use this one — the shipped models were trained on exactly that split.

    python -m hitl_sem.classification.split_data
        Draw a fresh random split (seed 42, 40% init) and write a new CSV.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "classification" / "images_all"
DEFAULT_INIT_DIR = REPO_ROOT / "data" / "classification" / "images_init"
DEFAULT_TEST_DIR = REPO_ROOT / "data" / "classification" / "images_test"
DEFAULT_CSV = REPO_ROOT / "classification_ground_truth_mapping.csv"

VALID_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def create_dataset_splits(src_dir, init_dir, test_dir, csv_path, init_ratio=0.4, seed=42):
    """Draw a random split and write the ground truth CSV."""
    random.seed(seed)

    src_path = Path(src_dir)
    init_path = Path(init_dir)
    test_path = Path(test_dir)

    init_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    records = []

    for root, dirs, files in os.walk(src_path):
        images = [f for f in files if Path(f).suffix.lower() in VALID_EXTS]

        if not images:
            continue

        rel_path = Path(root).relative_to(src_path)
        parts = rel_path.parts

        level_1 = parts[0] if len(parts) > 0 else "None"
        level_2 = parts[1] if len(parts) > 1 else "None"
        level_3 = parts[2] if len(parts) > 2 else "None"

        random.shuffle(images)
        split_idx = max(1, int(len(images) * init_ratio)) if len(images) > 1 else 1

        init_images = images[:split_idx]
        test_images = images[split_idx:]

        # Init images keep the folder hierarchy, which encodes their labels
        for img_name in init_images:
            dest_folder = init_path / rel_path
            dest_folder.mkdir(parents=True, exist_ok=True)

            dest_img_path = dest_folder / img_name
            shutil.copy2(Path(root) / img_name, dest_img_path)

            records.append({
                "filename": img_name, "split": "init",
                "level_1": level_1, "level_2": level_2, "level_3": level_3,
                "current_path": str(dest_img_path)
            })

        # Test images go into one flat folder; the CSV is their only label
        for img_name in test_images:
            dest_img_path = test_path / img_name

            if dest_img_path.exists():
                print(f"WARNING: File collision detected for {img_name}. Overwriting.")

            shutil.copy2(Path(root) / img_name, dest_img_path)

            records.append({
                "filename": img_name, "split": "test",
                "level_1": level_1, "level_2": level_2, "level_3": level_3,
                "current_path": str(dest_img_path)
            })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Data split complete! Processed {len(df)} total images.")
    print(f"Init images (structured): {len(df[df['split'] == 'init'])}")
    print(f"Test images (flat): {len(df[df['split'] == 'test'])}")
    print(f"Ground truth saved to: {csv_path}")


def apply_split_from_csv(src_dir, init_dir, test_dir, csv_path):
    """Materialize init/test folders from an existing ground truth CSV.

    Files are located by name anywhere under src_dir, so this works no matter how
    the Zenodo archive was unpacked, and reproduces the published split exactly.
    """
    src_path = Path(src_dir)
    init_path = Path(init_dir)
    test_path = Path(test_dir)

    init_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    index = {}
    for p in src_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            index.setdefault(p.name, p)

    df = pd.read_csv(csv_path)
    copied, missing = 0, []

    for _, row in df.iterrows():
        source = index.get(row["filename"])
        if source is None:
            missing.append(row["filename"])
            continue

        if row["split"] == "init":
            levels = [str(row[c]) for c in ("level_1", "level_2", "level_3")]
            levels = [lvl for lvl in levels if lvl not in ("None", "nan", "NaN")]
            dest_folder = init_path.joinpath(*levels)
        else:
            dest_folder = test_path

        dest_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest_folder / row["filename"])
        copied += 1

    print(f"Applied split from {csv_path}: {copied} images copied.")
    print(f"Init: {len(df[df['split'] == 'init'])} | Test: {len(df[df['split'] == 'test'])}")
    if missing:
        print(f"WARNING: {len(missing)} images listed in the CSV were not found under {src_path}:")
        for name in missing[:10]:
            print(f"  {name}")


def main():
    parser = argparse.ArgumentParser(description="Split the classification images into init and test sets.")
    parser.add_argument("--source-dir", type=str, default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--init-dir", type=str, default=str(DEFAULT_INIT_DIR))
    parser.add_argument("--test-dir", type=str, default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--from-csv", action="store_true",
                        help="Reproduce the published split from --csv-path instead of drawing a new one.")
    parser.add_argument("--init-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.from_csv:
        apply_split_from_csv(args.source_dir, args.init_dir, args.test_dir, args.csv_path)
    else:
        create_dataset_splits(args.source_dir, args.init_dir, args.test_dir, args.csv_path,
                              init_ratio=args.init_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
