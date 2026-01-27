"""
Unified manifest generation script for PTB-XL 10-second ECG data.

This script:
1. Reads .mat files directly from the ptbxl_10s directory
2. Extracts patient_id from each file to ensure no patient overlap between splits
3. Creates train/valid/test manifests with reproducible random seeds

Usage:
    python manifest_ptbxl_10s.py \
        --root /output/path/of/preprocess_ptbxl\
        --data-dir ptbxl_10s \
        --dest /output/path/of/preprocess_ptbxl/manifest/ptbxl_10s_manifest \
        --valid-percent 0.05 \
        --test-percent 0.15 \
        --seed 42
"""

import argparse
import glob
import os
import random
import scipy.io
import numpy as np
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate train/valid/test manifests for PTB-XL 10s data with patient-based splitting"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/path/to/preprocess_classification",
        help="Root directory (used as header in manifest files)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ptbxl_10s",
        help="Subdirectory containing .mat files (relative to root)"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="/path/to/preprocess_classification/manifest/ptbxl_10s_manifest",
        help="Output directory for manifest files"
    )
    parser.add_argument(
        "--valid-percent",
        type=float,
        default=0.1,
        help="Percentage of patients for validation set (default: 0.1)"
    )
    parser.add_argument(
        "--test-percent",
        type=float,
        default=0.1,
        help="Percentage of patients for test set (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="mat",
        help="File extension to search for (default: mat)"
    )
    return parser


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_patient_id(file_path):
    """Extract patient_id from a .mat file."""
    data = scipy.io.loadmat(file_path)
    return int(data['patient_id'][0][0])


def get_sample_count(file_path):
    """Get the number of samples (ECG length) from a .mat file."""
    data = scipy.io.loadmat(file_path)
    return data['feats'].shape[-1]


def write_manifest(file_list, dest_file, root_path, data_dir):
    """
    Write a manifest TSV file.

    Format:
        Line 1: root_path (header)
        Line 2+: relative_path<TAB>sample_count
    """
    with open(dest_file, "w") as f:
        # Write header (root path)
        print(root_path, file=f)

        # Write each file entry
        for file_path, sample_count in tqdm(file_list, desc=os.path.basename(dest_file)):
            # Get path relative to root
            rel_path = os.path.relpath(file_path, root_path)
            print(f"{rel_path}\t{sample_count}", file=f)


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Resolve paths
    root_path = os.path.realpath(args.root)
    data_path = os.path.join(root_path, args.data_dir)
    dest_path = os.path.realpath(args.dest)

    # Create output directory if needed
    os.makedirs(dest_path, exist_ok=True)

    # Find all .mat files
    search_pattern = os.path.join(data_path, f"*.{args.ext}")
    all_files = sorted(glob.glob(search_pattern))  # Sort for reproducibility

    if not all_files:
        print(f"No .{args.ext} files found in {data_path}")
        return

    print(f"Found {len(all_files)} files in {data_path}")

    # Group files by patient_id
    print("Extracting patient IDs and sample counts...")
    patient_to_files = {}
    file_to_samples = {}

    for file_path in tqdm(all_files, desc="Processing files"):
        try:
            patient_id = get_patient_id(file_path)
            sample_count = get_sample_count(file_path)

            if patient_id not in patient_to_files:
                patient_to_files[patient_id] = []
            patient_to_files[patient_id].append(file_path)
            file_to_samples[file_path] = sample_count

        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
            continue

    print(f"Found {len(patient_to_files)} unique patients")

    # Get sorted list of unique patients (sorted for reproducibility)
    unique_patients = sorted(patient_to_files.keys())

    # Shuffle with seed
    random.shuffle(unique_patients)

    # Calculate split sizes
    num_patients = len(unique_patients)
    num_test = int(num_patients * args.test_percent)
    num_valid = int(num_patients * args.valid_percent)
    num_train = num_patients - num_test - num_valid

    print(f"Splitting {num_patients} patients: {num_train} train, {num_valid} valid, {num_test} test")

    # Split patients (not files) into train/valid/test
    test_patients = set(unique_patients[:num_test])
    valid_patients = set(unique_patients[num_test:num_test + num_valid])
    train_patients = set(unique_patients[num_test + num_valid:])

    # Assign files to splits based on patient
    train_files = []
    valid_files = []
    test_files = []

    for patient_id, files in patient_to_files.items():
        # Create tuples of (file_path, sample_count)
        file_tuples = [(f, file_to_samples[f]) for f in files]

        if patient_id in train_patients:
            train_files.extend(file_tuples)
        elif patient_id in valid_patients:
            valid_files.extend(file_tuples)
        elif patient_id in test_patients:
            test_files.extend(file_tuples)

    # Sort files within each split for reproducibility
    train_files.sort(key=lambda x: x[0])
    valid_files.sort(key=lambda x: x[0])
    test_files.sort(key=lambda x: x[0])

    print(f"File counts: {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test")

    # Write manifest files
    print("\nWriting manifest files...")
    write_manifest(train_files, os.path.join(dest_path, "train.tsv"), root_path, args.data_dir)
    write_manifest(valid_files, os.path.join(dest_path, "valid.tsv"), root_path, args.data_dir)
    write_manifest(test_files, os.path.join(dest_path, "test.tsv"), root_path, args.data_dir)

    # Write split info for reproducibility verification
    info_file = os.path.join(dest_path, "split_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Root: {root_path}\n")
        f.write(f"Data dir: {args.data_dir}\n")
        f.write(f"Valid percent: {args.valid_percent}\n")
        f.write(f"Test percent: {args.test_percent}\n")
        f.write(f"Total files: {len(all_files)}\n")
        f.write(f"Total patients: {num_patients}\n")
        f.write(f"Train patients: {num_train}\n")
        f.write(f"Valid patients: {num_valid}\n")
        f.write(f"Test patients: {num_test}\n")
        f.write(f"Train files: {len(train_files)}\n")
        f.write(f"Valid files: {len(valid_files)}\n")
        f.write(f"Test files: {len(test_files)}\n")

    print(f"\nManifest files saved to: {dest_path}")
    print(f"Split info saved to: {info_file}")
    print("Done!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
