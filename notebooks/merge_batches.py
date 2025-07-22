#!/usr/bin/env python3

import argparse
import glob
import os

import numpy as np

def merge_batches(batch_dir, output_path):
    pattern = os.path.join(batch_dir, "batch_*.npy")
    batch_files = sorted(glob.glob(pattern))

    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {batch_dir}")

    first_batch = np.load(batch_files[0])
    embedding_dim = first_batch.shape[1]
    total_samples = first_batch.shape[0]

    for batch_file in batch_files[1:]:
        total_samples += np.load(batch_file, mmap_mode="r").shape[0]

    all_embeddings = np.empty((total_samples, embedding_dim), dtype=np.float32)
    current_idx = 0

    for batch_file in batch_files:
        batch = np.load(batch_file)
        batch_size = batch.shape[0]
        all_embeddings[current_idx : current_idx + batch_size] = batch
        current_idx += batch_size

    np.save(output_path, all_embeddings)
    return all_embeddings.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch_dir", required=True, help="Directory containing batch files"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Output path for merged file (must end with .npy)",
    )
    args = parser.parse_args()

    if not args.output_path.endswith(".npy"):
        raise ValueError("Output file must end with .npy extension")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    shape = merge_batches(args.batch_dir, args.output_path)
    print(f"Merged {shape[0]} samples with dimension {shape[1]}")


if __name__ == "__main__":
    main()
