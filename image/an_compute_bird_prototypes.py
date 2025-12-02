"""
an_compute_bird_prototypes.py (Anonymous Review Version)

This script computes simple class-level prototype vectors by averaging
all samples belonging to the same label. It is a general and simplified
version of the tool used in our system, with dataset-specific fields
removed for anonymous review.

The script remains functional and demonstrates the intended workflow,
but the input/output conventions have been made generic and do not
reflect the structure of the full implementation.
"""

import numpy as np
from collections import defaultdict
import argparse


def compute_prototypes(input_file, output_file):
    print(f"[INFO] Loading input vectors from: {input_file}")

    data = np.load(input_file, allow_pickle=True)

    # Generic field names for anonymization
    if "vectors" in data and "labels" in data:
        X = data["vectors"]    # (N, D)
        labels = data["labels"]  # (N,)
    else:
        # Fallback to the first two arrays (anonymous logic)
        keys = data.files
        if len(keys) < 2:
            raise ValueError("Input .npz must contain at least two arrays.")
        X = data[keys[0]]
        labels = data[keys[1]]

    print(f"[INFO] Loaded {X.shape[0]} samples with dim {X.shape[1]}")

    # Group by label
    grouped = defaultdict(list)
    for vec, lab in zip(X, labels):
        grouped[int(lab)].append(vec)

    proto_list = []
    label_list = []
    count_list = []

    for lab in sorted(grouped.keys()):
        arr = np.vstack(grouped[lab])
        proto = arr.mean(axis=0)
        proto_list.append(proto)
        label_list.append(lab)
        count_list.append(arr.shape[0])

    prototypes = np.vstack(proto_list)
    label_list = np.array(label_list)
    count_list = np.array(count_list)

    print(f"[INFO] Computed {len(label_list)} class prototypes.")

    # Save output in generic format
    np.savez(
        output_file,
        prototypes=prototypes.astype(np.float32),
        labels=label_list,
        counts=count_list
    )

    print(f"[INFO] Saved prototypes → {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Input .npz file containing 'vectors' and 'labels'.")
    parser.add_argument("--output", required=True,
                        help="Output .npz file to save computed prototypes.")
    args = parser.parse_args()

    compute_prototypes(args.input, args.output)


if __name__ == "__main__":
    main()
