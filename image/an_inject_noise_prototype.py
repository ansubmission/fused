"""
an_inject_noise_prototype.py (Anonymous Review Version)

This script adds Gaussian noise to a set of prototype-like vectors.
It is a simplified and generic version of a noise-injection utility
used in our system, with all experiment-specific field names removed
for anonymous review.

The script remains functional, but the input/output conventions have
been generalized and are intended only for illustrative purposes.
"""

import numpy as np
import argparse


def inject_noise(input_file, sigma, output_file):
    print(f"[INFO] Loading vectors from: {input_file}")
    data = np.load(input_file, allow_pickle=True)

    # Generic field names for anonymization
    if "prototypes" in data:
        X = data["prototypes"]
    elif "vectors" in data:
        X = data["vectors"]
    else:
        # fallback: pick first array
        key = data.files[0]
        X = data[key]

    labels = data.get("labels", None)

    print(f"[INFO] Vector shape: {X.shape}, sigma={sigma}")

    noise = np.random.randn(*X.shape) * sigma
    X_noisy = X + noise

    save_dict = {"vectors": X_noisy.astype(np.float32)}
    if labels is not None:
        save_dict["labels"] = labels

    np.savez(output_file, **save_dict)
    print(f"[INFO] Saved noisy vectors → {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Inject Gaussian noise into vectors.")
    parser.add_argument("--input", required=True, help="Input .npz file")
    parser.add_argument("--sigma", type=float, required=True, help="Noise stddev")
    parser.add_argument("--output", required=True, help="Output .npz file")
    args = parser.parse_args()

    inject_noise(args.input, args.sigma, args.output)


if __name__ == "__main__":
    main()
