"""
compress_prototypes.py (Anonymous Review Version)

This script demonstrates how prototype vectors can be reduced to
lower-dimensional representations using PCA. It is a simplified and
generic version of the compression tool used, with all
experiment-specific details removed for anonymous review.

The script remains functional and illustrates the intended workflow,
but the default dimensions, field names, and file conventions have been
generalized to avoid revealing dataset- or experiment-specific details.
"""

import numpy as np
import argparse
from sklearn.decomposition import PCA
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to input .npz file containing vectors.")
    parser.add_argument("--output_prefix", required=True,
                        help="Prefix for output compressed files.")
    parser.add_argument("--dims", nargs="+", type=int,
                        default=[16, 8, 4],
                        help="Target PCA dimensions to generate.")
    args = parser.parse_args()

    # Load prototype-like vectors (generic field name)
    data = np.load(args.input)
    if "vectors" in data:
        X = data["vectors"]
    else:
        # Fallback to common field names (safe and generic)
        for key in data.files:
            X = data[key]
            break

    print(f"[INFO] Loaded input vectors with shape: {X.shape}")

    max_dim = min(X.shape)
    print(f"[INFO] Max possible PCA dimension = {max_dim}")

    # Iterate over target dimensions
    for d in args.dims:
        d = min(d, max_dim)
        print(f"\n[INFO] Compressing to {d} dimensions...")

        pca = PCA(n_components=d)
        X_low = pca.fit_transform(X)

        # Save compressed vectors
        out_npz = f"{args.output_prefix}_{d}d.npz"
        np.savez(out_npz, vectors=X_low.astype(np.float32))
        print(f"[INFO] Saved compressed vectors → {out_npz}")

        # Save PCA projection matrix
        out_pkl = f"{args.output_prefix}_{d}d_pca.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(pca, f)
        print(f"[INFO] Saved PCA model → {out_pkl}")


if __name__ == "__main__":
    main()
