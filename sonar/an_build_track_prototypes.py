"""
an_build_track_prototypes.py  (Anonymous Review Version)

This script aggregates groups of clip-level vectors into prototype
representations.  All dataset-specific semantics (e.g., domain,
sequence, track) have been replaced with generic labels and indices.
The script remains functional with toy or public data.
"""

import argparse
import numpy as np
from pathlib import Path


def load_vectors(path):
    """
    Expected anonymous format:

        vectors : (N, D)
        labels  : (N,)      — group label (generic)
        indices : (N,)      — item identifier within each label

    This format matches outputs from an_extract_clip_embeddings.py.
    """
    data = np.load(path, allow_pickle=True)

    X = data.get("vectors")
    lbl = data.get("labels")
    idx = data.get("indices")

    if X is None or lbl is None or idx is None:
        raise ValueError("Input file must contain 'vectors', 'labels', and 'indices'.")

    return X, lbl, idx


def build_prototypes(vectors, labels, indices):
    """
    Aggregate vectors by (label, index) group.
    Produces one prototype per unique (label, index).
    """
    groups = {}

    for v, lab, ident in zip(vectors, labels, indices):
        key = (lab, ident)
        groups.setdefault(key, []).append(v)

    proto_list = []
    proto_labels = []
    proto_ids = []
    counts = []

    for (lab, ident), vecs in groups.items():
        arr = np.vstack(vecs)
        proto = arr.mean(axis=0)

        proto_list.append(proto)
        proto_labels.append(lab)
        proto_ids.append(ident)
        counts.append(len(vecs))

    return (
        np.vstack(proto_list),
        np.array(proto_labels),
        np.array(proto_ids),
        np.array(counts)
    )


def main():
    parser = argparse.ArgumentParser(description="Build prototype vectors (anonymous version).")
    parser.add_argument("--input", required=True, help="Input .npz of clip vectors.")
    parser.add_argument("--out", required=True, help="Output prototype .npz.")
    args = parser.parse_args()

    print(f"[INFO] Loading: {args.input}")
    X, lbl, idx = load_vectors(args.input)

    P, PL, PI, C = build_prototypes(X, lbl, idx)

    print(f"[INFO] Prototypes built: {len(P)}")
    print(f"[INFO] Saving → {args.out}")

    np.savez(
        args.out,
        vectors=P,
        labels=PL,
        indices=PI,
        counts=C
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
