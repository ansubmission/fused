"""
an_merge_prototypes.py  (Anonymous Review Version)

Merge multiple prototype files into a single prototype set.
All dataset-specific concepts have been replaced with generic
'vectors', 'labels', and 'indices' semantics for anonymity.
"""

import argparse
import numpy as np


def load_proto_file(path):
    data = np.load(path, allow_pickle=True)

    if "vectors" not in data or "labels" not in data or "indices" not in data:
        raise ValueError("Prototype file must contain 'vectors', 'labels', 'indices', 'counts'.")

    return (
        data["vectors"],     # (N_i, D)
        data["labels"],      # (N_i,)
        data["indices"],     # (N_i,)
        data["counts"],      # (N_i,)
    )


def main():
    parser = argparse.ArgumentParser(description="Merge prototype files (anonymous version).")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of prototype NPZ files.")
    parser.add_argument("--out", type=str, required=True, help="Output merged NPZ file.")
    args = parser.parse_args()

    print("[INFO] Merging prototype files:")
    all_V = []
    all_L = []
    all_I = []
    all_C = []

    for path in args.inputs:
        print(f"  - {path}")
        V, L, I, C = load_proto_file(path)
        all_V.append(V)
        all_L.append(L)
        all_I.append(I)
        all_C.append(C)

    V_final = np.vstack(all_V)
    L_final = np.concatenate(all_L)
    I_final = np.concatenate(all_I)
    C_final = np.concatenate(all_C)

    print("\n[INFO] Final merged sizes:")
    print(f"  Vectors: {V_final.shape}")
    print(f"  Labels:  {L_final.shape}")
    print(f"  Indices: {I_final.shape}")
    print(f"  Counts:  {C_final.shape}")

    print(f"\n[INFO] Saving merged file → {args.out}")
    np.savez(
        args.out,
        vectors=V_final,
        labels=L_final,
        indices=I_final,
        counts=C_final,
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
