"""
an_retrieval_eval_compressed_final.py  (Anonymous Review Version)

This script provides a generic example of evaluating vector–prototype
retrieval under dimensionality reduction (e.g., PCA). Field names and
experiment-specific conventions have been generalized for anonymous
review. The script remains functional when used with toy data.
"""

import numpy as np
import argparse
import pickle


def cosine_sim(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def recall_at_k(sim, labels_q, labels_p, ks=(1, 5, 10)):
    idx = np.argsort(-sim, axis=1)
    out = {}

    for k in ks:
        correct = 0
        for i in range(len(labels_q)):
            topk = labels_p[idx[i, :k]]
            if labels_q[i] in topk:
                correct += 1
        out[f"R@{k}"] = correct / len(labels_q)

    return out


def run_eval(emb_file, proto_file, pca_file, tag="Eval"):
    print(f"\n[INFO] {tag}")

    emb_data = np.load(emb_file, allow_pickle=True)
    proto_data = np.load(proto_file, allow_pickle=True)

    # generic field names for anonymization
    X = emb_data.get("vectors", emb_data[list(emb_data.files)[0]])
    Lq = emb_data.get("labels")
    if Lq is None:
        raise ValueError("Embedding file must contain 'labels'.")

    P = proto_data.get("vectors", proto_data.get("prototypes"))
    Lp = proto_data.get("labels")
    if Lp is None:
        raise ValueError("Prototype file must contain 'labels'.")

    # load PCA transform
    with open(pca_file, "rb") as f:
        pca = pickle.load(f)

    X_low = pca.transform(X)
    print(f"[INFO] X_low={X_low.shape}, P={P.shape}")

    sim = cosine_sim(X_low, P)
    results = recall_at_k(sim, Lq, Lp)

    print("  " + ", ".join([f"{k}={v:.4f}" for k, v in results.items()]))
    return results


def main():
    parser = argparse.ArgumentParser(description="Prototype-compressed retrieval (anonymous version).")
    parser.add_argument("--emb_a", required=True)
    parser.add_argument("--emb_b", required=True)
    parser.add_argument("--proto_a", required=True)
    parser.add_argument("--proto_b", required=True)
    parser.add_argument("--pca_a", required=True)
    parser.add_argument("--pca_b", required=True)
    args = parser.parse_args()

    print("========== Retrieval Under Prototype Compression (Anonymous) ==========")

    run_eval(args.emb_a, args.proto_a, args.pca_a, "A → A")
    run_eval(args.emb_b, args.proto_b, args.pca_b, "B → B")
    run_eval(args.emb_a, args.proto_b, args.pca_b, "A → B")
    run_eval(args.emb_b, args.proto_a, args.pca_a, "B → A")


if __name__ == "__main__":
    main()
