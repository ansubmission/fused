"""
an_retrieval_eval_birds_60.py (Anonymous Review Version)

This script provides a generic example of vector-to-prototype retrieval
evaluation using cosine similarity. Field names and experiment-specific
details have been generalized for anonymous review.

The script remains functional when used with toy embeddings and
prototypes, but does not reflect the structure of the full system.
"""

import numpy as np
import argparse


def cosine_sim(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def recall_at_k(sim, labels_q, labels_p, ks=(1, 5, 10)):
    idx = np.argsort(-sim, axis=1)
    stats = {}
    n = len(labels_q)

    for k in ks:
        correct = 0
        for i in range(n):
            topk = labels_p[idx[i, :k]]
            if labels_q[i] in topk:
                correct += 1
        stats[f"R@{k}"] = correct / n

    return stats


def run_eval(emb_file, proto_file, tag="Eval"):
    print(f"\n[INFO] Running: {tag}")

    emb_data = np.load(emb_file, allow_pickle=True)
    proto_data = np.load(proto_file, allow_pickle=True)

    # generic field names for anonymization
    E = emb_data.get("vectors", emb_data[list(emb_data.files)[0]])
    Lq = emb_data.get("labels", None)

    P = proto_data.get("vectors", proto_data.get("prototypes"))
    Lp = proto_data.get("labels", None)

    if Lq is None or Lp is None:
        raise ValueError("Both embedding and prototype files must contain 'labels'.")

    sim = cosine_sim(E, P)
    results = recall_at_k(sim, Lq, Lp)

    print("  " + ", ".join([f"{k}={v:.4f}" for k, v in results.items()]))
    return results


def main():
    parser = argparse.ArgumentParser(description="Generic vector–prototype retrieval evaluation.")
    parser.add_argument("--emb_a", required=True, help="Embeddings A (.npz)")
    parser.add_argument("--emb_b", required=True, help="Embeddings B (.npz)")
    parser.add_argument("--proto_a", required=True, help="Prototypes A (.npz)")
    parser.add_argument("--proto_b", required=True, help="Prototypes B (.npz)")
    args = parser.parse_args()

    print("=========== Retrieval Evaluation (Anonymous Version) ===========")

    run_eval(args.emb_a, args.proto_a, "A → A")
    run_eval(args.emb_b, args.proto_b, "B → B")
    run_eval(args.emb_a, args.proto_b, "A → B")
    run_eval(args.emb_b, args.proto_a, "B → A")


if __name__ == "__main__":
    main()
