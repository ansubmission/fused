"""
an_retrieval_eval_tracks.py  (Anonymous Review Version)

Evaluate prototype-level retrieval across label groups.
This script computes cosine-similarity retrieval among prototypes
and reports Recall@K metrics for each (query_label → target_label) pair.
"""

import argparse
import numpy as np
import torch
from torch.nn.functional import cosine_similarity


# ---------------------------------------------------------
# Load prototypes (anonymous format)
# ---------------------------------------------------------
def load_prototypes(path):
    data = np.load(path, allow_pickle=True)

    # Main vector array
    if "vectors" in data:
        P = torch.tensor(data["vectors"], dtype=torch.float32)
    elif "prototypes" in data:
        P = torch.tensor(data["prototypes"], dtype=torch.float32)
    else:
        raise KeyError("File missing 'vectors' or 'prototypes'.")

    # Anonymous label groups
    if "labels" in data:
        labels = data["labels"].tolist()
    elif "proto_domains" in data:
        labels = data["proto_domains"].tolist()
    elif "domains" in data:
        labels = data["domains"].tolist()
    else:
        raise KeyError("File missing 'labels'/'proto_domains'/'domains'.")

    # Anonymous indices (may represent original track ids)
    if "indices" in data:
        indices = data["indices"].tolist()
    elif "proto_track_ids" in data:
        indices = data["proto_track_ids"].tolist()
    elif "track_ids" in data:
        indices = data["track_ids"].tolist()
    else:
        raise KeyError("File missing 'indices'/'proto_track_ids'/'track_ids'.")

    return P, labels, indices


# ---------------------------------------------------------
# Compute Recall@K (label → label)
# ---------------------------------------------------------
def compute_recall_at_k(P, labels, Klist=[1, 5, 10]):
    device = P.device
    N = P.size(0)

    # Compute cosine similarity matrix
    sims = cosine_similarity(P.unsqueeze(1), P.unsqueeze(0), dim=2)  # (N, N)
    sims.fill_diagonal_(-1e9)   # never retrieve itself

    results = {}
    unique_labels = sorted(list(set(labels)))

    for q_lab in unique_labels:
        results[q_lab] = {}

        idx_q = [i for i, lb in enumerate(labels) if lb == q_lab]

        for t_lab in unique_labels:
            idx_t = [i for i, lb in enumerate(labels) if lb == t_lab]
            idx_t = torch.tensor(idx_t, device=device, dtype=torch.long)

            if len(idx_t) == 0:
                continue

            recalls = {k: [] for k in Klist}

            for qi in idx_q:
                scores = sims[qi][idx_t]
                topk_vals, topk_idx = torch.topk(scores, k=max(Klist))
                for k in Klist:
                    recalls[k].append(1 if qi in idx_t[topk_idx[:k]] else 0)

            results[q_lab][t_lab] = {k: float(np.mean(recalls[k])) for k in Klist}

    return results


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Anonymous prototype retrieval evaluation.")
    parser.add_argument("--prototype_npz", type=str, required=True)
    args = parser.parse_args()

    print(f"[INFO] Loading prototype file: {args.prototype_npz}")
    P, labels, indices = load_prototypes(args.prototype_npz)

    P = P.to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] #Vectors = {P.size(0)}, Dim = {P.size(1)}")
    print(f"[INFO] Unique label groups: {sorted(set(labels))}")

    Klist = [1, 5, 10]
    results = compute_recall_at_k(P, labels, Klist)

    print("\n==================== RETRIEVAL RESULTS ====================\n")
    for q_lab in results:
        print(f"▶ Query label: {q_lab}")
        for t_lab in results[q_lab]:
            r = results[q_lab][t_lab]
            print(f"    {q_lab} → {t_lab}")
            print(f"       R@1  = {r[1]:.3f}")
            print(f"       R@5  = {r[5]::.3f}")
            print(f"       R@10 = {r[10]:.3f}")
        print()

    print("==================== RETRIEVAL MATRIX (R@1) ====================")
    uniq = sorted(set(labels))
    print("Labels:", uniq)
    for q in uniq:
        row = [f"{results[q][t][1]:.3f}" for t in uniq]
        print(f"{q:10s}: {row}")


if __name__ == "__main__":
    main()
