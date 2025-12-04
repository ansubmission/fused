
"""
an_retrieval_eval.py  (Anonymous Review Version)

Implements retrieval evaluation:
1) In-domain retrieval = track-level Recall@K (same fish trajectory).
2) Cross-domain retrieval = prototype-level domain→domain semantic Recall@K.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm


# --------------------------------------------------------
# Utility to load embeddings / prototypes
# --------------------------------------------------------
def load_npz(path):
    data = np.load(path, allow_pickle=True)
    embs = data["embeddings"]
    metas = data["metadata"].tolist()
    return embs, metas


def load_prototypes(path):
    data = np.load(path, allow_pickle=True)
    P = data["prototypes"] if "prototypes" in data else data["vectors"]
    labels = data["labels"].tolist()
    return P, labels


# --------------------------------------------------------
# In-domain Recall@K (track-level)
# --------------------------------------------------------
def compute_track_recall(sim_row, pos_indices, k):
    if len(pos_indices) == 0:
        return None
    topk = np.argsort(-sim_row)[:k]
    return int(any(p in topk for p in pos_indices))


def evaluate_in_domain(domain, emb, meta, Ks=[1,5,10]):
    print(f"\n▶ In-domain Retrieval on {domain}")

    # Build track → indices dictionary
    track_map = {}
    for gi, m in enumerate(meta):
        tid = m["track_id"]
        track_map.setdefault(tid, []).append(gi)

    R = {k: [] for k in Ks}

    for qi, mq in enumerate(tqdm(meta, desc=f"{domain} (in-domain)")):
        q_track = mq["track_id"]
        q_clip = mq["clip_id"]

        # Positive = same track, but not same clip
        pos = [gi for gi in track_map.get(q_track, [])
               if meta[gi]["clip_id"] != q_clip]

        if len(pos) == 0:
            continue

        sims = emb @ emb[qi]

        # Remove self-match
        for gi, mg in enumerate(meta):
            if mg["track_id"] == q_track and mg["clip_id"] == q_clip:
                sims[gi] = -1e9

        for k in Ks:
            R[k].append(compute_track_recall(sims, pos, k))

    # Report
    for k in Ks:
        if len(R[k]):
            print(f"  R@{k} = {np.mean(R[k]):.4f}")
        else:
            print(f"  R@{k} = N/A (no positives)")

    return R


# --------------------------------------------------------
# Cross-domain prototype retrieval (semantic)
# --------------------------------------------------------
def compute_proto_recall(Pq, Lq, Pg, Lg, Ks=[1,5,10]):
    print("\n▶ Cross-domain Prototype Retrieval")

    sims = Pq @ Pg.T
    sims[np.arange(len(Pq)), np.arange(len(Pq))] = -1e9

    unique_labels = sorted(set(Lq))
    R = {k: [] for k in Ks}

    for qi, qlab in enumerate(Lq):
        # positives = same-domain prototypes
        pos_idx = [gi for gi, lab in enumerate(Lg) if lab == qlab]

        if len(pos_idx) == 0:
            continue

        ranked = np.argsort(-sims[qi])

        for k in Ks:
            topk = ranked[:k]
            hit = int(any(g in topk for g in pos_idx))
            R[k].append(hit)

    for k in Ks:
        print(f"  Proto R@{k} = {np.mean(R[k]):.4f}")

    return R


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    domains = ["dA", "dB", "dC"]   # anonymous domain names
    root = Path("embeddings")

    print("Loading embeddings...")
    emb_data = {}
    for d in domains:
        emb, meta = load_npz(root / f"{d}.npz")
        emb_data[d] = (emb, meta)
        print(f"  {d}: {len(emb)} embeddings loaded.")

    print("\n================ In-Domain Retrieval ================")
    in_results = {}
    for d in domains:
        emb, meta = emb_data[d]
        R = evaluate_in_domain(d, emb, meta)
        in_results[d] = R

    print("\n================ Cross-Domain Retrieval ================")

    # Load prototype file (anonymous)
    P, L = load_prototypes(root / "prototypes.npz")

    cross_results = compute_proto_recall(P, L, P, L)

    print("\n================ FINAL SUMMARY ================")
    print("In-domain R@1 matrix:")
    for d in domains:
        r1 = np.mean(in_results[d][1]) if len(in_results[d][1]) else np.nan
        print(f"  {d}: {r1:.3f}")

    print("\nCross-domain Prototype R@1 =", np.mean(cross_results[1]))


if __name__ == "__main__":
    main()
