"""
Paper-quality t-SNE visualization for US/EU SigLIP embeddings
- Color = species (class_id)
- Marker = domain (US = solid circles, EU = faint crosses)
- Designed for single-column figure width
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import argparse


def load_embeddings(npz_path):
    """Load embeddings + class_id"""
    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"]
    class_id = data["class_id"]
    return emb, class_id


def generate_tsne(
    us_file,
    eu_file,
    output,
    n_samples=None,
    perplexity=30,
    random_state=42
):
    print("=" * 60)
    print("Generating optimized t-SNE (paper-quality)")
    print("=" * 60)

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------
    print(f"[INFO] Loading: {us_file}")
    us_emb, us_class = load_embeddings(us_file)
    print(f"US: {us_emb.shape[0]} samples")

    print(f"[INFO] Loading: {eu_file}")
    eu_emb, eu_class = load_embeddings(eu_file)
    print(f"EU: {eu_emb.shape[0]} samples")

    # Sampling
    rng = np.random.default_rng(random_state)
    if n_samples is not None:
        if len(us_emb) > n_samples:
            idx = rng.choice(len(us_emb), n_samples, replace=False)
            us_emb, us_class = us_emb[idx], us_class[idx]
        if len(eu_emb) > n_samples:
            idx = rng.choice(len(eu_emb), n_samples, replace=False)
            eu_emb, eu_class = eu_emb[idx], eu_class[idx]

    # Combine for a SINGLE t-SNE space
    all_emb = np.vstack([us_emb, eu_emb])
    all_class = np.concatenate([us_class, eu_class])
    domains = np.concatenate([
        np.zeros(len(us_emb), dtype=np.int64),  # 0 = US
        np.ones(len(eu_emb), dtype=np.int64),   # 1 = EU
    ])

    # ------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------
    print(f"[INFO] Running t-SNE (perplexity={perplexity}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        verbose=1,
    )
    emb_2d = tsne.fit_transform(all_emb)

    # ------------------------------------------------------
    # Colors (species)
    # ------------------------------------------------------
    unique_classes = np.unique(all_class)
    n_classes = len(unique_classes)

    # High-quality colormap for clustering
    # cmap = plt.cm.get_cmap("gist_ncar", n_classes)
    # class_to_color = {cid: cmap(i) for i, cid in enumerate(unique_classes)}
    # colors = np.array([class_to_color[c] for c in all_class])

    # --- Low-saturation colormap ------------------------------------------
    base_cmap = plt.cm.get_cmap("viridis", n_classes)   # smooth + not too bright

    def soften(color, mix=0.30):
        """Mix with white to reduce saturation."""
        return tuple(color[:3] * (1 - mix) + np.array([1,1,1]) * mix)

    class_to_color = {
        cid: soften(base_cmap(i))    # apply white-mix softening
        for i, cid in enumerate(unique_classes)
    }

    colors = np.array([class_to_color[c] for c in all_class_ids])


    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.6, 3.6))  # single-column 3.5~3.6 inch

    mask_us = domains == 0
    mask_eu = domains == 1

    # US: solid circles
    ax.scatter(
        emb_2d[mask_us, 0],
        emb_2d[mask_us, 1],
        c=colors[mask_us],
        s=9,
        marker="o",
        alpha=0.75,
        linewidths=0.0,
        label="US"
    )

    # EU: faint crosses
    ax.scatter(
        emb_2d[mask_eu, 0],
        emb_2d[mask_eu, 1],
        c=colors[mask_eu],
        s=12,
        marker="x",
        alpha=0.40,
        linewidths=0.6,
        label="EU"
    )

    # Clean style
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.set_title("US/EU SigLIP Embedding Geometry", fontsize=10, fontweight="bold")

    # Legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="black", markersize=5, linestyle="None", label="US"),
        Line2D([0], [0], marker="x", color="black", markersize=5, linestyle="None", label="EU"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=8
    )

    plt.tight_layout()
    fig.savefig(output, dpi=450, bbox_inches="tight")
    print(f"[INFO] Saved figure to: {output}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate paper-quality US/EU t-SNE figure.")
    parser.add_argument("--us_emb", required=True)
    parser.add_argument("--eu_emb", required=True)
    parser.add_argument("--output", default="fig_tsne_paper.png")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_tsne(
        us_file=args.us_emb,
        eu_file=args.eu_emb,
        output=args.output,
        n_samples=args.n_samples,
        perplexity=args.perplexity,
        random_state=args.seed
    )


if __name__ == "__main__":
    main()
