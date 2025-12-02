"""
Figure 2: t-SNE of 32D and 8D Prototypes (clean, low-saturation)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse


def load_proto(path):
    d = np.load(path, allow_pickle=True)
    return d["prototypes"], d["class_id"]


def soften(color, mix=0.30):
    """Reduce saturation by mixing with white."""
    return tuple(color * (1 - mix) + np.array([1, 1, 1]) * mix)


def plot_two_proto_tsne(
    file32, file8, output="fig2_tsne_prototypes.png",
    perplexity=10, seed=42
):
    print("Loading:", file32)
    P32, C32 = load_proto(file32)

    print("Loading:", file8)
    P8, C8 = load_proto(file8)

    # validate same classes
    assert np.array_equal(np.sort(C32), np.sort(C8)), "class mismatch between 32D and 8D"

    classes = np.unique(C32)
    n_classes = len(classes)

    # low-saturation color map
    cmap = plt.cm.get_cmap("viridis", n_classes)
    class_to_color = {cid: soften(cmap(i)) for i, cid in enumerate(classes)}

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))  # single-column width ~7.2in

    # ---------------------------------------------
    # panel 1: 32D
    # ---------------------------------------------
    print("t-SNE 32D...")
    tsne_32 = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(P32) - 1),
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    P32_2d = tsne_32.fit_transform(P32)

    ax = axes[0]
    for cid in classes:
        mask = (C32 == cid)
        ax.scatter(
            P32_2d[mask, 0], P32_2d[mask, 1],
            s=22, marker="o", alpha=0.8,
            c=[class_to_color[cid]],
            linewidths=0
        )
    ax.set_title("32D Prototypes", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)

    # ---------------------------------------------
    # panel 2: 8D
    # ---------------------------------------------
    print("t-SNE 8D...")
    tsne_8 = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(P8) - 1),
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    P8_2d = tsne_8.fit_transform(P8)

    ax = axes[1]
    for cid in classes:
        mask = (C8 == cid)
        ax.scatter(
            P8_2d[mask, 0], P8_2d[mask, 1],
            s=22, marker="o", alpha=0.8,
            c=[class_to_color[cid]],
            linewidths=0
        )
    ax.set_title("8D Prototypes", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1", fontsize=9)

    fig.suptitle("Effect of Prototype Dimensionality on Latent Geometry",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output, dpi=450, bbox_inches="tight")
    print("Saved to:", output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto32", required=True)
    parser.add_argument("--proto8", required=True)
    parser.add_argument("--output", default="fig2_tsne_prototypes.png")
    parser.add_argument("--perplexity", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    plot_two_proto_tsne(
        args.proto32,
        args.proto8,
        output=args.output,
        perplexity=args.perplexity,
        seed=args.seed
    )
