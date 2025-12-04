"""
an_extract_clip_embeddings.py  (Anonymous Review Version)

This script demonstrates how to load short video clips (npz files),
pass them through a generic encoder, and save the resulting vectors
together with minimal metadata.  All dataset- and project-specific
details have been removed for anonymous review.  The script remains
functional when used with public or toy data.
"""

import os
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------
# A simple placeholder encoder (anonymous)
# ---------------------------------------------------------
class SimpleEncoder(torch.nn.Module):
    """
    A minimal example encoder for anonymous review purposes.
    Replace with any model that maps (B,1,T,H,W) → (B,D).
    """
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool3d((1,1,1))
        self.linear = torch.nn.Linear(1, embedding_dim)

    def forward(self, x):
        # x: (B,1,T,H,W)
        pooled = self.pool(x)               # (B,1,1,1,1)
        flat = pooled.view(x.size(0), -1)   # (B,1)
        return self.linear(flat)            # (B,D)


# ---------------------------------------------------------
# Utility: load clip from npz
# ---------------------------------------------------------
def load_clip(path: Path) -> np.ndarray:
    data = np.load(str(path))
    clip = data["clip"].astype(np.float32)
    if clip.max() > 1:
        clip /= 255.0
    return clip  # (T,H,W)


def resize_clip(clip: np.ndarray, h: int, w: int):
    T = clip.shape[0]
    frames = [cv2.resize(clip[t], (w, h)) for t in range(T)]
    return np.stack(frames, axis=0)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Extract clip embeddings (anonymous version).")

    parser.add_argument("--input_roots", nargs="+", required=True,
                        help="One or more root dirs containing clip_* and metadata.json.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional checkpoint file for the example encoder.")
    parser.add_argument("--dim", type=int, default=256,
                        help="Output embedding dimension.")
    parser.add_argument("--clip_size", type=int, nargs=2, default=[112,112])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, required=True)

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    H, W = args.clip_size
    os.makedirs(Path(args.out).parent, exist_ok=True)

    # ----------------- initialize encoder -----------------
    encoder = SimpleEncoder(embedding_dim=args.dim).to(device).eval()

    # optional checkpoint
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"], strict=False)
        print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    all_vecs: List[np.ndarray] = []
    all_labels: List[str] = []
    all_ids: List[int] = []

    # ----------------- scan roots -----------------
    with torch.no_grad():
        for root_str in args.input_roots:
            root = Path(root_str)
            meta_path = root / "metadata.json"
            if not meta_path.exists():
                continue

            with open(meta_path, "r") as f:
                meta_list = json.load(f)

            for m in tqdm(meta_list, desc=f"Processing {root.name}", leave=False):
                cid = int(m.get("clip_id", 0))
                clip_npz = root / f"clip_{cid}.npz"
                if not clip_npz.exists():
                    continue

                clip = load_clip(clip_npz)
                clip = resize_clip(clip, H, W)
                clip_tensor = (
                    torch.from_numpy(clip)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)  # (1,1,T,H,W)
                )

                z = encoder(clip_tensor).squeeze(0).cpu().numpy()
                all_vecs.append(z)

                # minimal metadata (anonymous)
                all_labels.append(root.name)
                all_ids.append(cid)

    vectors = np.stack(all_vecs, axis=0)
    labels = np.array(all_labels)
    ids = np.array(all_ids, dtype=np.int32)

    np.savez_compressed(
        args.out,
        vectors=vectors,
        labels=labels,
        indices=ids,
    )
    print(f"[INFO] Saved embeddings to: {args.out}")
    print(f"[INFO] Total clips: {vectors.shape[0]}, dim={vectors.shape[1]}")


if __name__ == "__main__":
    main()
