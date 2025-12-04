#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
an_viz_clips_cfc.py  (Anonymous Review Version)

Visualize extracted clips:
- Randomly sample K clips
- Produce:
  1) Multi-frame grid (with bounding boxes)
  2) Multi-frame grid (clean, no bbox)
  3) MP4 video (with bbox)
  4) MP4 video (clean)
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import argparse
import random


# -------------------------------------------------
# Load metadata
# -------------------------------------------------
def load_metadata(path: Path):
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------
# Resolve original image path
# -------------------------------------------------
def resolve_image_path(img_root: Path, seq: str, fid: int, fname: str | None):
    """
    Tries to resolve the raw image path using:
    1) Provided filename
    2) Heuristic fallback: seq_fid.jpg / seq_00012.png etc.
    """
    # direct filename
    if fname:
        p = img_root / fname
        if p.exists():
            return p

    # common fallback patterns
    for ext in [".jpg", ".png", ".jpeg"]:
        p1 = img_root / f"{seq}_{fid}{ext}"
        p2 = img_root / f"{seq}_{fid:05d}{ext}"
        if p1.exists():
            return p1
        if p2.exists():
            return p2

    return None


# -------------------------------------------------
# Visualize clip grid
# -------------------------------------------------
def visualize_clip(
    clip_path: Path,
    meta: dict,
    img_root: Path,
    out_path: Path,
    num_frames: int = 6,
    with_bbox: bool = True,
):
    clip = np.load(clip_path)["clip"]   # (T, H, W)
    frames = meta["frames"]
    seq = meta["sequence"]
    tid = meta["track_id"]
    cid = meta["clip_id"]

    T = len(frames)
    if T == 0:
        print(f"⚠️ Empty clip {clip_path.name}")
        return

    if num_frames >= T:
        idxs = list(range(T))
    else:
        idxs = np.linspace(0, T - 1, num_frames, dtype=int).tolist()

    cols = min(3, len(idxs))
    rows = (len(idxs) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)

    for ax_i, fi in enumerate(idxs):
        ax = axes[ax_i]
        fmeta = frames[fi]

        fid = fmeta["frame_id_0based"]
        fname = fmeta.get("img_filename")
        bbox = fmeta["bbox_xywh"]

        img_p = resolve_image_path(img_root, seq, fid, fname)

        if img_p and img_p.exists():
            raw = cv2.imread(str(img_p))
            if raw is not None:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                ax.imshow(raw)
                if with_bbox:
                    x, y, w, h = bbox
                    ax.add_patch(Rectangle((x, y), w, h,
                                           edgecolor="red", fill=False, linewidth=2))
                ax.set_title(f"{img_p.name}\nframe={fid}", fontsize=8)
            else:
                ax.imshow(clip[fi], cmap="gray")
                ax.set_title(f"ROI frame={fid} (read fail)", fontsize=8)
        else:
            ax.imshow(clip[fi], cmap="gray")
            ax.set_title(f"ROI frame={fid}", fontsize=8)

        ax.axis("off")

    for k in range(len(idxs), len(axes)):
        axes[k].axis("off")

    plt.suptitle(f"seq={seq} | track={tid} | clip={cid}", fontsize=10)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📷 Saved: {out_path}")


# -------------------------------------------------
# Generate clip video
# -------------------------------------------------
def generate_video(
    clip_path: Path,
    meta: dict,
    img_root: Path,
    out_path: Path,
    fps=10,
    with_bbox=True,
):
    clip = np.load(clip_path)["clip"]   # (T, H, W)
    frames = meta["frames"]
    seq = meta["sequence"]
    cid = meta["clip_id"]

    if not frames:
        print(f"⚠️ Empty clip {cid}")
        return

    # original frame size
    f0 = frames[0]
    H, W = f0["orig_h"], f0["orig_w"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    if not vw.isOpened():
        print(f"⚠️ Cannot open writer: {out_path}")
        return

    count = 0

    for idx, fmeta in enumerate(frames):
        fid = fmeta["frame_id_0based"]
        fname = fmeta.get("img_filename")
        bbox = fmeta["bbox_xywh"]

        img_p = resolve_image_path(img_root, seq, fid, fname)
        frame = None

        if img_p and img_p.exists():
            frame = cv2.imread(str(img_p))

        if frame is None:
            roi = clip[idx]
            frame = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame, (W, H))
        else:
            if with_bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (int(x), int(y)),
                              (int(x + w), int(y + h)), (0, 0, 255), 2)

            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))

        vw.write(frame)
        count += 1

    vw.release()

    if count > 0:
        print(f"  🎞 Saved ({count} frames): {out_path}")
    else:
        print(f"⚠️ No frames written for clip {cid}")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Anonymous Clip Visualizer")
    ap.add_argument("--clips_dir", type=str, required=True)
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="viz_out")
    ap.add_argument("--num_clips", type=int, default=5)
    ap.add_argument("--num_frames", type=int, default=6)
    args = ap.parse_args()

    clips_root = Path(args.clips_dir)
    img_root = Path(args.img_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect clips
    all_clips = []
    seq_dirs = [d for d in clips_root.iterdir() if d.is_dir()]
    print(f"Found {len(seq_dirs)} sequences")

    for sd in seq_dirs:
        meta_path = sd / "metadata.json"
        if not meta_path.exists():
            continue
        meta_list = load_metadata(meta_path)
        for m in meta_list:
            cid = m["clip_id"]
            cpath = sd / f"clip_{cid}.npz"
            if cpath.exists():
                all_clips.append((sd.name, cpath, m))

    print(f"Found {len(all_clips)} clips total")

    if not all_clips:
        print("❌ No clips found. Check --clips_dir.")
        return

    chosen = random.sample(all_clips, min(args.num_clips, len(all_clips)))
    print("\nSelected clips:")
    for seq, cp, m in chosen:
        print(f"  {seq}/clip_{m['clip_id']}.npz (track={m['track_id']})")

    print("\nGenerating outputs...")

    for seq, cp, m in chosen:
        tag = f"{seq}_t{m['track_id']}_c{m['clip_id']}"

        # Grid with bbox
        visualize_clip(cp, m, img_root, out_root / f"{tag}_bbox.png",
                       num_frames=args.num_frames, with_bbox=True)

        # Grid without bbox
        visualize_clip(cp, m, img_root, out_root / f"{tag}_clean.png",
                       num_frames=args.num_frames, with_bbox=False)

        # Video with bbox
        generate_video(cp, m, img_root, out_root / f"{tag}_bbox.mp4",
                       fps=10, with_bbox=True)

        # Video without bbox
        generate_video(cp, m, img_root, out_root / f"{tag}_clean.mp4",
                       fps=10, with_bbox=False)

    print(f"\n✅ Done. Results saved in: {out_root}")


if __name__ == "__main__":
    main()
