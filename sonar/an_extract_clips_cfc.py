"""
an_extract_clips_cfc.py  (Anonymous Review Version)

This script provides a generic example of extracting short video clips
from a set of image sequences with tracking annotations. Dataset-specific
details, naming conventions, and structural assumptions have been
generalized for anonymous review.

The script remains functional when used with any MOT-style annotations
and a folder of frame images, including synthetic or toy data.
"""

import os
import json
import glob
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_mot_annotations(gt_path):
    """
    Load tracking annotations in a generic MOT-format:

    frame_id, track_id, x, y, w, h, ...
    """
    tracks = {}
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue

            frame_id = int(parts[0]) - 1   # convert to 0-based index
            track_id = int(parts[1])
            x, y = float(parts[2]), float(parts[3])
            w, h = float(parts[4]), float(parts[5])

            tracks.setdefault(track_id, []).append(
                dict(frame=frame_id, x=x, y=y, w=w, h=h)
            )
    return tracks


def map_frames(images_root):
    """
    Build a generic mapping: frame_idx → image_path
    Assumes file names end with _<frame>.jpg or .png
    """
    frame_map = {}
    imgs = glob.glob(str(Path(images_root) / "*.jpg"))
    imgs += glob.glob(str(Path(images_root) / "*.png"))

    for path in imgs:
        stem = Path(path).stem
        try:
            fid = int(stem.split("_")[-1])
            frame_map[fid] = path
        except ValueError:
            continue

    return frame_map


def crop_and_resize(frame, box, out_size=112):
    x, y, w, h = box
    H, W = frame.shape[:2]

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    roi = frame[y1:y2, x1:x2]
    return cv2.resize(roi, (out_size, out_size))


def extract_clips_for_sequence(
    seq_dir,
    images_root,
    output_dir,
    clip_length=16,
    clip_stride=8,
    resize_size=112,
    max_clips_per_track=4,
):
    seq_dir = Path(seq_dir)
    seq_name = seq_dir.name

    gt_file = seq_dir / "gt.txt"
    if not gt_file.exists():
        print(f"[WARN] No gt.txt found in {seq_dir}")
        return

    out_dir = Path(output_dir) / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    tracks = load_mot_annotations(gt_file)
    frame_map = map_frames(images_root)

    for tid, dets in tracks.items():
        dets = sorted(dets, key=lambda x: x["frame"])

        if len(dets) < clip_length:
            continue

        clip_idx = 0
        for start in range(0, len(dets) - clip_length + 1, clip_stride):
            if clip_idx >= max_clips_per_track:
                break

            window = dets[start:start + clip_length]
            clip_frames = []
            ok = True

            for d in window:
                fid = d["frame"]
                if fid not in frame_map:
                    ok = False
                    break

                img = cv2.imread(frame_map[fid], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    ok = False
                    break

                crop = crop_and_resize(img, (d["x"], d["y"], d["w"], d["h"]), resize_size)
                clip_frames.append(crop)

            if not ok:
                continue

            clip_arr = np.stack(clip_frames, axis=0)
            save_path = out_dir / f"clip_{len(metadata)}.npz"
            np.savez_compressed(save_path, clip=clip_arr)

            metadata.append(
                dict(
                    sequence=seq_name,
                    track=tid,
                    clip_id=len(metadata),
                    frames=[d["frame"] for d in window],
                )
            )
            clip_idx += 1

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] {seq_name}: extracted {len(metadata)} clips")


def extract_all(
    mot_root,
    images_root,
    output_dir,
    clip_length=16,
    clip_stride=8,
    resize_size=112,
    max_clips_per_track=4,
):
    seq_dirs = [d for d in Path(mot_root).iterdir() if d.is_dir()]
    print(f"[INFO] Found {len(seq_dirs)} sequences")

    for sd in seq_dirs:
        extract_clips_for_sequence(
            sd,
            images_root,
            output_dir,
            clip_length,
            clip_stride,
            resize_size,
            max_clips_per_track,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generic clip extraction (anonymous version).")
    parser.add_argument("--mot_root", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=8)
    parser.add_argument("--resize_size", type=int, default=112)
    parser.add_argument("--max_clips_per_track", type=int, default=4)

    args = parser.parse_args()

    extract_all(
        args.mot_root,
        args.images_root,
        args.output_dir,
        args.clip_length,
        args.clip_stride,
        args.resize_size,
        args.max_clips_per_track,
    )
