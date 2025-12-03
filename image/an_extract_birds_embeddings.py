"""
an_extract_birds_embeddings.py (Anonymous Review Version)

This script demonstrates a generic pipeline for extracting image
embeddings using a vision model such as SigLIP. Dataset-specific
details, path conventions, and field names have been generalized
for anonymous review.

This version remains functional when provided with public or toy
datasets, but it does not reflect the structure of the full system.
"""

import csv
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModel


def load_csv(csv_path):
    """Load CSV as a list of generic records."""
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    return items


def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to load: {path} ({e})")
        return None


def extract_embeddings(csv_file, output_file, batch_size=16, model_name="google/siglip-base-patch16-224"):
    print("=" * 60)
    print(f"[INFO] Loading vision model: {model_name}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    records = load_csv(csv_file)
    print(f"[INFO] Loaded {len(records)} entries from {csv_file}")

    all_emb = []
    all_labels = []
    all_aux = []
    all_paths = []

    batch_imgs = []
    batch_meta = []

    for rec in tqdm(records):
        path = rec.get("path") or rec.get("image_path") or rec[list(rec.keys())[0]]
        img = load_image(path)
        if img is None:
            continue

        batch_imgs.append(img)
        batch_meta.append(rec)

        if len(batch_imgs) == batch_size:
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            all_emb.append(feats.cpu().numpy())

            for r in batch_meta:
                # Generic metadata fields (anonymized)
                lbl = int(r.get("label", r.get("class_id", 0)))
                aux = r.get("aux", r.get("domain", "unknown"))
                all_labels.append(lbl)
                all_aux.append(aux)
                all_paths.append(path)

            batch_imgs = []
            batch_meta = []

    # last batch
    if batch_imgs:
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
        all_emb.append(feats.cpu().numpy())

        for r in batch_meta:
            lbl = int(r.get("label", r.get("class_id", 0)))
            aux = r.get("aux", r.get("domain", "unknown"))
            path = r.get("path") or r.get("image_path") or ""
            all_labels.append(lbl)
            all_aux.append(aux)
            all_paths.append(path)

    emb = np.concatenate(all_emb, axis=0)
    labels = np.array(all_labels)
    aux = np.array(all_aux)
    paths = np.array(all_paths)

    print(f"[INFO] Saving → {output_file}")
    np.savez(
        output_file,
        vectors=emb.astype(np.float32),   # generic name
        labels=labels,
        aux=aux,
        paths=paths,
    )
    print("[INFO] Done.")


def main():
    parser = argparse.ArgumentParser(description="Extract generic image embeddings (anonymous version).")
    parser.add_argument("--csv", required=True, help="Input CSV file (generic format).")
    parser.add_argument("--out", required=True, help="Output NPZ file.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224")
    args = parser.parse_args()

    extract_embeddings(args.csv, args.out, args.batch, args.model)


if __name__ == "__main__":
    main()
