"""
InfoNCE contrastive learning training script for video clip representations.

Supports the following directory layout:

    clips_dir/
        ├── seq_1/
        │   ├── clip_0.npz
        │   ├── clip_1.npz
        │   └── metadata.json
        ├── seq_2/
        │   ├── clip_0.npz
        │   └── metadata.json

The input clips_dir may be a single directory or a list of directories
(for multi-domain training).
"""

import sys
import os
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import yaml
import argparse
import logging
from typing import Optional, Dict
from tqdm import tqdm

from models import R2Plus1D, PrototypeProjector
from dataset.clip_dataset import ClipDataset, collate_fn
from train.losses import InfoNCELoss
from train.utils import save_checkpoint, load_checkpoint


# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------
def clip_gradients(model: nn.Module, max_norm: float = 1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# ---------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------
def train_epoch(
    encoder: nn.Module,
    projector: nn.Module,
    dataloader: DataLoader,
    criterion: InfoNCELoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:

    encoder.train()
    projector.train()

    total_loss, total_acc, num_batches = 0.0, 0.0, 0
    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:

        # Basic safety checks
        if batch is None or ("clip1" not in batch) or ("clip2" not in batch):
            continue

        clip1 = batch["clip1"].to(device)
        clip2 = batch["clip2"].to(device)

        if clip1.ndim != 5 or clip2.ndim != 5:
            continue

        clip1 = clip1.contiguous()
        clip2 = clip2.contiguous()

        optimizer.zero_grad(set_to_none=True)

        # AMP forward pass
        if use_amp and scaler is not None:
            with autocast("cuda"):
                z1 = encoder(clip1)
                w1 = projector(z1)
                z2 = encoder(clip2)
                w2 = projector(z2)
                loss, info = criterion(w1, w2)

            # Backward + grad clip
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradients(encoder, max_norm=max_grad_norm)
            clip_gradients(projector, max_grad_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        else:
            z1 = encoder(clip1)
            w1 = projector(z1)
            z2 = encoder(clip2)
            w2 = projector(z2)
            loss, info = criterion(w1, w2)
            loss.backward()
            clip_gradients(encoder, max_norm=max_grad_norm)
            clip_gradients(projector, max_grad_norm=max_grad_norm)
            optimizer.step()

        total_loss += info["loss"]
        total_acc += info["accuracy"]
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{info['loss']:.4f}",
            "acc": f"{info['accuracy']:.4f}",
        })

    return {
        "loss": total_loss / max(1, num_batches),
        "accuracy": total_acc / max(1, num_batches),
    }


# ---------------------------------------------------------------
# Warmup + Cosine LR Scheduler
# ---------------------------------------------------------------
class WarmupCosineLR:
    """Epoch-level warmup + cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * float(epoch + 1) / float(self.warmup_epochs)
        else:
            cosine_epoch = epoch - self.warmup_epochs
            cosine_total = max(1, self.total_epochs - self.warmup_epochs)
            ratio = float(cosine_epoch) / float(cosine_total)
            lr = 0.5 * self.base_lr * (
                1.0 + torch.cos(torch.tensor(ratio * 3.1415926))
            ).item()

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        return lr


# ---------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="InfoNCE training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(config["training"]["log_dir"])
    logger.info(f"Using device: {device}")

    # Model
    encoder = R2Plus1D(
        input_channels=1,
        embedding_dim=config["model"]["encoder_dim"],
    ).to(device)

    projector = PrototypeProjector(
        in_dim=config["model"]["encoder_dim"],
        hidden_dim=config["model"]["projector_hidden"],
        out_dim=config["model"]["projector_dim"],
    ).to(device)

    logger.info(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    logger.info(f"Projector params: {sum(p.numel() for p in projector.parameters()):,}")

    # Dataset
    dataset = ClipDataset(
        clips_dir=config["data"]["clips_dir"],
        clip_length=config["data"]["clip_length"],
        clip_size=tuple(config["data"]["clip_size"]),
        augment=config["data"]["augment"],
        min_track_clips=config["data"].get("min_track_clips", 2),
        max_pairs_per_track=config["data"].get("max_pairs_per_track", None),
    )

    logger.info(f"Loaded {len(dataset)} clips.")

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"].get("pin_memory", True),
        collate_fn=collate_fn,
    )

    # Loss / optimizer / scheduler
    criterion = InfoNCELoss(temperature=config["training"]["temperature"])

    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = WarmupCosineLR(
        optimizer,
        warmup_epochs=config["training"].get("warmup_epochs", 5),
        total_epochs=config["training"]["num_epochs"],
        base_lr=config["training"]["learning_rate"],
    )

    use_amp = config["training"].get("use_amp", True)
    scaler = GradScaler("cuda") if use_amp else None

    # Resume
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, encoder, projector, optimizer)
        start_epoch = ckpt["epoch"] + 1

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, config["training"]["num_epochs"]):

        lr = scheduler.step(epoch)
        logger.info(f"Epoch {epoch + 1} | LR {lr:.6f}")

        metrics = train_epoch(
            encoder,
            projector,
            dataloader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            max_grad_norm=config["training"].get("grad_clip", 1.0),
        )

        logger.info(
            f"Epoch {epoch + 1} | Loss {metrics['loss']:.4f} | Acc {metrics['accuracy']:.4f}"
        )

        if (epoch + 1) % config["training"]["save_every"] == 0:
            ckpt_path = save_checkpoint(
                encoder,
                projector,
                optimizer,
                epoch,
                metrics["loss"],
                config["training"]["checkpoint_dir"],
                filename=f"checkpoint_epoch_{epoch + 1}.pth",
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_ckpt = save_checkpoint(
        encoder,
        projector,
        optimizer,
        config["training"]["num_epochs"] - 1,
        metrics["loss"],
        config["training"]["checkpoint_dir"],
        filename="final_checkpoint.pth",
    )
    logger.info(f"Training complete. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
