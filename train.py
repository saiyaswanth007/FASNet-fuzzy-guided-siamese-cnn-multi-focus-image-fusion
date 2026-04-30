#!/usr/bin/env python3
"""
train.py
=========
Training script for the Siamese FCNN model.

Exact hyperparameters from Section 3.2 of:
  "A Fuzzy Convolutional Neural Network for Multi-Focus Image Fusion"
  Bhalla et al., JVCIR 2022.

Hyperparameters (STRICT — paper Table):
  Optimizer    : SGD (NOT Adam)
  Learning rate: 0.002
  Momentum     : 0.9
  Weight decay : 0.0005
  Initializer  : Xavier (in model)
  Loss         : Cross-Entropy
  Batch size   : 64

Input:
  datasets/synthetic/training_dataset_norm.npz
    patches_A : (5000, 16, 16) float32  [0,1]
    patches_B : (5000, 16, 16) float32  [0,1]
    labels    : (5000,)        int8     {0,1}

Note: The NPZ contains raw [0,255] patches normalized to [0,1].
      We additionally fuzzify them using the S-shaped membership
      function before feeding to the network.

Usage:
  python3 train.py [--epochs 30] [--batch-size 64] [--no-fuzzify]
                   [--checkpoint checkpoints/] [--resume model.pt]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.siamese_fcnn import SiameseFCNN
from utils.fuzzy_preprocessing import fuzzify_image


# ---------------------------------------------------------------------------
# Hyperparameters (paper-exact)
# ---------------------------------------------------------------------------
LR           = 0.002
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005
BATCH_SIZE   = 64
VAL_SPLIT    = 0.1      # 10% validation


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PatchPairDataset(Dataset):
    """
    Loads (patch_A, patch_B, label) triples from the training NPZ.

    Optionally applies fuzzy S-membership fuzzification to each patch
    before returning, converting [0,1] pixel values to fuzzy domain.
    """

    def __init__(self, npz_path: str, fuzzify: bool = True):
        data = np.load(str(npz_path))
        self.A      = data["patches_A"].astype(np.float32)   # (N,16,16) [0,255] or [0,1]
        self.B      = data["patches_B"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)
        self.fuzzify = fuzzify

        # Normalize to [0,1] if still in [0,255]
        if self.A.max() > 1.0:
            self.A /= 255.0
            self.B /= 255.0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        a = self.A[idx]   # (16, 16)
        b = self.B[idx]   # (16, 16)

        if self.fuzzify:
            a = fuzzify_image(a)   # S-membership → still (16,16) [0,1]
            b = fuzzify_image(b)

        # Add channel dim → (1, 16, 16)
        a_t = torch.from_numpy(a).unsqueeze(0)
        b_t = torch.from_numpy(b).unsqueeze(0)
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        return a_t, b_t, lbl


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of correct predictions."""
    predicted = preds.argmax(dim=1)
    return (predicted == labels).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for patch_a, patch_b, labels in loader:
        patch_a = patch_a.to(device)
        patch_b = patch_b.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        probs = model(patch_a, patch_b)         # (B, 2)
        loss  = criterion(probs, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(probs, labels) * bs
        n          += bs

    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for patch_a, patch_b, labels in loader:
        patch_a = patch_a.to(device)
        patch_b = patch_b.to(device)
        labels  = labels.to(device)

        probs = model(patch_a, patch_b)
        loss  = criterion(probs, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(probs, labels) * bs
        n          += bs

    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Siamese FCNN for patch-level focus classification."
    )
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--momentum",   type=float, default=MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--no-fuzzify", action="store_true",
                        help="Skip fuzzy preprocessing (use raw normalized patches).")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints"),
                        help="Directory to save checkpoints.")
    parser.add_argument("--resume",  default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "datasets"),
                        help="Root dataset directory.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    npz_path = Path(args.data_root) / "synthetic" / "training_dataset_norm.npz"
    if not npz_path.exists():
        # Fall back to raw (un-normalised) dataset
        npz_path = Path(args.data_root) / "synthetic" / "training_dataset.npz"
    if not npz_path.exists():
        print(f"[ERROR] No training NPZ found in {Path(args.data_root)/'synthetic'}")
        sys.exit(1)

    print(f"[INFO] Loading dataset: {npz_path}")
    dataset  = PatchPairDataset(str(npz_path), fuzzify=not args.no_fuzzify)
    n_total  = len(dataset)
    n_val    = max(1, int(n_total * VAL_SPLIT))
    n_train  = n_total - n_val

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=False)

    print(f"[INFO] Train: {n_train}  Val: {n_val}  Fuzzify: {not args.no_fuzzify}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SiameseFCNN().to(device)

    # Print parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {n_params:,}")

    # ── Loss & Optimiser (paper-exact) ────────────────────────────────────────
    # CrossEntropyLoss works on raw logits but our model outputs softmax.
    # We use NLLLoss(log(probs)) which is equivalent to CE for softmax outputs.
    criterion = nn.NLLLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr           = args.lr,
        momentum     = args.momentum,
        weight_decay = args.weight_decay,
    )

    start_epoch = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[INFO] Resumed from epoch {start_epoch-1}")

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}  {'Time':>7}")
    print("-" * 65)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # NLLLoss requires log probabilities
        def _criterion(probs, labels):
            return criterion(torch.log(probs + 1e-8), labels)

        tr_loss, tr_acc = train_one_epoch(model, train_loader, _criterion,
                                          optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, _criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(f"{epoch+1:>6}  {tr_loss:>10.4f}  {tr_acc:>10.4f}  "
              f"{vl_loss:>10.4f}  {vl_acc:>10.4f}  {elapsed:>6.1f}s")

        # Save checkpoint every 5 epochs and best model
        if (epoch + 1) % 5 == 0 or vl_acc > best_val_acc:
            ckpt_path = ckpt_dir / f"fcnn_epoch{epoch+1:03d}.pt"
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc":   vl_acc,
                "history":   history,
            }, str(ckpt_path))

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                best_path = ckpt_dir / "fcnn_best.pt"
                torch.save({
                    "epoch":   epoch,
                    "model":   model.state_dict(),
                    "val_acc": vl_acc,
                }, str(best_path))
                print(f"         [BEST] val_acc={vl_acc:.4f} → saved {best_path.name}")

    # ── Save training curves ──────────────────────────────────────────────────
    np.save(str(ckpt_dir / "training_history.npy"), history)
    _plot_history(history, ckpt_dir / "training_curves.png")

    print(f"\n[DONE] Training complete.")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Checkpoints       : {ckpt_dir}")
    print(f"  Best model        : {ckpt_dir/'fcnn_best.pt'}")
    print(f"\nNext step: python3 inference/score_map.py")


def _plot_history(history: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                        facecolor="#1a1a2e")
        for ax in (ax1, ax2):
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#e0e0e0")
            for spine in ax.spines.values():
                spine.set_edgecolor("#0f3460")

        epochs = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs, history["train_loss"], color="#00d4ff", label="Train")
        ax1.plot(epochs, history["val_loss"],   color="#ff6b6b", label="Val")
        ax1.set_title("Cross-Entropy Loss", color="#00d4ff")
        ax1.set_xlabel("Epoch", color="#e0e0e0")
        ax1.legend(facecolor="#16213e", labelcolor="#e0e0e0")

        ax2.plot(epochs, history["train_acc"], color="#00d4ff", label="Train")
        ax2.plot(epochs, history["val_acc"],   color="#ff6b6b", label="Val")
        ax2.set_title("Accuracy", color="#00d4ff")
        ax2.set_xlabel("Epoch", color="#e0e0e0")
        ax2.set_ylim(0, 1)
        ax2.legend(facecolor="#16213e", labelcolor="#e0e0e0")

        fig.suptitle("Siamese FCNN — Training Curves",
                     color="#00d4ff", fontweight="bold")
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"[INFO] Training curves saved: {out_path}")
    except Exception as exc:
        print(f"[WARN] Could not plot training curves: {exc}")


if __name__ == "__main__":
    main()
