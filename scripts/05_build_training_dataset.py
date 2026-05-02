#!/usr/bin/env python3
"""
scripts/05_build_training_dataset.py
======================================
Assembles the final labelled training dataset (patches_A, patches_B, labels)
exactly as originally described.  A total of 5,000 images from an ImageNet database are retrieved, where
  2500 images are positive samples, whose output is set to be 1 and 2500
  negative samples with 0. Each sample consisted of two input image patches
  (16 × 16) and reference image labels, i.e., 1 or 0.

  Images are rotated by 90° and 180° in both horizontal and vertical
  directions. Data augmentation is done to increase the dataset size.

Label convention:
  Label = 1 → patch_A is FOCUSED   (patch_A from original, patch_B from blurred)
  Label = 0 → patch_A is BLURRED   (patch_A from blurred,  patch_B from original)

Input:
  datasets/synthetic/patches/focused/<stem>.npy
  datasets/synthetic/patches/blurred/blur_level_<N>/<stem>.npy

Output:
  datasets/synthetic/training_dataset.npz
    patches_A : (N_total, 16, 16) float32  [0,255]
    patches_B : (N_total, 16, 16) float32  [0,255]
    labels    : (N_total,)        int8     {0,1}

  datasets/synthetic/training_dataset_norm.npz
    Same but values normalized to [0, 1].

Usage:
  python3 scripts/05_build_training_dataset.py [--data-root ./datasets]
                                               [--n-positive 2500]
                                               [--n-negative 2500]
                                               [--augment]
                                               [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.image_utils import augment_patch, N_BLUR_LEVELS, PATCH_SIZE

# Default Values
N_POSITIVE = 2_500   # label=1 samples (focused A, blurred B)
N_NEGATIVE = 2_500   # label=0 samples (blurred A, focused B)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_patch_file(path: Path) -> np.ndarray | None:
    """Load an .npy patch file; returns None on error."""
    try:
        arr = np.load(str(path))
        if arr.ndim == 3 and arr.shape[1] == PATCH_SIZE and arr.shape[2] == PATCH_SIZE:
            return arr.astype(np.float32)
        return None
    except Exception:
        return None


def random_patch(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one random patch from the (N, H, W) array."""
    idx = rng.integers(0, len(arr))
    return arr[idx]


# ---------------------------------------------------------------------------
# Collect available stems (images that have BOTH focused and blurred patches)
# ---------------------------------------------------------------------------
def collect_matched_stems(
    focused_dir: Path,
    blurred_lvl_dirs: list[Path],
) -> list[str]:
    """
    Return stems (image IDs) for which BOTH focused and ≥1 blurred patch
    files exist.
    """
    focused_stems = {p.stem for p in focused_dir.glob("*.npy")}
    blurred_stems: set[str] = set()
    for lvl_dir in blurred_lvl_dirs:
        blurred_stems |= {p.stem for p in lvl_dir.glob("*.npy")}
    matched = sorted(focused_stems & blurred_stems)
    return matched


# ---------------------------------------------------------------------------
# Build sample arrays
# ---------------------------------------------------------------------------
def build_samples(
    focused_dir: Path,
    blurred_lvl_dirs: list[Path],
    stems: list[str],
    n_positive: int,
    n_negative: int,
    do_augment: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the (patches_A, patches_B, labels) arrays.

    Positive samples (label=1): A=focused, B=blurred
    Negative samples (label=0): A=blurred, B=focused

    With augmentation, each raw patch produces 5 variants
    (original, rot90, rot180, horizontal_flip, vertical_flip).
    """
    pos_A: list[np.ndarray] = []
    pos_B: list[np.ndarray] = []
    neg_A: list[np.ndarray] = []
    neg_B: list[np.ndarray] = []

    # ── distribute n_positive and n_negative across stems ────────────────────
    # Each stem can contribute multiple patches; we sample randomly.
    n_stems = len(stems)
    if n_stems == 0:
        raise RuntimeError("No matched stems found — cannot build dataset.")

    # Shuffle stems
    rng.shuffle(np.array(stems))   # inplace
    stem_list = stems.copy()

    # Divide needed patches across stems
    pos_per_stem = max(1, (n_positive + n_stems - 1) // n_stems)
    neg_per_stem = max(1, (n_negative + n_stems - 1) // n_stems)

    print(f"  Sampling ~{pos_per_stem} positive + ~{neg_per_stem} negative "
          f"patches per stem from {n_stems} stems …")

    for stem in tqdm(stem_list, desc="Building samples"):
        # Load focused patches
        f_path  = focused_dir / f"{stem}.npy"
        f_arr   = load_patch_file(f_path)
        if f_arr is None or len(f_arr) == 0:
            continue

        # Choose a random blur level for this stem
        valid_lvl_dirs = [d for d in blurred_lvl_dirs
                          if (d / f"{stem}.npy").exists()]
        if not valid_lvl_dirs:
            continue
        lvl_dir = rng.choice(np.array(valid_lvl_dirs))  # type: ignore[arg-type]
        b_arr = load_patch_file(lvl_dir / f"{stem}.npy")
        if b_arr is None or len(b_arr) == 0:
            continue

        # Positive samples
        for _ in range(min(pos_per_stem, len(f_arr))):
            fp = random_patch(f_arr, rng)
            bp = random_patch(b_arr, rng)
            if do_augment:
                for fp_aug, bp_aug in zip(augment_patch(fp), augment_patch(bp)):
                    pos_A.append(fp_aug)
                    pos_B.append(bp_aug)
            else:
                pos_A.append(fp)
                pos_B.append(bp)

        # Negative samples (swap roles: A=blurred, B=focused)
        for _ in range(min(neg_per_stem, len(f_arr))):
            fp = random_patch(f_arr, rng)
            bp = random_patch(b_arr, rng)
            if do_augment:
                for fp_aug, bp_aug in zip(augment_patch(fp), augment_patch(bp)):
                    neg_A.append(bp_aug)  # blurred as A
                    neg_B.append(fp_aug)  # focused as B
            else:
                neg_A.append(bp)
                neg_B.append(fp)

        # Early exit once we have enough
        if len(pos_A) >= n_positive and len(neg_A) >= n_negative:
            break

    # ── Trim to exact counts ──────────────────────────────────────────────────
    pos_A = pos_A[:n_positive]
    pos_B = pos_B[:n_positive]
    neg_A = neg_A[:n_negative]
    neg_B = neg_B[:n_negative]

    # ── Concatenate and shuffle ───────────────────────────────────────────────
    all_A  = np.stack(pos_A + neg_A, axis=0)            # (N, 16, 16)
    all_B  = np.stack(pos_B + neg_B, axis=0)
    labels = np.array([1] * len(pos_A) + [0] * len(neg_A), dtype=np.int8)

    perm = rng.permutation(len(labels))
    all_A  = all_A[perm]
    all_B  = all_B[perm]
    labels = labels[perm]

    return all_A, all_B, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble final labelled training dataset (NPZ)."
    )
    parser.add_argument(
        "--data-root", default=str(PROJECT_ROOT / "datasets"),
        help="Root datasets directory (default: ./datasets)"
    )
    parser.add_argument(
        "--n-positive", type=int, default=N_POSITIVE,
        help=f"Number of positive (label=1) samples (default: {N_POSITIVE})"
    )
    parser.add_argument(
        "--n-negative", type=int, default=N_NEGATIVE,
        help=f"Number of negative (label=0) samples (default: {N_NEGATIVE})"
    )
    parser.add_argument(
        "--augment", action="store_true", default=True,
        help="Apply 90°/180° rotation augmentation"
    )
    parser.add_argument(
        "--no-augment", dest="augment", action="store_false",
        help="Disable augmentation."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    data_root    = Path(args.data_root)
    synthetic    = data_root / "synthetic"
    patches_root = synthetic / "patches"
    focused_dir  = patches_root / "focused"

    # Collect blur level directories
    blurred_base = patches_root / "blurred"
    blurred_lvl_dirs = [
        blurred_base / f"blur_level_{lvl}"
        for lvl in range(1, N_BLUR_LEVELS + 1)
        if (blurred_base / f"blur_level_{lvl}").exists()
    ]

    if not focused_dir.exists() or not any(focused_dir.glob("*.npy")):
        print(f"[ERROR] No focused patches in {focused_dir}")
        print("  Run:  python3 scripts/04_extract_patches.py")
        sys.exit(1)

    if not blurred_lvl_dirs:
        print(f"[ERROR] No blurred patch directories found under {blurred_base}")
        print("  Run scripts 03 and 04 first.")
        sys.exit(1)

    # ── Match stems ───────────────────────────────────────────────────────────
    print("[INFO] Matching focused ↔ blurred patch files …")
    stems = collect_matched_stems(focused_dir, blurred_lvl_dirs)
    print(f"  {len(stems)} matched image stems found.")

    if len(stems) == 0:
        print("[ERROR] No matched stems — patches must exist for both "
              "focused and blurred versions of each image.")
        sys.exit(1)

    # ── Build dataset ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    print(f"\n[INFO] Building dataset …")
    print(f"  Positive samples (label=1) : {args.n_positive}")
    print(f"  Negative samples (label=0) : {args.n_negative}")
    print(f"  Augmentation (90°,180°)    : {args.augment}")
    print(f"  Random seed                : {args.seed}")

    A, B, labels = build_samples(
        focused_dir     = focused_dir,
        blurred_lvl_dirs = blurred_lvl_dirs,
        stems           = stems,
        n_positive      = args.n_positive,
        n_negative      = args.n_negative,
        do_augment      = args.augment,
        rng             = rng,
    )

    # ── Save raw (unnormalized) ───────────────────────────────────────────────
    out_raw = synthetic / "training_dataset.npz"
    np.savez_compressed(
        str(out_raw),
        patches_A = A,
        patches_B = B,
        labels    = labels,
    )

    # ── Save normalized [0,1] version ─────────────────────────────────────────
    out_norm = synthetic / "training_dataset_norm.npz"
    np.savez_compressed(
        str(out_norm),
        patches_A = (A / 255.0).astype(np.float32),
        patches_B = (B / 255.0).astype(np.float32),
        labels    = labels,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total    = len(labels)
    n_pos_act  = int(labels.sum())
    n_neg_act  = n_total - n_pos_act

    print(f"\n{'='*60}")
    print("TRAINING DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total samples       : {n_total:,}")
    print(f"  Positive (label=1)  : {n_pos_act:,}")
    print(f"  Negative (label=0)  : {n_neg_act:,}")
    print(f"  patches_A shape     : {A.shape}  dtype={A.dtype}")
    print(f"  patches_B shape     : {B.shape}  dtype={B.dtype}")
    print(f"  labels shape        : {labels.shape}  dtype={labels.dtype}")
    print(f"  A value range       : [{A.min():.1f}, {A.max():.1f}]")
    print(f"\n  Saved (raw)         : {out_raw}")
    print(f"  Saved (normalized)  : {out_norm}")
    print(f"\n[DONE] Training dataset ready.")
    print("Run verify_pipeline.py to validate all pipeline outputs.")


if __name__ == "__main__":
    main()
