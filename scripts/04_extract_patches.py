#!/usr/bin/env python3
"""
scripts/04_extract_patches.py
===============================
Extracts 16×16 non-overlapping patches from both focused (original) and
blurred images, and saves them as .npy arrays for training.

Paper reference (Section 3.2, Table 3):
  "The network training has been done with a patch size of 16 × 16."
  "Each image is cropped into patches of size 16 × 16 without overlapping
   using bicubic transformation."
  "The patch size was chosen to be 16 × 16, as explained in Table 3."

Two sets of patches are saved:
  datasets/synthetic/patches/focused/  ← from raw_imagenet (label=1 source)
  datasets/synthetic/patches/blurred/  ← from blur_level_N (label=0 source)

Each .npy file:
  shape : (N_patches, 16, 16)   dtype : float32   range : [0, 255]
  filename: <image_stem>.npy

Design note:
  We extract patches from ALL 5 blur levels to give the training dataset
  construction script (05) choice over which level pairs to use.
  The actual (focused, blurred) pairs with labels are assembled in 05.

Usage:
  python3 scripts/04_extract_patches.py [--data-root ./datasets]
                                        [--workers 4]
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.image_utils import (
    PATCH_SIZE,
    N_BLUR_LEVELS,
    extract_patches_no_overlap,
)


# ---------------------------------------------------------------------------
# Per-image worker
# ---------------------------------------------------------------------------
def _extract_and_save(args: tuple[Path, Path]) -> tuple[str, int, bool]:
    """
    Load one image, extract non-overlapping 16×16 patches, save as .npy.

    Parameters
    ----------
    args : (src_png_path, dst_npy_path)

    Returns
    -------
    (stem, n_patches, success)
    """
    src, dst = args
    try:
        if dst.exists():
            # Resume: count existing patches
            arr = np.load(str(dst), mmap_mode="r")
            return (src.stem, len(arr), True)

        arr = np.array(Image.open(str(src)).convert("L"), dtype=np.float32)
        patches = extract_patches_no_overlap(arr, patch_size=PATCH_SIZE)

        if not patches:
            return (src.stem, 0, False)

        patches_arr = np.stack(patches, axis=0)   # (N, 16, 16)
        np.save(str(dst), patches_arr)
        return (src.stem, len(patches), True)
    except Exception as exc:
        return (src.stem, 0, False)


# ---------------------------------------------------------------------------
# Batch-extract for one source directory
# ---------------------------------------------------------------------------
def extract_all(
    src_dir: Path,
    dst_dir: Path,
    workers: int,
    desc: str,
) -> dict[str, int]:
    """
    Extract patches from every PNG in *src_dir*, saving to *dst_dir*.

    Returns dict {stem → n_patches}.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_files = sorted(src_dir.glob("*.png"))
    if not src_files:
        print(f"  [WARN] No PNGs found in {src_dir}")
        return {}

    tasks = [
        (src, dst_dir / f"{src.stem}.npy")
        for src in src_files
    ]

    results: dict[str, int] = {}
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_extract_and_save, t): t for t in tasks}
        with tqdm(total=len(tasks), desc=desc) as pbar:
            for fut in as_completed(futs):
                stem, n, ok = fut.result()
                if ok and n > 0:
                    results[stem] = n
                else:
                    failed += 1
                pbar.update(1)

    if failed:
        print(f"  [WARN] {failed} images failed patch extraction.")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 16×16 non-overlapping patches from source and blurred images."
    )
    parser.add_argument(
        "--data-root", default=str(PROJECT_ROOT / "datasets"),
        help="Root datasets directory (default: ./datasets)"
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
        help="Number of parallel workers (default: half CPU cores)"
    )
    parser.add_argument(
        "--patch-size", type=int, default=PATCH_SIZE,
        help=f"Patch size (paper: {PATCH_SIZE}, default: {PATCH_SIZE})"
    )
    args = parser.parse_args()

    data_root    = Path(args.data_root)
    synthetic    = data_root / "synthetic"
    src_dir      = synthetic / "raw_imagenet"
    blurred_root = synthetic / "blurred"
    patches_root = synthetic / "patches"

    # Validate inputs
    if not src_dir.exists() or not any(src_dir.glob("*.png")):
        print(f"[ERROR] No source images in {src_dir}")
        print("  Run:  python3 scripts/02_download_imagenet_proxy.py")
        sys.exit(1)

    # ── Step 1: Extract patches from FOCUSED images ──────────────────────────
    print("\n[STEP 1] Extracting patches from focused (original) images …")
    focused_patch_dir = patches_root / "focused"
    focused_stats = extract_all(src_dir, focused_patch_dir, args.workers,
                                 desc="Focused patches")
    total_focused = sum(focused_stats.values())
    print(f"  {len(focused_stats)} images → {total_focused:,} patches "
          f"in {focused_patch_dir}")

    # ── Step 2: Extract patches from each BLUR LEVEL ─────────────────────────
    print("\n[STEP 2] Extracting patches from blurred images (5 levels) …")
    all_blurred_stats: dict[int, dict[str, int]] = {}

    for lvl in range(1, N_BLUR_LEVELS + 1):
        lvl_src = blurred_root / f"blur_level_{lvl}"
        if not lvl_src.exists() or not any(lvl_src.glob("*.png")):
            print(f"  [WARN] blur_level_{lvl} missing — skipping.")
            continue

        lvl_dst = patches_root / "blurred" / f"blur_level_{lvl}"
        stats = extract_all(lvl_src, lvl_dst, args.workers,
                             desc=f"Blur level {lvl}")
        all_blurred_stats[lvl] = stats
        total = sum(stats.values())
        print(f"  Level {lvl}: {len(stats)} images → {total:,} patches")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PATCH EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Patch size          : {args.patch_size} × {args.patch_size}")
    print(f"  Focused images      : {len(focused_stats)}")
    print(f"  Focused patches     : {total_focused:,}")
    for lvl, stats in all_blurred_stats.items():
        print(f"  Blur level {lvl} patches: {sum(stats.values()):,}")

    patches_per_img = total_focused // max(len(focused_stats), 1)
    print(f"\n  Approx patches/img  : {patches_per_img}")
    print(f"  (520÷16)² = {(520 // 16) ** 2} patches per full 520×520 image)")

    print(f"\n[DONE] Patches saved to: {patches_root}")
    print("Next step:  python3 scripts/05_build_training_dataset.py")


if __name__ == "__main__":
    main()
