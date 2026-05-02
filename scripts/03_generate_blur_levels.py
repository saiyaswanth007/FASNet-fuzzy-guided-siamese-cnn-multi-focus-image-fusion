#!/usr/bin/env python3
"""
scripts/03_generate_blur_levels.py
====================================
Generates 5 progressively blurred versions of each source image by
applying a Gaussian filter sequentially. Gaussian filters with a standard deviation of 0.5 are used.
  In the first level, the simulation uses a Gaussian filter with a 0.5
  standard deviation value. And in the second level, the blurred image has
  been obtained using the first level blurred image with the Gaussian filter
  again, and so on.

Input:
  datasets/synthetic/raw_imagenet/*.png   (5,000 grayscale 520×520 images)

Output:
  datasets/synthetic/blurred/
      blur_level_1/*.png
      blur_level_2/*.png
      blur_level_3/*.png
      blur_level_4/*.png
      blur_level_5/*.png

Usage:
  python3 scripts/03_generate_blur_levels.py [--data-root ./datasets]
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
    BLUR_SIGMA,
    N_BLUR_LEVELS,
    apply_gaussian_blur,
    save_grayscale,
)


# ---------------------------------------------------------------------------
# Per-image worker (runs in subprocess for multiprocessing)
# ---------------------------------------------------------------------------
def _process_one_image(args: tuple[Path, list[Path]]) -> tuple[str, bool]:
    """
    Apply 5 sequential blur levels to a single source image.

    Parameters
    ----------
    args : (src_path, [level_dir_1, ..., level_dir_5])

    Returns
    -------
    (stem, success)
    """
    src_path, level_dirs = args
    stem = src_path.stem

    try:
        # Load the already-grayscale source PNG
        arr = np.array(Image.open(str(src_path)).convert("L"),
                       dtype=np.float32)

        current = arr.copy()
        for level_idx, level_dir in enumerate(level_dirs, start=1):
            # Level k = blur(level k-1);  level 1 = blur(original)
            current = apply_gaussian_blur(current, sigma=BLUR_SIGMA)

            dst = level_dir / f"{stem}.png"
            # Skip if already written (resume support)
            if dst.exists():
                continue

            save_grayscale(current, str(dst))

        return (stem, True)
    except Exception as exc:
        return (stem, False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 5 sequential Gaussian blur levels for each source image."
    )
    parser.add_argument(
        "--data-root", default=str(PROJECT_ROOT / "datasets"),
        help="Root datasets directory (default: ./datasets)"
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
        help="Number of parallel worker processes (default: half CPU cores)"
    )
    parser.add_argument(
        "--sigma", type=float, default=BLUR_SIGMA,
        help=f"Gaussian blur sigma (default: {BLUR_SIGMA})"
    )
    parser.add_argument(
        "--n-levels", type=int, default=N_BLUR_LEVELS,
        help=f"Number of blur levels (default: {N_BLUR_LEVELS})"
    )
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    src_dir    = data_root / "synthetic" / "raw_imagenet"
    blurred_root = data_root / "synthetic" / "blurred"

    # Validate source directory
    src_images = sorted(src_dir.glob("*.png"))
    if not src_images:
        print(f"[ERROR] No PNG images found in {src_dir}")
        print("  Run script 02 first:  python3 scripts/02_download_imagenet_proxy.py")
        sys.exit(1)

    print(f"[INFO] Found {len(src_images)} source images in {src_dir}")
    print(f"[INFO] Blur sigma = {args.sigma}  |  Levels = {args.n_levels}")
    print(f"[INFO] Workers = {args.workers}")

    # Create output level directories
    level_dirs: list[Path] = []
    for lvl in range(1, args.n_levels + 1):
        lvl_dir = blurred_root / f"blur_level_{lvl}"
        lvl_dir.mkdir(parents=True, exist_ok=True)
        level_dirs.append(lvl_dir)

    # Check how many are already done (resume support)
    done_count = sum(1 for p in src_images
                     if (level_dirs[-1] / f"{p.stem}.png").exists())
    if done_count == len(src_images):
        print(f"[OK] All {done_count} images already blurred — skipping.")
        print("Next step:  python3 scripts/04_extract_patches.py")
        return

    todo = [(p, level_dirs) for p in src_images
            if not (level_dirs[-1] / f"{p.stem}.png").exists()]
    print(f"[INFO] Processing {len(todo)} images ({done_count} already done).")

    success = 0
    failed  = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one_image, t): t[0] for t in todo}
        with tqdm(total=len(todo), desc="Blurring images") as pbar:
            for fut in as_completed(futures):
                stem, ok = fut.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix({"ok": success, "fail": failed})

    print(f"\n[DONE] Blur generation complete.")
    print(f"  Success : {success}")
    print(f"  Failed  : {failed}")
    print(f"  Output  : {blurred_root}")
    print()
    print("Blur level summary:")
    for lvl, lvl_dir in enumerate(level_dirs, 1):
        n = len(list(lvl_dir.glob("*.png")))
        print(f"  Level {lvl}: {n} images")

    print("\nNext step:  python3 scripts/04_extract_patches.py")


if __name__ == "__main__":
    main()
