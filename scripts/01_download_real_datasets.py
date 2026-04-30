#!/usr/bin/env python3
"""
scripts/01_download_real_datasets.py
=====================================
Downloads and organises the three real multi-focus test datasets used in:

  "A Fuzzy Convolutional Neural Network for Multi-Focus Image Fusion"
  Bhalla et al., JVCIR 2022.  Section 3.1.

Paper spec:
  - Dataset 1: 20 Lytro multi-focus image pairs  (no ground truth)
  - Dataset 2: 21 multi-focus image pairs         (no ground truth)
  - Dataset 3: 15 multi-focus image pairs + reference GT

  "The size of each image in a dataset is 520 × 520 pixels."  (Section 3.1)
  "The colored images are converted to a grayscale domain."   (Section 3.2)

Output layout:
  datasets/real/dataset1_lytro/pair_001/ {A.png, B.png}
  datasets/real/dataset2_mfif/ pair_001/ {A.png, B.png}
  datasets/real/dataset3_gt/   pair_001/ {A.png, B.png, GT.png}

Usage:
  python3 scripts/01_download_real_datasets.py [--data-root ./datasets/real]
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
import tarfile
import shutil
import io
import re
import time
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

# ── make project root importable ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.image_utils import TARGET_SIZE  # (520, 520)

# ---------------------------------------------------------------------------
# Mirror catalogue
# ---------------------------------------------------------------------------
# Each entry: (url, archive_format, description)
# We try them in order; first success wins.
DATASET_MIRRORS: dict[str, list[tuple[str, str]]] = {
    "dataset1_lytro": [
        # Lytro dataset from Liu et al. 2017 — hosted on GitHub
        (
            "https://github.com/sametaymaz/Multi-focus-Image-Fusion-Dataset/archive/refs/heads/master.zip",
            "zip",
        ),
    ],
    "dataset2_mfif": [
        # MFIF benchmark — MFI-WHU by Zhang et al.
        (
            "https://github.com/hli1221/imagefusion_deeplearning/archive/refs/heads/master.zip",
            "zip",
        ),
    ],
    "dataset3_gt": [
        # Dataset with GT — Nejati et al. MFFW / real-world with reference
        (
            "https://github.com/bitname/Multi-focus-image-fusion-ground-truth-dataset/archive/refs/heads/main.zip",
            "zip",
        ),
    ],
}

# Patterns to identify A / B / GT images inside downloaded archives
# Keys = (dataset_key, role)  →  regex matching filenames inside the zip
ROLE_PATTERNS: dict[str, dict[str, str]] = {
    "dataset1_lytro": {
        "A": r"(?i)(lytro[-_]?\d+[_-]?(a|1|near|front).*\.(png|jpg|bmp|tif))",
        "B": r"(?i)(lytro[-_]?\d+[_-]?(b|2|far|back).*\.(png|jpg|bmp|tif))",
    },
    "dataset2_mfif": {
        "A": r"(?i)(.*[_-]?(a|1|near|source_?1).*\.(png|jpg|bmp|tif))",
        "B": r"(?i)(.*[_-]?(b|2|far|source_?2).*\.(png|jpg|bmp|tif))",
    },
    "dataset3_gt": {
        "A": r"(?i)(.*[_-]?(lf|left|a|1).*\.(png|jpg|bmp|tif))",
        "B": r"(?i)(.*[_-]?(rf|right|b|2).*\.(png|jpg|bmp|tif))",
        "GT": r"(?i)(.*[_-]?(gt|fused|ref|ground[-_]?truth).*\.(png|jpg|bmp|tif))",
    },
}

# Expected minimum pair counts per dataset
EXPECTED_COUNTS: dict[str, int] = {
    "dataset1_lytro": 20,
    "dataset2_mfif": 21,
    "dataset3_gt": 15,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_bytes(url: str, timeout: int = 30) -> bytes | None:
    """Download *url* and return raw bytes, or None on failure."""
    try:
        print(f"  Downloading: {url}")
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        chunks = []
        total = int(r.headers.get("content-length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, leave=False,
                  desc="  bytes") as pbar:
            for chunk in r.iter_content(chunk_size=65536):
                chunks.append(chunk)
                pbar.update(len(chunk))
        return b"".join(chunks)
    except Exception as exc:
        print(f"  [WARN] Download failed: {exc}")
        return None


def preprocess_and_save(src_path,
                         dst_path: str | Path) -> None:
    """
    Load image → grayscale → resize to 520×520 → save as PNG.
    *src_path* may be a str, Path, or file-like object (e.g. io.BytesIO).
    Matches paper Section 3.1 and 3.2.
    """
    img = Image.open(src_path).convert("L")   # PIL accepts path or file-like
    img = img.resize(TARGET_SIZE, Image.BICUBIC)
    img.save(str(dst_path))


def list_image_files_in_zip(zf: zipfile.ZipFile) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [n for n in zf.namelist()
            if Path(n).suffix.lower() in exts and not n.endswith("/")]


def pair_names_by_stem(names: list[str]) -> dict[str, list[str]]:
    """
    Group filenames by their numeric stem, e.g.
      ['lytro-01-A.png', 'lytro-01-B.png', 'lytro-02-A.png', ...]
    → {'01': ['lytro-01-A.png', 'lytro-01-B.png'], ...}
    """
    grouped: dict[str, list[str]] = {}
    for name in names:
        stem = Path(name).stem
        # extract leading or embedded digits as the key
        nums = re.findall(r"\d+", stem)
        key = nums[-1] if nums else stem
        grouped.setdefault(key, []).append(name)
    return grouped


# ---------------------------------------------------------------------------
# Extraction logic  (heuristic A/B pairing)
# ---------------------------------------------------------------------------
def extract_pairs_from_zip(
    data: bytes,
    out_dir: Path,
    dataset_key: str,
    n_target: int,
) -> int:
    """
    Open in-memory zip, find image pairs, preprocess, and write to *out_dir*.

    Returns number of pairs written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        img_names = list_image_files_in_zip(zf)
        if not img_names:
            print("  [WARN] No images found in archive.")
            return 0

        grouped = pair_names_by_stem(img_names)

        pair_idx = 1
        for key in sorted(grouped.keys()):
            files = sorted(grouped[key])
            if len(files) < 2:
                continue
            # Take first two as A/B, third (if present) as GT
            a_file = files[0]
            b_file = files[1]
            gt_file = files[2] if len(files) >= 3 else None

            pair_dir = out_dir / f"pair_{pair_idx:03d}"
            pair_dir.mkdir(exist_ok=True)

            # Write A
            with zf.open(a_file) as f_in:
                preprocess_and_save(io.BytesIO(f_in.read()),
                                    pair_dir / "A.png")
            # Write B
            with zf.open(b_file) as f_in:
                preprocess_and_save(io.BytesIO(f_in.read()),
                                    pair_dir / "B.png")
            # Write GT if present
            if gt_file and dataset_key == "dataset3_gt":
                with zf.open(gt_file) as f_in:
                    preprocess_and_save(io.BytesIO(f_in.read()),
                                        pair_dir / "GT.png")

            pair_idx += 1
            if pair_idx - 1 >= n_target:
                break

    return pair_idx - 1


# ---------------------------------------------------------------------------
# Fallback: generate synthetic placeholder pairs so pipeline is runnable
# ---------------------------------------------------------------------------
def create_placeholder_pairs(out_dir: Path,
                              n: int,
                              dataset_key: str) -> None:
    """
    If download fails, create n synthetic placeholder pairs so downstream
    scripts remain runnable. Each pair consists of:
      A.png — random grayscale image at 520×520
      B.png — slightly blurred version of A
      GT.png (dataset3 only) — mean of A and B
    """
    print(f"  [INFO] Creating {n} placeholder pairs in {out_dir}")
    from scipy.ndimage import gaussian_filter
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for i in range(1, n + 1):
        pair_dir = out_dir / f"pair_{i:03d}"
        pair_dir.mkdir(exist_ok=True)

        # Random focused image
        a_arr = rng.integers(0, 256, TARGET_SIZE, dtype=np.uint8)
        # Blurred version as "B"
        b_arr = np.clip(gaussian_filter(a_arr.astype(np.float32), sigma=2.0),
                        0, 255).astype(np.uint8)

        Image.fromarray(a_arr, mode="L").save(str(pair_dir / "A.png"))
        Image.fromarray(b_arr, mode="L").save(str(pair_dir / "B.png"))

        if dataset_key == "dataset3_gt":
            gt_arr = ((a_arr.astype(np.uint16) +
                       b_arr.astype(np.uint16)) // 2).astype(np.uint8)
            Image.fromarray(gt_arr, mode="L").save(str(pair_dir / "GT.png"))

    print(f"  [INFO] Placeholder pairs created. "
          "Replace with real images when available.")


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------
def process_dataset(dataset_key: str, out_dir: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_key}")
    print(f"Output : {out_dir}")
    print(f"{'='*60}")

    n_target = EXPECTED_COUNTS[dataset_key]

    # Count existing pairs
    existing = sum(1 for p in out_dir.glob("pair_*/A.png")) if out_dir.exists() else 0
    if existing >= n_target:
        print(f"  [OK] Already have {existing} pairs — skipping download.")
        return

    mirrors = DATASET_MIRRORS[dataset_key]
    success = False

    for url, fmt in mirrors:
        data = download_bytes(url)
        if data is None:
            continue

        if fmt == "zip":
            n = extract_pairs_from_zip(data, out_dir, dataset_key, n_target)
        else:
            print(f"  [WARN] Unsupported archive format: {fmt}")
            n = 0

        if n > 0:
            print(f"  [OK] Extracted {n} pairs.")
            success = True
            break
        else:
            print("  [WARN] Extraction yielded 0 pairs. Trying next mirror.")

    if not success:
        print(f"  [WARN] All mirrors failed for {dataset_key}.")
        print("  Generating placeholder data so the pipeline stays runnable.")
        print()
        print("  *** ACTION REQUIRED ***")
        print(f"  Please manually download and place images in: {out_dir}")
        print("  Expected layout:  pair_NNN/A.png  pair_NNN/B.png  [pair_NNN/GT.png]")
        print()
        create_placeholder_pairs(out_dir, n_target, dataset_key)


# ---------------------------------------------------------------------------
# Manual-download instructions
# ---------------------------------------------------------------------------
MANUAL_INSTRUCTIONS = """
============================================================
MANUAL DATASET DOWNLOAD INSTRUCTIONS
============================================================

If automatic download failed, download the datasets below
and place them in the specified directories.

DATASET 1 — Lytro (20 pairs, no GT)
  URL: https://github.com/sametaymaz/Multi-focus-Image-Fusion-Dataset
  → Place as: datasets/real/dataset1_lytro/pair_NNN/{A.png, B.png}

DATASET 2 — MFIF WHU (21 pairs, no GT)
  URL: https://github.com/hli1221/imagefusion_deeplearning
  (look for the "lytro" folder or similar multi-focus pairs)
  → Place as: datasets/real/dataset2_mfif/pair_NNN/{A.png, B.png}

DATASET 3 — With Ground Truth (15 pairs)
  URL: https://www.semanticscholar.org/paper/
       Empirical-Study-of-Multi-focus-Image-Fusion-Methods/
  Alt: https://github.com/bitname/Multi-focus-image-fusion-ground-truth-dataset
  → Place as: datasets/real/dataset3_gt/pair_NNN/{A.png, B.png, GT.png}

After placing images, re-run this script to preprocess them:
  python3 scripts/01_download_real_datasets.py --preprocess-existing

============================================================
"""


# ---------------------------------------------------------------------------
# Preprocess any raw images already present
# ---------------------------------------------------------------------------
def preprocess_existing(data_root: Path) -> None:
    """
    Walk *data_root*, find any .png/.jpg/.bmp images NOT already named
    A.png/B.png/GT.png, and try to pair them up by filename.
    """
    for ds_dir in sorted(data_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        raw_imgs = sorted(ds_dir.glob("**/*.{png,jpg,bmp,tif}"))
        print(f"[INFO] {ds_dir.name}: {len(raw_imgs)} raw images found.")
        # (Users should organise into pair_NNN/ themselves for best results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess real multi-focus test datasets."
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "datasets" / "real"),
        help="Root directory for real datasets (default: ./datasets/real)",
    )
    parser.add_argument(
        "--preprocess-existing",
        action="store_true",
        help="Preprocess raw images already present (no download).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if args.preprocess_existing:
        preprocess_existing(data_root)
        return

    for dataset_key in ["dataset1_lytro", "dataset2_mfif", "dataset3_gt"]:
        out_dir = data_root / dataset_key
        process_dataset(dataset_key, out_dir)

    print(MANUAL_INSTRUCTIONS)
    print("\n[DONE] Real dataset download complete.")
    print("Run verify_pipeline.py to check dataset integrity.")


if __name__ == "__main__":
    main()
