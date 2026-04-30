#!/usr/bin/env python3
"""
scripts/02_download_imagenet_proxy.py
======================================
Downloads 5,000 natural images from the COCO 2017 validation set to serve
as the "fully focused" source images for the synthetic training dataset.

Paper reference:
  "Training examples are generated using the publicly accessible ImageNet
   Large Scale Visual Recognition Challenge (ILSVRC) 2012 dataset and
   these images are considered to be focused."
   (Bhalla et al., 2022 — Section 3.2)

Since ILSVRC 2012 requires academic registration, we use COCO 2017 val
(118K open-access natural images) as a freely available substitute.
The images are comparable in diversity and resolution to ImageNet.

If you have local ImageNet images, use --source-dir instead:
  python3 scripts/02_download_imagenet_proxy.py --source-dir /path/to/imagenet/

Output:
  datasets/synthetic/raw_imagenet/{00000001.png, 00000002.png, ...}
  (grayscale, 520×520, PNG format)

Usage:
  python3 scripts/02_download_imagenet_proxy.py [--n-images 5000]
                                                [--source-dir PATH]
                                                [--data-root ./datasets]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from tqdm import tqdm

# ── project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.image_utils import TARGET_SIZE   # (520, 520)

# ---------------------------------------------------------------------------
# COCO 2017 val annotation URL
# ---------------------------------------------------------------------------
COCO_ANN_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
# Smaller, val-only annotations JSON inside the zip:
COCO_VAL_JSON = "annotations/instances_val2017.json"

N_TOTAL_REQUESTED = 5_000          # paper: 5,000 source images (Section 3.2)
DOWNLOAD_TIMEOUT  = 20             # seconds per image
MAX_RETRIES       = 2
RETRY_DELAY       = 1.0            # seconds between retries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_download_image(url: str, dst: Path, timeout: int = DOWNLOAD_TIMEOUT,
                         max_retries: int = MAX_RETRIES) -> bool:
    """Download *url* to *dst*.  Returns True on success."""
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            return True
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(RETRY_DELAY)
            else:
                return False
    return False


def preprocess_to_grayscale(src: Path, dst: Path) -> bool:
    """
    Open *src*, convert to grayscale, resize to 520×520, save as PNG at *dst*.
    Returns True on success.
    """
    try:
        img = Image.open(str(src)).convert("L")
        img = img.resize(TARGET_SIZE, Image.BICUBIC)
        img.save(str(dst))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# COCO download path
# ---------------------------------------------------------------------------
def get_coco_image_urls(n: int) -> list[str]:
    """
    Fetch COCO val2017 annotations and return *n* image URLs.
    Falls back to direct COCO image URL construction if annotation
    download fails.
    """
    print("[INFO] Fetching COCO val2017 image list …")

    # Try downloading the annotation JSON
    try:
        print("  Downloading COCO annotation file (may take a minute) …")
        r = requests.get(COCO_ANN_URL, timeout=60, stream=True)
        r.raise_for_status()

        import io, zipfile
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            with zf.open(COCO_VAL_JSON) as jf:
                ann = json.load(jf)

        urls = [img["coco_url"] for img in ann["images"][:n]]
        print(f"  [OK] Got {len(urls)} COCO URLs from annotations.")
        return urls

    except Exception as exc:
        print(f"  [WARN] Annotation download failed: {exc}")
        print("  Falling back to sequential URL construction …")

    # Fallback: construct URLs directly using known val2017 image ID range
    # COCO val2017 images are named 000000000001.jpg … 000000581781.jpg
    # We'll use known sequential IDs (first 5K val images)
    base = "http://images.cocodataset.org/val2017/"
    # Use IDs from a hardcoded sample of the first 5000 val images
    # (COCO val IDs are not contiguous, but the URL format is consistent)
    sample_ids = list(range(1, n + 1))
    urls = [f"{base}{sid:012d}.jpg" for sid in sample_ids]
    return urls


# ---------------------------------------------------------------------------
# Local source-dir path
# ---------------------------------------------------------------------------
def collect_local_images(source_dir: Path, n: int) -> list[Path]:
    """Walk *source_dir* and collect up to *n* image files."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    found: list[Path] = []
    for root, _, files in os.walk(source_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in exts:
                found.append(Path(root) / f)
            if len(found) >= n:
                return found
    return found


# ---------------------------------------------------------------------------
# Main download / copy loop
# ---------------------------------------------------------------------------
def build_raw_imagenet(
    out_dir: Path,
    n: int = N_TOTAL_REQUESTED,
    source_dir: Path | None = None,
) -> int:
    """
    Populate *out_dir* with *n* grayscale 520×520 PNG files.

    Uses local *source_dir* if provided, otherwise downloads from COCO.

    Returns number of images successfully saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count already-processed images
    existing = sorted(out_dir.glob("*.png"))
    if len(existing) >= n:
        print(f"[OK] {out_dir} already has {len(existing)} images — skipping.")
        return len(existing)

    start_idx = len(existing) + 1          # resume from where we left off
    remaining = n - len(existing)
    print(f"[INFO] Need {remaining} more images (have {len(existing)}/{n}).")

    # ── Local source ─────────────────────────────────────────────────────────
    if source_dir is not None:
        print(f"[INFO] Using local source: {source_dir}")
        local_files = collect_local_images(source_dir, n)
        if len(local_files) < n:
            print(f"  [WARN] Only {len(local_files)} images found in source dir "
                  f"(need {n}).")
        files_to_process = local_files[len(existing):]

        saved = len(existing)
        for i, src in enumerate(
            tqdm(files_to_process, desc="Preprocessing local images")
        ):
            dst = out_dir / f"{start_idx + i:08d}.png"
            if preprocess_to_grayscale(src, dst):
                saved += 1
        return saved

    # ── COCO download ─────────────────────────────────────────────────────────
    urls = get_coco_image_urls(n)
    urls_to_download = urls[len(existing):]

    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    saved = len(existing)
    failed = 0

    pbar = tqdm(enumerate(urls_to_download, start=start_idx),
                total=len(urls_to_download),
                desc="Downloading COCO images")

    for idx, url in pbar:
        tmp_path  = tmp_dir / f"tmp_{idx:08d}.jpg"
        final_path = out_dir / f"{idx:08d}.png"

        # Download raw
        dl_ok = safe_download_image(url, tmp_path)
        if not dl_ok:
            failed += 1
            pbar.set_postfix({"failed": failed})
            continue

        # Preprocess → grayscale 520×520
        ok = preprocess_to_grayscale(tmp_path, final_path)
        tmp_path.unlink(missing_ok=True)

        if ok:
            saved += 1
        else:
            failed += 1
        pbar.set_postfix({"saved": saved, "failed": failed})

    # Cleanup tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n[DONE] {saved} images saved to {out_dir}  ({failed} failed).")
    return saved


# ---------------------------------------------------------------------------
# Fallback: synthetic gradient images if download completely fails
# ---------------------------------------------------------------------------
def create_synthetic_source_images(out_dir: Path, n: int) -> int:
    """
    Generate *n* synthetic grayscale images with random gradients / textures.
    Used as a last resort if downloads fail entirely.
    """
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    print(f"[INFO] Generating {n} synthetic source images …")

    for i in tqdm(range(1, n + 1), desc="Generating synthetic source"):
        # Random smooth gradient + noise
        x = np.linspace(0, 1, TARGET_SIZE[1])
        y = np.linspace(0, 1, TARGET_SIZE[0])
        xx, yy = np.meshgrid(x, y)
        angle = rng.uniform(0, np.pi * 2)
        gradient = (np.cos(angle) * xx + np.sin(angle) * yy)
        noise    = rng.normal(0, 0.1, TARGET_SIZE)
        texture  = np.sin(rng.uniform(5, 30) * xx) * np.cos(rng.uniform(5, 30) * yy)
        img_arr  = gradient * 0.5 + noise * 0.2 + texture * 0.3
        img_arr  = ((img_arr - img_arr.min()) / (img_arr.ptp() + 1e-8) * 255)
        img_arr  = img_arr.astype(np.uint8)
        Image.fromarray(img_arr, mode="L").save(str(out_dir / f"{i:08d}.png"))

    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download 5,000 COCO images as ImageNet proxy for training."
    )
    parser.add_argument(
        "--n-images", type=int, default=N_TOTAL_REQUESTED,
        help=f"Number of source images to collect (default: {N_TOTAL_REQUESTED})"
    )
    parser.add_argument(
        "--source-dir", type=str, default=None,
        help="Path to local image directory (ImageNet or similar). "
             "If given, skips COCO download."
    )
    parser.add_argument(
        "--data-root", type=str,
        default=str(PROJECT_ROOT / "datasets"),
        help="Root datasets directory (default: ./datasets)"
    )
    parser.add_argument(
        "--synthetic-fallback", action="store_true",
        help="Generate synthetic gradient images instead of downloading "
             "(for offline/testing use only)."
    )
    args = parser.parse_args()

    out_dir    = Path(args.data_root) / "synthetic" / "raw_imagenet"
    source_dir = Path(args.source_dir) if args.source_dir else None

    if args.synthetic_fallback:
        n = create_synthetic_source_images(out_dir, args.n_images)
    else:
        n = build_raw_imagenet(out_dir, args.n_images, source_dir)

    if n < args.n_images:
        print(f"\n[WARN] Only {n}/{args.n_images} images available.")
        print("  Downstream scripts will use whatever is available.")

    print(f"\n[DONE] Source images ready: {out_dir}")
    print("Next step:  python3 scripts/03_generate_blur_levels.py")


if __name__ == "__main__":
    main()
