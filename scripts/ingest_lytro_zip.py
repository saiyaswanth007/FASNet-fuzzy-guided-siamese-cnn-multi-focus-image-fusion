#!/usr/bin/env python3
"""
scripts/ingest_lytro_zip.py
============================
Ingests the downloaded Multi-focus-Image-Fusion-Dataset-master.zip
(which contains MF_Dataset.rar inside) into the pipeline's Dataset 1
directory structure.

The dataset contains multi-image sequences (3-4 images per scene).
The paper uses PAIRS, so we take image[0] as A and image[1] as B
for each scene group.

Usage:
  python3 scripts/ingest_lytro_zip.py \\
      --zip datasets/Multi-focus-Image-Fusion-Dataset-master.zip \\
      --out datasets/real/dataset1_lytro
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.image_utils import TARGET_SIZE   # (520, 520)

# Images to skip (not part of multi-focus sequences)
SKIP_STEMS = {
    "5fused1", "5fused2", "5fused3", "5fused",   # fused outputs
    "truckorj", "orjcars",                         # originals, not sequences
    "OscarSelfie1pg", "OscarSelfie2orj", "OscarSelfie2pg",  # mismatched
    "bookC",                                        # single image (no number)
    "cemyılmaz", "cemyılmaz1",                      # encoding issues
    "source02C",                                    # standalone
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def scene_key_and_index(stem: str) -> tuple[str, int] | None:
    """
    Parse a filename stem into (scene_key, focus_index).

    Examples:
      'a1'            → ('a',          1)
      'boats-l2'      → ('boats-l',    2)
      'source02C1'    → ('source02C',  1)
      'clock31'       → ('clock3',     1)   ← trailing digit = index
      'Streetb1'      → ('Streetb',    1)
      'livingroom3'   → ('livingroom', 3)
    """
    if stem in SKIP_STEMS:
        return None
    # Match trailing integer as focus index
    m = re.match(r'^(.+?)(\d+)$', stem)
    if not m:
        return None
    key, idx = m.group(1), int(m.group(2))
    return (key, idx)


def group_by_scene(names: list[str]) -> dict[str, list[tuple[int, str]]]:
    """
    Group image filenames by scene key.
    Returns { scene_key: [(focus_idx, filename), ...] } sorted by focus_idx.
    """
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for name in names:
        stem = Path(name).stem
        ext  = Path(name).suffix.lower()
        if ext not in IMAGE_EXTS:
            continue
        parsed = scene_key_and_index(stem)
        if parsed is None:
            continue
        key, idx = parsed
        groups[key].append((idx, name))

    # Sort each group by index
    return {k: sorted(v) for k, v in groups.items()}


def preprocess(data: bytes, dst: Path) -> bool:
    """Open image bytes, convert to grayscale 520×520, save to dst."""
    try:
        img = Image.open(io.BytesIO(data)).convert("L")
        img = img.resize(TARGET_SIZE, Image.BICUBIC)
        img.save(str(dst))
        return True
    except Exception as exc:
        print(f"  [WARN] Could not process → {dst.name}: {exc}")
        return False


def extract_rar(rar_path: Path, out_dir: Path) -> Path | None:
    """
    Extract RAR using `unar` (handles all RAR compression variants).
    Returns the extracted subfolder path, or None on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["unar", "-o", str(out_dir), "-f", str(rar_path)],
        capture_output=True, text=True
    )
    # unar exits 0 even with warnings; check that files were actually written
    candidates = [d for d in out_dir.iterdir() if d.is_dir()]
    if candidates:
        return candidates[0]
    # Fallback: files extracted directly (no subdirectory)
    files = list(out_dir.iterdir())
    if files:
        return out_dir
    print(f"[ERROR] unar produced no output.\nstdout: {result.stdout}\nstderr: {result.stderr}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Lytro MF Dataset zip into pipeline Dataset 1."
    )
    parser.add_argument(
        "--zip",
        default=str(PROJECT_ROOT / "datasets" / "Multi-focus-Image-Fusion-Dataset-master.zip"),
        help="Path to the downloaded zip file"
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "datasets" / "real" / "dataset1_lytro"),
        help="Output directory for Dataset 1 pairs"
    )
    parser.add_argument(
        "--n-pairs", type=int, default=20,
        help="Number of scene pairs to extract (paper: 20, default: 20)"
    )
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir  = Path(args.out)

    if not zip_path.exists():
        print(f"[ERROR] Zip not found: {zip_path}")
        sys.exit(1)

    # ── Step 1: Extract MF_Dataset.rar from the outer zip ────────────────────
    print(f"[1] Opening outer zip: {zip_path.name}")
    with zipfile.ZipFile(str(zip_path)) as zf:
        rar_member = next(
            (n for n in zf.namelist() if n.endswith(".rar")), None
        )
        if rar_member is None:
            print("[ERROR] No .rar file found inside the zip.")
            sys.exit(1)
        print(f"    Found RAR inside zip: {rar_member}")
        rar_bytes = zf.read(rar_member)

    tmp_rar = zip_path.parent / "_tmp_mf.rar"
    tmp_dir = zip_path.parent / "_tmp_mf_extracted"

    with open(str(tmp_rar), "wb") as f:
        f.write(rar_bytes)
    print(f"    RAR size: {len(rar_bytes) / 1e6:.1f} MB")

    # ── Step 2: Extract the RAR ───────────────────────────────────────────────
    print(f"[2] Extracting RAR → {tmp_dir} …")
    mf_dir = extract_rar(tmp_rar, tmp_dir)
    if mf_dir is None:
        print("[ERROR] RAR extraction failed.")
        tmp_rar.unlink(missing_ok=True)
        sys.exit(1)

    # Collect all image files
    all_images = []
    for root, _, files in os.walk(str(mf_dir)):
        for f in files:
            full = Path(root) / f
            if full.suffix.lower() in IMAGE_EXTS:
                all_images.append(full)

    print(f"    Found {len(all_images)} image files in RAR.")

    # ── Step 3: Group into scenes ─────────────────────────────────────────────
    print("[3] Grouping images by scene …")
    names_only = [str(p.relative_to(tmp_dir)) for p in all_images]
    path_map   = {str(p.relative_to(tmp_dir)): p for p in all_images}

    groups = group_by_scene([p.name for p in all_images])
    # Rebuild with full paths
    full_groups: dict[str, list[tuple[int, Path]]] = {}
    for key, items in groups.items():
        full_paths = []
        for idx, fname in items:
            # Find matching full path
            match = next((p for p in all_images if p.name == fname), None)
            if match:
                full_paths.append((idx, match))
        if len(full_paths) >= 2:
            full_groups[key] = full_paths

    print(f"    {len(full_groups)} valid scenes with ≥2 images found:")
    for k in sorted(full_groups):
        imgs = [Path(p).name for _, p in full_groups[k]]
        print(f"      {k}: {imgs}")

    # ── Step 4: Clean old placeholder pairs ──────────────────────────────────
    print(f"\n[4] Clearing existing content in {out_dir} …")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # ── Step 5: Write pair_NNN/{A.png, B.png} ────────────────────────────────
    print(f"[5] Writing up to {args.n_pairs} pairs …")
    pair_idx = 1
    written  = 0

    for scene_key in sorted(full_groups.keys()):
        if written >= args.n_pairs:
            break
        items = full_groups[scene_key]
        if len(items) < 2:
            continue

        # A = lowest index (most-focused foreground or standard first view)
        # B = second-lowest index (different focus plane)
        _, path_a = items[0]
        _, path_b = items[1]

        pair_dir = out_dir / f"pair_{pair_idx:03d}"
        pair_dir.mkdir()

        ok_a = preprocess(path_a.read_bytes(), pair_dir / "A.png")
        ok_b = preprocess(path_b.read_bytes(), pair_dir / "B.png")

        if ok_a and ok_b:
            print(f"  pair_{pair_idx:03d}: {path_a.name} ↔ {path_b.name}")
            pair_idx += 1
            written  += 1
        else:
            shutil.rmtree(str(pair_dir))

    # ── Step 6: Cleanup temp files ────────────────────────────────────────────
    print("\n[6] Cleaning up temporary files …")
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
    tmp_rar.unlink(missing_ok=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    actual = len(list(out_dir.glob("pair_*/A.png")))
    print(f"\n{'='*55}")
    print(f"DONE  — {actual} pairs written to {out_dir}")
    if actual < args.n_pairs:
        print(f"[NOTE] Only {actual}/{args.n_pairs} pairs available in this dataset.")
        print("       The paper uses 20 Lytro pairs; this dataset has multi-view")
        print("       scenes. All valid A/B pairs have been extracted.")
    print(f"{'='*55}")
    print("\nNext: python3 verify_pipeline.py --min-source-count 100")


if __name__ == "__main__":
    main()
