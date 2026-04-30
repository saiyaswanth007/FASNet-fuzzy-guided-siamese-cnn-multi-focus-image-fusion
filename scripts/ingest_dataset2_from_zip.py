#!/usr/bin/env python3
"""
scripts/ingest_dataset2_from_zip.py
=====================================
Builds Dataset 2 (21 pairs) from the same zip already used for Dataset 1.

Dataset 1 used scenes 1-20 (first 20 alphabetical scenes, focus levels 1&2).
Dataset 2 uses the remaining 15 scenes (21-35) PLUS 6 extra pairs taken
from existing scenes using focus level combinations 1vs3 and 1vs4,
giving 21 valid multi-focus pairs total.

All pairs have real SSIM > 0.7 because they come from the same scene.

Usage:
  python3 scripts/ingest_dataset2_from_zip.py
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
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.image_utils import TARGET_SIZE

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

SKIP_STEMS = {
    "5fused1", "5fused2", "5fused3", "5fused",
    "truckorj", "orjcars",
    "OscarSelfie1pg", "OscarSelfie2orj", "OscarSelfie2pg",
    "bookC", "cemyılmaz", "cemyılmaz1", "source02C",
}

# Scenes already used for Dataset 1 (first 20 alphabetical)
DATASET1_SCENES = {
    "Streetb", "a", "astronot", "boats-l", "bookC",
    "c", "cameramanC", "clock", "d", "flowerC",
    "g", "h", "hoed_C", "house", "j",
    "jetplane", "k", "lake", "livingroom", "mandril_gray",
}


def scene_key_and_index(stem: str) -> tuple[str, int] | None:
    if stem in SKIP_STEMS:
        return None
    m = re.match(r'^(.+?)(\d+)$', stem)
    if not m:
        return None
    return (m.group(1), int(m.group(2)))


def preprocess(data: bytes, dst: Path) -> bool:
    try:
        img = Image.open(io.BytesIO(data)).convert("L")
        img = img.resize(TARGET_SIZE, Image.BICUBIC)
        img.save(str(dst))
        return True
    except Exception as exc:
        print(f"  [WARN] {dst.name}: {exc}")
        return False


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = a.mean(), b.mean()
    s1, s2 = a.std(), b.std()
    cov = ((a - mu1) * (b - mu2)).mean()
    return float(((2*mu1*mu2+C1)*(2*cov+C2)) / ((mu1**2+mu2**2+C1)*(s1**2+s2**2+C2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        default=str(PROJECT_ROOT / "datasets" / "Multi-focus-Image-Fusion-Dataset-master.zip"),
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "datasets" / "real" / "dataset2_mfif"),
    )
    parser.add_argument("--n-pairs", type=int, default=21)
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir  = Path(args.out)

    # ── Extract RAR from zip ──────────────────────────────────────────────────
    print(f"[1] Extracting RAR from {zip_path.name} …")
    with zipfile.ZipFile(str(zip_path)) as zf:
        rar_member = next(n for n in zf.namelist() if n.endswith(".rar"))
        rar_bytes  = zf.read(rar_member)

    tmp_rar = zip_path.parent / "_tmp_ds2.rar"
    tmp_dir = zip_path.parent / "_tmp_ds2_extracted"
    tmp_rar.write_bytes(rar_bytes)

    # ── Extract with unar ─────────────────────────────────────────────────────
    print("[2] Extracting RAR with unar …")
    subprocess.run(["unar", "-o", str(tmp_dir), "-f", str(tmp_rar)],
                   capture_output=True)

    all_images = list(Path(tmp_dir).rglob("*"))
    all_images = [p for p in all_images
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    print(f"    {len(all_images)} images found.")

    # ── Group by scene ────────────────────────────────────────────────────────
    print("[3] Grouping into scenes …")
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in all_images:
        parsed = scene_key_and_index(p.stem)
        if parsed:
            key, idx = parsed
            groups[key].append((idx, p))

    for k in groups:
        groups[k].sort()

    # ── Split: scenes NOT in Dataset 1 ───────────────────────────────────────
    remaining_scenes = {k: v for k, v in groups.items()
                        if k not in DATASET1_SCENES and len(v) >= 2}
    ds1_scenes_with_extra = {k: v for k, v in groups.items()
                              if k in DATASET1_SCENES and len(v) >= 3}

    print(f"\n    Scenes NOT in Dataset 1: {len(remaining_scenes)}")
    for k, v in sorted(remaining_scenes.items()):
        print(f"      {k}: {[i for i, _ in v]}")

    print(f"\n    Dataset1 scenes with extra levels (1vs3/1vs4): {len(ds1_scenes_with_extra)}")

    # ── Build pair list ───────────────────────────────────────────────────────
    # Priority 1: remaining scenes (take focus 1 vs 2)
    pair_candidates: list[tuple[Path, Path, str]] = []
    for key, items in sorted(remaining_scenes.items()):
        idxs = [idx for idx, _ in items]
        paths = {idx: p for idx, p in items}
        min_idx = min(idxs)
        second_idx = sorted(idxs)[1]
        pair_candidates.append((paths[min_idx], paths[second_idx],
                                 f"{key}{min_idx} ↔ {key}{second_idx}"))

    # Priority 2: extra level pairs from Dataset1 scenes (1vs3 or 1vs4)
    for key, items in sorted(ds1_scenes_with_extra.items()):
        idxs = [idx for idx, _ in items]
        paths = {idx: p for idx, p in items}
        min_idx = min(idxs)
        # Take 1vs3 (skip the 1vs2 already used in Dataset1)
        extra_idxs = sorted(i for i in idxs if i != min_idx)[1:]  # skip second
        if extra_idxs:
            third_idx = extra_idxs[0]
            pair_candidates.append((paths[min_idx], paths[third_idx],
                                     f"{key}{min_idx} ↔ {key}{third_idx} [extra]"))
        if len(pair_candidates) >= args.n_pairs:
            break

    pair_candidates = pair_candidates[:args.n_pairs]

    # ── Write pairs ───────────────────────────────────────────────────────────
    print(f"\n[4] Clearing {out_dir} …")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"[5] Writing {len(pair_candidates)} pairs …")
    written = 0
    ssim_vals = []

    for i, (path_a, path_b, label) in enumerate(pair_candidates, 1):
        pair_dir = out_dir / f"pair_{i:03d}"
        pair_dir.mkdir()

        ok_a = preprocess(path_a.read_bytes(), pair_dir / "A.png")
        ok_b = preprocess(path_b.read_bytes(), pair_dir / "B.png")

        if ok_a and ok_b:
            # Verify SSIM
            arr_a = np.array(Image.open(pair_dir / "A.png"), dtype=float)
            arr_b = np.array(Image.open(pair_dir / "B.png"), dtype=float)
            s = ssim(arr_a, arr_b)
            ssim_vals.append(s)
            print(f"  pair_{i:03d}: {label}  SSIM={s:.3f}")
            written += 1
        else:
            shutil.rmtree(str(pair_dir))

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\n[6] Cleaning up …")
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
    tmp_rar.unlink(missing_ok=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_ssim = sum(ssim_vals) / len(ssim_vals) if ssim_vals else 0
    print(f"\n{'='*55}")
    print(f"DONE  — {written} pairs written to {out_dir}")
    print(f"Average SSIM between A and B: {avg_ssim:.3f}")
    print(f"(real multi-focus pairs should be > 0.7)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
