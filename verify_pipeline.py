#!/usr/bin/env python3
"""
verify_pipeline.py
===================
End-to-end sanity check for the FCNN-MFIF dataset pipeline.

Validates:
  1. Real datasets: correct pair counts, image sizes, grayscale
  2. Synthetic source images: count and size
  3. Blur levels: 5 levels present, SSIM decreasing 1→5
  4. Patch files: shape (N, 16, 16), dtype float32
  5. Training dataset NPZ: shapes, label distribution, value ranges

Generates:
  pipeline_verification_report.txt  — text summary
  pipeline_verification_plots.png   — visual inspection grid

Usage:
  python3 verify_pipeline.py [--data-root ./datasets]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.image_utils import TARGET_SIZE, PATCH_SIZE, N_BLUR_LEVELS

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# SSIM helper (no scikit-image required)
# ---------------------------------------------------------------------------
def ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute a simplified SSIM between two grayscale arrays [0,255]."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = img1.mean(), img2.mean()
    s1, s2 = img1.std(), img2.std()
    cov = ((img1 - mu1) * (img2 - mu2)).mean()
    num = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)
    return float(num / (den + 1e-8))


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------
def check_real_datasets(data_root: Path) -> dict[str, dict]:
    """Validate Dataset 1, 2, 3."""
    expected = {
        "dataset1_lytro": {"n_pairs": 20, "has_gt": False},
        "dataset2_mfif":  {"n_pairs": 21, "has_gt": False},
    }
    results = {}
    real_root = data_root / "real"

    for ds_key, spec in expected.items():
        ds_dir = real_root / ds_key
        rec: dict = {"exists": ds_dir.exists()}
        if not ds_dir.exists():
            results[ds_key] = rec
            continue

        pair_dirs = sorted(ds_dir.glob("pair_*/"))
        a_images = [d / "A.png" for d in pair_dirs if (d / "A.png").exists()]
        b_images = [d / "B.png" for d in pair_dirs if (d / "B.png").exists()]
        gt_images = [d / "GT.png" for d in pair_dirs if (d / "GT.png").exists()]

        rec["n_pairs"]        = len(a_images)
        rec["expected_pairs"] = spec["n_pairs"]
        rec["pairs_ok"]       = len(a_images) >= spec["n_pairs"]

        # Check first image size and grayscale
        size_checks = []
        gray_checks = []
        for img_path in a_images[:3]:
            try:
                img = Image.open(str(img_path))
                size_checks.append(img.size == TARGET_SIZE)
                gray_checks.append(img.mode == "L")
            except Exception:
                size_checks.append(False)
        rec["size_ok"] = all(size_checks) if size_checks else False
        rec["gray_ok"] = all(gray_checks) if gray_checks else False

        if spec["has_gt"]:
            rec["gt_count"] = len(gt_images)
            rec["gt_ok"]    = len(gt_images) >= spec["n_pairs"]

        results[ds_key] = rec

    return results


def check_synthetic_source(data_root: Path) -> dict:
    """Validate raw_imagenet directory."""
    src_dir = data_root / "synthetic" / "raw_imagenet"
    if not src_dir.exists():
        return {"exists": False}

    imgs = sorted(src_dir.glob("*.png"))
    rec: dict = {"exists": True, "count": len(imgs), "target": 5000}

    if imgs:
        try:
            sample = Image.open(str(imgs[0]))
            rec["size"] = sample.size
            rec["mode"] = sample.mode
            rec["size_ok"] = sample.size == TARGET_SIZE
            rec["gray_ok"] = sample.mode == "L"
        except Exception:
            rec["size_ok"] = False

    return rec


def check_blur_levels(data_root: Path) -> dict:
    """Validate blur levels and check SSIM decreases."""
    blurred_root = data_root / "synthetic" / "blurred"
    rec: dict = {"levels_found": [], "ssim_decreasing": None}

    if not blurred_root.exists():
        return rec

    # Collect level directories
    levels_present = []
    for lvl in range(1, N_BLUR_LEVELS + 1):
        lvl_dir = blurred_root / f"blur_level_{lvl}"
        if lvl_dir.exists() and any(lvl_dir.glob("*.png")):
            levels_present.append(lvl)
    rec["levels_found"] = levels_present

    # SSIM check: pick one image and compare across levels
    src_dir = data_root / "synthetic" / "raw_imagenet"
    src_imgs = sorted(src_dir.glob("*.png"))[:1] if src_dir.exists() else []

    if src_imgs and len(levels_present) >= 2:
        stem  = src_imgs[0].stem
        orig  = np.array(Image.open(str(src_imgs[0])).convert("L"),
                         dtype=np.float32)
        ssims = []
        prev  = orig
        for lvl in levels_present:
            lvl_path = blurred_root / f"blur_level_{lvl}" / f"{stem}.png"
            if not lvl_path.exists():
                continue
            lvl_arr = np.array(Image.open(str(lvl_path)).convert("L"),
                               dtype=np.float32)
            ssims.append(ssim_simple(orig, lvl_arr))

        rec["ssim_values"] = ssims
        rec["ssim_decreasing"] = all(
            ssims[i] > ssims[i + 1] for i in range(len(ssims) - 1)
        )

    return rec


def check_patches(data_root: Path) -> dict:
    """Validate extracted patch .npy files."""
    patches_root = data_root / "synthetic" / "patches"
    rec: dict = {}

    focused_dir = patches_root / "focused"
    if focused_dir.exists():
        npy_files = sorted(focused_dir.glob("*.npy"))
        rec["focused_files"] = len(npy_files)
        if npy_files:
            arr = np.load(str(npy_files[0]))
            rec["focused_shape"] = arr.shape
            rec["focused_dtype"] = str(arr.dtype)
            rec["patch_size_ok"] = arr.shape[1:] == (PATCH_SIZE, PATCH_SIZE)

    for lvl in range(1, N_BLUR_LEVELS + 1):
        lvl_dir = patches_root / "blurred" / f"blur_level_{lvl}"
        if lvl_dir.exists():
            n = len(list(lvl_dir.glob("*.npy")))
            rec[f"blur_level_{lvl}_files"] = n

    return rec


def check_training_dataset(data_root: Path) -> dict:
    """Validate the final training NPZ."""
    npz_path = data_root / "synthetic" / "training_dataset.npz"
    rec: dict = {"exists": npz_path.exists()}
    if not npz_path.exists():
        return rec

    try:
        data = np.load(str(npz_path))
        A      = data["patches_A"]
        B      = data["patches_B"]
        labels = data["labels"]

        rec["patches_A_shape"] = A.shape
        rec["patches_B_shape"] = B.shape
        rec["labels_shape"]    = labels.shape
        rec["total_samples"]   = len(labels)
        rec["n_positive"]      = int((labels == 1).sum())
        rec["n_negative"]      = int((labels == 0).sum())
        rec["A_min"]           = float(A.min())
        rec["A_max"]           = float(A.max())
        rec["patch_shape_ok"]  = (A.shape[1:] == (PATCH_SIZE, PATCH_SIZE) and
                                  B.shape[1:] == (PATCH_SIZE, PATCH_SIZE))
        rec["balance_ok"]      = abs(rec["n_positive"] - rec["n_negative"]) <= 100
        rec["labels_valid"]    = set(np.unique(labels)).issubset({0, 1})
    except Exception as exc:
        rec["error"] = str(exc)

    return rec


# ---------------------------------------------------------------------------
# Visual plots
# ---------------------------------------------------------------------------
def make_plots(data_root: Path, out_path: Path) -> None:
    """Generate a visual verification grid saved to *out_path*."""
    if not HAS_MATPLOTLIB:
        return

    fig = plt.figure(figsize=(20, 14), facecolor="#1a1a2e")

    # Colour theme
    TEXT_COLOR = "#e0e0e0"
    ACCENT     = "#00d4ff"
    plt.rcParams.update({
        "text.color":      TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color":     TEXT_COLOR,
        "ytick.color":     TEXT_COLOR,
        "axes.titlecolor": ACCENT,
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":  "#16213e",
        "axes.edgecolor":  "#0f3460",
    })

    gs = gridspec.GridSpec(3, 7, figure=fig,
                           hspace=0.5, wspace=0.3,
                           left=0.04, right=0.98,
                           top=0.92, bottom=0.05)

    fig.suptitle("FCNN-MFIF Dataset Pipeline — Verification Report",
                 fontsize=16, color=ACCENT, fontweight="bold", y=0.97)

    # ── Row 0: Blur level progression ────────────────────────────────────────
    src_dir      = data_root / "synthetic" / "raw_imagenet"
    blurred_root = data_root / "synthetic" / "blurred"
    src_imgs     = sorted(src_dir.glob("*.png")) if src_dir.exists() else []

    titles_row0 = ["Original"] + [f"Blur L{i}" for i in range(1, 6)]
    for col, title in enumerate(titles_row0):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(title, fontsize=9)
        ax.axis("off")

        if col == 0 and src_imgs:
            try:
                img = np.array(Image.open(str(src_imgs[0])).convert("L"))
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            except Exception:
                pass
        elif col > 0 and src_imgs:
            lvl_path = (blurred_root / f"blur_level_{col}" /
                        f"{src_imgs[0].stem}.png")
            if lvl_path.exists():
                try:
                    img = np.array(Image.open(str(lvl_path)).convert("L"))
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                except Exception:
                    pass

    # ── Row 1: Real dataset samples ───────────────────────────────────────────
    real_root = data_root / "real"
    ds_samples = [
        ("DS1 Lytro", real_root / "dataset1_lytro"),
        ("DS2 MFIF",  real_root / "dataset2_mfif"),
    ]
    col = 0
    for ds_label, ds_dir in ds_samples:
        pairs = sorted(ds_dir.glob("pair_*/")) if ds_dir.exists() else []
        for role in ("A", "B", "GT"):
            ax = fig.add_subplot(gs[1, col])
            ax.set_title(f"{ds_label}\n{role}", fontsize=8)
            ax.axis("off")
            if pairs:
                img_path = pairs[0] / f"{role}.png"
                if img_path.exists():
                    try:
                        img = np.array(Image.open(str(img_path)).convert("L"))
                        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                    except Exception:
                        pass
            col += 1
            if col >= 7:
                break
        if col >= 7:
            break

    # ── Row 2: Sample patch pairs from training dataset ───────────────────────
    npz_path = data_root / "synthetic" / "training_dataset.npz"
    if npz_path.exists():
        try:
            data   = np.load(str(npz_path))
            A_all  = data["patches_A"]
            B_all  = data["patches_B"]
            labels = data["labels"]

            pos_idx = np.where(labels == 1)[0][:4]
            neg_idx = np.where(labels == 0)[0][:3]
            sample_idx = list(pos_idx) + list(neg_idx)

            for col, idx in enumerate(sample_idx[:7]):
                ax = fig.add_subplot(gs[2, col])
                lbl = labels[idx]
                ax.set_title(
                    f"A={'focused' if lbl==1 else 'blurred'}\nlabel={lbl}",
                    fontsize=7
                )
                ax.axis("off")
                # Show A and B side-by-side in one axes using composite
                pair_img = np.hstack([A_all[idx], B_all[idx]])
                ax.imshow(pair_img, cmap="gray", vmin=0, vmax=255,
                          interpolation="nearest", aspect="equal")
        except Exception:
            pass

    fig.text(0.5, 0.01,
             "Row 0: Gaussian blur progression (σ=0.5, 5 levels sequential) | "
             "Row 1: Real dataset samples (520×520 grayscale) | "
             "Row 2: Training patch pairs (16×16, label={1=focused, 0=blurred})",
             ha="center", fontsize=7.5, color="#888888")

    plt.savefig(str(out_path), dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
INFO = "[INFO]"


def print_and_write(msg: str, fh) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def generate_report(data_root: Path, report_path: Path,
                    min_source_count: int = 5000) -> bool:
    """Generate text report. Returns True if all critical checks pass."""
    all_ok = True

    with open(str(report_path), "w") as fh:
        header = (
            "=" * 70 + "\n"
            "FCNN-MFIF DATASET PIPELINE — VERIFICATION REPORT\n"
            "=" * 70
        )
        print_and_write(header, fh)

        # ── Real datasets ─────────────────────────────────────────────────────
        print_and_write("\n[A] REAL MULTI-FOCUS DATASETS", fh)
        print_and_write("-" * 40, fh)
        ds_results = check_real_datasets(data_root)

        for ds_key, rec in ds_results.items():
            if not rec.get("exists"):
                print_and_write(f"  {SKIP} {ds_key}: directory not found", fh)
                continue

            n       = rec.get("n_pairs", 0)
            exp     = rec.get("expected_pairs", "?")
            ok_n    = rec.get("pairs_ok", False)
            ok_sz   = rec.get("size_ok", False)
            ok_gr   = rec.get("gray_ok", False)

            status = PASS if (ok_n and ok_sz and ok_gr) else FAIL
            if status == FAIL:
                all_ok = False

            print_and_write(
                f"  {status} {ds_key}: {n}/{exp} pairs  "
                f"size={TARGET_SIZE[0]}×{TARGET_SIZE[1]}:{['NO','OK'][ok_sz]}  "
                f"grayscale:{['NO','OK'][ok_gr]}", fh
            )

            if "gt_ok" in rec:
                gt_s = PASS if rec["gt_ok"] else FAIL
                print_and_write(
                    f"         {gt_s} ground-truth images: {rec.get('gt_count',0)}/{exp}", fh
                )

        # ── Synthetic source ──────────────────────────────────────────────────
        print_and_write("\n[B] SYNTHETIC SOURCE IMAGES", fh)
        print_and_write("-" * 40, fh)
        syn_rec = check_synthetic_source(data_root)

        if not syn_rec.get("exists"):
            print_and_write(f"  {SKIP} raw_imagenet dir not found", fh)
        else:
            c   = syn_rec.get("count", 0)
            t   = min_source_count
            ok  = c >= t
            if not ok:
                all_ok = False
            print_and_write(
                f"  {PASS if ok else FAIL} Count: {c}/{t}  "
                f"size:{['NO','OK'][syn_rec.get('size_ok',False)]}  "
                f"grayscale:{['NO','OK'][syn_rec.get('gray_ok',False)]}", fh
            )

        # ── Blur levels ───────────────────────────────────────────────────────
        print_and_write("\n[C] BLUR LEVEL GENERATION", fh)
        print_and_write("-" * 40, fh)
        blur_rec = check_blur_levels(data_root)

        found = blur_rec.get("levels_found", [])
        ok_lvl = len(found) == N_BLUR_LEVELS
        if not ok_lvl:
            all_ok = False
        print_and_write(
            f"  {PASS if ok_lvl else FAIL} Levels found: {found} "
            f"(expected {list(range(1, N_BLUR_LEVELS+1))})", fh
        )

        if "ssim_values" in blur_rec:
            ssims = blur_rec["ssim_values"]
            dec   = blur_rec.get("ssim_decreasing", False)
            print_and_write(
                f"  {PASS if dec else FAIL} SSIM decreasing: {dec}  "
                f"values={[round(s,4) for s in ssims]}", fh
            )
            if not dec:
                all_ok = False

        # ── Patches ───────────────────────────────────────────────────────────
        print_and_write("\n[D] PATCH EXTRACTION", fh)
        print_and_write("-" * 40, fh)
        patch_rec = check_patches(data_root)

        n_focused = patch_rec.get("focused_files", 0)
        ok_shape  = patch_rec.get("patch_size_ok", False)
        ok_dtype  = patch_rec.get("focused_dtype", "") == "float32"

        print_and_write(
            f"  {PASS if n_focused>0 else FAIL} Focused patch files : {n_focused}", fh
        )
        if "focused_shape" in patch_rec:
            print_and_write(
                f"  {PASS if ok_shape else FAIL} Patch shape     : "
                f"{patch_rec['focused_shape']}  "
                f"(expected (N,{PATCH_SIZE},{PATCH_SIZE}))", fh
            )
            print_and_write(
                f"  {PASS if ok_dtype else FAIL} dtype           : "
                f"{patch_rec.get('focused_dtype','?')}  (expected float32)", fh
            )

        for lvl in range(1, N_BLUR_LEVELS + 1):
            key = f"blur_level_{lvl}_files"
            n   = patch_rec.get(key, 0)
            print_and_write(
                f"  {INFO} Blur level {lvl} patch files: {n}", fh
            )

        # ── Training dataset ──────────────────────────────────────────────────
        print_and_write("\n[E] TRAINING DATASET (NPZ)", fh)
        print_and_write("-" * 40, fh)
        npz_rec = check_training_dataset(data_root)

        if not npz_rec.get("exists"):
            print_and_write(f"  {SKIP} training_dataset.npz not found", fh)
        else:
            total = npz_rec.get("total_samples", 0)
            n_pos = npz_rec.get("n_positive", 0)
            n_neg = npz_rec.get("n_negative", 0)
            ok_ps = npz_rec.get("patch_shape_ok", False)
            ok_bl = npz_rec.get("balance_ok", False)
            ok_lb = npz_rec.get("labels_valid", False)

            if not (ok_ps and ok_bl and ok_lb):
                all_ok = False

            print_and_write(
                f"  {PASS if total>0 else FAIL} Total samples   : {total:,}", fh
            )
            print_and_write(
                f"  {PASS if n_pos>0 else FAIL} Positive (lbl=1): {n_pos:,}", fh
            )
            print_and_write(
                f"  {PASS if n_neg>0 else FAIL} Negative (lbl=0): {n_neg:,}", fh
            )
            print_and_write(
                f"  {PASS if ok_ps else FAIL} Patch shape     : "
                f"{npz_rec.get('patches_A_shape','?')}", fh
            )
            print_and_write(
                f"  {PASS if ok_bl else FAIL} Class balance   : "
                f"|pos-neg|={abs(n_pos-n_neg)}  (threshold ≤100)", fh
            )
            print_and_write(
                f"  {PASS if ok_lb else FAIL} Labels ∈ {{0,1}} : {ok_lb}", fh
            )
            print_and_write(
                f"  {INFO} Value range     : "
                f"[{npz_rec.get('A_min',0):.1f}, {npz_rec.get('A_max',255):.1f}]", fh
            )

        # ── Final verdict ─────────────────────────────────────────────────────
        print_and_write("\n" + "=" * 70, fh)
        verdict = PASS if all_ok else FAIL
        print_and_write(
            f"OVERALL: {verdict}  ({'All checks passed' if all_ok else 'Some checks failed — see above'})",
            fh
        )
        print_and_write("=" * 70, fh)

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the FCNN-MFIF dataset pipeline outputs."
    )
    parser.add_argument(
        "--data-root", default=str(PROJECT_ROOT / "datasets"),
        help="Root datasets directory (default: ./datasets)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib plot generation."
    )
    parser.add_argument(
        "--min-source-count", type=int, default=5000,
        help="Minimum expected source images (default: 5000). "
             "Use 100 when running the quick smoke-test."
    )
    args = parser.parse_args()

    data_root   = Path(args.data_root)
    report_path = PROJECT_ROOT / "pipeline_verification_report.txt"
    plot_path   = PROJECT_ROOT / "pipeline_verification_plots.png"

    print(f"\n{'='*70}", flush=True)
    print("FCNN-MFIF Pipeline Verification", flush=True)
    print(f"{'='*70}\n", flush=True)

    all_ok = generate_report(data_root, report_path,
                             min_source_count=args.min_source_count)

    if not args.no_plot:
        print("\n[INFO] Generating visual verification plots …")
        make_plots(data_root, plot_path)

    print(f"\n[INFO] Text report : {report_path}")
    print(f"[INFO] Visual plot : {plot_path}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
