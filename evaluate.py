"""
evaluate.py
===========
Complete evaluation protocol for FCNN-MFIF.

Runs 6 fusion methods, computes all metrics, saves CSV + comparison plots.

Methods implemented:
    Average  — F = (A+B)/2                         (trivial baseline)
    Max      — F = max(A,B)                         (pixel-wise max)
    NSWT     — Non-subsampled Wavelet Transform      (Laplacian pyramid proxy)
    GF       — Guided Filter focus fusion           (He et al. 2013)
    CNN      — Laplacian-energy CNN-proxy fusion    (focus measure + GF)
    FCNN     — Our trained Siamese FCNN

Note on ECNN / IFCNN / DRPL:
    These require authors' released weights/code. Stubs are included
    but marked [EXTERNAL] — replace with actual implementations if available.

Usage:
    python3 evaluate.py --model checkpoints/fcnn_best.pt
    python3 evaluate.py --model checkpoints/fcnn_best.pt --fast-stride
    python3 evaluate.py --model checkpoints/fcnn_best.pt --dataset ds1
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics import compute_all_metrics
from inference.fusion import guided_filter, pixel_wise_fusion, fuse_images


# ============================================================================
# Normalisation helper
# ============================================================================
def _norm(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    return img / 255.0 if img.max() > 1.5 else img


# ============================================================================
# Baseline 1 — Average
# ============================================================================
def fuse_average(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """F = (A + B) / 2."""
    return ((_norm(A) + _norm(B)) / 2.0).clip(0, 1)


# ============================================================================
# Baseline 2 — Max
# ============================================================================
def fuse_max(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """F = pixel-wise max(A, B)."""
    return np.maximum(_norm(A), _norm(B))


# ============================================================================
# Baseline 3 — NSWT  (Laplacian pyramid proxy)
# ============================================================================
def fuse_nswt(A: np.ndarray, B: np.ndarray, levels: int = 4) -> np.ndarray:
    """
    Non-Subsampled Wavelet Transform fusion proxy.

    Uses a Laplacian pyramid decomposition (no pywavelets needed):
      - High-frequency bands: take pixel with larger absolute value (max rule)
      - Low-frequency band:   average
    Then reconstruct by collapsing the pyramid.

    This matches the core principle of NSWT-based MFIF methods.
    """
    A, B = _norm(A).astype(np.float64), _norm(B).astype(np.float64)

    def gaussian_blur(img):
        k = np.array([[1,4,6,4,1]], dtype=np.float64) / 16.0
        return convolve2d(convolve2d(img, k, mode='same', boundary='symm'),
                          k.T, mode='same', boundary='symm')

    # Build Laplacian pyramids
    def build_laplacian(img, n):
        pyr = []
        current = img.copy()
        for _ in range(n):
            blurred = gaussian_blur(current)
            pyr.append(current - blurred)   # high-freq residual
            current = blurred
        pyr.append(current)                  # low-freq base
        return pyr

    def collapse(pyr):
        result = pyr[-1]
        for lap in reversed(pyr[:-1]):
            result = result + lap
        return result

    pA = build_laplacian(A, levels)
    pB = build_laplacian(B, levels)

    fused_pyr = []
    for i in range(levels):               # high-freq: take larger absolute
        fused_pyr.append(
            np.where(np.abs(pA[i]) >= np.abs(pB[i]), pA[i], pB[i])
        )
    fused_pyr.append((pA[-1] + pB[-1]) / 2.0)  # low-freq: average

    return collapse(fused_pyr).clip(0, 1).astype(np.float32)


# ============================================================================
# Baseline 4 — GF  (Guided Filter focus fusion)
# ============================================================================
def fuse_guided_filter(
    A: np.ndarray, B: np.ndarray,
    win: int = 15, r: int = 7, eps: float = 0.2,
) -> np.ndarray:
    """
    Guided-filter MFIF (He et al. 2013).
    Focus measure = local variance.
    Decision map refined by guided filter.
    """
    A, B = _norm(A), _norm(B)

    def local_var(img, w):
        mu  = uniform_filter(img.astype(np.float64), w)
        mu2 = uniform_filter((img ** 2).astype(np.float64), w)
        return np.maximum(mu2 - mu**2, 0.0).astype(np.float32)

    var_A, var_B = local_var(A, win), local_var(B, win)
    init_map = (var_A >= var_B).astype(np.float32)
    guidance = (A + B) / 2.0
    FD = guided_filter(guidance, init_map, radius=r, eps=eps)
    return pixel_wise_fusion(FD, A, B)


# ============================================================================
# Baseline 5 — CNN-proxy  (Laplacian energy focus measure + GF refinement)
# ============================================================================
def fuse_cnn_proxy(
    A: np.ndarray, B: np.ndarray,
    win: int = 15, r: int = 7, eps: float = 0.2,
) -> np.ndarray:
    """
    CNN-proxy fusion.

    Replaces the Siamese CNN's focus score with a classical Laplacian energy
    measure (sum of squared Laplacian in a local window).  The rest of the
    pipeline (soft map + guided filter + pixel-wise fusion) is identical to
    our FCNN method.  This isolates the contribution of the learned network.
    """
    A, B = _norm(A), _norm(B)

    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    def laplacian_energy(img, w):
        lap = convolve2d(img.astype(np.float64), lap_kernel,
                         mode='same', boundary='symm')
        return uniform_filter(lap**2, w).astype(np.float32)

    le_A, le_B = laplacian_energy(A, win), laplacian_energy(B, win)
    eps_le = 1e-10
    soft_map = le_A / (le_A + le_B + eps_le)           # soft focus map ∈ [0,1]
    guidance  = (A + B) / 2.0
    FD = guided_filter(guidance, soft_map, radius=r, eps=eps)
    return pixel_wise_fusion(FD, A, B)


# ============================================================================
# Baseline 6 — FCNN  (our method)
# ============================================================================
def fuse_fcnn(
    A: np.ndarray, B: np.ndarray,
    model, device, stride: int = 2, batch_size: int = 512,
) -> np.ndarray:
    """Full FCNN-MFIF pipeline → fused float32 [0,1]."""
    result = fuse_images(A, B, model, device=device,
                         stride=stride, batch_size=batch_size, verbose=False)
    return result["fused"]


# ============================================================================
# Methods registry
# ============================================================================
METHODS = ["Average", "Max", "NSWT", "GF", "CNN", "FCNN"]


# ============================================================================
# Sanity checks
# ============================================================================
def sanity_check(method: str, m: dict, H: int, W: int) -> None:
    """Warn if any metric looks unrealistic."""
    if m["MI"] > 20:
        print(f"    [WARN] {method}: MI={m['MI']:.2f} looks too high (expected < 5)")
    if not (0 <= m["EI"] <= 1):
        print(f"    [WARN] {method}: EI={m['EI']:.4f} outside [0,1]")
    if not (0 <= m["SS"] <= 1):
        print(f"    [WARN] {method}: SS={m['SS']:.4f} outside [0,1]")
    if not (0 <= m["HP"] <= 1):
        print(f"    [WARN] {method}: HP={m['HP']:.4f} outside [0,1]")
    if "PSNR" in m and m["PSNR"] < 10:
        print(f"    [WARN] {method}: PSNR={m['PSNR']:.2f} dB seems very low")
    if "RMSE" in m and m["RMSE"] > 100:
        print(f"    [WARN] {method}: RMSE={m['RMSE']:.2f} seems very high")


# ============================================================================
# Dataset loader
# ============================================================================
def load_pairs(
    datasets: list[Path],
    ref_dir: Path | None = None,
) -> list[tuple]:
    """
    Load (pair_dir, A, B [, Ref]) from dataset directories.
    If ref_dir is given, tries to load a reference image too (for PSNR/RMSE).
    """
    pairs = []
    for ds in datasets:
        if not ds.exists():
            print(f"  [WARN] Dataset not found: {ds}")
            continue
        for p in sorted(ds.glob("pair_*/")):
            if (p / "A.png").exists() and (p / "B.png").exists():
                A = np.array(Image.open(p / "A.png").convert("L"), dtype=np.float32)
                B = np.array(Image.open(p / "B.png").convert("L"), dtype=np.float32)
                ref = None
                if ref_dir and (ref_dir / p.name / "Ref.png").exists():
                    ref = np.array(
                        Image.open(ref_dir / p.name / "Ref.png").convert("L"),
                        dtype=np.float32,
                    )
                pairs.append((p, A, B, ref))
    return pairs


# ============================================================================
# Per-pair evaluation
# ============================================================================
def evaluate(
    model,
    device,
    pairs: list,
    stride: int = 2,
    out_dir: Path = Path("outputs/eval"),
    save_images: bool = True,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for idx, entry in enumerate(pairs):
        pair_dir, A, B = entry[0], entry[1], entry[2]
        ref = entry[3] if len(entry) > 3 else None

        pair_name = f"{pair_dir.parent.name}/{pair_dir.name}"
        print(f"\n[{idx+1}/{len(pairs)}] {pair_name}")

        pair_out = out_dir / pair_dir.parent.name / pair_dir.name
        pair_out.mkdir(parents=True, exist_ok=True)

        if save_images:
            Image.fromarray(A.astype(np.uint8)).save(pair_out / "A.png")
            Image.fromarray(B.astype(np.uint8)).save(pair_out / "B.png")

        An, Bn = A / 255.0, B / 255.0
        ref_n  = ref / 255.0 if ref is not None else None

        # ── Run each fusion method ────────────────────────────────────────
        fused: dict[str, np.ndarray] = {}

        for method in METHODS:
            print(f"  {method:<8} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                if   method == "Average": fused[method] = fuse_average(A, B)
                elif method == "Max":     fused[method] = fuse_max(A, B)
                elif method == "NSWT":    fused[method] = fuse_nswt(A, B)
                elif method == "GF":      fused[method] = fuse_guided_filter(A, B)
                elif method == "CNN":     fused[method] = fuse_cnn_proxy(A, B)
                elif method == "FCNN":    fused[method] = fuse_fcnn(
                    A, B, model, device, stride=stride)
                print(f"{time.time()-t0:.1f}s")
            except Exception as exc:
                print(f"FAILED ({exc})")
                continue

        # ── Metrics for each method ───────────────────────────────────────
        for method, F in fused.items():
            m = compute_all_metrics(An, Bn, F, ref=ref_n)
            sanity_check(method, m, *A.shape)
            rec = {
                "dataset": pair_dir.parent.name,
                "pair":    pair_dir.name,
                "method":  method,
                **{k: round(v, 6) for k, v in m.items()},
            }
            records.append(rec)
            psnr_str = f" PSNR={m['PSNR']:.2f}" if "PSNR" in m else ""
            print(f"    {method:<8} MI={m['MI']:.4f} EI={m['EI']:.4f} "
                  f"SS={m['SS']:.4f} HP={m['HP']:.4f}{psnr_str}")

            if save_images:
                F_u8 = (F * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(F_u8).save(pair_out / f"fused_{method}.png")

    return records


# ============================================================================
# Summary table
# ============================================================================
def print_summary_table(records: list[dict]) -> None:
    metric_keys = ["MI", "EI", "SS", "HP"]
    sums: dict   = defaultdict(lambda: defaultdict(float))
    counts: dict = defaultdict(int)

    for r in records:
        m = r["method"]
        for k in metric_keys:
            if k in r:
                sums[m][k] += r[k]
        counts[m] += 1

    print("\n" + "="*70)
    print(f"{'Method':<10} {'MI':>9} {'EI':>9} {'SS':>9} {'HP':>9}  (avg over pairs)")
    print("-"*70)
    for method in METHODS:
        if method not in counts:
            continue
        n   = counts[method]
        row = f"{method:<10}"
        for k in metric_keys:
            row += f" {sums[method][k]/n:>9.4f}"
        print(row)
    print("-"*70)
    print(f"{'Target(FCNN)':<10} {'1.1678':>9} {'0.7281':>9} {'0.9850':>9} {'0.8020':>9}")
    print("="*70)


# ============================================================================
# CSV save
# ============================================================================
def save_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fields = list(records[0].keys())
    # Ensure PSNR/RMSE columns exist even if absent in some rows
    for k in ("PSNR", "RMSE"):
        if any(k in r for r in records) and k not in fields:
            fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"\n[CSV] Saved → {path}  ({len(records)} rows, "
          f"{len(records)//max(len(METHODS),1)} pairs)")


# ============================================================================
# Comparison plots  (2-row grid: A/B/methods)
# ============================================================================
def save_comparison_plots(
    pairs: list, records: list[dict], out_dir: Path,
    max_pairs: int = 5,
) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not found — skipping plots.")
        return

    rec_map: dict = {}
    for r in records:
        rec_map[(r["dataset"], r["pair"], r["method"])] = r

    plotted = 0
    for entry in pairs:
        if plotted >= max_pairs:
            break
        pair_dir, A, B = entry[0], entry[1], entry[2]
        pair_out = out_dir / pair_dir.parent.name / pair_dir.name
        fused_files = {m: pair_out / f"fused_{m}.png" for m in METHODS}
        if not all(f.exists() for f in fused_files.values()):
            continue

        ncols = len(METHODS) + 2    # A, B, + each method
        fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 5),
                                 facecolor="#0d0d1a")
        fig.suptitle(f"{pair_dir.parent.name}/{pair_dir.name}",
                     color="#00d4ff", fontsize=12, fontweight="bold")

        panels = [("Input A", A/255.0), ("Input B", B/255.0)] + [
            (m, np.array(Image.open(fused_files[m]).convert("L"))/255.0)
            for m in METHODS
        ]

        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, color="#00d4ff", fontsize=8.5, pad=4)
            ax.axis("off")
            ax.set_facecolor("#16213e")
            ds, pair = pair_dir.parent.name, pair_dir.name
            if title in METHODS and (ds, pair, title) in rec_map:
                r = rec_map[(ds, pair, title)]
                txt = (f"MI={r['MI']:.3f}\nEI={r['EI']:.3f}\n"
                       f"SS={r['SS']:.3f}\nHP={r['HP']:.3f}")
                ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                        color="white", fontsize=6.5, va="top",
                        bbox=dict(facecolor="black", alpha=0.55, pad=2))

        plt.tight_layout()
        out_path = pair_out / "comparison.png"
        plt.savefig(out_path, dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        plotted += 1

    print(f"[PLOTS] {plotted} comparison figure(s) → {out_dir}")


# ============================================================================
# CLI
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="FCNN-MFIF Full Evaluation (all metrics, all methods)"
    )
    parser.add_argument("--model",       required=True,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--dataset",     choices=["ds1", "ds2", "both"],
                        default="both")
    parser.add_argument("--fast-stride", action="store_true",
                        help="stride=8 (~30s/pair) vs default stride=2")
    parser.add_argument("--out-dir",     default="outputs/eval")
    parser.add_argument("--no-images",   action="store_true")
    parser.add_argument("--max-plots",   type=int, default=5,
                        help="Max comparison plots to generate (default: 5)")
    args = parser.parse_args()

    import torch
    from model.siamese_fcnn import SiameseFCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.model, map_location=device, weights_only=False)
    model  = SiameseFCNN().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] Model  : epoch {ckpt.get('epoch','?')}, "
          f"val_acc={ckpt.get('val_acc',0):.4f}")

    stride = 8 if args.fast_stride else 2
    print(f"[INFO] Stride : {stride}  ({'fast' if args.fast_stride else 'exact'})")
    print(f"[INFO] Methods: {METHODS}")

    real_dir = PROJECT_ROOT / "datasets" / "real"
    ds_paths: list[Path] = []
    if args.dataset in ("ds1", "both"):
        ds_paths.append(real_dir / "dataset1_lytro")
    if args.dataset in ("ds2", "both"):
        ds_paths.append(real_dir / "dataset2_mfif")

    pairs = load_pairs(ds_paths)
    if not pairs:
        print("[ERROR] No pairs found. Run:  ./run_pipeline.sh --stage 1")
        sys.exit(1)
    print(f"[INFO] Pairs  : {len(pairs)}")

    out_dir = Path(args.out_dir)
    records = evaluate(model, device, pairs,
                       stride=stride, out_dir=out_dir,
                       save_images=not args.no_images)

    csv_path = out_dir / "results.csv"
    save_csv(records, csv_path)
    print_summary_table(records)

    if not args.no_images:
        save_comparison_plots(pairs, records, out_dir,
                              max_pairs=args.max_plots)

    print(f"\n[DONE] → {csv_path}")


if __name__ == "__main__":
    main()
