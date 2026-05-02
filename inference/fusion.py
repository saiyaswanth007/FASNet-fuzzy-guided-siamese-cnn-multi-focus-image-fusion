"""
inference/fusion.py
====================
Final fusion stage for the FCNN-MFIF pipeline.

Implements the core fusion Algorithm for FCNN-MFIF.

Full pipeline (after CNN produces Score Map):
─────────────────────────────────────────────────────────────────────
  Step 1  Focus Map (FM)       ← overlap-averaging of SM → full res
  Step 2  Binary Map (BM)      ← threshold FM at T = 0.4
  Step 3  Initial Decision Map ← bwareaopen: remove regions < 1% of image
           (ID')
  Step 4  Final Decision Map   ← Guided Filter (r=7, ε=0.2) on ID'
           (FD)
  Step 5  Fused Image (F)      ← F = FD·A' + (1−FD)·B'   pixel-wise
─────────────────────────────────────────────────────────────────────

Public API:
  fuse_images(A_gray, B_gray, model, device=None)
      → dict with keys: SM, FM, BM, ID, FD, fused

  visualize_fusion(...)
      → saves 7-panel figure

  guided_filter(guidance, src, radius, eps)
  remove_small_regions(bm, min_area)
  generate_binary_map(fm, threshold)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label, uniform_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.fuzzy_preprocessing import fuzzify_pair
from utils.image_utils import TARGET_SIZE
from inference.score_map import generate_score_map, generate_focus_map, PATCH_SIZE, STRIDE

#
THRESHOLD     = 0.4   # binary segmentation threshold
GUIDED_RADIUS = 7     # guided filter window radius
GUIDED_EPS    = 0.2   # guided filter regularisation


# ---------------------------------------------------------------------------
# Step 2: Binary Map
# ---------------------------------------------------------------------------
def generate_binary_map(
    fm: np.ndarray,
    threshold: float = THRESHOLD,
) -> np.ndarray:
    """
    Apply threshold T=0.4 to Focus Map to produce Binary Map.


        BM(x,y) = 1  if FM(x,y) > T
                  0  otherwise

    Parameters
    ----------
    fm        : (H, W) float32 Focus Map, values ∈ [0,1]
    threshold : 0.4

    Returns
    -------
    BM : (H, W) uint8  {0, 1}
    """
    BM = (fm > threshold).astype(np.uint8)
    return BM


# ---------------------------------------------------------------------------
# Step 3: Small Region Removal  (bwareaopen equivalent)
# ---------------------------------------------------------------------------
def remove_small_regions(
    bm: np.ndarray,
    min_area: float | None = None,
) -> np.ndarray:
    """
    Remove misclassified small connected regions from Binary Map.


        ID' = bwareaopen(BM, area)
        area = 0.01 × H × W

    bwareaopen (MATLAB behaviour) removes connected components of
    FOREGROUND pixels (value=1) whose area is smaller than the
    threshold.  It does NOT fill small background holes.
    This is exactly what we implement here.

    Parameters
    ----------
    bm       : (H, W) uint8 Binary Map {0,1}
    min_area : minimum region size in pixels.
               Default = 0.01 × H × W

    Returns
    -------
    ID : (H, W) uint8  Initial Decision Map (noise-free binary map)
    """
    H, W = bm.shape
    if min_area is None:
        min_area = 0.01 * H * W   # area = 0.01 × ht × wt

    ID = bm.copy()

    # bwareaopen: remove small FOREGROUND (1) connected components only
    labeled_fg, n_fg = label(ID)
    for comp_id in range(1, n_fg + 1):
        mask = labeled_fg == comp_id
        if mask.sum() < min_area:
            ID[mask] = 0   # remove small foreground region → set to background

    return ID.astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 4: Guided Filter   (He et al. 2013)
# ---------------------------------------------------------------------------
def _box_filter(img: np.ndarray, radius: int) -> np.ndarray:
    """Uniform (box) mean filter of window size (2r+1) × (2r+1)."""
    size = 2 * radius + 1
    return uniform_filter(img.astype(np.float64), size=size, mode="reflect")


def guided_filter(
    guidance: np.ndarray,
    src: np.ndarray,
    radius: int = GUIDED_RADIUS,
    eps: float   = GUIDED_EPS,
) -> np.ndarray:
    """
    Guided Image Filter (He et al. 2013).

    Removes edge artifacts and refines boundaries of the binary decision map.


        r = 7,  ε = 0.2
        Guidance image = mean(A', B')  — average of both fuzzified sources

                          is the standard MFIF convention so neither source
                          image biases the decision boundary)

    Algorithm:
        mean_I  = box_filter(I)
        mean_p  = box_filter(p)
        cov_Ip  = box_filter(I·p) − mean_I · mean_p
        var_I   = box_filter(I·I) − mean_I²
        a       = cov_Ip / (var_I + ε)
        b       = mean_p − a · mean_I
        FD      = box_filter(a) · I + box_filter(b)

    Parameters
    ----------
    guidance : (H, W) float  guidance image — use mean(A', B') in fuse_images()
    src      : (H, W) float  input to filter (Initial Decision Map ID')
    radius   : 7
    eps      : 0.2

    Returns
    -------
    FD : (H, W) float32  Final Decision Map, values ∈ [0, 1]
    """
    I = guidance.astype(np.float64)
    p = src.astype(np.float64)

    mean_I  = _box_filter(I,     radius)
    mean_p  = _box_filter(p,     radius)
    mean_Ip = _box_filter(I * p, radius)
    mean_II = _box_filter(I * I, radius)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I  = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)

    FD = mean_a * I + mean_b
    FD = np.clip(FD, 0.0, 1.0)
    return FD.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5: Final Pixel-Wise Fusion
# ---------------------------------------------------------------------------
def pixel_wise_fusion(
    FD:  np.ndarray,
    A:   np.ndarray,
    B:   np.ndarray,
) -> np.ndarray:
    """
    Final image fusion using

        F(x,y) = FD(x,y) · A'(x,y) + (1 − FD(x,y)) · B'(x,y)

    FD ≈ 1 → take from A' (A is focused at that pixel)
    FD ≈ 0 → take from B' (B is focused at that pixel)

    Parameters
    ----------
    FD : (H, W) float  Final Decision Map ∈ [0,1]
    A  : (H, W) float  Fuzzified image A' ∈ [0,1]
    B  : (H, W) float  Fuzzified image B' ∈ [0,1]

    Returns
    -------
    F  : (H, W) float32  Fused image, values ∈ [0,1]
    """
    F = FD * A + (1.0 - FD) * B
    F = np.clip(F, 0.0, 1.0)
    return F.astype(np.float32)


# ---------------------------------------------------------------------------
# Full End-to-End Fusion Pipeline
# ---------------------------------------------------------------------------
def fuse_images(
    A_gray: np.ndarray,
    B_gray: np.ndarray,
    model,
    device=None,
    stride:    int   = 2,          #
    threshold: float = THRESHOLD,
    gf_radius: int   = GUIDED_RADIUS,
    gf_eps:    float = GUIDED_EPS,
    batch_size: int  = 256,
    verbose: bool    = True,
) -> dict:
    """
    Complete FCNN-MFIF fusion pipeline from raw grayscale images to
    fused output.

    Pipeline:
        A_gray, B_gray
          → Fuzzification (S-membership)        → A', B'
          → Siamese CNN sliding window           → Score Map (SM)
          → Overlap averaging                    → Focus Map (FM)
          → Threshold T=0.4                      → Binary Map (BM)
          → bwareaopen (area=0.01·H·W)           → Initial Decision Map (ID')
          → Guided Filter (r=7, ε=0.2)           → Final Decision Map (FD)
          → Pixel-wise: FD·A' + (1−FD)·B'       → Fused Image (F)

    Parameters
    ----------
    A_gray    : (H, W) float32 or uint8  grayscale image A  (raw, [0,255])
    B_gray    : (H, W) float32 or uint8  grayscale image B  (raw, [0,255])
    model     : trained SiameseFCNN
    device    : torch.device (auto-detected if None)
    stride    : 2
    threshold : 0.4
    gf_radius : 7
    gf_eps    : 0.2
    batch_size: patches processed per forward pass
    verbose   : print progress

    Returns
    -------
    dict with keys:
        A_fuzz  : (H,W) float32  fuzzified A'
        B_fuzz  : (H,W) float32  fuzzified B'
        SM      : (n_rows, n_cols) float32  Score Map
        FM      : (H,W) float32  Focus Map
        BM      : (H,W) uint8    Binary Map
        ID      : (H,W) uint8    Initial Decision Map (after small region removal)
        FD      : (H,W) float32  Final Decision Map (after guided filter)
        fused   : (H,W) float32  Fused image ∈ [0,1]
        fused_uint8 : (H,W) uint8  Fused image ∈ [0,255]
    """
    import torch
    if device is None:
        device = next(model.parameters()).device

    H, W = A_gray.shape[:2]

    def _log(msg):
        if verbose:
            print(f"  [FCNN] {msg}", flush=True)

    # ── Step 0: Fuzzification ────────────────────────────────────────────────
    _log("Fuzzifying images A and B ...")
    A_fuzz, B_fuzz = fuzzify_pair(
        A_gray.astype(np.float32),
        B_gray.astype(np.float32),
    )

    # ── Step 1: Score Map (sliding window CNN) ───────────────────────────────
    _log(f"Generating Score Map (stride={stride}) ...")
    SM = generate_score_map(
        model, A_fuzz, B_fuzz,
        patch_size=PATCH_SIZE,
        stride=stride,
        batch_size=batch_size,
        device=device,
    )
    _log(f"  SM shape={SM.shape}  range=[{SM.min():.3f}, {SM.max():.3f}]")

    # ── Step 2: Focus Map (overlap averaging → full resolution) ─────────────
    _log("Reconstructing Focus Map (overlap averaging) ...")
    FM = generate_focus_map(SM, original_size=(H, W),
                            patch_size=PATCH_SIZE, stride=stride)
    _log(f"  FM shape={FM.shape}  range=[{FM.min():.3f}, {FM.max():.3f}]")

    # ── Step 3: Binary Map (threshold T=0.4) ────────────────────────────────
    _log(f"Thresholding FM at T={threshold} → Binary Map ...")
    BM = generate_binary_map(FM, threshold=threshold)
    _log(f"  BM foreground={BM.sum()} / {H*W} pixels "
         f"({100*BM.mean():.1f}% A-focused)")

    # ── Step 4: Small Region Removal (bwareaopen, area = 0.01·H·W) ──────────
    min_area = 0.01 * H * W
    _log(f"Removing small regions (min_area={int(min_area)} px) ...")
    ID = remove_small_regions(BM, min_area=min_area)
    _log(f"  ID foreground={ID.sum()} px after filtering")

    # ── Step 5: Guided Filter (r=7, ε=0.2) ──────────────────────────────────
    _log(f"Applying Guided Filter (r={gf_radius}, ε={gf_eps}) ...")
    # Guidance = mean(A', B'): preserves edges from BOTH sources equally.
    #
    # convention so neither source image biases the decision boundary.
    guidance = (A_fuzz + B_fuzz) / 2.0
    FD = guided_filter(guidance, ID.astype(np.float32),
                       radius=gf_radius, eps=gf_eps)
    _log(f"  FD range=[{FD.min():.3f}, {FD.max():.3f}]")

    # ── Step 6: Pixel-wise Fusion ───────────────────────────────────
    _log("Fusing: F = FD·A' + (1−FD)·B' ...")
    fused = pixel_wise_fusion(FD, A_fuzz, B_fuzz)

    fused_uint8 = (fused * 255.0).clip(0, 255).astype(np.uint8)
    _log(f"  Fused image range=[{fused_uint8.min()}, {fused_uint8.max()}]")
    _log("Done.")

    return {
        "A_fuzz":      A_fuzz,
        "B_fuzz":      B_fuzz,
        "SM":          SM,
        "FM":          FM,
        "BM":          BM,
        "ID":          ID,
        "FD":          FD,
        "fused":       fused,
        "fused_uint8": fused_uint8,
    }


# ---------------------------------------------------------------------------
# Visualisation — 7-panel figure
# ---------------------------------------------------------------------------
def visualize_fusion(
    result: dict,
    out_path: str | None = None,
    pair_name: str = "pair",
) -> None:
    """
    7-panel visualisation of the complete fusion pipeline:
    [A' | B' | SM | FM | BM | FD | Fused]
    """
    try:
        import matplotlib
        matplotlib.use("Agg" if out_path else "TkAgg")
        import matplotlib.pyplot as plt

        panels = [
            (result["A_fuzz"],  "A' (fuzzified)",      "gray",    0, 1),
            (result["B_fuzz"],  "B' (fuzzified)",      "gray",    0, 1),
            (result["SM"],      "Score Map (SM)",       "hot",     0, 1),
            (result["FM"],      "Focus Map (FM)",       "RdBu_r",  0, 1),
            (result["BM"],      "Binary Map (BM)",      "gray",    0, 1),
            (result["FD"],      "Decision Map (FD)",    "RdBu_r",  0, 1),
            (result["fused"],   "Fused Image (F)",      "gray",    0, 1),
        ]

        fig, axes = plt.subplots(1, 7, figsize=(28, 4.5),
                                 facecolor="#0d0d1a")
        fig.suptitle(
            f"FCNN-MFIF Fusion Pipeline — {pair_name}",
            color="#00d4ff", fontweight="bold", fontsize=13, y=1.01
        )

        for ax, (img, title, cmap, vmin, vmax) in zip(axes, panels):
            ax.set_facecolor("#16213e")
            ax.set_title(title, color="#00d4ff", fontsize=8.5, pad=4)
            ax.axis("off")
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         shrink=0.85).ax.yaxis.set_tick_params(color="#888")

        fig.text(0.5, -0.03,
                 "T=0.4 | bwareaopen(0.01·H·W) | Guided Filter(r=7,ε=0.2) | "
                 "F=FD·A'+(1−FD)·B'",
                 ha="center", fontsize=7.5, color="#666")

        plt.tight_layout()

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=130, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  [FCNN] Saved: {out_path}")
        else:
            plt.show()
        plt.close()

    except Exception as exc:
        print(f"  [WARN] Visualisation failed: {exc}")


def save_outputs(result: dict, out_dir: str, pair_name: str = "pair") -> None:
    """Save all intermediate and final outputs as PNG files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _save(arr: np.ndarray, name: str) -> None:
        if arr.dtype in (np.float32, np.float64):
            img_data = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            img_data = arr.astype(np.uint8)
            if img_data.max() <= 1:
                img_data = img_data * 255
        Image.fromarray(img_data).save(str(out / f"{pair_name}_{name}.png"))

    _save(result["A_fuzz"],  "A_fuzz")
    _save(result["B_fuzz"],  "B_fuzz")
    _save(result["FM"],      "focus_map")
    _save(result["BM"],      "binary_map")
    _save(result["ID"],      "initial_decision")
    _save(result["FD"],      "final_decision")
    _save(result["fused"],   "fused")
    print(f"  [FCNN] Outputs saved → {out}/")


# ---------------------------------------------------------------------------
# CLI: run full fusion on a dataset pair
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(
        description="Full FCNN-MFIF fusion pipeline on a multi-focus pair."
    )
    parser.add_argument("--model",    required=True,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--pair-dir", default=None,
                        help="Directory containing A.png and B.png. "
                             "Default: dataset1_lytro/pair_001/")
    parser.add_argument("--out-dir",  default="outputs",
                        help="Output directory (default: ./outputs)")
    parser.add_argument("--stride",   type=int, default=STRIDE)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--all-pairs", action="store_true",
                        help="Run on ALL pairs in dataset1_lytro and dataset2_mfif")
    args = parser.parse_args()

    from model.siamese_fcnn import SiameseFCNN

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.model, map_location=device, weights_only=False)
    model  = SiameseFCNN().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] Model loaded: {args.model}  (epoch {ckpt.get('epoch','?')})")

    # Collect pairs to process
    if args.all_pairs:
        pairs = []
        for ds in ["dataset1_lytro", "dataset2_mfif"]:
            ds_dir = PROJECT_ROOT / "datasets" / "real" / ds
            if ds_dir.exists():
                for p in sorted(ds_dir.glob("pair_*/")):
                    if (p / "A.png").exists() and (p / "B.png").exists():
                        pairs.append((p, ds + "_" + p.name))
    else:
        pair_dir = Path(args.pair_dir) if args.pair_dir else (
            PROJECT_ROOT / "datasets" / "real" / "dataset1_lytro" / "pair_001"
        )
        pairs = [(pair_dir, pair_dir.parent.name + "_" + pair_dir.name)]

    print(f"[INFO] Processing {len(pairs)} pair(s)...")

    for pair_dir, pair_name in pairs:
        print(f"\n{'─'*50}")
        print(f"[PAIR] {pair_name}")

        A_gray = np.array(Image.open(pair_dir / "A.png").convert("L"),
                          dtype=np.float32)
        B_gray = np.array(Image.open(pair_dir / "B.png").convert("L"),
                          dtype=np.float32)

        result = fuse_images(
            A_gray, B_gray, model, device=device,
            stride=args.stride, batch_size=args.batch_size,
        )

        out_dir = Path(args.out_dir) / pair_name
        save_outputs(result, str(out_dir), pair_name=pair_name)
        visualize_fusion(result,
                         out_path=str(out_dir / f"{pair_name}_pipeline.png"),
                         pair_name=pair_name)

    print(f"\n[DONE] All outputs saved to: {args.out_dir}")
