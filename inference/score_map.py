"""
inference/score_map.py
========================
Score Map (SM) and Focus Map (FM) generation for the Siamese FCNN.

Implements the core score map logic for FCNN-MFIF.


  SM size = (⌈H/2⌉ - conv_patchsize + 1) × (⌈W/2⌉ - conv_patchsize + 1)
  where conv_patchsize = 8  (half of 16 due to overlapping)

  "It is observed from the experimental analysis that the size of the SM
   is reduced to half of the original image size. The reason is the
   overlapping of the pixels."

  → stride = 8 (half of patch_size=16) achieves the described overlap.

Sliding window (stride=8, patch=16):
  For 520×520 image: floor((520-16)/8)+1 = 64 positions → SM = 64×64
  Then averaged back to 520×520 using bilinear upsampling.

Usage:
  from inference.score_map import generate_score_map, generate_focus_map

  SM = generate_score_map(model, A_fuzz, B_fuzz)
  FM = generate_focus_map(SM, original_size=(520, 520))
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.image_utils import PATCH_SIZE   # 16

STRIDE = PATCH_SIZE // 2   # 8 — produces SM ≈ H/2 × W/2


# ---------------------------------------------------------------------------
# Core: Score Map
# ---------------------------------------------------------------------------
def generate_score_map(
    model: torch.nn.Module,
    img_a: np.ndarray,
    img_b: np.ndarray,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Slide a window over fuzzified images A' and B', feed each patch pair
    to the Siamese FCNN, and build the Score Map.

    Parameters
    ----------
    model     : trained SiameseFCNN (or any module with same signature)
    img_a     : (H, W) float32 fuzzified image A', values ∈ [0,1]
    img_b     : (H, W) float32 fuzzified image B', values ∈ [0,1]
    patch_size: 16
    stride    : 8
    batch_size: how many patches to process at once (memory trade-off)
    device    : CPU or CUDA

    Returns
    -------
    SM : (n_rows, n_cols) float32 array, values ∈ [0,1]
         SM[r,c] = p(A is more focused) at position (r,c)
         Value closer to 1 → A is focused at that location
         Value closer to 0 → B is focused at that location
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    H, W = img_a.shape[:2]

    # Compute SM grid dimensions
    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1

    # Extract all patch pairs into arrays
    patches_a: list[np.ndarray] = []
    patches_b: list[np.ndarray] = []

    for r in range(n_rows):
        for c in range(n_cols):
            rr = r * stride
            cc = c * stride
            patches_a.append(img_a[rr:rr+patch_size, cc:cc+patch_size])
            patches_b.append(img_b[rr:rr+patch_size, cc:cc+patch_size])

    patches_a = np.stack(patches_a, axis=0).astype(np.float32)   # (N,16,16)
    patches_b = np.stack(patches_b, axis=0).astype(np.float32)

    # Batch inference
    scores: list[float] = []
    n_patches = len(patches_a)

    with torch.no_grad():
        for start in range(0, n_patches, batch_size):
            end  = min(start + batch_size, n_patches)
            pa   = torch.from_numpy(patches_a[start:end]).unsqueeze(1).to(device)
            pb   = torch.from_numpy(patches_b[start:end]).unsqueeze(1).to(device)
            probs = model(pa, pb)          # (batch, 2)
            # probs[:,0] = p(B_focused),  probs[:,1] = p(A_focused)
            # NLLLoss(label=1) trains probs[:,1] = p(A_focused)
            score = probs[:, 1].cpu().numpy()   # p(A is focused)
            scores.extend(score.tolist())

    SM = np.array(scores, dtype=np.float32).reshape(n_rows, n_cols)
    return SM


# ---------------------------------------------------------------------------
# Focus Map: upsample SM → original resolution
# ---------------------------------------------------------------------------
def generate_focus_map(
    sm: np.ndarray,
    original_size: tuple[int, int],
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    """
    Map SM back to original resolution using OVERLAP AVERAGING.


            original size by the averaging method."

    For each SM position (r,c), the score is added to every pixel
    in its patch footprint in the accumulation map, then divided by
    the count to produce the averaged Focus Map.

    Parameters
    ----------
    sm            : (n_rows, n_cols) float32 Score Map
    original_size : (H, W) of the source images
    patch_size    : 16
    stride        : 8

    Returns
    -------
    FM : (H, W) float32 Focus Map, values ∈ [0, 1]
         FM pixel > 0.5  →  Image A is more focused at that pixel
         FM pixel ≤ 0.5  →  Image B is more focused at that pixel
    """
    H, W = original_size
    n_rows, n_cols = sm.shape

    accum = np.zeros((H, W), dtype=np.float32)  # sum of scores
    count = np.zeros((H, W), dtype=np.float32)  # number of patches covering each pixel

    for r in range(n_rows):
        for c in range(n_cols):
            rr = r * stride
            cc = c * stride
            r_end = min(rr + patch_size, H)
            c_end = min(cc + patch_size, W)
            accum[rr:r_end, cc:c_end] += sm[r, c]
            count[rr:r_end, cc:c_end] += 1.0

    # Avoid division by zero for any uncovered border pixels
    FM = np.where(count > 0, accum / count, 0.5)
    return FM.astype(np.float32)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def visualize_maps(
    img_a: np.ndarray,
    img_b: np.ndarray,
    sm: np.ndarray,
    fm: np.ndarray,
    out_path: str | None = None,
) -> None:
    """
    Display a 5-panel figure:
      [A' | B' | Score Map (SM) | Focus Map (FM) | Overlay on A]
    """
    try:
        import matplotlib
        matplotlib.use("Agg" if out_path else "TkAgg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 5, figsize=(22, 4), facecolor="#1a1a2e")
        titles = ["Image A' (fuzzified)", "Image B' (fuzzified)",
                  "Score Map (SM)", "Focus Map (FM)", "FM overlay on A"]
        images = [img_a, img_b, sm, fm, None]
        cmaps  = ["gray", "gray", "hot", "RdBu_r", "gray"]

        for ax, title, img, cmap in zip(axes, titles, images, cmaps):
            ax.set_facecolor("#16213e")
            ax.set_title(title, color="#00d4ff", fontsize=9)
            ax.axis("off")

        # Panels 0-3
        for i, (ax, img, cmap) in enumerate(zip(axes[:4],
                                                  [img_a, img_b, sm, fm],
                                                  cmaps[:4])):
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Panel 4: overlay FM as alpha mask on A
        ax = axes[4]
        ax.imshow(img_a, cmap="gray", vmin=0, vmax=1)
        ax.imshow(fm, cmap="hot", alpha=0.4, vmin=0, vmax=1)

        fig.suptitle("FCNN-MFIF Score Map & Focus Map",
                     color="#00d4ff", fontweight="bold", fontsize=12)
        plt.tight_layout()

        if out_path:
            plt.savefig(out_path, dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[INFO] Saved: {out_path}")
        else:
            plt.show()
        plt.close()

    except Exception as exc:
        print(f"[WARN] Visualisation failed: {exc}")


# ---------------------------------------------------------------------------
# CLI: run on a real dataset pair
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Score Map and Focus Map for a multi-focus pair."
    )
    parser.add_argument("--model",  required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--pair-dir", default=None,
                        help="Path to pair directory with A.png and B.png. "
                             "If not given, uses dataset1_lytro/pair_001/")
    parser.add_argument("--out",    default="outputs/score_map.png",
                        help="Output visualisation path.")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help=f"Sliding window stride (default: {STRIDE})")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    from model.siamese_fcnn import SiameseFCNN
    from utils.fuzzy_preprocessing import fuzzify_pair

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.model, map_location=device)
    model  = SiameseFCNN().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] Loaded model from {args.model}")

    # Load image pair
    if args.pair_dir:
        pair_dir = Path(args.pair_dir)
    else:
        pair_dir = (PROJECT_ROOT / "datasets" / "real" /
                    "dataset1_lytro" / "pair_001")

    img_a_raw = np.array(Image.open(pair_dir / "A.png").convert("L"),
                          dtype=np.float32)
    img_b_raw = np.array(Image.open(pair_dir / "B.png").convert("L"),
                          dtype=np.float32)
    print(f"[INFO] Images loaded: {pair_dir}")

    # Fuzzify
    A_fuzz, B_fuzz = fuzzify_pair(img_a_raw, img_b_raw)
    print(f"[INFO] Fuzzified: A shape={A_fuzz.shape}  B shape={B_fuzz.shape}")

    # Generate SM and FM
    print("[INFO] Generating Score Map …")
    SM = generate_score_map(model, A_fuzz, B_fuzz, stride=args.stride)
    FM = generate_focus_map(SM, original_size=A_fuzz.shape[:2])

    print(f"[INFO] SM shape={SM.shape}  min={SM.min():.3f}  max={SM.max():.3f}")
    print(f"[INFO] FM shape={FM.shape}  min={FM.min():.3f}  max={FM.max():.3f}")

    # Visualise
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    visualize_maps(A_fuzz, B_fuzz, SM, FM, out_path=args.out)
    print(f"\n[DONE] Outputs saved to {args.out}")
