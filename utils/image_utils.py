"""
utils/image_utils.py
====================
Shared image processing utilities for the FCNN-MFIF dataset pipeline.

All functions exactly match the preprocessing steps described in:
  "A Fuzzy Convolutional Neural Network for Multi-Focus Image Fusion"
  Bhalla et al., Journal of Visual Communication and Image Representation, 2022.

Key paper specs implemented here:
  - Image size  : 520 × 520 pixels  (Section 3.1)
  - Color space : Grayscale          (Section 3.2)
  - Blur        : Gaussian σ = 0.5, 5 sequential levels (Section 3.2 / Fig 4)
  - Patch size  : 16 × 16, no overlap, bicubic crop    (Section 3.2 / Table 3)
  - Augment     : 90° and 180° rotations               (Section 3.2)
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Constants (paper-defined)
# ---------------------------------------------------------------------------
TARGET_SIZE: tuple[int, int] = (520, 520)   # Section 3.1
PATCH_SIZE:  int = 16                        # Table 3
BLUR_SIGMA:  float = 0.5                     # Section 3.2
N_BLUR_LEVELS: int = 5                       # Section 3.2 / Fig 4


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_grayscale(path: str,
                   size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Load an image from *path*, convert to grayscale, and resize to *size*.

    Returns
    -------
    np.ndarray of shape (H, W), dtype float32, values in [0, 255].
    """
    img = Image.open(path).convert("L")          # 'L' = single-channel grayscale
    img = img.resize(size, Image.BICUBIC)         # bicubic as per Section 3.2
    return np.array(img, dtype=np.float32)


def save_grayscale(arr: np.ndarray, path: str) -> None:
    """
    Save a float32 grayscale array (H, W) as a PNG.
    Clips to [0, 255] before saving.
    """
    clipped = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(clipped, mode="L")
    img.save(path)


# ---------------------------------------------------------------------------
# Blurring  (Section 3.2, Fig 4)
# ---------------------------------------------------------------------------
def apply_gaussian_blur(img: np.ndarray, sigma: float = BLUR_SIGMA) -> np.ndarray:
    """
    Apply a Gaussian filter with the given *sigma* to a 2-D grayscale array.

    Matches paper:
        "Gaussian filters with a standard deviation of 0.5"
        "In the first level, the simulation uses a Gaussian filter with
         a 0.5 standard deviation value. And in the second level, the
         blurred image has been obtained using the first level blurred
         image with the Gaussian filter and the same procedure has been
         continued till five levels."

    Returns
    -------
    np.ndarray of same shape and dtype=float32.
    """
    return gaussian_filter(img, sigma=sigma).astype(np.float32)


def generate_blur_levels(img: np.ndarray,
                          n_levels: int = N_BLUR_LEVELS,
                          sigma: float = BLUR_SIGMA) -> list[np.ndarray]:
    """
    Generate *n_levels* progressively blurred versions of *img* by applying
    the Gaussian filter sequentially (each level blurs the previous output).

    Returns
    -------
    list of n_levels np.ndarrays  [level1, level2, ..., level5]
    The original image is NOT included.
    """
    levels: list[np.ndarray] = []
    current = img.copy()
    for _ in range(n_levels):
        current = apply_gaussian_blur(current, sigma=sigma)
        levels.append(current.copy())
    return levels


# ---------------------------------------------------------------------------
# Patch extraction  (Section 3.2, Table 3)
# ---------------------------------------------------------------------------
def extract_patches_no_overlap(img: np.ndarray,
                                patch_size: int = PATCH_SIZE) -> list[np.ndarray]:
    """
    Crop *img* into non-overlapping *patch_size* × *patch_size* patches.

    Paper: "Each image is cropped into patches of size 16 × 16 without
            overlapping using bicubic transformation."

    Only complete patches are kept (border pixels that don't fill a full
    patch are discarded, matching standard CNN patch extraction practice).

    Returns
    -------
    list of np.ndarrays, each of shape (patch_size, patch_size), dtype=float32.
    """
    H, W = img.shape[:2]
    patches: list[np.ndarray] = []
    for r in range(0, H - patch_size + 1, patch_size):
        for c in range(0, W - patch_size + 1, patch_size):
            patch = img[r:r + patch_size, c:c + patch_size]
            patches.append(patch.astype(np.float32))
    return patches


# ---------------------------------------------------------------------------
# Data augmentation  (Section 3.2)
# ---------------------------------------------------------------------------
def augment_patch(patch: np.ndarray) -> list[np.ndarray]:
    """
    Return augmented copies of *patch*.

    Paper: "Images are rotated by 90° and 180° in both horizontal and
            vertical directions. Data augmentation is done to increase
            the dataset size."

    Returns 5 variants:
      [original, rot90, rot180, horizontal_flip, vertical_flip]
    Each shape: (patch_size, patch_size), dtype preserved.
    """
    rot90  = np.rot90(patch, k=1)           # 90° counter-clockwise
    rot180 = np.rot90(patch, k=2)           # 180°
    hflip  = np.fliplr(patch)               # horizontal flip (left-right)
    vflip  = np.flipud(patch)               # vertical flip (up-down)
    return [patch, rot90, rot180, hflip, vflip]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Normalize a patch to [0, 1] range."""
    return patch / 255.0


def image_pairs_in_dir(directory: str) -> list[tuple[str, str]]:
    """
    Discover (A, B) image pairs inside *directory*.

    Convention used in this pipeline:
        pair_NNN/A.png   — focused source image A
        pair_NNN/B.png   — defocused source image B

    Returns list of (path_A, path_B) sorted by pair index.
    """
    pairs: list[tuple[str, str]] = []
    for entry in sorted(os.listdir(directory)):
        pair_dir = os.path.join(directory, entry)
        if not os.path.isdir(pair_dir):
            continue
        path_a = os.path.join(pair_dir, "A.png")
        path_b = os.path.join(pair_dir, "B.png")
        if os.path.exists(path_a) and os.path.exists(path_b):
            pairs.append((path_a, path_b))
    return pairs


def image_pairs_with_gt(directory: str) -> list[tuple[str, str, str]]:
    """
    Like image_pairs_in_dir but also returns the ground-truth path.

    Convention:
        pair_NNN/A.png
        pair_NNN/B.png
        pair_NNN/GT.png
    """
    triples: list[tuple[str, str, str]] = []
    for entry in sorted(os.listdir(directory)):
        pair_dir = os.path.join(directory, entry)
        if not os.path.isdir(pair_dir):
            continue
        path_a  = os.path.join(pair_dir, "A.png")
        path_b  = os.path.join(pair_dir, "B.png")
        path_gt = os.path.join(pair_dir, "GT.png")
        if os.path.exists(path_a) and os.path.exists(path_b):
            triples.append((path_a, path_b,
                            path_gt if os.path.exists(path_gt) else ""))
    return triples
