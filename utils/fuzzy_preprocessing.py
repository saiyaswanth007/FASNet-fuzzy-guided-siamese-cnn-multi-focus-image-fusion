"""
utils/fuzzy_preprocessing.py
==============================
S-shaped fuzzy membership fuzzification of grayscale images, exactly as
described here:


─────────────────────────────────────────────────────────────────────────────
Parameters (computed per image P):
  γ'  = Entropy(P)
  χ'  = median(P)
  ν   = (max(P) + mean(P)) / 2

Fuzzification (S-shaped membership function):
  μ(P(i,j)) =
    0,                                          if 0 < μ ≤ γ'
    (μ - γ')² / (χ' - γ')(ν - γ'),             if γ' < μ ≤ χ'
    1 - (μ - γ')² / (ν - χ')(ν - γ'),          if χ' < μ ≤ ν
    1,                                          if μ > ν

where μ(P(i,j)) ∈ [0,1] is the pixel value first normalized from [0,255]→[0,1].
─────────────────────────────────────────────────────────────────────────────

Usage:
    from utils.fuzzy_preprocessing import fuzzify_image, fuzzify_pair

    A_fuzz = fuzzify_image(A_gray)   # input: (H,W) uint8 or float32
    B_fuzz = fuzzify_image(B_gray)   # output: (H,W) float32 in [0,1]
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Parameter computation
# ---------------------------------------------------------------------------
def _compute_gamma(img_normalized: np.ndarray) -> float:
    """
    γ' = Entropy(P)

    Computed on the normalized [0,1] image using the Shannon entropy of
    the pixel intensity histogram (256 bins).

    We use natural log (nats), which
    is the standard for scipy.stats.entropy.
    """
    # Histogram over 256 bins on [0,1]
    hist, _ = np.histogram(img_normalized.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist / (hist.sum() + 1e-12)          # normalise to probability
    h = scipy_entropy(hist + 1e-12)              # Shannon entropy (nats)
    # Clamp to [0,1] so it can serve as a fuzzy threshold
    return float(np.clip(h / np.log(256), 0.0, 1.0))


def _compute_chi(img_normalized: np.ndarray) -> float:
    """χ' = median(P)"""
    return float(np.median(img_normalized))


def _compute_nu(img_normalized: np.ndarray) -> float:
    """ν = (max(P) + mean(P)) / 2"""
    m1 = float(img_normalized.max())
    m2 = float(img_normalized.mean())
    return (m1 + m2) / 2.0


# ---------------------------------------------------------------------------
# S-shaped membership function
# ---------------------------------------------------------------------------
def _s_membership(mu: np.ndarray,
                  gamma: float,
                  chi: float,
                  nu: float) -> np.ndarray:
    """
    Apply the S-shaped membership function pixel-wise.

    Parameters
    ----------
    mu    : (H, W) float32 array, values in [0, 1]
    gamma : γ'   lower crossover
    chi   : χ'   midpoint
    nu    : ν    upper crossover

    Returns
    -------
    (H, W) float32 array in [0, 1]
    """
    out = np.zeros_like(mu, dtype=np.float32)

    # Guard against degenerate parameters (e.g. uniform image)
    denom_lower = (chi - gamma) * (nu - gamma) + 1e-12
    denom_upper = (nu - chi)   * (nu - gamma) + 1e-12

    # Region 1: μ ≤ γ'  →  0
    mask1 = mu <= gamma
    out[mask1] = 0.0

    # Region 2: γ' < μ ≤ χ'  →  (μ - γ')² / [(χ' - γ')(ν - γ')]
    mask2 = (mu > gamma) & (mu <= chi)
    out[mask2] = ((mu[mask2] - gamma) ** 2) / denom_lower

    # Region 3: χ' < μ ≤ ν  →  1 - (μ - γ')² / [(ν - χ')(ν - γ')]
    mask3 = (mu > chi) & (mu <= nu)
    out[mask3] = 1.0 - ((mu[mask3] - gamma) ** 2) / denom_upper

    # Region 4: μ > ν  →  1
    mask4 = mu > nu
    out[mask4] = 1.0

    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fuzzify_image(img: np.ndarray) -> np.ndarray:
    """
    Fuzzify a single grayscale image using the S-shaped membership function.

    Steps:
      1. Normalise pixel values [0,255] → [0,1]
      2. Compute γ' (entropy), χ' (median), ν (max+mean)/2
      3. Apply S-shaped membership function pixel-wise

    Parameters
    ----------
    img : (H, W) array, uint8 [0,255] or float32 [0,255] or float32 [0,1]

    Returns
    -------
    (H, W) float32 array in [0, 1]  — the fuzzified image A' or B'
    """
    # Normalise to [0, 1]
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    # Compute parameters
    gamma = _compute_gamma(arr)
    chi   = _compute_chi(arr)
    nu    = _compute_nu(arr)

    # Ensure ordering: gamma ≤ chi ≤ nu (degenerate case protection)
    if not (gamma <= chi <= nu):
        # Fall back to simple normalization if parameters are degenerate
        return arr.copy()

    # Apply S-shaped membership
    fuzz = _s_membership(arr, gamma, chi, nu)
    return fuzz


def fuzzify_pair(img_a: np.ndarray,
                 img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuzzify a multi-focus image pair (A, B) → (A', B').

    Each image is fuzzified independently with its own parameters,
    Each image is fuzzified independently with its own parameters.

    Parameters
    ----------
    img_a, img_b : (H, W) grayscale arrays [0,255] or [0,1]

    Returns
    -------
    (A_fuzz, B_fuzz) : tuple of (H,W) float32 arrays in [0,1]
    """
    return fuzzify_image(img_a), fuzzify_image(img_b)


def get_fuzzy_parameters(img: np.ndarray) -> dict[str, float]:
    """
    Return the three fuzzy parameters for an image (for inspection/logging).

    Returns
    -------
    {'gamma': γ', 'chi': χ', 'nu': ν}
    """
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return {
        "gamma": _compute_gamma(arr),
        "chi":   _compute_chi(arr),
        "nu":    _compute_nu(arr),
    }
