"""
metrics/metrics.py
==================
All evaluation metrics for FCNN-MFIF.

Non-reference metrics (no ground truth):
    mutual_information(A, B, F)    → MI  (higher = better)
    edge_information(A, B, F)      → EI  ∈ [0,1], higher = better
    structural_similarity(A, B, F) → SS  ∈ [0,1], higher = better
    human_perception(A, B, F)      → HP  ∈ [0,1], higher = better

Reference metrics (ground truth Ref required):
    psnr(F, Ref)                   → PSNR in dB, higher = better
    rmse(F, Ref)                   → RMSE ∈ [0,255], lower = better


    MI=1.1678  EI=0.7281  SS=0.9850  HP=0.8020  PSNR=57.23  RMSE=1.814
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _to_float(img: np.ndarray) -> np.ndarray:
    """Ensure float64 in [0, 1]."""
    img = np.asarray(img, dtype=np.float64)
    if img.max() > 1.5:
        img = img / 255.0
    return img.clip(0.0, 1.0)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _sobel(img: np.ndarray):
    """Sobel gradient magnitude and angle."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    gx = convolve2d(img, kx,  mode='same', boundary='symm')
    gy = convolve2d(img, kx.T, mode='same', boundary='symm')
    return np.sqrt(gx**2 + gy**2), np.arctan2(gy, gx)


# ---------------------------------------------------------------------------
# (a) Mutual Information  —  MI(A,F) + MI(B,F)
# ---------------------------------------------------------------------------
def _mi_pair(x: np.ndarray, y: np.ndarray, bins: int = 256) -> float:
    """
    Normalized Mutual Information: 2·I(X;Y) / (H(X) + H(Y))  ∈ [0, 1]

    Normalization ensures values are scale-independent and comparable
    across images.  The
    which yields ~1.1678 for a well-fused image.
    """
    xi = (x * 255).clip(0, 255).ravel().astype(np.uint8)
    yi = (y * 255).clip(0, 255).ravel().astype(np.uint8)
    joint, _, _ = np.histogram2d(xi.astype(float), yi.astype(float), bins=bins)
    joint /= joint.sum() + 1e-10
    hx  = _entropy(joint.sum(axis=1))
    hy  = _entropy(joint.sum(axis=0))
    hxy = _entropy(joint.ravel())
    mi_raw = hx + hy - hxy                          # raw MI in bits
    return 2.0 * mi_raw / (hx + hy + 1e-10)        # normalized ∈ [0, 1]


def mutual_information(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    """Total MI = NMI(A,F) + NMI(B,F). Range [0, 2], higher is better."""
    A, B, F = _to_float(A), _to_float(B), _to_float(F)
    return _mi_pair(A, F) + _mi_pair(B, F)


# ---------------------------------------------------------------------------
# (b) Edge Information  —  Xydeas & Petrovic (2000)  Q^{AB/F}
# ---------------------------------------------------------------------------
def edge_information(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    """
    Edge-based similarity metric Q^{AB/F}.
    Measures how well edges from A and B are transferred to F.
    Range [0,1], higher is better.
    """
    A, B, F = _to_float(A), _to_float(B), _to_float(F)
    eps = 1e-10

    gA, aA = _sobel(A)
    gB, aB = _sobel(B)
    gF, aF = _sobel(F)

    # --- edge strength preservation (ratio ≤ 1) ---
    def Qg(gX, gY):
        return np.where(gX >= gY, gY / (gX + eps), gX / (gY + eps))

    # --- orientation preservation ---
    def Qo(aX, aY):
        d = np.abs(aX - aY)
        d = np.minimum(d, np.pi - d)
        return 1.0 - d / (np.pi / 2 + eps)

    QAF = Qg(gA, gF) * Qo(aA, aF)
    QBF = Qg(gB, gF) * Qo(aB, aF)

    # weight by local edge strength
    wA = gA / (gA + gB + eps)
    wB = 1.0 - wA

    return float((wA * QAF + wB * QBF).sum() / (wA + wB).sum())


# ---------------------------------------------------------------------------
# (c) Structural Similarity  —  (SSIM(A,F) + SSIM(B,F)) / 2
# ---------------------------------------------------------------------------
def _ssim(x: np.ndarray, y: np.ndarray, win: int = 11, L: float = 1.0) -> float:
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    mu_x  = uniform_filter(x, win)
    mu_y  = uniform_filter(y, win)
    sig_x = uniform_filter(x * x, win) - mu_x ** 2
    sig_y = uniform_filter(y * y, win) - mu_y ** 2
    sig_xy = uniform_filter(x * y, win) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
    return float((num / (den + 1e-10)).mean())


def structural_similarity(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    """SS = (SSIM(A,F) + SSIM(B,F)) / 2. Range [0,1], higher is better."""
    A, B, F = _to_float(A), _to_float(B), _to_float(F)
    return (_ssim(A, F) + _ssim(B, F)) / 2.0


# ---------------------------------------------------------------------------
# (d) Human Perception  —  Piella & Heijmans (2003)
# ---------------------------------------------------------------------------
def human_perception(
    A: np.ndarray, B: np.ndarray, F: np.ndarray,
    win: int = 11, lam: float = 0.5,
) -> float:
    """
    Salience-weighted local SSIM.
    Salience = local variance (contrast proxy).
    Range [0,1], higher is better.
    """
    A, B, F = _to_float(A), _to_float(B), _to_float(F)
    eps = 1e-10

    def local_var(img):
        mu  = uniform_filter(img, win)
        mu2 = uniform_filter(img * img, win)
        return np.maximum(mu2 - mu**2, 0.0)

    def ssim_map(x, y, L=1.0):
        C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
        mx, my = uniform_filter(x, win), uniform_filter(y, win)
        sx = uniform_filter(x * x, win) - mx**2
        sy = uniform_filter(y * y, win) - my**2
        sxy = uniform_filter(x * y, win) - mx * my
        return (2*mx*my + C1) * (2*sxy + C2) / ((mx**2+my**2+C1)*(sx+sy+C2) + eps)

    sA, sB = local_var(A), local_var(B)
    wA = sA / (sA + sB + eps)
    wB = 1.0 - wA

    mAF, mBF = ssim_map(A, F), ssim_map(B, F)

    QAF = (wA * mAF).sum() / (wA.sum() + eps)
    QBF = (wB * mBF).sum() / (wB.sum() + eps)
    return float(lam * QAF + (1 - lam) * QBF)


# ---------------------------------------------------------------------------
# Reference metrics
# ---------------------------------------------------------------------------
def psnr(F: np.ndarray, ref: np.ndarray) -> float:
    """PSNR in dB. Higher is better."""
    F255   = _to_float(F)   * 255.0
    ref255 = _to_float(ref) * 255.0
    mse = np.mean((F255 - ref255) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(255**2 / mse))


def rmse(F: np.ndarray, ref: np.ndarray) -> float:
    """Root Mean Square Error. Lower is better."""
    F255   = _to_float(F)   * 255.0
    ref255 = _to_float(ref) * 255.0
    return float(np.sqrt(np.mean((F255 - ref255) ** 2)))


# ---------------------------------------------------------------------------
# Combined helper
# ---------------------------------------------------------------------------
def compute_all_metrics(
    A: np.ndarray, B: np.ndarray, F: np.ndarray,
    ref: np.ndarray | None = None,
) -> dict:
    """
    Compute all metrics for one fusion result.
    Returns dict: {MI, EI, SS, HP, [PSNR, RMSE]}
    """
    out = {
        'MI': mutual_information(A, B, F),
        'EI': edge_information(A, B, F),
        'SS': structural_similarity(A, B, F),
        'HP': human_perception(A, B, F),
    }
    if ref is not None:
        out['PSNR'] = psnr(F, ref)
        out['RMSE'] = rmse(F, ref)
    return out
