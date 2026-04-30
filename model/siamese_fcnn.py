"""
model/siamese_fcnn.py
======================
Siamese Convolutional Neural Network for Multi-Focus Image Fusion.

Exact architecture from Section 2.2 of:
  "A Fuzzy Convolutional Neural Network for Multi-Focus Image Fusion"
  Bhalla et al., JVCIR 2022.

Architecture (Table 2 / Fig. 2):
─────────────────────────────────────────────────────────────────────
  Two SHARED-weight branches (one per input patch):

  Branch (applied identically to patch_A' and patch_B'):
    Conv Block 1: Conv2D(1→64,  3×3, stride=1, pad=1) + BN + ReLU
    Conv Block 2: Conv2D(64→128, 3×3, stride=1, pad=1) + BN + ReLU
    MaxPool2D(2×2, stride=2)
    Conv Block 3: Conv2D(128→256, 3×3, stride=1, pad=1) + BN + ReLU
    Flatten  →  256 × 8 × 8 = 16,384 features

  Fusion:
    Concatenate(branch_A, branch_B)  →  32,768 features
    FC(32768 → 512) + ReLU
    FC(512   →   2)
    Softmax  →  [p(B_focused), p(A_focused)]
              index:    0            1

Input patches must be fuzzified (A', B') ∈ [0,1], shape (B,1,16,16).
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """
    Single convolutional block: Conv2D → BatchNorm → ReLU.

    Paper notation: N3kKs1  where K = number of feature maps.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)          # bias=False when using BN
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Shared-weight branch
# ---------------------------------------------------------------------------
class SharedBranch(nn.Module):
    """
    Single Siamese branch — processes one 16×16 patch.

    Architecture:
        Conv Block 1: 1  → 64  FM  (N3k64s1)
        Conv Block 2: 64 → 128 FM  (N3k128s1)
        MaxPool2D (2×2, stride=2)   → 128 × 8 × 8
        Conv Block 3: 128→ 256 FM  (N3k256s1)
        Flatten                     → 16,384-d vector
    """
    def __init__(self):
        super().__init__()
        self.block1   = ConvBlock(1,   64)
        self.block2   = ConvBlock(64,  128)
        self.maxpool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3   = ConvBlock(128, 256)
        self.flatten  = nn.Flatten()

        # Feature dimension: 256 channels × 8 × 8 spatial = 16,384
        self.out_features = 256 * 8 * 8   # = 16,384

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, 16, 16)  fuzzified patch, values ∈ [0, 1]

        Returns
        -------
        (B, 16384)  feature vector
        """
        x = self.block1(x)    # (B, 64,  16, 16)
        x = self.block2(x)    # (B, 128, 16, 16)
        x = self.maxpool(x)   # (B, 128,  8,  8)
        x = self.block3(x)    # (B, 256,  8,  8)
        x = self.flatten(x)   # (B, 16384)
        return x


# ---------------------------------------------------------------------------
# Full Siamese FCNN
# ---------------------------------------------------------------------------
class SiameseFCNN(nn.Module):
    """
    Siamese Fuzzy Convolutional Neural Network for patch-level focus
    classification.

    Both branches share IDENTICAL weights (implemented via a single
    SharedBranch instance called twice).

    Forward pass:
        patch_A', patch_B'  →  [p(A_focused), p(B_focused)]

    where:
        p(A_focused) > 0.5  →  predict patch_A is more in focus
        p(B_focused) > 0.5  →  predict patch_B is more in focus
    """

    def __init__(self):
        super().__init__()

        # Shared-weight feature extractor (called for both A and B)
        self.branch = SharedBranch()

        feat_dim = self.branch.out_features   # 16,384 per branch

        # Fusion: concatenated features → FC → Softmax
        self.fc1 = nn.Linear(feat_dim * 2, 512)   # 32,768 → 512
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.Linear(512, 2)              # 512    →   2

        # Xavier weight initialisation (paper Section 3.2)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier (Glorot) uniform initialisation for all Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,
                patch_a: torch.Tensor,
                patch_b: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patch_a : (B, 1, 16, 16)  fuzzified patch A', values ∈ [0,1]
        patch_b : (B, 1, 16, 16)  fuzzified patch B', values ∈ [0,1]

        Returns
        -------
        (B, 2)  softmax probabilities [p(A_focused), p(B_focused)]
        """
        feat_a = self.branch(patch_a)          # (B, 16384)
        feat_b = self.branch(patch_b)          # (B, 16384)

        fused = torch.cat([feat_a, feat_b], dim=1)   # (B, 32768)

        out = self.fc1(fused)                  # (B, 512)
        out = self.relu(out)
        out = self.fc2(out)                    # (B, 2)
        out = F.softmax(out, dim=1)            # (B, 2)
        # out[:,0] = p(label=0) = p(B_focused)
        # out[:,1] = p(label=1) = p(A_focused)  ← NLLLoss convention
        return out

    def extract_features(self,
                          patch_a: torch.Tensor,
                          patch_b: torch.Tensor) -> torch.Tensor:
        """Return concatenated features (before FC) — for debugging."""
        feat_a = self.branch(patch_a)
        feat_b = self.branch(patch_b)
        return torch.cat([feat_a, feat_b], dim=1)

    def predict_focus(self,
                       patch_a: torch.Tensor,
                       patch_b: torch.Tensor) -> torch.Tensor:
        """
        Return a scalar focus score ∈ [0,1] for each sample.
        Score → 1 means A is more focused.
        Score → 0 means B is more focused.

        Uses probs[:,1] because NLLLoss(label=1) trains probs[:,1]
        to represent p(A_focused).
        """
        probs = self.forward(patch_a, patch_b)
        return probs[:, 1]   # p(A_focused) — class index 1 = label 1
