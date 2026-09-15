"""
State classifier model.

Frozen DINOv3 ViT-L backbone + small MLP head. Predicts state
(NSW/Tas/Vic/WA) from a single 1024x1024 image using the same
preprocessing as the biomass model (typically the left half of the L/R
split).

Used by ``postprocessing.py`` to gate per-state rules: per-state scaling,
WA->Dead=0 and NSW->Clover=0 hard rules, and per-state max clipping.

Classifying state from a pasture image is easy — the four Australian
states have visually distinct pasture appearances (species mix, colour
palette, soil colour). This is essentially a linear probe on top of
pretrained DINOv3 features and reaches ~97-98% CV accuracy on this
dataset.
"""

import torch
import torch.nn as nn

from config import STATE_LABELS, N_STATES, DINOV3_DIM


class StateClassifier(nn.Module):
    """
    Frozen DINOv3 backbone + small classifier head.

    Input: ``(B, 3, H, W)`` — one image per sample (typically the left
    half at 1024x1024).
    Output: ``(B, N_STATES)`` — logits. Apply softmax to get probabilities.
    """

    def __init__(self, backbone, freeze_backbone: bool = True):
        """
        Args:
            backbone: a timm-style feature extractor whose output is a
                pooled ``(B, DINOV3_DIM)`` feature vector.
            freeze_backbone: if True (default), freeze the backbone
                parameters and keep it in eval mode so BatchNorm/Dropout
                statistics don't drift during head training.
        """
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(DINOV3_DIM),
            nn.Linear(DINOV3_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, N_STATES),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, x):
        """
        Compute logits for the input batch.

        Always runs the backbone in the appropriate no-grad / eval-mode
        context when frozen, even during training of the head, so the
        frozen weights don't accumulate gradients or receive updates
        from dropout etc.
        """
        if not self.training:
            feat = self.backbone(x)
        else:
            with torch.no_grad():
                feat = self.backbone(x)
        return self.head(feat)


def state_idx_to_name(idx: int) -> str:
    """Map a state class index to its canonical name."""
    return STATE_LABELS[int(idx)]


def state_name_to_idx(name: str) -> int:
    """Map a canonical state name to its class index."""
    return STATE_LABELS.index(name)
