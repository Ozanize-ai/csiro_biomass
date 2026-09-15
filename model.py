"""
Biomass model architecture.

DINOv3 ViT-L backbone with an L/R split strategy: the left and right halves
of the image are passed through a shared backbone, features are summed, and
a 3-layer MLP head produces the 3 atomic biomass predictions (green, dead,
clover). Outputs are passed through softplus to enforce non-negativity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import config


def _make_3layer_head(in_dim: int, out_dim: int, hidden: int = config.HEAD_HIDDEN,
                      dropout: float = 0.1) -> nn.Module:
    """
    Build the regression head: LayerNorm -> Linear -> GELU -> Dropout
    -> Linear -> GELU -> Dropout -> Linear.

    The intermediate width tapers from ``hidden`` to ``hidden // 2`` before
    the final projection to ``out_dim``.
    """
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden // 2, out_dim),
    )


class BiomassModel(nn.Module):
    """
    DINOv3 ViT-L + L/R sum-pool + 3-layer regression head.

    Forward takes ``left`` and ``right`` tensors (each ``(B, 3, H, W)``)
    and returns ``(B, num_atomic_targets)`` non-negative predictions for
    the atomic targets [Dry_Green_g, Dry_Dead_g, Dry_Clover_g].
    """

    def __init__(self, backbone_name: str = config.BACKBONE,
                 num_atomic_targets: int = 3,
                 pretrained: bool = True,
                 local_weights_path=None):
        """
        Args:
            backbone_name: timm model name for the backbone.
            num_atomic_targets: number of scalar outputs from the head
                (default 3 for green/dead/clover).
            pretrained: load pretrained DINOv3 weights from
                ``config.DINOV3_WEIGHTS_LOCAL`` (or
                ``local_weights_path`` if provided). If False, backbone
                is randomly initialized.
            local_weights_path: override for the pretrained weights path.
        """
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        if pretrained:
            weights_path = local_weights_path or config.DINOV3_WEIGHTS_LOCAL
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"DINOv3 weights not found at {weights_path}. "
                    f"Update config.DINOV3_WEIGHTS_LOCAL or set the "
                    f"DINOV3_WEIGHTS environment variable."
                )
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing:
                print(f"WARNING: {len(missing)} missing keys: {missing[:3]}...")
            if unexpected:
                print(f"WARNING: {len(unexpected)} unexpected keys: {unexpected[:3]}...")

        self.feat_dim = config.DINOV3_DIM
        self.head = _make_3layer_head(self.feat_dim, num_atomic_targets)

    def freeze_backbone(self):
        """Set ``requires_grad=False`` on all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Set ``requires_grad=True`` on all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def enable_grad_checkpointing(self):
        """
        Turn on gradient checkpointing in the backbone (trades compute
        for lower activation memory). Required to fit ViT-L at 1024 in
        24GB VRAM at the training batch size.
        """
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=True)
            print("Enabled gradient checkpointing on backbone")

    def get_param_groups(self, lr_backbone: float, lr_head: float,
                         weight_decay: float):
        """
        Build layer-wise parameter groups for the optimizer.

        The backbone gets ``lr_backbone`` and the head gets ``lr_head``,
        both with the same weight decay. Typical usage: backbone LR is
        ~10x smaller than head LR to avoid destroying pretrained features.

        Args:
            lr_backbone: learning rate for backbone parameters.
            lr_head:     learning rate for head parameters.
            weight_decay: weight decay applied to both groups.

        Returns:
            list of two dicts suitable for passing to an ``AdamW``
            optimizer's ``params`` argument.
        """
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": self.head.parameters(),     "lr": lr_head,     "weight_decay": weight_decay},
        ]

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Run inference on paired left/right image halves.

        Args:
            left:  ``(B, 3, H, W)`` tensor of left halves.
            right: ``(B, 3, H, W)`` tensor of right halves.

        Returns:
            ``(B, num_atomic_targets)`` tensor of non-negative
            (post-softplus) atomic biomass predictions in grams.
        """
        feat_l = self.backbone(left)
        feat_r = self.backbone(right)
        feat = feat_l + feat_r
        return F.softplus(self.head(feat))


def build_model() -> BiomassModel:
    """Convenience constructor: returns ``BiomassModel()`` with defaults."""
    return BiomassModel()
