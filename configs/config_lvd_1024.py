"""
Configuration for the DINOv3 ViT-L @ 1024 baseline.

This is the primary shipping configuration. All 5-fold checkpoints in
``outputs/dinov3_lvd_1024/foldN/`` were trained under these settings.

To adapt to your local environment, edit the paths in the ``Paths`` block
below (or set the corresponding environment variables). Everything else is
tuned and should not be modified for reproducibility of the published results.
"""

import os
from pathlib import Path


# ============================================================================
# Paths — EDIT THESE FOR YOUR ENVIRONMENT
# ============================================================================
# Root of the Kaggle CSIRO Biomass competition data. Should contain
# `train.csv` and the `train/` image directory.
DATA_ROOT = Path(os.environ.get(
    "CSIRO_DATA_ROOT",
    "./data/csiro-biomass",
))

TRAIN_CSV = DATA_ROOT / "train.csv"
IMAGE_DIR = DATA_ROOT

# Where training artifacts (checkpoints, logs, OOF predictions) are written.
OUTPUT_DIR = Path(os.environ.get("CSIRO_OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Path to a locally-downloaded DINOv3 ViT-L LVD-1689M checkpoint. If you
# cannot download the weights ahead of time, set this to a directory and
# modify model.py to allow ``timm.create_model(..., pretrained=True)`` for
# automatic download instead.
DINOV3_WEIGHTS_LOCAL = Path(os.environ.get(
    "DINOV3_WEIGHTS",
    "./weights/dinov3/vit_large_patch16_dinov3_lvd1689m.pth",
))


# ============================================================================
# Model
# ============================================================================
BACKBONE = "vit_large_patch16_dinov3.lvd1689m"
DINOV3_DIM = 1024
HEAD_HIDDEN = 512


# ============================================================================
# Image processing
# ============================================================================
IMG_SIZE = 1024


# ============================================================================
# Cross-validation
# ============================================================================
# StratifiedGroupKFold stratifies on State and groups on Sampling_Date to
# prevent date-based leakage across folds. Seed found via search_seeds.py
# for maximum per-fold balance on the competition-weighted metric.
CV_SEED = 6821
N_FOLDS = 5


# ============================================================================
# Training
# ============================================================================
TRAIN_SEED = 42

# Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 8.
# At 1024 on a 24GB GPU with gradient checkpointing + bf16, batch 2 fits.
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
NUM_WORKERS = 4

# Two-stage training: freeze backbone for the first EPOCHS_FROZEN epochs
# (head-only warmup), then unfreeze for the remaining EPOCHS_UNFROZEN.
EPOCHS_FROZEN = 5
EPOCHS_UNFROZEN = 22
EPOCHS_TOTAL = EPOCHS_FROZEN + EPOCHS_UNFROZEN

LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 0.02


# ============================================================================
# Output organization
# ============================================================================
# Each config writes to outputs/{RUN_NAME}/foldN/. Different experiments
# should use different RUN_NAMEs to keep artifacts cleanly separated.
RUN_NAME = "dinov3_lvd_1024"


# ============================================================================
# Targets
# ============================================================================
# The model predicts the three atomic targets; GDM and Dry_Total are derived
# from those atomics via:
#     GDM_g       = Dry_Green_g + Dry_Clover_g
#     Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g
ATOMIC_TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]
DERIVED_TARGETS = ["GDM_g", "Dry_Total_g"]
ALL_TARGETS = ATOMIC_TARGETS + DERIVED_TARGETS

# Competition metric weights. Total dominates (0.5) since it captures the
# summed prediction quality; GDM (0.2) captures the live-plant mass; each
# atomic gets 0.1.
TARGET_WEIGHTS = {
    "Dry_Green_g":  0.1,
    "Dry_Dead_g":   0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g":        0.2,
    "Dry_Total_g":  0.5,
}


# ============================================================================
# States — kept for postprocessing gates and fold reporting
# ============================================================================
STATE_LABELS = ["NSW", "Tas", "Vic", "WA"]
N_STATES = len(STATE_LABELS)


# ============================================================================
# Augmentation
# ============================================================================
# Spatial augmentation — applied to the full 2:1 image before L/R split.
USE_RANDOM_RESIZED_CROP = True
RRC_PROB = 0.5
RRC_SCALE = (0.85, 1.0)

USE_SMALL_ROTATION = True
ROTATION_LIMIT_DEG = 5
ROTATION_PROB = 0.3

# Reflections and per-pixel color augmentation.
P_HFLIP = 0.5
P_VFLIP = 0.5
P_BRIGHTNESS_CONTRAST = 0.5
P_HSV = 0.5
P_GAMMA = 0.3
P_CLAHE_OR_EQ = 0.15
P_SHARPEN_OR_EMBOSS = 0.15

# Heavier techniques — evaluated and found not to help on this small dataset.
USE_MIXUP = False
USE_CUTMIX = False
USE_RANDOM_ERASING = False


# ============================================================================
# Image cleaning
# ============================================================================
# The training set includes small orange-text timestamps in the corner of
# many images. Removed via clean.py to prevent the model from over-attending
# to date artifacts.
APPLY_TIMESTAMP_REMOVAL = True

# Trim this many pixels off each border before further processing (removes
# camera-lens vignette and border compression artifacts).
DEFAULT_EDGE_TRIM_PX = 5

# Images excluded from training due to visual quality issues (obstructions,
# out-of-focus captures, etc.) identified during manual inspection.
EXCLUDED_IMAGES = {
    "train/ID473494649.jpg",
    "train/ID1889150649.jpg",
}


# ============================================================================
# Inference-time test-time augmentation (TTA)
# ============================================================================
# 4 reflection views: identity + horizontal flip + vertical flip + both.
# Rotations were evaluated and found to hurt (edge effects from resizing).
USE_TTA = True
TTA_REFLECTIONS = True
