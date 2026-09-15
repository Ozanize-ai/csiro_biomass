"""
Kaggle inference notebook for CSIRO Biomass.

This is a self-contained script that runs as a Kaggle notebook to produce
``submission.csv`` on the competition's test set. All model architecture,
preprocessing, and postprocessing logic is inlined here so the notebook
can run without importing any local project modules.

Pipeline:
    1. Biomass 5-fold ensemble inference at 1024 with 4-view reflection
       TTA. Predictions from all 5 folds are averaged per image.
       Cached to /tmp/atomic_predictions.npy.
    2. State classifier inference (both L and R halves, averaged) to
       predict the state (NSW/Tas/Vic/WA) of each test image. Cached
       to /tmp/state_predictions.npy and /tmp/state_probs.npy.
    3. Postprocessing: per-state scaling, asymmetric dead correction,
       WA->Dead=0 hard rule, NSW->Clover=0 hard rule, per-state and
       global max clipping. Rules gate on state-classifier confidence
       and per-target model confidence to protect rare outliers.
    4. Write submission.csv and a submission_no_postproc.csv baseline.

To iterate on postprocessing parameters cheaply:
    - Run the notebook once (steps 1-2 take ~35 min for the full pipeline)
    - Modify POSTPROC_CONFIG at the top of this file
    - Re-run: steps 1 and 2 read from /tmp cache, only step 3 re-runs

Required Kaggle inputs (attach as datasets):
    - Competition:  /kaggle/input/competitions/csiro-biomass/
    - DINOv3:       /kaggle/input/datasets/<user>/dinov3-vitl16-weights/
    - Biomass:      /kaggle/input/datasets/<user>/csiro-winning-ensemble/
                        fold_0_best.pth ... fold_4_best.pth
    - State clf:    /kaggle/input/datasets/<user>/csiro-state-clssifier/
                        state_classifier_best.pth

The state classifier checkpoint contains only the head state_dict. Its
backbone is the same DINOv3 ViT-L used by the biomass model — built fresh
from the DINOv3 weights and the state head attached at inference time.
"""

import gc
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Paths — Kaggle-side layout
# ============================================================================

COMP_DIR = Path("/kaggle/input/competitions/csiro-biomass")
TEST_DIR = COMP_DIR / "test"
SAMPLE_SUB_PATH = COMP_DIR / "sample_submission.csv"

DINOV3_WEIGHTS = Path("/kaggle/input/datasets/ozanize/dinov3-vitl16-weights/dinov3_vitl16.pth")

STATE_CLASSIFIER_CKPT = Path("/kaggle/input/datasets/ozanize/csiro-state-clssifier/state_classifier_best.pth")

OUTPUT_DIR = Path("/kaggle/working")

# Intermediate .npy files land in /tmp so Kaggle's submission UI doesn't
# see them as candidate output files.
CACHE_DIR = Path("/tmp")


# ============================================================================
# Biomass ensemble configuration
# ============================================================================

EXPERIMENTS = [
    {
        "name": "dinov3_lvd_1024",
        "img_size": 1024,
        "checkpoint_dir": Path("/kaggle/input/datasets/ozanize/csiro-winning-ensemble"),
        "batch_size_per_gpu": 4,
    },
]

CHECKPOINT_NAMES = [
    "fold_0_best.pth", "fold_1_best.pth", "fold_2_best.pth",
    "fold_3_best.pth", "fold_4_best.pth",
]


# ============================================================================
# Postprocessing configuration
#
# Every rule here is gated on state-classifier confidence AND a per-target
# safety threshold (max_pred) to protect rare outliers where the model's
# prediction should be trusted even if the state gate says otherwise.
# ============================================================================

POSTPROC_CONFIG = {
    # Per-state multiplicative scales applied to atomic predictions.
    # Values derived from an OOF sweep to correct per-state systematic
    # bias. Keyed by state name — must match STATE_LABELS order below.
    "per_state_scales": {
        "NSW": {"green": 1.03, "dead": 1.0,  "clover": 1.0},
        "Tas": {"green": 1.0,  "dead": 1.0,  "clover": 1.0},
        "Vic": {"green": 1.0,  "dead": 1.0,  "clover": 0.85},
        "WA":  {"green": 0.97, "dead": 0.80, "clover": 0.80},
    },

    # Universal clover multiplier (applied AFTER per-state scaling).
    # 1.0 = disabled.
    "clover_universal_mult": 1.0,

    # Asymmetric dead scaling: shrink small dead predictions further
    # (typically noise around zero) and boost large dead predictions
    # (typically underpredicted in training). Threshold values chosen
    # from OOF residual analysis.
    "dead_asymmetric_enabled": True,
    "dead_high_threshold": 20.0,
    "dead_high_mult": 1.1,
    "dead_low_threshold": 10.0,
    "dead_low_mult": 0.9,

    # WA -> Dead = 0 hard rule. Training WA samples all have dead=0
    # (max observed dead in WA = 0.00). Only zero if state classifier
    # is confident (>= min_prob) and the model's own dead prediction
    # is small (< max_pred, safety margin).
    "wa_dead_zero_enabled": True,
    "wa_dead_zero_min_prob": 0.75,
    "wa_dead_zero_max_pred": 10.0,

    # NSW -> Clover = 0 hard rule. Training NSW is 99% clover-zero
    # (74/75 samples). The one nonzero-clover NSW sample has clover=10.10,
    # so max_pred=5.0 protects against wrongly zeroing similar high-clover
    # NSW test samples (if any) — the model's prediction is trusted when
    # it exceeds the safety threshold.
    "nsw_clover_zero_enabled": True,
    "nsw_clover_zero_min_prob": 0.75,
    "nsw_clover_zero_max_pred": 5.0,

    # Clipping — per-state max clipping uses training-derived per-state
    # maxes; global max clipping applies training-set-wide maxes.
    "per_state_max_clip_enabled": True,
    "global_max_clip_enabled": True,
    "min_clip": 0.0,
}


# ============================================================================
# Constants
# ============================================================================

BACKBONE_NAME = "vit_large_patch16_dinov3.lvd1689m"
DEFAULT_EDGE_TRIM_PX = 5
HEAD_HIDDEN = 512
DINOV3_DIM = 1024
N_STATES = 4
STATE_LABELS = ["NSW", "Tas", "Vic", "WA"]

USE_TTA = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ATOMIC_TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]
ALL_TARGETS = ATOMIC_TARGETS + ["GDM_g", "Dry_Total_g"]

# Per-state training-set maxes used by the per_state_max_clip rule.
PER_STATE_MAX = {
    "NSW": {"Dry_Green_g": 157.98, "Dry_Dead_g": 83.84, "Dry_Clover_g": 10.10},
    "Tas": {"Dry_Green_g":  69.61, "Dry_Dead_g": 58.75, "Dry_Clover_g": 71.79},
    "Vic": {"Dry_Green_g":  89.74, "Dry_Dead_g": 38.86, "Dry_Clover_g": 67.90},
    "WA":  {"Dry_Green_g":  72.10, "Dry_Dead_g":  0.00, "Dry_Clover_g": 58.88},
}

# Global training-set maxes used by the global_max_clip rule.
GLOBAL_MAX = {
    "Dry_Green_g": 157.98,
    "Dry_Dead_g":   83.84,
    "Dry_Clover_g": 71.79,
}


# ============================================================================
# Image preprocessing
# ============================================================================

def load_and_preprocess_one(path: Path, img_size: int) -> np.ndarray:
    """
    Load, edge-trim, resize, normalize.

    Returns a ``(3, H, 2W)`` float32 numpy array in NCHW form (ready to
    be split into L/R halves).
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    d = DEFAULT_EDGE_TRIM_PX
    if d > 0:
        img = img[d:-d, d:-d]
    img = cv2.resize(img, (2 * img_size, img_size), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    img = np.transpose(img, (2, 0, 1))
    return img


class TestDataset(Dataset):
    """
    Yield ``{'left', 'right', 'image_id'}`` per test image, where left
    and right are the two ``(3, img_size, img_size)`` halves of the
    resized image.
    """

    def __init__(self, image_paths, img_size):
        self.image_paths = list(image_paths)
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel = self.image_paths[idx]
        full = load_and_preprocess_one(TEST_DIR / rel, self.img_size)
        left = torch.from_numpy(full[:, :, : self.img_size]).contiguous()
        right = torch.from_numpy(full[:, :, self.img_size:]).contiguous()
        return {"left": left, "right": right, "image_id": rel}


# ============================================================================
# Biomass model — mirrors model.BiomassModel; inlined for self-contained notebook
# ============================================================================

def _make_3layer_head(in_dim, out_dim, hidden=HEAD_HIDDEN, dropout=0.1):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, out_dim),
    )


class BiomassModel(nn.Module):
    """DINOv3 backbone + L/R sum-pool + 3-layer regression head."""

    def __init__(self, backbone, num_atomic=3):
        super().__init__()
        self.backbone = backbone
        self.head = _make_3layer_head(DINOV3_DIM, num_atomic)

    def forward(self, left, right):
        feat_l = self.backbone(left)
        feat_r = self.backbone(right)
        feat = feat_l + feat_r
        return F.softplus(self.head(feat))


# ============================================================================
# State classifier head — mirrors state_classifier_model.StateClassifier
# ============================================================================

class StateClassifierHead(nn.Module):
    """
    Small MLP that turns pooled DINOv3 features into state logits.

    Applied to features already produced by the shared backbone (unlike
    training, where the backbone is wrapped inside the classifier module).
    Keeping it separate here lets us reuse a single backbone forward pass
    for whatever we want.
    """

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(DINOV3_DIM),
            nn.Linear(DINOV3_DIM, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, N_STATES),
        )

    def forward(self, features):
        return self.head(features)


# ============================================================================
# Backbone loading (shared between biomass and state classifier)
# ============================================================================

def build_backbone():
    """Build a fresh DINOv3 ViT-L and load pretrained weights from disk."""
    import timm
    backbone = timm.create_model(BACKBONE_NAME, pretrained=False,
                                 num_classes=0, global_pool="avg")
    state = torch.load(DINOV3_WEIGHTS, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    target_keys = set(backbone.state_dict().keys())
    overlap = len(set(state.keys()) & target_keys)
    print(f"  DINOv3 weights: {overlap}/{len(target_keys)} keys overlap")
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print(f"  Loaded with {len(missing)} missing, {len(unexpected)} unexpected")
    return backbone


def build_biomass_model():
    """Build a BiomassModel with a fresh DINOv3 backbone."""
    backbone = build_backbone()
    return BiomassModel(backbone)


def load_fold_checkpoint(model, ckpt_path):
    """
    Load a fold checkpoint into an existing BiomassModel (or DataParallel
    wrapper). Fails loudly if more than 5 keys are missing after loading
    (a hard sanity check — architecture mismatch would produce many
    more missing keys).
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    target = model.module if isinstance(model, nn.DataParallel) else model
    target_keys = set(target.state_dict().keys())
    overlap = len(set(state.keys()) & target_keys)
    missing, unexpected = target.load_state_dict(state, strict=False)
    print(f"    Loaded {ckpt_path.name}: {overlap}/{len(target_keys)} matched, "
          f"{len(missing)} missing")
    if len(missing) > 5:
        raise RuntimeError(f"Fold checkpoint has {len(missing)} missing keys.")


# ============================================================================
# Step 1: Biomass inference (per-fold TTA -> ensemble average)
# ============================================================================

@torch.no_grad()
def biomass_predict_one_fold(model, loader, device):
    """
    Run one fold's model over the entire test set with 4-view reflection
    TTA (identity, hflip, vflip, both). Returns a dict mapping image_id
    to the averaged atomic prediction.
    """
    model.eval()
    all_preds = {}
    tta_specs = [(None, None), ([3], [3]), ([2], [2]), ([2, 3], [2, 3])]

    for batch_idx, batch in enumerate(loader):
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        cpu_sum = None
        for left_dims, right_dims in tta_specs:
            if left_dims is None:
                L, R = left, right
            else:
                L = torch.flip(left, dims=left_dims)
                R = torch.flip(right, dims=right_dims)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                p = model(L, R)
            p_cpu = p.float().cpu()
            cpu_sum = p_cpu if cpu_sum is None else cpu_sum + p_cpu
            del p, p_cpu, L, R

        pred = (cpu_sum / len(tta_specs)).numpy()
        for i, img_id in enumerate(batch["image_id"]):
            all_preds[img_id] = pred[i]
        del left, right, cpu_sum
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return all_preds


def run_biomass_experiment(experiment, test_image_paths, device, n_gpus):
    """
    Run one experiment (a 5-fold model group) over the test set: load
    each fold checkpoint in turn into the same BiomassModel, run TTA
    inference, accumulate the averaged per-image predictions.

    Returns a dict mapping image_id to the mean prediction across all
    5 folds.
    """
    name = experiment["name"]
    img_size = experiment["img_size"]
    ckpt_dir = experiment["checkpoint_dir"]
    batch_per_gpu = experiment["batch_size_per_gpu"]
    total_batch = batch_per_gpu * n_gpus

    print(f"\n{'='*60}")
    print(f"BIOMASS EXPERIMENT: {name}  (img_size={img_size}, batch={total_batch})")
    print(f"{'='*60}")

    test_ds = TestDataset(test_image_paths, img_size=img_size)
    test_loader = DataLoader(test_ds, batch_size=total_batch, shuffle=False,
                             num_workers=2, pin_memory=True)
    print(f"  {len(test_ds)} test images")

    print(f"  Building model with DINOv3 weights...")
    t0 = time.time()
    model = build_biomass_model().to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"  Wrapped in DataParallel across {n_gpus} GPUs")
    print(f"  Model build: {time.time()-t0:.1f}s")

    accumulated = None
    for fold_idx, ckpt_name in enumerate(CHECKPOINT_NAMES):
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Not found: {ckpt_path}")
        t_fold = time.time()
        print(f"\n  Fold {fold_idx} ({ckpt_name}):")
        load_fold_checkpoint(model, ckpt_path)
        fold_preds = biomass_predict_one_fold(model, test_loader, device)

        if accumulated is None:
            accumulated = {k: v.copy() for k, v in fold_preds.items()}
        else:
            for k, v in fold_preds.items():
                accumulated[k] += v
        print(f"    Fold {fold_idx} TTA inference: {time.time()-t_fold:.1f}s")
        del fold_preds
        gc.collect()
        torch.cuda.empty_cache()

    n_folds = len(CHECKPOINT_NAMES)
    for k in accumulated:
        accumulated[k] /= n_folds

    del model, test_loader, test_ds
    gc.collect()
    torch.cuda.empty_cache()
    return accumulated


# ============================================================================
# Step 2: State classifier inference (both halves averaged)
# ============================================================================

@torch.no_grad()
def predict_states(test_image_paths, device, n_gpus):
    """
    Run the state classifier on all test images. Predicts on both L and
    R halves separately and averages the softmax probabilities.

    Returns:
        state_idx: (N,) argmax state index per image.
        probs:     (N, N_STATES) averaged softmax probabilities.
        all_ids:   list of image_ids in the same order.
    """
    print(f"\n{'='*60}")
    print(f"STATE CLASSIFIER INFERENCE")
    print(f"{'='*60}")

    if not STATE_CLASSIFIER_CKPT.exists():
        raise FileNotFoundError(
            f"State classifier checkpoint not found: {STATE_CLASSIFIER_CKPT}\n"
            f"Train it locally with train_state_classifier.py and upload to Kaggle."
        )

    print("  Building state classifier (DINOv3 backbone + head)...")
    t0 = time.time()
    backbone = build_backbone()
    head = StateClassifierHead()

    ckpt = torch.load(STATE_CLASSIFIER_CKPT, map_location="cpu", weights_only=True)
    head.head.load_state_dict(ckpt["head_state_dict"])
    print(f"  Loaded state classifier head (CV acc: {ckpt.get('cv_accuracy_mean', 'N/A')})")

    backbone.eval()
    head.eval()
    backbone = backbone.to(device)
    head = head.to(device)
    if n_gpus > 1:
        backbone = nn.DataParallel(backbone)
    print(f"  Build: {time.time()-t0:.1f}s")

    # State classifier was trained at 1024 — keep at training resolution.
    img_size = 1024
    test_ds = TestDataset(test_image_paths, img_size=img_size)
    batch_size = 8 * n_gpus
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    all_probs_left = []
    all_probs_right = []
    all_ids = []
    t0 = time.time()
    for batch in test_loader:
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            features_l = backbone(left)
            features_r = backbone(right)
            logits_l = head(features_l.float())
            logits_r = head(features_r.float())
        probs_l = torch.softmax(logits_l.float(), dim=1).cpu().numpy()
        probs_r = torch.softmax(logits_r.float(), dim=1).cpu().numpy()
        all_probs_left.append(probs_l)
        all_probs_right.append(probs_r)
        all_ids.extend(batch["image_id"])
    print(f"  State inference (both halves): {time.time()-t0:.1f}s")

    probs_left = np.concatenate(all_probs_left, axis=0)
    probs_right = np.concatenate(all_probs_right, axis=0)
    probs = (probs_left + probs_right) / 2.0
    state_idx = probs.argmax(axis=1)

    state_idx_l = probs_left.argmax(axis=1)
    state_idx_r = probs_right.argmax(axis=1)
    agreement = (state_idx_l == state_idx_r).mean()
    print(f"  L/R prediction agreement: {agreement:.3f} "
          f"({(state_idx_l == state_idx_r).sum()}/{len(state_idx)} images)")

    print(f"\n  Predicted state distribution:")
    for s_idx in range(N_STATES):
        n = (state_idx == s_idx).sum()
        avg_conf = probs[state_idx == s_idx, s_idx].mean() if n > 0 else 0.0
        print(f"    {STATE_LABELS[s_idx]}: {n:3d} images  (avg prob {avg_conf:.3f})")

    del backbone, head, test_loader, test_ds
    gc.collect()
    torch.cuda.empty_cache()

    return state_idx, probs, all_ids


# ============================================================================
# Step 3: Postprocessing
# ============================================================================

def postprocess_atomic(atomic, predicted_states, state_probs, config):
    """
    Apply the postprocessing pipeline to atomic biomass predictions.

    Rules applied in order:
      1. Per-state multiplicative scaling.
      2. Universal clover multiplier (if != 1.0).
      3. Asymmetric dead correction (low mult below threshold, high mult
         above threshold).
      4. WA -> Dead = 0 hard rule (gated by state confidence + max_pred).
      5. NSW -> Clover = 0 hard rule (gated by state confidence + max_pred).
      6. Per-state max clipping.
      7. Global max clipping.
      8. Min clipping (typically 0.0).

    Args:
        atomic:           (N, 3) predicted atomic values.
        predicted_states: (N,) array of state name strings.
        state_probs:      (N, N_STATES) softmax probs from state classifier.
        config:           dict with rule enables and parameters.

    Returns:
        (N, 3) postprocessed atomic predictions.
    """
    out = atomic.astype(np.float32).copy()

    print(f"\n=== Postprocessing ({len(out)} predictions) ===")
    _print_summary(out, "Before")

    for state_name, scales in config["per_state_scales"].items():
        mask = predicted_states == state_name
        if mask.any():
            out[mask, 0] *= scales.get("green", 1.0)
            out[mask, 1] *= scales.get("dead", 1.0)
            out[mask, 2] *= scales.get("clover", 1.0)
    _print_summary(out, "After per-state scaling")

    if config["clover_universal_mult"] != 1.0:
        out[:, 2] *= config["clover_universal_mult"]
        _print_summary(out, f"After clover x {config['clover_universal_mult']}")

    if config["dead_asymmetric_enabled"]:
        high_thresh = config["dead_high_threshold"]
        low_thresh = config["dead_low_threshold"]
        out[out[:, 1] > high_thresh, 1] *= config["dead_high_mult"]
        out[out[:, 1] < low_thresh, 1] *= config["dead_low_mult"]
        _print_summary(out, "After dead asymmetric")

    if config["wa_dead_zero_enabled"]:
        wa_idx = STATE_LABELS.index("WA")
        wa_prob = state_probs[:, wa_idx]
        max_pred = config.get("wa_dead_zero_max_pred", float("inf"))
        mask = (
            (predicted_states == "WA")
            & (wa_prob >= config["wa_dead_zero_min_prob"])
            & (out[:, 1] < max_pred)
        )
        n_zeroed = mask.sum()
        out[mask, 1] = 0.0
        print(f"  WA->Dead=0 applied to {n_zeroed} samples (max_pred={max_pred})")
        _print_summary(out, "After WA->Dead=0")

    if config.get("nsw_clover_zero_enabled", False):
        nsw_idx = STATE_LABELS.index("NSW")
        nsw_prob = state_probs[:, nsw_idx]
        max_pred = config.get("nsw_clover_zero_max_pred", float("inf"))
        mask = (
            (predicted_states == "NSW")
            & (nsw_prob >= config["nsw_clover_zero_min_prob"])
            & (out[:, 2] < max_pred)
        )
        n_zeroed = mask.sum()
        out[mask, 2] = 0.0
        print(f"  NSW->Clover=0 applied to {n_zeroed} samples (max_pred={max_pred})")
        _print_summary(out, "After NSW->Clover=0")

    if config["per_state_max_clip_enabled"]:
        for state_name, maxes in PER_STATE_MAX.items():
            mask = predicted_states == state_name
            if mask.any():
                out[mask, 0] = np.minimum(out[mask, 0], maxes["Dry_Green_g"])
                out[mask, 1] = np.minimum(out[mask, 1], maxes["Dry_Dead_g"])
                out[mask, 2] = np.minimum(out[mask, 2], maxes["Dry_Clover_g"])
        _print_summary(out, "After per-state max clip")

    if config["global_max_clip_enabled"]:
        out[:, 0] = np.minimum(out[:, 0], GLOBAL_MAX["Dry_Green_g"])
        out[:, 1] = np.minimum(out[:, 1], GLOBAL_MAX["Dry_Dead_g"])
        out[:, 2] = np.minimum(out[:, 2], GLOBAL_MAX["Dry_Clover_g"])
        _print_summary(out, "After global max clip")

    if config["min_clip"] is not None:
        out = np.maximum(out, config["min_clip"])

    _print_summary(out, "FINAL")
    return out


def _print_summary(atomic, label):
    """Print a mean/std/min/max summary of the three atomic columns."""
    g, d, c = atomic[:, 0], atomic[:, 1], atomic[:, 2]
    print(f"  {label}:")
    print(f"    Green:  mean={g.mean():6.2f}  std={g.std():5.2f}  "
          f"min={g.min():5.2f}  max={g.max():6.2f}")
    print(f"    Dead:   mean={d.mean():6.2f}  std={d.std():5.2f}  "
          f"min={d.min():5.2f}  max={d.max():6.2f}")
    print(f"    Clover: mean={c.mean():6.2f}  std={c.std():5.2f}  "
          f"min={c.min():5.2f}  max={c.max():6.2f}")


def derive_5d(atomic):
    """
    Derive the 5-target output from 3 atomic predictions.

    Args:
        atomic: (N, 3) atomic predictions [Green, Dead, Clover].

    Returns:
        (N, 5) array: [Green, Dead, Clover, GDM=Green+Clover,
        Total=Green+Dead+Clover].
    """
    n = atomic.shape[0]
    out = np.zeros((n, 5), dtype=np.float32)
    out[:, 0] = atomic[:, 0]
    out[:, 1] = atomic[:, 1]
    out[:, 2] = atomic[:, 2]
    out[:, 3] = atomic[:, 0] + atomic[:, 2]
    out[:, 4] = atomic[:, 0] + atomic[:, 1] + atomic[:, 2]
    return out


# ============================================================================
# Submission writing
# ============================================================================

def write_submission(image_ids, predictions_5d, sub_path):
    """
    Write a submission CSV in the competition's long format: one row per
    (image, target) pair with columns ``sample_id`` and ``target``.
    """
    rows = []
    for i, rel_path in enumerate(image_ids):
        img_id = Path(rel_path).stem
        for j, tgt in enumerate(ALL_TARGETS):
            rows.append({
                "sample_id": f"{img_id}__{tgt}",
                "target": float(predictions_5d[i, j]),
            })
    sub = pd.DataFrame(rows)
    sub.to_csv(sub_path, index=False)
    print(f"\nWrote {sub_path} ({len(sub)} rows)")
    print(f"  Per-target stats:")
    for j, tgt in enumerate(ALL_TARGETS):
        col = predictions_5d[:, j]
        print(f"    {tgt:15s}: mean={col.mean():6.2f}  std={col.std():5.2f}  "
              f"min={col.min():5.2f}  max={col.max():6.2f}")


# ============================================================================
# Main
# ============================================================================

def main():
    overall_t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = max(1, torch.cuda.device_count() if device == "cuda" else 0)
    print(f"Device: {device}, n_gpus={n_gpus}")
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory/1e9:.1f}GB")

    # Assemble the test image list from sample_submission if available
    # (guarantees order matches), else scan the test directory.
    if SAMPLE_SUB_PATH.exists():
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
        ids = sample_sub["sample_id"].str.split("__").str[0].drop_duplicates().tolist()
        test_image_paths = []
        for img_id in ids:
            for cand in [f"{img_id}.jpg", f"{img_id}.JPG", f"{img_id}.jpeg"]:
                if (TEST_DIR / cand).exists():
                    test_image_paths.append(cand)
                    break
            else:
                raise FileNotFoundError(f"No image for {img_id}")
        print(f"Found {len(test_image_paths)} test images via sample_submission")
    else:
        test_image_paths = sorted([p.name for p in TEST_DIR.iterdir()
                                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        print(f"Found {len(test_image_paths)} test images via dir listing")

    # ========================================================================
    # STEP 1: Biomass inference (cached to /tmp on re-runs)
    # ========================================================================
    atomic_path = CACHE_DIR / "atomic_predictions.npy"
    ids_path = CACHE_DIR / "image_ids.npy"

    if atomic_path.exists() and ids_path.exists():
        print(f"\n[Step 1] Loading cached biomass predictions: {atomic_path}")
        atomic = np.load(atomic_path)
        image_ids = np.load(ids_path, allow_pickle=True).tolist()
        print(f"  Loaded {len(image_ids)} cached predictions")
    else:
        print(f"\n[Step 1] Running biomass inference...")
        per_experiment_preds = []
        for experiment in EXPERIMENTS:
            preds = run_biomass_experiment(experiment, test_image_paths, device, n_gpus)
            per_experiment_preds.append(preds)
            print(f"  Total elapsed: {(time.time()-overall_t0)/60:.1f} min")

        n_exp = len(per_experiment_preds)
        if n_exp == 1:
            final_preds = per_experiment_preds[0]
        else:
            final_preds = {}
            sample_keys = list(per_experiment_preds[0].keys())
            for k in sample_keys:
                stacked = np.stack([exp[k] for exp in per_experiment_preds], axis=0)
                final_preds[k] = stacked.mean(axis=0)

        image_ids = list(test_image_paths)
        atomic = np.stack([final_preds[img_id] for img_id in image_ids], axis=0)

        np.save(atomic_path, atomic)
        np.save(ids_path, np.array(image_ids, dtype=object))
        print(f"  Saved {atomic_path} (shape {atomic.shape})")

    # ========================================================================
    # STEP 2: State classifier inference (cached to /tmp on re-runs)
    # ========================================================================
    state_idx_path = CACHE_DIR / "state_predictions.npy"
    state_probs_path = CACHE_DIR / "state_probs.npy"

    if state_idx_path.exists() and state_probs_path.exists():
        print(f"\n[Step 2] Loading cached state predictions: {state_idx_path}")
        state_idx = np.load(state_idx_path)
        state_probs = np.load(state_probs_path)
    else:
        print(f"\n[Step 2] Running state classifier...")
        state_idx, state_probs, state_ids = predict_states(test_image_paths, device, n_gpus)
        assert state_ids == image_ids, "Order mismatch between biomass and state predictions"
        np.save(state_idx_path, state_idx)
        np.save(state_probs_path, state_probs)
        print(f"  Saved {state_idx_path} and {state_probs_path}")

    predicted_states = np.array([STATE_LABELS[int(i)] for i in state_idx])

    # ========================================================================
    # STEP 3: Postprocessing + write submission(s)
    # ========================================================================
    print(f"\n[Step 3] Applying postprocessing...")
    atomic_pp = postprocess_atomic(atomic, predicted_states, state_probs, POSTPROC_CONFIG)
    predictions_5d = derive_5d(atomic_pp)

    sub_path = OUTPUT_DIR / "submission.csv"
    write_submission(image_ids, predictions_5d, sub_path)

    # Also write a no-postprocessing baseline for A/B comparison on the LB.
    baseline_5d = derive_5d(np.maximum(atomic, 0.0))
    baseline_path = OUTPUT_DIR / "submission_no_postproc.csv"
    print(f"\nWriting baseline (no postprocessing) submission for comparison:")
    write_submission(image_ids, baseline_5d, baseline_path)

    print(f"\n{'='*60}")
    print(f"DONE - total runtime {(time.time()-overall_t0)/60:.1f} min")
    print(f"{'='*60}")
    print(f"\nMain submission:    {sub_path}")
    print(f"No-postproc backup: {baseline_path}")


if __name__ == "__main__":
    main()
