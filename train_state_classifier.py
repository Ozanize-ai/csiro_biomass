"""
Train the state classifier used by inference-time postprocessing.

Standalone training: DINOv3 ViT-L backbone frozen, small MLP head trained
to predict state (NSW/Tas/Vic/WA). The saved head is loaded by
``kaggle_inference.py`` and used to gate the per-state postprocessing
rules (per-state scales, WA->Dead=0, NSW->Clover=0, per-state max clips).

CV strategy: StratifiedKFold on ``State`` ONLY — no date grouping.
The biomass CV uses StratifiedGroupKFold with date-grouping to prevent
biomass-amount leakage, but that grouping hurts state classification:
WA has only ~3-4 distinct dates, so date-grouping pulls all WA samples
from a single date into the same fold and the model never trains on
visually similar WA samples to those in validation. A per-state split
gives each fold ~6 WA samples drawn from ALL dates, with training
seeing visually similar WA from overlapping dates.

Usage (Windows or Linux):
    python train_state_classifier.py

Outputs:
    outputs/state_classifier/state_classifier_best.pth
    outputs/state_classifier/training_log.txt

Run time on RTX 3090: ~10 minutes (frozen backbone, small head, 20 epochs).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows OpenMP workaround

import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

import config
from clean import remove_timestamp_orange
from dataset import IMAGENET_MEAN, IMAGENET_STD, load_manual_crops
from state_classifier_model import StateClassifier
from config import STATE_LABELS, N_STATES


OUTPUT_DIR = config.OUTPUT_DIR / "state_classifier"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CKPT_PATH = OUTPUT_DIR / "state_classifier_best.pth"
LOG_PATH = OUTPUT_DIR / "training_log.txt"


# ============================================================================
# Dataset — image-level (single half per sample, not L/R paired like biomass)
# ============================================================================

class StateDataset(Dataset):
    """
    Return ``(image_tensor, state_idx)`` for state classification.

    During training (``training_mode=True``): randomly picks the LEFT or
    RIGHT half of the resized image. This gives 2x effective variety per
    epoch — the same image is sampled at different halves on different
    epochs.

    During eval (``training_mode=False``): always uses the LEFT half
    (deterministic).

    Preprocessing mirrors the biomass dataset: universal 5px trim,
    optional manual per-image crop, timestamp removal, resize to
    ``(2*img_size, img_size)``, then take one half.
    """

    def __init__(self, df, image_dir, transform, img_size=1024,
                 edge_trim_px=5, apply_timestamp_removal=True,
                 manual_crops=None, training_mode=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.img_size = img_size
        self.edge_trim_px = edge_trim_px
        self.apply_timestamp_removal = apply_timestamp_removal
        self.manual_crops = manual_crops or {}
        self.training_mode = training_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["image_path"]

        img = cv2.imread(str(self.image_dir / rel_path))
        if img is None:
            raise FileNotFoundError(f"Could not read {rel_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        d = self.edge_trim_px
        if d > 0:
            img = img[d:-d, d:-d]

        if rel_path in self.manual_crops:
            t, b, l, r = self.manual_crops[rel_path]
            h, w = img.shape[:2]
            img = img[t:h-b if b > 0 else h, l:w-r if r > 0 else w]

        if self.apply_timestamp_removal:
            img = remove_timestamp_orange(img)

        img = cv2.resize(img, (2 * self.img_size, self.img_size),
                         interpolation=cv2.INTER_CUBIC)

        if self.training_mode and np.random.random() < 0.5:
            half = img[:, self.img_size:]
        else:
            half = img[:, :self.img_size]

        tensor = self.transform(image=half)["image"]
        state_idx = STATE_LABELS.index(row["State"])
        return tensor, state_idx


def get_train_transform():
    """
    Training augmentation for state classification.

    Slightly heavier colour augmentation than biomass training because
    state-distinguishing features are largely colour and texture, and we
    want the classifier to be robust to lighting variation rather than
    learn spurious colour cues.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=20, p=0.4),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15,
                      hue=0.05, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transform():
    """Deterministic eval transform: normalize + to tensor. No augmentation."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ============================================================================
# Data loading
# ============================================================================

def make_state_table():
    """
    Load train.csv and produce one row per image with state metadata.

    The raw CSV is in long format (one row per target); this function
    dedupes on image and keeps just the metadata columns needed for
    state classification.

    Returns:
        DataFrame with columns ``image_id``, ``image_path``, ``State``,
        ``Sampling_Date`` — one row per non-excluded image.
    """
    df = pd.read_csv(config.TRAIN_CSV)
    df["image_id"] = df["sample_id"].str.split("__").str[0]

    metadata_cols = ["image_id", "image_path", "State", "Sampling_Date"]
    df_meta = df[metadata_cols].drop_duplicates("image_id").reset_index(drop=True)

    df_meta = df_meta[~df_meta["image_path"].isin(config.EXCLUDED_IMAGES)].reset_index(drop=True)

    return df_meta


# ============================================================================
# Backbone loading
# ============================================================================

def build_backbone():
    """
    Instantiate DINOv3 ViT-L via timm and load the local pretrained
    weights. Backbone is returned in whatever mode timm defaults to
    (typically train); the caller wraps it in ``StateClassifier`` which
    switches it to eval + no-grad when ``freeze_backbone=True``.
    """
    backbone = timm.create_model(config.BACKBONE, pretrained=False,
                                 num_classes=0, global_pool="avg")

    state = torch.load(config.DINOV3_WEIGHTS_LOCAL, map_location="cpu",
                       weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    target_keys = set(backbone.state_dict().keys())
    src_keys = set(state.keys())
    overlap = len(src_keys & target_keys)
    print(f"DINOv3 weights: {overlap}/{len(target_keys)} keys overlap")

    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print(f"  Loaded with {len(missing)} missing, {len(unexpected)} unexpected")

    return backbone


# ============================================================================
# Training loop
# ============================================================================

def train_one_fold(model, train_loader, val_loader, device,
                   epochs=15, lr=1e-3, weight_decay=1e-4,
                   class_weights=None, log_fn=print):
    """
    Train the head for ``epochs`` epochs. Backbone stays frozen (in eval mode).

    If ``val_loader`` is provided, evaluates each epoch and returns the
    best validation accuracy. If ``val_loader`` is None (final training
    on all data), just trains without evaluation.

    Args:
        model: StateClassifier instance.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation, or None to skip eval.
        device: CUDA device or "cpu".
        epochs: number of epochs to train.
        lr: learning rate for the head.
        weight_decay: weight decay for AdamW.
        class_weights: optional per-class weight tensor for CrossEntropyLoss
            (used to balance the imbalanced state distribution).
        log_fn: callable used for logging (defaults to print).

    Returns:
        (best_val_acc, best_epoch) — best accuracy and the epoch it was
        achieved in. When ``val_loader`` is None, returns (0.0, -1).
    """
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(epochs):
        # Train — keep backbone in eval mode even during head training
        # (so its BatchNorm/Dropout statistics don't drift)
        model.train()
        if hasattr(model, "backbone"):
            model.backbone.eval()

        total_loss = 0.0
        n_train = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            n_train += images.size(0)

        train_loss = total_loss / n_train

        if val_loader is not None:
            model.eval()
            correct = 0
            total = 0
            per_state_correct = [0] * N_STATES
            per_state_total = [0] * N_STATES
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(images)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    for s in range(N_STATES):
                        mask = (labels == s)
                        per_state_total[s] += mask.sum().item()
                        per_state_correct[s] += ((preds == labels) & mask).sum().item()

            val_acc = correct / total if total > 0 else 0.0
            per_state_acc = [
                f"{STATE_LABELS[s]}={per_state_correct[s]}/{per_state_total[s]}"
                for s in range(N_STATES) if per_state_total[s] > 0
            ]
            log_fn(f"  ep {epoch+1:2d}/{epochs}: train_loss={train_loss:.4f}  "
                   f"val_acc={val_acc:.4f}  ({', '.join(per_state_acc)})")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
        else:
            log_fn(f"  ep {epoch+1:2d}/{epochs}: train_loss={train_loss:.4f}")

        scheduler.step()

    return best_val_acc, best_epoch


# ============================================================================
# Main
# ============================================================================

def main():
    """
    Run 5-fold CV to measure classifier accuracy, then train a final
    head on ALL data and save the checkpoint.
    """
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== State Classifier Training ===")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    df = make_state_table()
    log(f"Loaded {len(df)} images")
    log("State distribution:")
    for state, count in df["State"].value_counts().sort_index().items():
        log(f"  {state}: {count}")

    manual_crops = load_manual_crops()
    log(f"Manual crops loaded: {len(manual_crops)} entries")

    # Class weights to counteract the WA (~32) vs Tas (~138) imbalance.
    state_counts = df["State"].value_counts().reindex(STATE_LABELS).values
    class_weights = torch.tensor(
        state_counts.sum() / (N_STATES * state_counts), dtype=torch.float32
    )
    log("Class weights: " + ", ".join(
        f"{s}={w:.2f}" for s, w in zip(STATE_LABELS, class_weights.tolist())
    ))

    # Build the backbone once and reuse it across all folds + final training.
    log("\nBuilding backbone...")
    backbone = build_backbone()

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    log("\n=== Cross-validation (5-fold StratifiedKFold on State only) ===")
    log("  (no date grouping — see module docstring for rationale)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.CV_SEED)
    fold_accs = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["State"])):
        log(f"\nFold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")
        val_states = df.iloc[val_idx]["State"].value_counts().to_dict()
        log("  Val state distribution: " + ", ".join(
            f"{s}={val_states.get(s, 0)}" for s in STATE_LABELS
        ))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_ds = StateDataset(
            train_df, image_dir=config.IMAGE_DIR,
            transform=get_train_transform(),
            img_size=config.IMG_SIZE,
            edge_trim_px=config.DEFAULT_EDGE_TRIM_PX,
            apply_timestamp_removal=config.APPLY_TIMESTAMP_REMOVAL,
            manual_crops=manual_crops,
            training_mode=True,
        )
        val_ds = StateDataset(
            val_df, image_dir=config.IMAGE_DIR,
            transform=get_eval_transform(),
            img_size=config.IMG_SIZE,
            edge_trim_px=config.DEFAULT_EDGE_TRIM_PX,
            apply_timestamp_removal=config.APPLY_TIMESTAMP_REMOVAL,
            manual_crops=manual_crops,
            training_mode=False,
        )
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                                  num_workers=config.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                                num_workers=config.NUM_WORKERS, pin_memory=True)

        model = StateClassifier(backbone, freeze_backbone=True).to(device)
        best_acc, _ = train_one_fold(model, train_loader, val_loader, device,
                                     epochs=20, class_weights=class_weights, log_fn=log)
        fold_accs.append(best_acc)
        log(f"  Fold {fold_idx} best val_acc: {best_acc:.4f}")

        del model
        torch.cuda.empty_cache()

    log(f"\nCV accuracy mean: {np.mean(fold_accs):.4f}, std: {np.std(fold_accs):.4f}")
    log("Per-fold: " + ", ".join(f"{a:.4f}" for a in fold_accs))

    # ------------------------------------------------------------------
    # Final training on ALL data, then save head weights
    # ------------------------------------------------------------------
    log("\n=== Final training on ALL data ===")

    full_ds = StateDataset(
        df, image_dir=config.IMAGE_DIR,
        transform=get_train_transform(),
        img_size=config.IMG_SIZE,
        edge_trim_px=config.DEFAULT_EDGE_TRIM_PX,
        apply_timestamp_removal=config.APPLY_TIMESTAMP_REMOVAL,
        manual_crops=manual_crops,
        training_mode=True,
    )
    full_loader = DataLoader(full_ds, batch_size=8, shuffle=True,
                             num_workers=config.NUM_WORKERS, pin_memory=True)

    model = StateClassifier(backbone, freeze_backbone=True).to(device)
    train_one_fold(model, full_loader, None, device, epochs=20,
                   class_weights=class_weights, log_fn=log)

    # Only the head is saved (backbone is shared with biomass model and
    # loaded fresh in kaggle_inference.py from DINOv3 weights).
    torch.save({
        "head_state_dict": model.head.state_dict(),
        "state_labels": STATE_LABELS,
        "cv_accuracy_mean": float(np.mean(fold_accs)),
        "cv_accuracy_per_fold": [float(a) for a in fold_accs],
    }, CKPT_PATH)
    log(f"\nSaved checkpoint: {CKPT_PATH}")
    log(f"  CV accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog saved: {LOG_PATH}")


if __name__ == "__main__":
    main()
