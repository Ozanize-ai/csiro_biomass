"""
Training driver for the biomass regression model.

Runs 5-fold cross-validation using the fold assignments produced by
``cv_split.py``. For each requested fold:

  Stage 1 (``config.EPOCHS_FROZEN`` epochs): backbone frozen, head-only
    warmup so the head can reach a reasonable state before touching the
    pretrained features.
  Stage 2 (``config.EPOCHS_UNFROZEN`` epochs): backbone unfrozen, full
    fine-tuning with layer-wise LR (backbone LR << head LR) and cosine
    schedule.
  Final eval: reload the best checkpoint and evaluate with 4-view
    reflection TTA (identity, hflip, vflip, both) if ``config.USE_TTA``
    is on.

Each fold saves its best checkpoint, per-epoch log, and OOF predictions
under ``outputs/{RUN_NAME}/fold{N}/``.

Usage:
    python cv_split.py                           # ONCE — generates folds.csv
    python train.py --config lvd_1024 --folds 0  # train a single fold
    python train.py --folds 0,1,2,3,4            # train all 5 folds
    python train.py --folds 0-4                  # range syntax also works
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
# Parse --config FIRST and set CSIRO_CONFIG before importing config, so that
# helper modules pick up the right config at import time.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", default=os.environ.get("CSIRO_CONFIG", "lvd_1024"))
_pre_args, _ = _pre.parse_known_args()
os.environ["CSIRO_CONFIG"] = _pre_args.config

import csv
import gc
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import (BiomassDataset, get_train_transforms, get_valid_transforms,
                     make_image_label_table, load_manual_crops)
from metric import (weighted_r2_global, per_target_r2, derive_targets_from_atomic)
from model import build_model


# Per-target SmoothL1 loss weights. Tuned to give roughly equal gradient
# magnitude across the 3 atomic targets given their different value scales
# (Green typically largest, Dead smallest). Sums to ~1.
LOSS_WEIGHTS = torch.tensor([0.36, 0.27, 0.36], dtype=torch.float32)


# ============================================================================
# Helpers
# ============================================================================

def vram_str(label=""):
    """Return a one-line VRAM allocation report, or empty string on CPU."""
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return (f"VRAM{' ' + label if label else ''}: "
            f"alloc={alloc:.1f}GB  reserved={reserved:.1f}GB  total={total:.1f}GB")


def cleanup_between_stages(*objs_to_release):
    """Delete references and empty the CUDA cache. Call between Stage 1 and
    Stage 2 (or before final eval) to free training-loop memory."""
    for o in objs_to_release:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_epoch(log_path: Path, row: dict):
    """Append a per-epoch metrics row to a CSV, writing the header on the
    first call."""
    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_folds(s: str) -> list[int]:
    """
    Parse a fold specification string.

    Accepts comma-separated integers and inclusive ranges:
        "0"         -> [0]
        "0,2,4"     -> [0, 2, 4]
        "0-4"       -> [0, 1, 2, 3, 4]
        "0,2-3"     -> [0, 2, 3]
    """
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            for i in range(int(a), int(b) + 1):
                out.add(i)
        else:
            out.add(int(part))
    return sorted(out)


# ============================================================================
# Loss
# ============================================================================

def compute_loss(pred, targets, loss_weights, device):
    """
    Weighted SmoothL1 loss over the 3 atomic targets.

    Args:
        pred:         (B, 3) predicted atomic values.
        targets:      (B, 3) true atomic values.
        loss_weights: (3,) per-target weights.
        device:       CUDA device to place ``loss_weights`` on.

    Returns:
        Scalar loss = per-sample sum of weighted SmoothL1, then mean over
        the batch.
    """
    smooth_l1 = F.smooth_l1_loss(pred, targets, reduction="none")  # (B, 3)
    loss = (smooth_l1 * loss_weights.to(device)).sum(dim=1).mean()
    return loss


# ============================================================================
# Train / evaluate
# ============================================================================

def train_one_epoch(model, loader, optimizer, device, grad_accum_steps,
                    scheduler=None, epoch_desc=""):
    """
    Run one training epoch: forward, weighted SmoothL1 loss, backward
    (with gradient accumulation), gradient clipping, optimizer step, and
    optional scheduler step.

    Returns:
        Mean training loss over all batches in the epoch.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=epoch_desc)
    for step, batch in enumerate(pbar):
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(left, right)
            loss = compute_loss(pred, targets, LOSS_WEIGHTS, device)
            loss_to_backward = loss / grad_accum_steps

        loss_to_backward.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device, use_tta=False):
    """
    Evaluate the model on a validation loader.

    If ``use_tta`` is True, runs 4-view reflection TTA (identity, hflip,
    vflip, both) and averages predictions. Both halves of each image
    receive the same reflection.

    Returns:
        wr2:       scalar competition metric (weighted global R²).
        per_t:     dict of per-target R² (diagnostic).
        pred_dict: dict of 5 target arrays (atomics + derived).
        true_dict: dict of 5 target arrays for the true values.
        all_ids:   list of image_ids in evaluation order.
    """
    model.eval()
    all_pred_atomic = []
    all_true_atomic = []
    all_ids = []

    for batch in loader:
        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)

        if use_tta:
            views = [
                (left, right),
                (torch.flip(left, dims=[3]), torch.flip(right, dims=[3])),
                (torch.flip(left, dims=[2]), torch.flip(right, dims=[2])),
                (torch.flip(left, dims=[2, 3]), torch.flip(right, dims=[2, 3])),
            ]
            preds = []
            for L, R in views:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    p = model(L, R)
                preds.append(p.float())
            pred = torch.stack(preds, dim=0).mean(dim=0)
        else:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(left, right).float()

        all_pred_atomic.append(pred.cpu().numpy())
        all_true_atomic.append(batch["targets"].cpu().numpy())
        all_ids.extend(batch["image_id"])

    pred = np.concatenate(all_pred_atomic, axis=0)
    true = np.concatenate(all_true_atomic, axis=0)

    pred_dict = derive_targets_from_atomic(pred[:, 0], pred[:, 1], pred[:, 2])
    true_dict = derive_targets_from_atomic(true[:, 0], true[:, 1], true[:, 2])

    wr2 = weighted_r2_global(true_dict, pred_dict)
    per_t = per_target_r2(true_dict, pred_dict)

    return wr2, per_t, pred_dict, true_dict, all_ids


def save_oof(pred_dict, true_dict, ids, out_path):
    """Save OOF predictions and ground truth for all 5 targets to CSV."""
    df = pd.DataFrame({"image_path": ids})
    for tgt in config.ALL_TARGETS:
        df[f"pred_{tgt}"] = pred_dict[tgt]
        df[f"true_{tgt}"] = true_dict[tgt]
    df.to_csv(out_path, index=False)


# ============================================================================
# Per-fold training
# ============================================================================

def train_one_fold(fold, df_full, manual_crops, device):
    """
    Train one fold end-to-end: Stage 1 (frozen head-only) + Stage 2
    (unfrozen full fine-tune) + final TTA eval on the best checkpoint.

    Args:
        fold:         integer fold index (0..N_FOLDS-1).
        df_full:      DataFrame with a ``fold`` column assigning each image
                      to a fold.
        manual_crops: dict from ``dataset.load_manual_crops``.
        device:       CUDA device or "cpu".

    Returns:
        dict summarising the fold: best mid-train R², best-with-TTA R²,
        best epoch, and per-target TTA R².
    """
    torch.manual_seed(config.TRAIN_SEED + fold)
    np.random.seed(config.TRAIN_SEED + fold)

    train_df = df_full[df_full["fold"] != fold].reset_index(drop=True)
    valid_df = df_full[df_full["fold"] == fold].reset_index(drop=True)
    print(f"\nFold {fold}: train={len(train_df)}, valid={len(valid_df)}")
    print("Train state dist:", train_df["State"].value_counts().to_dict())
    print("Valid state dist:", valid_df["State"].value_counts().to_dict())

    train_ds = BiomassDataset(
        train_df, config.IMAGE_DIR, get_train_transforms(config.IMG_SIZE),
        apply_timestamp_removal=config.APPLY_TIMESTAMP_REMOVAL,
        manual_crops=manual_crops, img_size=config.IMG_SIZE,
    )
    valid_ds = BiomassDataset(
        valid_df, config.IMAGE_DIR, get_valid_transforms(config.IMG_SIZE),
        apply_timestamp_removal=config.APPLY_TIMESTAMP_REMOVAL,
        manual_crops=manual_crops, img_size=config.IMG_SIZE,
    )
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel: {config.BACKBONE}  total={n_params:.1f}M")
    print(f"  {vram_str('after model load')}")

    out_dir = config.OUTPUT_DIR / config.RUN_NAME / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"
    if log_path.exists():
        log_path.unlink()
    # Fold-numbered filename so multiple checkpoints can be uploaded to
    # Kaggle without renaming.
    best_ckpt_path = out_dir / f"fold_{fold}_best.pth"

    best_r2 = -math.inf
    best_epoch_global = -1
    overall_epoch = 0

    # ------------------------------------------------------------------
    # Stage 1: backbone frozen, head-only warmup
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Fold {fold} Stage 1: backbone frozen, head-only")
    print("=" * 60)
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LR_HEAD, weight_decay=config.WEIGHT_DECAY,
    )
    print(vram_str("Stage 1 start"))

    for epoch in range(config.EPOCHS_FROZEN):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device,
                               config.GRAD_ACCUM_STEPS, scheduler=None,
                               epoch_desc=f"f{fold} S1 ep{epoch}")
        wr2, per_t, _, _, _ = evaluate(model, valid_loader, device)
        epoch_time = time.time() - t0

        improved = wr2 > best_r2
        if improved:
            best_r2 = wr2
            best_epoch_global = overall_epoch
            torch.save(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                best_ckpt_path,
            )

        flag = " *NEW BEST*" if improved else ""
        print(f"f{fold} S1 ep{epoch}  loss={loss:.4f}  R²={wr2:.4f}{flag}  "
              f"time={epoch_time:.1f}s")

        log_epoch(log_path, {
            "stage": 1, "epoch": epoch, "global_epoch": overall_epoch,
            "loss": loss, "wr2": wr2, "improved": int(improved),
            "time_s": round(epoch_time, 1),
            **{f"r2_{t}": round(v, 4) for t, v in per_t.items()},
        })
        overall_epoch += 1

    # ------------------------------------------------------------------
    # Stage 2: backbone unfrozen, full fine-tune
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Fold {fold} Stage 2: full fine-tune")
    print("=" * 60)
    cleanup_between_stages(optimizer)
    print(vram_str("after Stage 1 cleanup"))

    model.unfreeze_backbone()
    model.enable_grad_checkpointing()

    optimizer = torch.optim.AdamW(
        model.get_param_groups(config.LR_BACKBONE, config.LR_HEAD, config.WEIGHT_DECAY)
    )
    optim_steps_per_epoch = max(1, len(train_loader) // config.GRAD_ACCUM_STEPS)
    total_optim_steps = optim_steps_per_epoch * config.EPOCHS_UNFROZEN
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_optim_steps, eta_min=config.LR_BACKBONE * 0.1,
    )
    print(f"Cosine: {total_optim_steps} steps over {config.EPOCHS_UNFROZEN} epochs")
    print(vram_str("Stage 2 start"))

    for epoch in range(config.EPOCHS_UNFROZEN):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device,
                               config.GRAD_ACCUM_STEPS, scheduler=scheduler,
                               epoch_desc=f"f{fold} S2 ep{epoch}")
        wr2, per_t, _, _, _ = evaluate(model, valid_loader, device)
        epoch_time = time.time() - t0

        improved = wr2 > best_r2
        if improved:
            best_r2 = wr2
            best_epoch_global = overall_epoch
            torch.save(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                best_ckpt_path,
            )

        flag = " *NEW BEST*" if improved else ""
        cur_lr_bb = optimizer.param_groups[0]["lr"]
        cur_lr_hd = optimizer.param_groups[1]["lr"]
        print(f"f{fold} S2 ep{epoch}  loss={loss:.4f}  R²={wr2:.4f}{flag}  "
              f"lr_bb={cur_lr_bb:.2e}  lr_head={cur_lr_hd:.2e}  time={epoch_time:.1f}s")

        log_epoch(log_path, {
            "stage": 2, "epoch": epoch, "global_epoch": overall_epoch,
            "loss": loss, "wr2": wr2, "improved": int(improved),
            "time_s": round(epoch_time, 1),
            "lr_backbone": cur_lr_bb, "lr_head": cur_lr_hd,
            **{f"r2_{t}": round(v, 4) for t, v in per_t.items()},
        })
        overall_epoch += 1

    # ------------------------------------------------------------------
    # Final eval with TTA
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Fold {fold} Final eval with TTA")
    print("=" * 60)
    cleanup_between_stages(optimizer, scheduler)
    print(vram_str("after Stage 2 cleanup"))

    print(f"Loading best from {best_ckpt_path} (epoch {best_epoch_global}, R²={best_r2:.4f})")
    state = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    wr2_tta, per_t_tta, pd_tta, td_tta, ids_tta = evaluate(
        model, valid_loader, device, use_tta=config.USE_TTA
    )
    print(f"\nFOLD {fold} FINAL:")
    print(f"  best mid-train R²: {best_r2:.4f}  (epoch {best_epoch_global})")
    print(f"  best + TTA   R²:   {wr2_tta:.4f}")
    print(f"  per-target R² (TTA): {per_t_tta}")

    save_oof(pd_tta, td_tta, ids_tta, out_dir / "oof.csv")
    print(f"  Saved checkpoint, log, OOF in {out_dir}/")

    del model, optimizer, scheduler, train_loader, valid_loader, train_ds, valid_ds
    cleanup_between_stages()

    return {
        "fold": fold, "best_r2": best_r2, "best_r2_tta": wr2_tta,
        "best_epoch": best_epoch_global, "per_target_tta": per_t_tta,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Parse args, load data + fold assignments, train each requested fold,
    print an overall summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lvd_1024",
                        help="Config name (loads configs/config_<name>.py). "
                             "Default lvd_1024.")
    parser.add_argument("--folds", type=str, default="0",
                        help="Folds to train. Accepts comma-separated ints "
                             "and inclusive ranges: '0', '0,1,3', '0-4', '0,2-3'.")
    parser.add_argument("--folds_csv", type=Path,
                        default=config.OUTPUT_DIR / "folds.csv",
                        help="Path to folds.csv from cv_split.py.")
    args = parser.parse_args()

    folds_to_run = parse_folds(args.folds)
    if not folds_to_run:
        raise ValueError(f"No folds parsed from --folds={args.folds!r}")
    print(f"Folds to train: {folds_to_run}")

    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    df = make_image_label_table(config.TRAIN_CSV)
    print(f"\nTotal images: {len(df)}")

    n_before = len(df)
    df = df[~df["image_path"].isin(config.EXCLUDED_IMAGES)].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} excluded images")
        print(f"Remaining: {len(df)}")

    folds_df = pd.read_csv(args.folds_csv)
    df = df.merge(folds_df[["image_path", "fold"]], on="image_path", how="left")
    if df["fold"].isna().any():
        missing = df[df["fold"].isna()]["image_path"].tolist()
        raise ValueError(
            f"{len(missing)} non-excluded images don't have fold assignments: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    df["fold"] = df["fold"].astype(int)

    manual_crops = load_manual_crops()
    if manual_crops:
        print(f"Loaded {len(manual_crops)} manual crop rectangles from manual_crops.csv")

    print(f"\nTraining settings (RUN_NAME={config.RUN_NAME}):")
    print(f"  Image size: {config.IMG_SIZE}  (per half {config.IMG_SIZE}x{config.IMG_SIZE})")
    print(f"  Batch size: {config.BATCH_SIZE}, grad accum: {config.GRAD_ACCUM_STEPS} "
          f"(effective batch {config.BATCH_SIZE * config.GRAD_ACCUM_STEPS})")
    print(f"  Augmentation: RRC=p~{config.RRC_PROB}, "
          f"Gamma=p{config.P_GAMMA}, "
          f"Bright/Contrast=p{config.P_BRIGHTNESS_CONTRAST}, "
          f"HSV=p{config.P_HSV}, "
          f"CLAHE/Eq=p{config.P_CLAHE_OR_EQ}, "
          f"Sharpen/Emboss=p{config.P_SHARPEN_OR_EMBOSS}")
    print(f"  Epochs: {config.EPOCHS_FROZEN} frozen + {config.EPOCHS_UNFROZEN} unfrozen")
    print(f"  Output: outputs/{config.RUN_NAME}/foldN/fold_N_best.pth")

    results = []
    failures = []
    overall_t0 = time.time()
    for i, fold in enumerate(folds_to_run):
        print("\n" + "#" * 70)
        print(f"# FOLD {fold}  ({i+1} of {len(folds_to_run)})")
        print(f"# Elapsed total: {(time.time()-overall_t0)/60:.1f} min")
        print("#" * 70)
        try:
            result = train_one_fold(fold, df, manual_crops, device)
            results.append(result)
        except Exception as e:
            import traceback
            print(f"\n!!! FOLD {fold} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append((fold, str(e)))
            cleanup_between_stages()

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total wall time: {(time.time()-overall_t0)/60:.1f} min")
    print(f"Folds attempted: {folds_to_run}")
    print(f"Folds succeeded: {[r['fold'] for r in results]}")
    if failures:
        print(f"Folds FAILED: {[f[0] for f in failures]}")
    print()
    if results:
        print(f"{'fold':>5}  {'best_r2':>10}  {'r2_TTA':>10}  {'best_ep':>8}")
        for r in results:
            print(f"{r['fold']:>5}  {r['best_r2']:>10.4f}  {r['best_r2_tta']:>10.4f}  "
                  f"{r['best_epoch']:>8}")
        if len(results) >= 2:
            mean_r2 = np.mean([r["best_r2_tta"] for r in results])
            std_r2 = np.std([r["best_r2_tta"] for r in results])
            print(f"\nMean TTA R²: {mean_r2:.4f} ± {std_r2:.4f}")


if __name__ == "__main__":
    main()
