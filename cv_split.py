"""
Cross-validation split.

Uses ``StratifiedGroupKFold(stratify=State, groups=Sampling_Date)`` to build
5 folds with two properties:

* **No date leakage** — every Sampling_Date appears in exactly one fold, so
  images taken on the same day at the same farm never appear in both train
  and validation. A simple StratifiedKFold on this dataset would leak
  farm-day clusters and produce over-optimistic CV scores.

* **Balanced state representation** — folds have roughly proportional
  NSW/Tas/Vic/WA counts, within the constraint imposed by grouping.

The seed (``config.CV_SEED``) was chosen by evaluating candidate seeds on
a metric-weighted balance objective (competition weights + a clover-presence
balance term) to produce folds with comparable competition-weighted
statistics.

Running this module produces ``outputs/folds.csv`` with per-image fold
assignments and prints a diagnostic report.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config
from dataset import make_image_label_table


def split_stratified_grouped(df: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    """
    Split ``df`` into ``n_folds`` folds using StratifiedGroupKFold.

    Stratifies on ``State`` (each fold gets roughly proportional
    NSW/Tas/Vic/WA counts) and groups on ``Sampling_Date`` (each date is
    kept entirely within one fold).

    Args:
        df: DataFrame with ``State`` and ``Sampling_Date`` columns, one row
            per image.
        n_folds: number of folds to produce.
        seed: random_state for StratifiedGroupKFold.

    Returns:
        1-D int array of length ``len(df)``, where each entry is the fold
        index (0..n_folds-1) for that row.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = -np.ones(len(df), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(df, df["State"], groups=df["Sampling_Date"])
    ):
        folds[val_idx] = fold_idx
    return folds


def report_split(df: pd.DataFrame, folds: np.ndarray):
    """
    Print a diagnostic report of the fold split: sizes, state distribution,
    per-target statistics, clover-presence rate, and a date-overlap check.
    """
    n_folds = len(np.unique(folds))

    print("\nPer-fold sizes and date counts:")
    rows = []
    for f in range(n_folds):
        sub = df[folds == f]
        rows.append({
            "fold": f,
            "n": len(sub),
            "n_dates": sub["Sampling_Date"].nunique(),
            **sub["State"].value_counts().to_dict(),
        })
    print(pd.DataFrame(rows).fillna(0).astype(
        {"fold": int, "n": int, "n_dates": int}
    ).to_string(index=False))

    print("\nPer-fold target distributions (mean ± std):")
    rows = []
    for f in range(n_folds):
        sub = df[folds == f]
        row = {"fold": f, "n": len(sub)}
        for tgt in config.ALL_TARGETS:
            row[tgt] = f"{sub[tgt].mean():.1f}±{sub[tgt].std():.1f}"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nPer-fold clover-presence rate:")
    for f in range(n_folds):
        sub = df[folds == f]
        rate = (sub["Dry_Clover_g"] > 0).mean()
        n_with = (sub["Dry_Clover_g"] > 0).sum()
        print(f"  fold {f}: {rate:.1%}  ({n_with}/{len(sub)} images have clover)")

    print("\nDate-overlap check:")
    date_to_folds = {}
    for i in range(len(df)):
        d = df.iloc[i]["Sampling_Date"]
        date_to_folds.setdefault(d, set()).add(folds[i])
    overlapping = [(d, fs) for d, fs in date_to_folds.items() if len(fs) > 1]
    if overlapping:
        print(f"  WARNING: {len(overlapping)} dates appear in multiple folds!")
        for d, fs in overlapping[:3]:
            print(f"    {d}: folds {fs}")
    else:
        print(f"  OK: all {len(date_to_folds)} dates appear in exactly one fold")


def main():
    """
    Regenerate fold assignments using the current config, save to
    ``outputs/folds.csv``, and print a diagnostic report.
    """
    df = make_image_label_table(config.TRAIN_CSV)
    print(f"Loaded {len(df)} images, "
          f"{df['Sampling_Date'].nunique()} unique dates, "
          f"{df['State'].nunique()} states")
    print(f"Applying CV_SEED={config.CV_SEED} for {config.N_FOLDS}-fold "
          f"StratifiedGroupKFold (stratify=State, groups=Sampling_Date)")

    folds = split_stratified_grouped(df, config.N_FOLDS, config.CV_SEED)

    if len(np.unique(folds)) < config.N_FOLDS:
        raise RuntimeError(
            f"Some folds are empty (got {len(np.unique(folds))}/{config.N_FOLDS}). "
            f"Try a different CV_SEED value in the config."
        )

    report_split(df, folds)

    out_df = df[["image_path", "State", "Sampling_Date", "Species"]].copy()
    out_df["fold"] = folds
    out_path = config.OUTPUT_DIR / "folds.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved fold assignments to {out_path}")


if __name__ == "__main__":
    main()
