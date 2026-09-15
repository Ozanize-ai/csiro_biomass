"""
Competition metric and target-derivation utilities.

The competition uses a **globally weighted R²** across all (image, target)
pairs — not the mean of per-target R² values. Every (image, target) row is
weighted by its target's weight, and a single ``y_mean`` is computed as the
weighted mean across everything.

A practical consequence: ``Dry_Total_g`` (largest values, weight 0.5)
dominates the metric both through its weight and through its scale
contributing to SS_res and SS_tot. Improvements on Total move the global
metric far more than improvements on the atomic targets in isolation.
"""

import numpy as np

from config import TARGET_WEIGHTS, ALL_TARGETS


def weighted_r2_global(y_true_dict, y_pred_dict):
    """
    Compute the competition's globally weighted R².

    Pools all (image, target) rows together, weights each row by its target's
    weight, and computes a single weighted R² using the weighted global mean.

    Args:
        y_true_dict: mapping from target name to 1-D array of true values.
        y_pred_dict: mapping from target name to 1-D array of predictions.
            Both dicts must contain every entry in ``config.ALL_TARGETS``
            with the same length arrays.

    Returns:
        Scalar weighted R² as defined by the competition.
    """
    all_y_true = []
    all_y_pred = []
    all_w = []

    for tgt in ALL_TARGETS:
        y_t = np.asarray(y_true_dict[tgt], dtype=np.float64)
        y_p = np.asarray(y_pred_dict[tgt], dtype=np.float64)
        w = TARGET_WEIGHTS[tgt]
        all_y_true.append(y_t)
        all_y_pred.append(y_p)
        all_w.append(np.full_like(y_t, w))

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    weights = np.concatenate(all_w)

    y_mean = np.sum(weights * y_true) / np.sum(weights)
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - y_mean) ** 2)

    return 1.0 - ss_res / ss_tot


def per_target_r2(y_true_dict, y_pred_dict):
    """
    Compute R² for each target independently — a diagnostic complement to
    the pooled weighted metric.

    Useful during training to see which targets the model is handling well
    vs poorly. Not the competition score.

    Args:
        y_true_dict: mapping from target name to 1-D array of true values.
        y_pred_dict: mapping from target name to 1-D array of predictions.

    Returns:
        dict mapping target name to its independent R² (NaN if the target's
        true variance is zero).
    """
    out = {}
    for tgt in ALL_TARGETS:
        y_t = np.asarray(y_true_dict[tgt], dtype=np.float64)
        y_p = np.asarray(y_pred_dict[tgt], dtype=np.float64)
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - y_t.mean()) ** 2)
        out[tgt] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return out


def derive_targets_from_atomic(green, dead, clover):
    """
    Derive the 5 target dict from the 3 predicted atomic targets.

    The competition's 5 targets are not independent — GDM and Dry_Total are
    exact sums of the atomics:

        GDM_g       = Dry_Green_g + Dry_Clover_g
        Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g

    Predicting only the 3 atomics and deriving the other two guarantees
    internally consistent predictions.

    Args:
        green:  1-D array of Dry_Green_g predictions.
        dead:   1-D array of Dry_Dead_g predictions.
        clover: 1-D array of Dry_Clover_g predictions.

    Returns:
        dict with all 5 targets keyed by their competition column names.
    """
    green = np.asarray(green)
    dead = np.asarray(dead)
    clover = np.asarray(clover)
    return {
        "Dry_Green_g":  green,
        "Dry_Dead_g":   dead,
        "Dry_Clover_g": clover,
        "GDM_g":        green + clover,
        "Dry_Total_g":  green + dead + clover,
    }
