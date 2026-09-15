# Next Steps

This document captures what to try next when picking this project back up. It's ordered by (expected gain × probability of success) / (effort).

## Current State

- **Personal best: 0.59726 private LB / 0.70377 public LB** (Phase 3)
- **Gold medal cutoff: ~0.65 private** (still ~+0.05 away)
- **Key blocker:** OOF has been misleading on model-side changes (3 consecutive submissions regressed despite positive OOF)

## Priority 1: Diagnostic — determine what's transferable

Before spending more compute on model changes, run diagnostics that isolate causes.

### 1a. Submit aux-only ensemble (5 models, no baseline)

**Why:** The Phase 5 submission averaged baseline + aux (10 models) and regressed. We don't know if aux itself transfers or if the ensemble combination is the problem.

**Setup:**
```python
# In kaggle_inference.py, change EXPERIMENTS to just the aux entry:
EXPERIMENTS = [
    {
        "name": "dinov3_lvd_1024_aux",
        "img_size": 1024,
        "checkpoint_dir": Path("/kaggle/input/datasets/ozanize/csiro-aux-ensemble"),
        "batch_size_per_gpu": 4,
    },
]
```

**Cost:** 1 Kaggle submission, ~30 min runtime.

**Decision:**
- Aux-only ≥ 0.597: aux transfers, but ensembling with baseline dilutes postprocessing effect. Try weighted ensemble or target-specific ensembling.
- Aux-only < 0.597: aux itself doesn't transfer. Skip aux entirely, look elsewhere.

### 1b. Re-tune postprocessing on aux OOF distribution

If aux-only works well, its bias pattern is different from baseline. The postprocessing rules (per-state scales, gates) were tuned on baseline OOF. Re-run the OOF gate sweep using aux OOF predictions and see if the optimal parameters differ.

**Cost:** ~30 min of analysis (no training needed).

## Priority 2: Weighted / target-specific ensembling

The 10-model ensemble hurt because equal averaging doesn't respect per-target strengths:

- Aux was stronger on Total (+0.028) and GDM (+0.020)
- Baseline was much stronger on Dead (+0.103 for baseline)

**Idea:** ensemble target-by-target with different weights:
```python
# For each atomic prediction (Green, Dead, Clover), choose the weighted
# combination that maximized OOF R² for that specific target
final_green  = 0.4 * baseline_green  + 0.6 * aux_green
final_dead   = 0.8 * baseline_dead   + 0.2 * aux_dead  # baseline dominant
final_clover = 0.6 * baseline_clover + 0.4 * aux_clover
# Then derive GDM, Total from these blended atomics
```

**Cost:** No new training. Just script an OOF weight search + one Kaggle submission.

## Priority 3: Hybrid Test-Time Training submission

This was locally validated (+0.003 R² OOF across 2 folds) but never submitted.

**Approach:** Per-image adaptation using MSE-to-reference-prediction loss. Adapt only Dead prediction; keep no-TTT for Green and Clover.

**Cost:**
- ~1 hour to integrate into kaggle_inference.py
- ~90 min Kaggle runtime (~3× normal)
- 1 submission slot

**Expected gain:** +0.002 to +0.008 private.

**Risk:** Low (never made things worse in local validation). Not a high-payoff swing but reliable small gain.

Files: `validate_ttt_local.py` has the working local implementation. Adapt for Kaggle.

## Priority 4: Alternative backbones for ensemble diversity

The current ensemble is 100% DINOv3 features. All 5 folds + aux 5 folds share the same feature representation, so ensemble diversity is limited.

**Idea:** Train 3 folds with a different backbone (DINOv2-Large or CLIP-ViT-L). Ensemble with existing DINOv3.

**Cost:** ~7-10 hours training (3 folds), + Kaggle notebook integration.

**Expected gain:** +0.005 to +0.015 IF the backbones produce genuinely different feature representations.

**Risk:** 5A's writeup explicitly noted "high resolution > bigger backbone." Different backbones might not add meaningful diversity. Uncertain.

## Priority 5: Density estimation architecture (major rewrite)

5B's team reported the highest CV-LB correlation of any team (r=0.831) using a fundamentally different architecture:

- Predict per-pixel density maps instead of per-image regression
- Aggregate density × area → total mass
- Use purely local loss (no image-level bias)

**Why it's promising:** Local-loss architecture may be more robust to distribution shift. Our regression models have shown train-test overfitting; 5B's approach might generalize better.

**Cost:** 5-7 days rewrite. Different data pipeline, different training loop, different loss function. Basically starting a second model from scratch.

**Expected gain:** +0.01 to +0.03 private if it works; but the compute cost is high.

**Risk:** Major rewrite, uncertain payoff. Only consider if smaller options have been exhausted.

## Priority 6: Extended aux (Variant B — species included)

4th place used NDVI + Height only. 5A used those + species classification (GaussianNLL, weight=0.05). If aux itself works (Priority 1 diagnostic determines this), adding species might compound.

**Cost:**
- Add species head to aux_pretraining.py
- Re-run Stage A (~3 hours)
- Re-run Stage B 5 folds (~11 hours)
- Kaggle notebook update + submission

**Expected gain:** +0.005 to +0.015 on top of any aux gain.

**Risk:** Species is 15 classes with some having only 2-4 samples. Weighted CE or GaussianNLL needed. 2nd, 5B, 7th all tried species aux and failed to get value.

**Prerequisite:** Priority 1 confirms aux-only transfers.

## Priorities NOT Worth Attempting

Based on this project's data, these were tried and are unlikely to yield value:

- **Interval classification heads** (1st place technique): ruled out over 3 attempts, disrupts Green predictions via shared backbone
- **1536 inference**: regressed private LB by 0.022
- **Lucerne gate**: redundant with NSW state gate
- **Progressive 1024→1536 fine-tuning**: destabilized converged model
- **Full TTT (all 3 targets)**: hurt Green/Clover to help Dead, net negative

## Diagnostic Data to Regenerate If Needed

If picking up cold, these OOF files will need to be regenerated:
- `outputs/dinov3_lvd_1024/foldN/oof.csv` (baseline 5-fold OOF)
- `outputs/dinov3_lvd_1024_aux/foldN/oof.csv` (aux 5-fold OOF)

Regenerating requires re-running the training. Alternatively, saved model checkpoints can be run through `evaluate()` in train.py to regenerate OOF quickly.

## Kaggle Environment Reminder

- Test images are ONLY available inside Kaggle notebooks — cannot be downloaded
- This blocks classical pseudo-labeling workflows
- Two selected final submissions on Kaggle should both be Phase 3 PB (0.59726)
- Kaggle inference notebook expects 4-5 datasets attached — see `kaggle_inference.py` docstring

## Meta: Deciding When to Stop

This project has hit a common competition pattern: the initial big improvements came easy (postprocessing +0.019), and subsequent attempts have shown diminishing returns or regressions. The gap to gold (~0.05) is substantial and likely requires either:

1. A fundamentally different architecture (density estimation)
2. Combining several small improvements simultaneously (aux + weighted ensemble + TTT)
3. Access to test images for pseudo-labeling (which the competition blocks)

At some point the effort/hour of value drops below other opportunities. If picking this back up:
- If you have 1-2 days: try Priority 1a + 1b + 2 (diagnostic + weighted ensemble)
- If you have a week: add Priority 3 + 4 (TTT + alternative backbone)
- If you have a month: add Priority 5 (density estimation)
