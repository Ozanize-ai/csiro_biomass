# Results & Experimental Journal

This document is the honest record of every submission and experiment for the CSIRO Biomass competition. It shows what worked, what failed, and what was learned. The final personal best is 0.59726 private LB from a base of 0.578.

## Submission History

| # | Phase | Change | Public | Private | Δ private |
|---|---|---|---|---|---|
| 1 | Starting point | DINOv2-Giant + VAE (from prior work) | 0.629 | 0.580 | — |
| 2 | Phase 1 | Full rebuild: DINOv3 ViT-L, L/R split | 0.678 | 0.564 | -0.016 |
| 3 | Phase 2 | Light augmentation @ 1024 | 0.693 | 0.578 | +0.014 |
| 4 | Phase 3 | State classifier + full postprocessing | **0.70377** | **0.59726** | **+0.019** |
| 5 | Phase 4a | 1536 inference of 1024 checkpoints | 0.682 | 0.575 | -0.022 |
| 6 | Phase 4b | Lucerne species gate added | 0.70356 | 0.59621 | -0.001 |
| 7 | Phase 5 | 10-model ensemble (baseline + aux) | 0.68823 | 0.58738 | -0.010 |

**Personal best**: Phase 3 (0.59726 private / 0.70377 public).

## Detailed Analysis Per Phase

### Phase 1: Rebuild from scratch

The starting point (0.580 private) used a DINOv2-Giant backbone with a VAE. We rebuilt from scratch with cleaner architecture:
- DINOv3 ViT-L (LVD-1689M pretrained) instead of DINOv2-Giant
- L/R split with shared backbone + sum-pooling
- Simple 3-layer regression head with softplus output

Result: private LB dropped from 0.580 to 0.564. The rebuild lost ground initially because we hadn't yet added the augmentation strategy and postprocessing that would eventually make it competitive.

**Lesson:** starting simpler and building up is worth the short-term regression.

### Phase 2: Light augmentation

Added augmentation aligned with what the winners reported: RandomResizedCrop (p~0.5), Gamma (p=0.3), Brightness/Contrast (p=0.5), HSV shifts (p=0.5), CLAHE/Equalize (p=0.15), Sharpen/Emboss (p=0.15). No MixUp/CutMix/RandomErasing.

Result: private LB 0.578, essentially matching the old baseline. Model was now well-trained and comparable to the starting point.

### Phase 3: State classifier + postprocessing (**+0.019 private, our biggest single win**)

Three changes together, each validated on OOF gate sweeps:

1. **State classifier**: separate DINOv3 head trained on all 355 images to predict state (NSW/Tas/Vic/WA). Achieved 97.75% CV accuracy.

2. **Per-state postprocessing**: apply state-conditional scaling and hard rules:
   - `WA → Dead = 0` (all WA training samples have dead=0)
   - `NSW → Clover = 0` (74/75 NSW training samples have clover=0)
   - Per-state scales to correct systematic bias
   - Asymmetric dead correction (0.9× when pred < 10, 1.1× when pred > 20)
   - Per-state max clipping to training maxes
   - Global max clip

3. **Confidence gating**: each rule requires state classifier confidence ≥ 0.75 AND a max_pred safety threshold (to protect rare outliers where the model gives a large prediction we should trust).

Result: **private 0.59726, public 0.70377** — our personal best. Both improved simultaneously, with roughly 1.5× private effect vs public effect (indicating good generalization).

**Lesson:** Domain knowledge + confidence-gated hard rules dominated model improvements. The postprocessing win came from understanding that pastures have systematic composition patterns by state — leveraging this with a strong state classifier + rules that respect model confidence.

### Phase 4a: 1536 inference of 1024 checkpoints (-0.022 private, regressed)

**Hypothesis:** Position embeddings interpolate at inference, giving the model more effective compute per image without retraining.

**Local validation:** OOF mean R² across 5 folds showed +0.028 improvement from 1024 → 1536 inference. Some folds showed large gains (fold 2 clover 0.03 → 0.94), but this was clearly an R² artifact on a low-variance subset.

**Result:** Regressed by -0.022 on both public and private. The OOF signal was misleading.

**Root cause:** Fold 2's outlier gain inflated the mean. Excluding fold 2, the average was +0.028; but this still overstated the transfer to test. The bicubic-interpolated position embeddings created a mismatch the training-fold validation didn't detect.

**Lesson:** Watch for OOF metrics dominated by variance artifacts on subset composition. Cross-fold direction (+/-) is more informative than magnitude.

### Phase 4b: Lucerne species gate (-0.001 private, marginal regression)

**Hypothesis:** All 22 Lucerne training samples have clover=0. If the species classifier can identify Lucerne on test, we can zero their clover predictions.

**Approach:** Trained a binary Lucerne vs not-Lucerne DINOv3 classifier. Achieved 100% precision at every threshold 0.3-0.8 on 5-fold CV.

**Concern noted before submission:** All 22 Lucerne training samples are in NSW. The `NSW → Clover = 0` gate already handles them. The Lucerne gate is only additive when the state classifier misidentifies an NSW image, which is rare.

**Result:** -0.001 on both public and private. Essentially neutral, consistent with the prediction that most Lucerne effect was already captured by the NSW gate.

**Lesson:** Redundant gates that overlap with existing coverage don't add value. Before adding a new rule, quantify its incremental effect vs already-active rules.

### Phase 5: Aux pretraining + 10-model ensemble (-0.010 private, regressed)

**Hypothesis:** 4th place's Stage 2 pretraining (NDVI + Height auxiliary tasks) transferred well; our aux 5-fold ensemble should improve when averaged with the baseline 5-fold.

**Local validation:**
| Fold | Baseline wr² | Aux wr² | Δ |
|---|---|---|---|
| 0 | 0.8211 | 0.8432 | +0.022 |
| 1 | 0.7461 | 0.7769 | +0.031 |
| 2 | 0.7751 | 0.7668 | -0.008 |
| 3 | 0.7132 | 0.7571 | +0.044 |
| 4 | 0.7404 | 0.7345 | -0.006 |
| **Mean** | 0.7592 | 0.7757 | **+0.017** |

3 of 5 folds showed clear improvement; 2 tied. Mean +0.017 in the right direction.

Per-target picture was more nuanced: Green +0.013, Dead **-0.103**, Clover +0.002, GDM +0.020, Total +0.028. Dead R² dropped substantially, but the higher-weighted targets (Total, GDM) improved enough that the weighted metric was positive.

**Result:** private 0.58738, -0.010 vs PB.

**Analysis:** Three consecutive model-side changes have regressed private LB despite positive OOF. This is a systematic pattern, not noise. Possibilities:
1. Our OOF is overfitting to training-set characteristics that don't hold on test
2. Postprocessing rules (tuned on original 5-fold OOF bias pattern) don't apply cleanly to the ensemble's averaged predictions with different bias signatures
3. Test set has structural differences not captured by our fold splits

**Lesson:** When OOF and LB disagree three times in a row, trust the LB. Consider that model-side improvements might require re-tuning the postprocessing rules for the new prediction distribution.

## Experiments That Didn't Reach Submission

### Interval classification heads (three attempts, all regressed on OOF)

1st place's technique — add classification heads alongside regression, predicting which of 7 value bins each target falls into.

| Attempt | Config | Fold 0 R² | vs Baseline (0.8283) |
|---|---|---|---|
| v1 | cls_weight=0.3, all 3 targets classified | 0.8161 | -0.012 |
| v2 | cls_weight=0.2, Dead+Clover only | 0.8162 | -0.012 |
| v3 | Two-stage (regression converged first, then add cls) | 0.8234 | -0.005 |

Every variant hurt Green R² by ~0.16, even when Green was excluded from classification. Root cause: the shared backbone gradients get disrupted by classification-task signal, and Green's converged features are the most fragile to disruption.

**Not submitted.** Conclusive negative result across three configurations.

### Test-Time Training (validated OOF, never submitted)

Per-image self-supervised adaptation using consistency loss across augmented views.

- **Full TTT** (all 3 targets): Fold 0 mean per-image MAE Δ = +0.127 (worse). Helped Dead, hurt Green/Clover.
- **Hybrid TTT** (TTT-Dead only, no-TTT Green/Clover): Fold 0 Δ = -0.013, Fold 3 Δ = -0.024 (both better).

Hybrid TTT worked consistently but the gain was small (+0.003 to +0.005 R² across two folds), and Kaggle runtime would be ~3× (30 min → 90 min). Given the small expected gain and compute cost, submission was deprioritized.

### Progressive 1024 → 1536 fine-tuning (local negative)

Fine-tuning a converged 1024 model at 1536 for 5 additional epochs hurt val R² on every epoch. Root cause: model was already converged, additional epochs with position embedding interpolation just destabilized it.

## Kaggle Submissions to Preserve (Do Not Overwrite)

Two selected submissions on Kaggle should be the PB (0.59726 private):
- Primary: Phase 3 submission with full postprocessing
- Backup: Same

**Do NOT let any of the following become a final pick:**
- Phase 5 aux ensemble (0.58738)
- Phase 4a 1536 (0.575)

## Statistics

- Total training runs: ~40+ (across baseline 5-fold, aux 5-fold, various experiments, classifier heads)
- Total local compute: estimated 100-120 GPU hours on RTX 3090
- Total Kaggle submissions: 7
- Days of active development: ~4 weeks
- Final rank movement: from #1038 upward (specific final rank not tracked)
