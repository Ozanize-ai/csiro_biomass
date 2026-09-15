# CSIRO Biomass Estimation — Kaggle Competition

A vision transformer approach for estimating pasture biomass composition (green, dead, clover) from top-down farm photographs, developed for the CSIRO Biomass Kaggle competition.

**Final personal best: 0.59726 private LB / 0.70377 public LB** — an improvement of +0.019 private over the starting baseline (0.578). Rank improvement from #1038/3806 upward.

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Approach](#approach)
- [Repository Structure](#repository-structure)
- [Reproducing Results](#reproducing-results)
- [Key Insights & Lessons](#key-insights--lessons)
- [What's Next](#whats-next)

## Overview

The CSIRO Biomass competition asks competitors to predict five quantities per image of Australian pasture:
- `Dry_Green_g` — mass of green vegetation (grams)
- `Dry_Dead_g` — mass of dead vegetation (grams)
- `Dry_Clover_g` — mass of clover (grams)
- `GDM_g` — green dry matter = Green + Clover (derived)
- `Dry_Total_g` — total = Green + Dead + Clover (derived)

Training data is small: 355 labeled images from 4 Australian states (NSW, Tas, Vic, WA) collected in 2015. Each image is a wide-angle top-down photo of a specific pasture area, roughly 1000×2000 pixels, sometimes with cardboard used to isolate the measurement area.

**Metric:** weighted R² across the 5 targets: `Total × 0.5 + GDM × 0.2 + atomic × 0.1 each`.

The competition places a premium on:
- Robustness to small training set + significant domain variety
- Handling of extreme class imbalance in some targets (e.g. 99% of NSW samples have clover=0)
- Careful architectural choices that don't overfit

## Results

| Milestone | Public LB | Private LB | Δ private |
|---|---|---|---|
| Starting baseline (DINOv2-Giant + VAE) | 0.629 | 0.580 | — |
| Phase 1 rebuild (DINOv3 ViT-L, L/R split) | 0.678 | 0.564 | -0.016 |
| Phase 2 (light augmentation @ 1024) | 0.693 | 0.578 | +0.014 |
| Phase 3 (state classifier + postprocessing) — **PB** | **0.70377** | **0.59726** | **+0.019** |
| Phase 4a (1536 inference — regressed) | 0.682 | 0.575 | -0.022 |
| Phase 4b (Lucerne gate added — marginal regression) | 0.70356 | 0.59621 | -0.001 |
| Phase 5 (aux-pretrain + ensemble — regressed) | 0.68823 | 0.58738 | -0.010 |

Full submission history and analysis: [RESULTS.md](RESULTS.md).

## Approach

### Data preprocessing

- Images resized to 1024 × 2048, then split into left/right halves (1024 × 1024 each)
- Timestamp orange-text removal (a competition-known issue)
- Manual crops for 10 problem images with visual artifacts
- Edge trimming (5px) to remove border artifacts

### Model architecture

- **Backbone**: DINOv3 ViT-L (LVD-1689M pretrained), shared between left/right halves
- **Feature aggregation**: sum-pool features from left + right (equivalent to averaging up to a scale)
- **Regression head**: 3-layer MLP with LayerNorm → GELU → Dropout, softplus output for non-negativity
- **Output**: 3 atomic targets (Green, Dead, Clover); GDM and Total derived

### Training

- 5-fold StratifiedGroupKFold (stratified by State, grouped by Sampling_Date, seed=6821)
- Two-stage training: 5 epochs with frozen backbone (head-only), then 22 epochs with unfrozen backbone
- Cosine LR schedule, layer-wise LR (backbone 1e-5, head 1e-4)
- SmoothL1 loss with equal weights across atomic targets
- Mixed precision (bfloat16), gradient checkpointing to fit ViT-L at 1024 in 24GB VRAM
- Effective batch size 8 (batch 2 × gradient accumulation 4)
- TTA at inference: 4 reflection views (identity, hflip, vflip, both)

### Postprocessing (biggest single win: +0.019 private)

A separate DINOv3 head classifies the state per image with 97.75% CV accuracy. This enables:
- **Per-state scaling** on the atomic predictions (empirically derived from state-specific bias analysis)
- **WA → Dead = 0** hard rule (WA has 0.00 max dead in training)
- **NSW → Clover = 0** hard rule (74/75 NSW training samples have clover=0)
- **Asymmetric dead correction** (0.9× multiplier below 10g, 1.1× above 20g)
- **Per-state max clipping** to training maxes

Each rule is gated by state classifier confidence (min_prob=0.75) and a max_pred safety threshold to protect rare outliers.

## Repository Structure

```
├── configs/                        # Training configuration
│   └── config_lvd_1024.py          # Primary config (DINOv3 @ 1024)
├── config.py                       # Config loader
├── clean.py                        # Image preprocessing (timestamp removal etc.)
├── cv_split.py                     # StratifiedGroupKFold split
├── dataset.py                      # BiomassDataset + augmentation
├── metric.py                       # Weighted R² + target derivation
│
├── model.py                        # BiomassModel: DINOv3 backbone + regression head
├── train.py                        # Biomass 5-fold training loop
│
├── state_classifier_model.py       # State classifier model
├── train_state_classifier.py       # State classifier training
│
├── kaggle_inference.py             # Self-contained Kaggle notebook
│
├── README.md                       # This file
├── RESULTS.md                      # Full submission history + experiment analysis
├── NEXT_STEPS.md                   # Prioritized next-steps guide
├── LICENSE                         # MIT
├── requirements.txt                # Pinned dependencies
└── .gitignore
```

Not committed (see `.gitignore`):
- `outputs/` — model checkpoints (each ~1.2GB), OOF CSVs, training logs
- `data/` — training images and CSVs
- Any `.pth`, `.npy` files

## Reproducing Results

### Prerequisites

- CUDA GPU with 24GB+ VRAM (tested on RTX 3090)
- Python 3.10+
- Conda environment or pip venv

### Setup

```bash
conda create -n csiro python=3.10
conda activate csiro
pip install torch==2.1 timm==1.0 albumentations==2.0.8 opencv-python-headless==4.12.0.88 \
            pandas scikit-learn numpy tqdm
```

Download:
- Kaggle competition data → `data/csiro-biomass/`
- DINOv3 ViT-L LVD-1689M weights → `weights/dinov3_vitl16.pth`

Edit `configs/config_lvd_1024.py` to point at your local paths.

### Training

```bash
# 1) Generate fold assignments (writes outputs/folds.csv)
python cv_split.py

# 2) Train the 5-fold biomass ensemble (~12-15 hours on RTX 3090)
python train.py --folds 0,1,2,3,4

# 3) Train the state classifier (~10 min)
python train_state_classifier.py
```

### Inference on Kaggle

The `kaggle_inference.py` file is run as a Kaggle notebook with the trained checkpoints uploaded as datasets. See the file's docstring for required Kaggle inputs and expected paths.

## Key Insights & Lessons

Three things I'd want another ML practitioner to take away from this project:

### 1. Postprocessing was our biggest single win (+0.019 private)

Domain knowledge outperformed model tweaks. The state classifier + hard rules (WA→Dead=0, NSW→Clover=0) leveraged the fact that pastures have systematic composition differences by region. The state classifier gets 97.75% CV accuracy, then applies confidence-gated hard rules with safety thresholds. This alone improved private LB by more than every model change combined.

### 2. OOF was systematically misleading on model-side changes

Three separate model changes showed positive OOF improvement but regressed on private LB:
- 1536 inference: +0.028 OOF, -0.022 private
- Lucerne gate: +0.005 OOF (predicted), -0.001 private
- Aux ensemble: +0.017 OOF, -0.010 private

Every model architectural change hurt private LB. Only postprocessing rules (based on population-level state statistics) transferred reliably. This suggests our OOF was overfitting to training-set-specific characteristics that don't generalize to the test distribution.

**Takeaway:** with small datasets (355 samples) and complex validation (5-fold with variance-dominated metrics), OOF is a noisy signal. Multiple independent validations (train-time, OOF, held-out fold subset) are needed before trusting a change enough to submit.

### 3. Winner techniques don't always transfer

We attempted three winner techniques from the writeups:
- **1st place's interval classification heads** (+0.03 in their report): -0.012 for us
- **1st place's TTT technique** (+0.02 in their report): +0.003 OOF, never submitted
- **4th place's aux pretraining** (+0.01 in their report): +0.017 OOF, -0.010 private on ensemble

The base 5-fold + postprocessing pipeline is where the reliable gains came from. Attempting to replicate exotic winning techniques as a small competitor without their specific full-stack often doesn't work.

## What's Next

See [NEXT_STEPS.md](NEXT_STEPS.md) for a detailed picking-up-later guide.

Short version — high-value moves not yet tried:
1. **Density estimation architecture** (5B's approach, r=0.831 CV-LB correlation, largest theoretical upside)
2. **Test-Time Training submission** (validated at +0.003 OOF, never submitted)
3. **Weighted ensemble** (aux underperformed at equal weight; asymmetric weighting per-target)
4. **Alternative backbone diversity** (DINOv2 or CLIP-based ensemble members)

## Acknowledgments

- Competition organized by CSIRO Australia
- DINOv3 pretrained weights from Meta AI Research
- Insights from the top winners' writeups
- Special thanks to the open-source `timm`, `albumentations`, and PyTorch communities
