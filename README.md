# RWM — Real World Mandate for Cervical Cancer Detection

> *A model that only works in a lab is not a model that saves lives.*

Cervical cancer is the fourth most common cancer in women worldwide —
with 94% of deaths occurring in low and middle income countries where
diagnostic infrastructure is limited and inconsistent.

RWM is a hybrid diagnostic pipeline built around a single question:
**does this model hold up when it sees clinical data it has never
trained on?**

Not a new dataset split. A completely different institution, different
hardware, different staining — zero retraining.

---

## Architecture

**Phase 1 — Deep Feature Extraction**
Four CNN backbones extract semantic feature vectors from 224×224 cell images:

| Backbone | Feature Dims | Characteristic |
|----------|:------------:|----------------|
| VGG16 | 512 | Local high-frequency textures |
| ResNet50 | 2048 | Deep residual semantic features |
| Xception | 2048 | Depthwise separable cross-channel mapping |
| EfficientNetB0 | 1280 | Compound-scaled spatial representation |

**Phase 2 — Global Context (ConvNeXtTiny)**
Standard CNNs miss long-range spatial relationships — nucleus position
relative to cytoplasm, cellular arrangement patterns. ConvNeXtTiny adds
a 768-dim ViT-style global context vector to every backbone's output.

**Phase 3 — Handcrafted Textural Analysis**
Two parallel feature streams capture what deep filters blur:
- **GLCM** (32-dim): contrast, correlation, energy, homogeneity across
  4 orientations and 2 distances
- **DWT Haar** (8-dim): mean + std across LL, LH, HL, HH sub-bands —
  frequency fingerprinting of lesion roughness

**Phase 4 — Fusion & Classification**
- PCA compresses deep features → 128-dim (variance-preserving, not lossy)
- Concatenated with 40 handcrafted features → **168-dim hybrid vector**
- XGBoost classifier (n_estimators=300, L2 regularization, balanced class weights)

---

## Results

### Internal Benchmark (Multi-Cancer Dataset)

| Model | Test Accuracy |
|-------|:-------------:|
| ResNet50 + ConvNeXtTiny | **97.41%** |
| Xception + ConvNeXtTiny | 97.01% |
| VGG16 + ConvNeXtTiny | 96.91% |
| EfficientNetB0 + ConvNeXtTiny | 96.81% |
| ResNet50 (standalone) | 97.21% |
| Xception (standalone) | 96.51% |
| VGG16 (standalone) | 96.41% |
| EfficientNetB0 (standalone) | 94.52% |

### Cross-Dataset Generalization — The Real Test

Trained on Multi-Cancer. Tested on SIPaKMeD (966 images). Zero retraining.

| Model | Cross-Dataset Accuracy |
|-------|:----------------------:|
| ResNet50 + ConvNeXtTiny | **85.92%** |
| Xception + ConvNeXtTiny | 84.06% |
| VGG16 + ConvNeXtTiny | 81.78% |
| Xception (standalone) | 68.53% |

> ConvNeXtTiny integration lifted Xception's cross-dataset accuracy
> by +15.53 percentage points on data it had never seen.

---

## Dataset

| Dataset | Images | Classes | Role |
|---------|:------:|:-------:|------|
| Multi-Cancer | 25,000 | 5 | Training + internal test |
| SIPaKMeD | 966 | 5 | Cross-dataset validation only |

**5 diagnostic classes:**
`cervix_dyk` · `cervix_koc` · `cervix_mep` · `cervix_pab` · `cervix_sfi`

80/20 stratified train-test split · fixed seed=42 · fully reproducible

---

## Tech Stack
Python · TensorFlow · Keras · XGBoost · scikit-learn
OpenCV · PyWavelets · scikit-image · Pandas · NumPy
Matplotlib · Seaborn
---

## Key Findings

- ConvNeXtTiny consistently improved every backbone on cross-dataset inference
- EfficientNetB0 showed the strongest *relative* improvement (+2.29%) — best generalization growth per parameter added
- ResNet50 + ConvNeXtTiny leads overall at 85.92% on unseen clinical data
- PCA acts as a precision filter here, not a bottleneck — enabled by the high feature density of the hybrid pool

---

## Why It Matters

Most cervical cancer detection research reports high accuracy on clean,
isolated, single-institution datasets. RWM deliberately tests the harder
question — domain shift resilience — because real clinical deployment
means different microscopes, different labs, different staining protocols.

The cross-dataset tier exists because a model that can't generalize
cannot save lives.

---

*Built as part of ongoing research into clinical generalization of
hybrid ML-DL pipelines for medical imaging.*