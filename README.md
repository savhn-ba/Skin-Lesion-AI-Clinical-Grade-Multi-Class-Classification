# Skin Lesion AI — Clinical-Grade Multi-Class Classification

A deep learning pipeline that classifies dermatoscopic images into seven diagnostic
categories using the **HAM10000** dataset.  The system is designed to act as a
clinical decision-support tool, giving **asymmetrically higher penalty to missing
high-risk lesions** (primarily Melanoma) through a cost-sensitive loss function.

---

## Dataset — HAM10000 & Class Imbalance

The HAM10000 ("Human Against Machine with 10000 training images") dataset contains
10,015 dermoscopy images labelled across seven skin-lesion categories:

| Code  | Full Name                          | Approx. share |
|-------|------------------------------------|---------------|
| MEL   | Melanoma                           | 11 %          |
| NV    | Melanocytic nevi                   | 67 %          |
| BCC   | Basal cell carcinoma               | 5 %           |
| AKIEC | Actinic keratoses / Bowen's disease| 3 %           |
| BKL   | Benign keratosis-like lesions      | 11 %          |
| DF    | Dermatofibroma                     | 1 %           |
| VASC  | Vascular lesions                   | 1 %           |

The dataset is **heavily imbalanced**: NV accounts for roughly 67 % of all images,
while DF and VASC together represent only ~2 %.  A naïve classifier trained with
standard cross-entropy will learn to predict NV almost exclusively and appear to
achieve high accuracy (~70 %) while completely missing the clinically dangerous
classes.

### Clinical Risk of Missing Melanoma

Melanoma is responsible for ~75 % of skin-cancer deaths despite representing only
~5 % of all skin-cancer diagnoses.  A **false negative** for Melanoma (i.e., the
model predicts a benign lesion when Melanoma is present) can delay treatment by
months and dramatically worsen prognosis.  Therefore, **recall / sensitivity for
Melanoma must be maximised** even at the cost of additional false positives.

---

## Approach

### 1. Weighted Sampling

A `WeightedRandomSampler` is constructed so that each class is drawn in proportion
to the inverse of its frequency, ensuring that every mini-batch contains a roughly
balanced representation of all seven classes.

### 2. Asymmetric Cost-Sensitive Loss

A custom `AsymmetricCostSensitiveLoss` wraps the standard cross-entropy with a
**cost matrix** `C[true, pred]` that encodes the clinical cost of every
misclassification.  The key design choice is:

```
C[MEL, *] >> C[NV, *]   # missing a Melanoma is far more costly than missing a naevus
```

The full 7 × 7 matrix is defined in `configs/default.yaml` and can be tuned
without changing the source code.

### 3. Model Architecture

`EfficientNet-B3` pre-trained on ImageNet serves as the backbone.  The final
classifier head is replaced with a 7-way linear layer.  Only the classifier head
and the last two `MBConv` blocks are fine-tuned during the first training phase;
all layers are unfrozen in the second phase.

### 4. Training Schedule

| Phase | Epochs | Layers unfrozen | LR     |
|-------|--------|-----------------|--------|
| 1     | 10     | Head + block 6–7| 1e-3   |
| 2     | 20     | All layers      | 1e-4   |

A `CosineAnnealingLR` scheduler is used in both phases.

### 5. Evaluation

Beyond overall accuracy the pipeline reports per-class **sensitivity** (recall),
**specificity**, **F1-score**, **ROC-AUC** and a **confusion matrix** so that
clinical performance can be assessed for each lesion type independently.

---

## Project Layout

```
.
├── configs/
│   └── default.yaml          # All hyperparameters and cost matrix
├── src/
│   ├── data/
│   │   ├── dataset.py        # HAM10000Dataset + WeightedRandomSampler helper
│   │   └── transforms.py     # Train / validation image transforms
│   ├── losses/
│   │   └── cost_sensitive.py # AsymmetricCostSensitiveLoss
│   ├── models/
│   │   └── classifier.py     # EfficientNet-B3 fine-tuned classifier
│   ├── train.py              # Training entry-point
│   └── evaluate.py           # Evaluation entry-point
├── tests/
│   ├── test_dataset.py
│   ├── test_loss.py
│   └── test_model.py
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download HAM10000 and put it under data/
#    Expected layout:
#      data/HAM10000_images_part1/  (*.jpg)
#      data/HAM10000_images_part2/  (*.jpg)
#      data/HAM10000_metadata.csv

# 3. Train
python -m src.train --config configs/default.yaml

# 4. Evaluate
python -m src.evaluate --config configs/default.yaml --checkpoint outputs/best_model.pth
```

---

## Requirements

See `requirements.txt`.  Core dependencies: `torch`, `torchvision`, `timm`,
`scikit-learn`, `pandas`, `Pillow`, `PyYAML`, `tqdm`.
