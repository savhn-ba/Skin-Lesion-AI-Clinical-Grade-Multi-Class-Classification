# Skin-Lesion-AI-Clinical-Grade-Multi-Class-Classification

## Project Overview

This repository contains a deep learning pipeline designed to classify dermatoscopic images into seven diagnostic categories using the **HAM10000 dataset**. The project mimics a clinical decision-support system, prioritizing the detection of high-risk lesions like Melanoma through asymmetric cost-sensitive learning.

### Key Results

  * **Macro-AUC:** 0.89
  * **Melanoma Recall:** 85%
  * **Architecture:** EfficientNet-B4


## Technical Stack & Architecture

  * **Backbone:** `EfficientNet-B4` (chosen for its compound scaling of depth, width, and resolution).
  * **Optimizer:** `AdamW` (with decoupled weight decay for superior regularization).
  * **Loss Function:** `Focal Loss` ($\gamma=2.0$) to focus training on "hard" misclassified examples.
  * **Pre-processing:** Custom implementation of the **DullRazor algorithm** for hair artifact removal.
  * **Augmentation:** `Albumentations` library (Vertical/Horizontal Flips, Random Rotation, Color Jitter).


## Deep Dive: The Imbalance Challenge

The primary obstacle in this project was the extreme class imbalance of the HAM10000 dataset (Melanoma is outnumbered by benign Nevi nearly 10:1).

### The "Volume vs. Weight" Conflict

In this implementation, I assigned **Melanoma a 3.0x weight penalty** in the loss function while keeping common Nevi at 1.0.

  * **The Observation:** Despite the 3x penalty, the model still showed a statistical bias toward the majority class in ambiguous cases.
  * **The Analysis:** This highlights a classic hurdle in Medical AI—the **Volume Gap**. The sheer frequency of 6,700 Nevus images can "dilute" the gradient updates of 1,100 Melanoma images, even with high Focal Loss weights.
  * **The Solution:** I utilized a smaller **Batch Size (16)** to introduce gradient noise, which acted as a regularizer and forced the model to learn more robust features rather than defaulting to the majority class.


## Data Integrity: GroupKFold

To ensure a scientifically valid evaluation, I implemented **GroupKFold cross-validation split by `lesion_id`**.

> **Why this matters:** Medical datasets often contain multiple images of the same lesion. A standard random split causes "Data Leakage," where the model "memorizes" a patient's skin profile in training and recognizes it in testing. My approach ensures the model generalizes to entirely unseen patients.

## How to Run

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/your-username/skin-lesion-ai.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Notebook:** Open `deep-learning-model.ipynb` in Kaggle or Jupyter.

## Future Roadmap

  * [ ] Implement **SMOTE** or Synthetic Data Augmentation to bridge the volume gap.
  * [ ] Ensemble **EfficientNet** with **Vision Transformers (ViT)** for hybrid feature extraction.
  * [ ] Deploy a lightweight version via **Streamlit** for real-time image testing.

## Author

Suruchi Kuamri

  * Undergraduate Student @ FLAME University
  * Major: Business Analytics | Minor: Digital Marketing & Communication
