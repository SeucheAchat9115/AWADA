# AWADA Domain Adaptation for Object Detection: Evaluation Protocol

This document outlines the experimental setup and evaluation specifications to compare four different training pipelines for unsupervised domain adaptation (UDA) in object detection: Non-Adaptive (Baseline), standard CycleGAN, AWADA, and an Oracle (Upper Bound).

## 1. Experimental Setup

### 1.1 Datasets
To evaluate the domain shift, the dataset must be split into distinct Source and Target domains.
* **Source Domain ($D_S$):** Fully annotated dataset (e.g., synthetic data like GTA5 or Sim10k).
* **Target Domain ($D_T$):** Unannotated dataset for training, annotated for evaluation only (e.g., real-world data like Cityscapes). *Note: For the Oracle experiment, the training split of $D_T$ must be temporarily treated as fully annotated.*

### 1.2 Architectures
* **Object Detector:** Faster R-CNN (ResNet-50 backbone). This two-stage detector will be used consistently across all experiments.
* **GAN Generators:** ResNet-based generator (9 blocks).
* **GAN Discriminators:** PatchGAN discriminator (operating on 128x128 patches).

---

## 2. Evaluation Pipelines

### Experiment A: Non-Adaptive (Source-Only Baseline)
1.  Train Faster R-CNN exclusively on the Source Domain $D_S$ using ground-truth labels.
2.  Evaluate the trained detector directly on the Target Domain validation set $D_T$.

### Experiment B: Standard CycleGAN
1.  Train standard CycleGAN to translate between $D_S$ and $D_T$ (using 128x128 patches).
2.  Generate a stylized dataset $D_{S \to T}$ by passing full Source images through the frozen Generator $G_{S \to T}$.
3.  Train a *new* Faster R-CNN model from scratch on $D_{S \to T}$, using the original $D_S$ bounding box labels.
4.  Evaluate the detector on the Target Domain validation set $D_T$.

### Experiment C: AWADA (Attention-Weighted)
This pipeline uses offline, RPN-guided attention maps to spatially mask adversarial training.
1.  **Offline Attention Generation:** * Patch the trained Faster R-CNN from Experiment A to output intermediate **RPN proposals** rather than final predictions.
    * Run all $D_S$ images through this patched network.
    * Generate full-resolution **binary attention maps** $A(x)$ by setting pixels inside the top RPN proposal boxes to 1 (foreground) and the rest to 0 (background).
2.  **Synchronized Patch Training:** Train the AWADA CycleGAN on 128x128 patches from $D_S$ and $D_T$. During the dataloader phase, synchronously crop the image patch and its corresponding 128x128 attention patch $A_{patch}(x)$.
3.  **Masked Adversarial Loss:** Compute the GAN losses. Apply the binary attention map $A_{patch}(x)$ to **spatially mask only the adversarial losses** (Generator and Discriminator). The cycle-consistency and identity losses remain unmasked to allow global style transfer.
4.  **Stylization:** Generate the stylized dataset $D_{S \to T\_awada}$ by passing full, uncropped Source images through the frozen AWADA Generator $G_{S \to T}$.
5.  **Detector Training:** Train a *new* Faster R-CNN model on $D_{S \to T\_awada}$ using the original $D_S$ labels.
6.  **Evaluation:** Evaluate the detector on the Target Domain validation set $D_T$.

### Experiment D: Oracle (Target-Only Upper Bound)
1.  Train Faster R-CNN exclusively on the Target Domain training set $D_T$ using its ground-truth labels.
2.  Evaluate the trained detector on the Target Domain validation set $D_T$.

---

## 3. Evaluation Metrics

The primary metric for comparison is **Mean Average Precision (mAP)**. 

### Expected Results Table

| Method | Training Data | Evaluation Data | Adapt Setup | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Lower Bound (Baseline)** | $D_S$ | $D_T$ | None | - | - |
| **Standard CycleGAN** | $D_{S \to T}$ | $D_T$ | Global | - | - |
| **AWADA** | $D_{S \to T\_awada}$| $D_T$ | RPN Patch Masking | - | - |
| **Upper Bound (Oracle)** | $D_T$ | $D_T$ | None (Fully Supervised) | - | - |
