# AWADA Implementation Plan

This document identifies every gap between the current repository implementation and the
AWADA paper (**Menke et al., "AWADA: Foreground-Focused Adversarial Learning for
Cross-Domain Object Detection"**, CVIU 2024) and provides a concrete plan for closing
each gap.

---

## 1  Gap Analysis

### 1.1  Missing Semantic Consistency Loss (L_sem)

**Paper**  
AWADA builds on CyCADA, whose total style-transfer objective is

```
L_total = w_gen · L_gen + w_disc · L_disc + w_cyc · L_cyc + w_sem · L_sem
```

`L_sem` is a semantic (feature-level) consistency loss that forces the translated image
to preserve high-level content by matching intermediate feature maps of a frozen,
pre-trained network (e.g. VGG-16) between the original and translated image.

The ablation study (Section 5) explicitly shows that **masking `L_sem` degrades
results** — confirming it is active during training and is kept *unmasked* (applied to
the full image, not just foreground).

**Current implementation**  
`CycleGAN.compute_generator_loss()` and `AWADACycleGAN.compute_generator_loss()`
compute only `L_gen + L_cyc`. There is no semantic loss term, no `lambda_sem`
hyper-parameter, and no feature-extractor network anywhere in the codebase.
`configs/awada.yaml` and `configs/cyclegan.yaml` have no `lambda_sem` entry.

**Required changes**

| File | Change |
|------|--------|
| `src/models/cyclegan.py` | Add optional `SemanticLoss` module (frozen VGG-16 features); compute `L_sem` in `compute_generator_loss()`; add `lambda_sem` parameter (default `1.0`) |
| `src/models/awada_cyclegan.py` | Inherit the semantic loss from `CycleGAN`; ensure `L_sem` is **not** masked by the AWM (consistent with the ablation finding) |
| `configs/awada.yaml` | Add `lambda_sem: 1.0` |
| `configs/cyclegan.yaml` | Add `lambda_sem: 1.0` |
| `train_awada.py` | Expose `--lambda_sem` CLI flag; pass it to `compute_generator_loss()` |
| `train_cyclegan.py` | Expose `--lambda_sem` CLI flag; pass it to `compute_generator_loss()` |
| `tests/test_awada_cyclegan.py` | Add tests that verify `L_sem` is present in loss dict and is unmasked |
| `tests/test_cyclegan.py` | Add tests for semantic loss path |

---

### 1.2  Incomplete 5-Step Training Pipeline

**Paper** (Section 3)

The full AWADA training pipeline has **five** steps:

1. Train a baseline detector on original source images → generate **source** attention maps.
2. Train a **preliminary** style-transfer model *without* AWMs (standard CycleGAN) →
   produce roughly stylised source images.
3. Train a **second** detector on the stylised source images → run it on **target** images
   to generate **target** attention maps.
4. Train the **final** AWADA style-transfer network using AWMs and attention maps for
   **both** domains.
5. Train the final detector on AWADA-stylised source images for target-domain inference.

**Current implementation**  
`scripts/exp_c_awada.sh` implements only four steps:

1. Generate source attention maps from the Experiment A detector.
2. Train AWADA CycleGAN (with AWMs, but using *only* source attention maps).
3. Stylise source images.
4. Train detector; evaluate.

Steps 2 and 3 of the paper's pipeline are entirely absent. Although `train_awada.py`
accepts an optional `--target_attention_dir` argument, the shell script never passes it,
so the AWADA network trains without target-domain attention maps — contradicting the
paper.

**Required changes**

| File | Change |
|------|--------|
| `scripts/exp_c_awada.sh` | Add Step 2: train preliminary CycleGAN (no AWMs) and stylise source images to a temporary directory |
| `scripts/exp_c_awada.sh` | Add Step 3: train a second detector on the stylised source images, then call `generate_attention_maps.py` on the **target** images to produce target attention maps |
| `scripts/exp_c_awada.sh` | Pass `--target_attention_dir` to `train_awada.py` in the existing AWADA training step |
| `train_awada.py` | Ensure `--target_attention_dir` is clearly documented as part of the full pipeline |

---

### 1.3  Missing BDD100k Benchmark (Cityscapes → BDD100k)

**Paper** (Section 4)

AWADA is evaluated on **three** domain-adaptation benchmarks:

| Benchmark | Classes | AWADA mAP@0.5 |
|-----------|---------|---------------|
| sim10k → Cityscapes | 1 (car) | 54.1 |
| Cityscapes → Foggy Cityscapes | 8 | 44.8 |
| **Cityscapes → BDD100k** | **7** | **31.5** |

**Current implementation**  
Only two benchmarks are implemented. There is no BDD100k dataset class, no experiment
script for this benchmark, and BDD100k is not an accepted choice in
`generate_attention_maps.py` or `train_detector.py`.

**Required changes**

| File | Change |
|------|--------|
| `src/datasets/bdd100k.py` | New dataset class supporting the 7 classes shared with Cityscapes: person, rider, car, truck, bus, motorcycle, bicycle (no `train` class) |
| `generate_attention_maps.py` | Add `bdd100k` to `--dataset` choices |
| `train_detector.py` | Add `bdd100k` to `--dataset` choices |
| `scripts/exp_c_awada.sh` | Add `cityscapes_to_bdd100k` benchmark variant |
| `scripts/exp_a_baseline.sh` | Add `cityscapes_to_bdd100k` benchmark variant |
| `scripts/exp_b_cyclegan.sh` | Add `cityscapes_to_bdd100k` benchmark variant |
| `scripts/exp_d_oracle.sh` | Add `cityscapes_to_bdd100k` benchmark variant |
| `tests/test_bdd100k.py` | Unit tests for the new dataset class |
| `README.md` | Document BDD100k dataset setup and the new benchmark variant |

---

### 1.4  Missing Identity Loss Support in `train_cyclegan.py`

**Paper / CyCADA**  
The CyCADA baseline (on which AWADA is built) uses an identity loss (`L_idt`) that
penalises the generator when it changes images already in the target domain. This is a
standard CycleGAN extension.

**Current implementation**  
`scripts/exp_b_cyclegan.sh` already passes `--lambda_idt 5.0` to `train_cyclegan.py`,
but `train_cyclegan.py` does **not** define that CLI argument. Running Experiment B as
written will raise an `argparse` error. `CycleGAN.compute_generator_loss()` has no
identity loss branch. The README configuration example shows `lambda_idt: 5.0`, but
neither `configs/cyclegan.yaml` nor `configs/awada.yaml` contains that key.

**Required changes**

| File | Change |
|------|--------|
| `src/models/cyclegan.py` | Add identity loss: `L_idt_A = L1(G_BA(real_A), real_A)`, `L_idt_B = L1(G_AB(real_B), real_B)`; add `lambda_idt` parameter to `compute_generator_loss()` |
| `train_cyclegan.py` | Add `--lambda_idt` CLI argument; pass it to `compute_generator_loss()` |
| `configs/cyclegan.yaml` | Add `lambda_idt: 5.0` |
| `configs/awada.yaml` | Add `lambda_idt: 5.0` (AWADA also inherits this from CycleGAN) |
| `train_awada.py` | Add `--lambda_idt` CLI argument for consistency |
| `tests/test_cyclegan.py` | Add tests verifying identity loss keys and values |

---

### 1.5  README / Documentation Inconsistencies

Several discrepancies exist between the README and the actual code:

| Location | Documented | Actual code | Fix |
|----------|-----------|-------------|-----|
| README "Attention Map Generation" table | `--top_k` (default `10`) | `--score_threshold` (default `0.5`) | Update README table to `--score_threshold` |
| README "Optional environment overrides" | `TOP_K=10` | Not used anywhere | Remove `TOP_K` from README |
| README "Attention Map Generation" prose | "top-K RPN proposals" | Score-threshold filtering | Update prose to match implementation |
| README config example | `lambda_idt: 5.0` | Key absent in `configs/awada.yaml` | Add `lambda_idt` to config (see §1.4) or remove from README |
| `scripts/exp_c_awada.sh` usage comment | Only 2 benchmark options | Three in paper | Add BDD100k option (see §1.3) |

**Required changes**

| File | Change |
|------|--------|
| `README.md` | Replace `--top_k` / `TOP_K` references with `--score_threshold` / `SCORE_THRESHOLD` |
| `README.md` | Add BDD100k dataset setup section and `cityscapes_to_bdd100k` usage examples |
| `README.md` | Align the config example block with the actual YAML files after §1.4 changes |

---

## 2  Prioritised Implementation Roadmap

| Priority | Item | Estimated effort | Impact |
|----------|------|-----------------|--------|
| P1 | **§1.4** – Identity loss bug in `train_cyclegan.py` | Small | Unblocks Experiment B |
| P2 | **§1.2** – Complete 5-step AWADA pipeline in `exp_c_awada.sh` | Medium | Core paper contribution |
| P3 | **§1.1** – Semantic consistency loss (L_sem) | Medium–Large | Core paper contribution |
| P4 | **§1.3** – BDD100k dataset + benchmark scripts | Medium | Third evaluation benchmark |
| P5 | **§1.5** – README/documentation fixes | Small | Correctness & usability |

---

## 3  Detailed Implementation Notes

### 3.1  Semantic Loss Module

The recommended implementation mirrors CyCADA (Hoffman et al., 2018):

```python
# src/models/semantic_loss.py
import torch
import torch.nn as nn
from torchvision.models import vgg16

class SemanticConsistencyLoss(nn.Module):
    """Feature-level perceptual loss using frozen VGG-16 relu3_3 activations."""

    def __init__(self):
        super().__init__()
        vgg = vgg16(weights="IMAGENET1K_V1")
        # Use features up to relu3_3 (index 15)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16])
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)
        self.criterion = nn.L1Loss()

    def forward(self, translated: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        feats_t = self.feature_extractor(translated)
        feats_o = self.feature_extractor(original)
        return self.criterion(feats_t, feats_o)
```

Integration into `CycleGAN.compute_generator_loss()`:

```python
# Both directions; loss is NOT masked (global image regularisation)
if lambda_sem > 0 and hasattr(self, "criterion_sem"):
    loss_sem_AB = self.criterion_sem(self.fake_B, self.real_A) * lambda_sem
    loss_sem_BA = self.criterion_sem(self.fake_A, self.real_B) * lambda_sem
    total = total + loss_sem_AB + loss_sem_BA
```

### 3.2  Completing the 5-Step Pipeline (`exp_c_awada.sh`)

The shell script needs two new steps inserted between the existing Step 1 (source
attention map generation) and the existing AWADA training step:

```bash
# [New Step 2] Train preliminary CycleGAN (no attention / no AWMs)
python train_cyclegan.py \
    --source_dir "$SOURCE_IMAGES" \
    --target_dir "$TARGET_IMAGES" \
    --output_dir "$PRELIM_GAN_OUTPUT" \
    --epochs "${PRELIM_GAN_EPOCHS:-100}" \
    --device "${DEVICE:-cuda}"

# Stylise source images with preliminary generator
PRELIM_GAN=$(ls -t "$PRELIM_GAN_OUTPUT"/cyclegan_epoch_*.pth | head -1)
python stylize_dataset.py \
    --generator_checkpoint "$PRELIM_GAN" \
    --source_dir "$SOURCE_IMAGES" \
    --output_dir "$STYLIZED_PRELIM_DIR" \
    --device "${DEVICE:-cuda}"

# [New Step 3] Train second detector on stylised images → target attention maps
python train_detector.py \
    --dataset "$SOURCE_DATASET" \
    --data_root "$SOURCE_ROOT" \
    --stylized_images_dir "$STYLIZED_PRELIM_DIR" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$SECOND_DETECTOR_OUTPUT" \
    --epochs "${DET_EPOCHS:-10}" \
    --device "${DEVICE:-cuda}"

python generate_attention_maps.py \
    --detector_checkpoint "$SECOND_DETECTOR_OUTPUT/detector_final.pth" \
    --dataset "$TARGET_DATASET" \
    --data_root "$TARGET_ROOT" \
    --output_dir "$TARGET_ATTENTION_DIR" \
    --score_threshold "${SCORE_THRESHOLD:-0.5}" \
    --num_classes "$NUM_CLASSES" \
    --split train \
    --device "${DEVICE:-cuda}"
```

Then pass `--target_attention_dir "$TARGET_ATTENTION_DIR"` to the existing
`train_awada.py` call.

### 3.3  BDD100k Dataset Class

BDD100k stores annotations in a single JSON file
(`bdd100k/labels/det_20/det_train.json`). The 7 classes shared with Cityscapes are:

```python
BDD100K_LABEL_MAP = {
    "person":     1,
    "rider":      2,
    "car":        3,
    "truck":      4,
    "bus":        5,
    "motorcycle": 6,
    "bicycle":    7,
}
```

The dataset class should follow the same interface as `CityscapesDetectionDataset`
(returns `(image_tensor, target_dict)` where `target_dict` contains `boxes`, `labels`,
and `image_id`).

### 3.4  Identity Loss Integration

```python
# In CycleGAN.compute_generator_loss()
if lambda_idt > 0:
    loss_idt_A = self.criterion_cycle(self.G_BA(self.real_A), self.real_A) * lambda_idt
    loss_idt_B = self.criterion_cycle(self.G_AB(self.real_B), self.real_B) * lambda_idt
    total = total + loss_idt_A + loss_idt_B
```

The AWM in `AWADACycleGAN` should **not** mask the identity loss (same reasoning as for
`L_cyc` and `L_sem` — it is a global regulariser, not an adversarial loss).

---

## 4  Summary Table

| # | Gap | Paper section | Files affected | Status |
|---|-----|--------------|----------------|--------|
| 1 | Semantic consistency loss `L_sem` missing | §2, eq. (total loss) | `cyclegan.py`, `awada_cyclegan.py`, both configs, training scripts | ❌ Not implemented |
| 2 | 5-step pipeline incomplete (no preliminary GAN, no target attention maps) | §3 | `exp_c_awada.sh` | ❌ Not implemented |
| 3 | BDD100k benchmark missing | §4 | `bdd100k.py`, all experiment scripts, `train_detector.py`, `generate_attention_maps.py` | ❌ Not implemented |
| 4 | Identity loss (`L_idt`) missing; `--lambda_idt` flag breaks `exp_b_cyclegan.sh` | CyCADA baseline | `cyclegan.py`, `train_cyclegan.py`, both configs | ❌ Not implemented / broken |
| 5 | README `--top_k` / `TOP_K` do not match code (`--score_threshold`) | — | `README.md` | ❌ Documentation error |
