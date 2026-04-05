# AWADA: Foreground-focused adversarial learning for cross-domain object detection

AWADA is an unsupervised domain adaptation framework for object detection that leverages attention-weighted CycleGAN training to focus style transfer on semantically meaningful regions. By using RPN (Region Proposal Network) attention maps from a source-domain Faster R-CNN detector to bias the adversarial training, AWADA produces more faithful style-translated images for cross-domain object detection benchmarks including **sim10k → Cityscapes**, **Cityscapes → Foggy Cityscapes**, and **Cityscapes → BDD100K**.

## Model Hierarchy

Three models are implemented, each building on the previous one:

1. **CycleGAN** — standard unpaired image-to-image translation with GAN, cycle-consistency, and optional identity losses.
2. **CyCada** *(inherits from CycleGAN)* — adds an unmasked semantic consistency loss backed by a frozen DeepLabV3 segmentation network, encouraging the generator to preserve semantic structure across domains (Hoffman et al., CyCADA, ICML 2018).
3. **AWADA** *(inherits from CyCada)* — replaces unmasked GAN losses with attention-masked adversarial losses so that adversarial training focuses on foreground objects identified by RPN proposals from a source-trained detector.

## Method Overview

1. **Baseline detector**: Train Faster R-CNN on the source domain.
2. **Attention map generation**: Extract RPN proposals from the baseline detector to create binary foreground attention masks for each source image.
3. **Style transfer (choose one)**:
   - *CycleGAN*: Standard unpaired translation, no domain knowledge.
   - *CyCada*: CycleGAN + semantic consistency loss to preserve semantics.
   - *AWADA*: CyCada + attention-masked adversarial losses to focus on foreground objects.
4. **Stylized training**: Stylize the source domain images using the trained generator, then train a new detector on the stylized images.

## Installation

**Requirements**: Python 3.11+, CUDA-capable GPU recommended.

Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or install in an existing environment:

```bash
uv pip install -e .
```

### pyproject.toml

```toml
[project]
name = "awada"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0.0,<2.5.0",
    "torchvision>=0.15.0,<0.20.0",
    "numpy>=1.21.0",
    "Pillow>=9.0.0",
    "scipy>=1.7.0",
    "tqdm>=4.62.0",
    "pycocotools>=2.0.7",
    "pyyaml>=6.0",
]
```

## Dataset Setup

### sim10k (Driving in the Matrix)

Download from the [Driving in the Matrix](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) project. Expected structure:

```
/data/sim10k/
├── images/       # JPEG images: 00001.jpg, 00002.jpg, ...
└── Annotations/  # PASCAL VOC XML files (one per image, car class only)
```

### Cityscapes

Download from the [Cityscapes dataset](https://www.cityscapes-dataset.com/) (requires registration). Expected structure:

```
/data/cityscapes/
├── leftImg8bit/
│   ├── train/{city}/*_leftImg8bit.png
│   └── val/{city}/*_leftImg8bit.png
└── gtFine/
    ├── train/{city}/*_gtFine_instanceIds.png
    └── val/{city}/*_gtFine_instanceIds.png
```

### Foggy Cityscapes

Download from the [Cityscapes dataset](https://www.cityscapes-dataset.com/) (Foggy Cityscapes package). Expected structure:

```
/data/foggy_cityscapes/
├── leftImg8bit_foggy/
│   ├── train/{city}/*_leftImg8bit_foggy_beta_0.02.png
│   └── val/{city}/*_leftImg8bit_foggy_beta_0.02.png
└── gtFine/
    ├── train/{city}/*_gtFine_instanceIds.png
    └── val/{city}/*_gtFine_instanceIds.png
```

### BDD100K

Download from the [BDD100K dataset](https://bdd-data.berkeley.edu/) (requires registration).

The dataloader **automatically detects** the annotation format — no manual preprocessing is required.  Three annotation layouts are supported (tried in order):

#### Option 1 — Pre-generated det_20 JSON (fastest)

Use the detection-specific download (`bdd100k_det_20_labels_trainval.zip`):

```
/data/bdd100k/
├── images/
│   └── 100k/
│       ├── train/    # JPEG images
│       └── val/
└── labels/
    └── det_20/
        ├── det_train.json
        └── det_val.json
```

#### Option 2 — Full scalabel labels (raw download)

Use the full labels download (`bdd100k_labels_images_trainval.zip`).  The
dataloader extracts the detection boxes automatically:

```
/data/bdd100k/
├── images/
│   └── 100k/
│       ├── train/
│       └── val/
└── labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

#### Option 3 — CSV raw format (non-JSON)

Prepare a flat CSV file with one bounding-box per row
(columns: `name,category,x1,y1,x2,y2`).  This is useful when annotations
come from custom tooling or are prepared manually:

```
/data/bdd100k/
├── images/
│   └── 100k/
│       ├── train/
│       └── val/
└── labels/
    └── det_20/
        ├── det_train.csv
        └── det_val.csv
```

Example CSV content:

```
name,category,x1,y1,x2,y2
0000f77c-6257be58.jpg,car,498.1,155.5,512.0,169.5
0000f77c-6257be58.jpg,pedestrian,100.0,50.0,130.0,150.0
0001a0e2-8d4ad1e7.jpg,truck,20.0,30.0,100.0,80.0
```

#### Generating det_20 JSON from raw annotations

You can also pre-generate the det_20 JSON from either of the raw formats
(Options 2 or 3) using the `generate_det_json` utility — for example to
cache the filtered result for faster subsequent loads:

```python
from awada.datasets.bdd100k import generate_det_json

# From full scalabel labels
generate_det_json(
    "/data/bdd100k/labels/bdd100k_labels_images_train.json",
    "/data/bdd100k/labels/det_20/det_train.json",
)

# From CSV
generate_det_json(
    "/data/bdd100k/labels/det_20/det_train.csv",
    "/data/bdd100k/labels/det_20/det_train.json",
)
```

The Cityscapes → BDD100K benchmark uses 7 shared classes: **pedestrian, rider, car, truck, bus, motorcycle, bicycle**. The `train` class present in Cityscapes has no reliable equivalent in BDD100K and is excluded from both sides of the benchmark.

## Quick Start

Set your data paths as environment variables, then run any of the four experiment scripts:

```bash
export SIM10K_ROOT=/data/sim10k
export CITYSCAPES_ROOT=/data/cityscapes
export FOGGY_ROOT=/data/foggy_cityscapes
export BDD100K_ROOT=/data/bdd100k
export OUTPUT_ROOT=./outputs
export DEVICE=cuda
```

### Experiment A: Non-Adaptive Baseline

Train on source domain, evaluate directly on target domain (no adaptation):

```bash
# sim10k → Cityscapes
bash scripts/exp_a_baseline.sh sim10k_to_cityscapes

# Cityscapes → Foggy Cityscapes
bash scripts/exp_a_baseline.sh cityscapes_to_foggy

# Cityscapes → BDD100K
bash scripts/exp_a_baseline.sh cityscapes_to_bdd100k
```

### Experiment B: Standard CycleGAN

Train CycleGAN, stylize source images, train detector on stylized images:

```bash
bash scripts/exp_b_cyclegan.sh sim10k_to_cityscapes
bash scripts/exp_b_cyclegan.sh cityscapes_to_foggy
bash scripts/exp_b_cyclegan.sh cityscapes_to_bdd100k
```

### Experiment B2: CyCada (CycleGAN + semantic consistency loss)

Same pipeline as Experiment B but with semantic consistency loss enabled:

```bash
bash scripts/exp_b_cycada.sh sim10k_to_cityscapes
bash scripts/exp_b_cycada.sh cityscapes_to_foggy
bash scripts/exp_b_cycada.sh cityscapes_to_bdd100k
```

### Experiment C: AWADA (Attention-Weighted)

Requires Experiment A checkpoint. Generates attention maps, trains AWADA CycleGAN, stylizes and retrains detector:

```bash
bash scripts/exp_c_awada.sh sim10k_to_cityscapes
bash scripts/exp_c_awada.sh cityscapes_to_foggy
bash scripts/exp_c_awada.sh cityscapes_to_bdd100k
```

### Experiment D: Oracle (Upper Bound)

Train and evaluate directly on the target domain with labels:

```bash
bash scripts/exp_d_oracle.sh sim10k_to_cityscapes
bash scripts/exp_d_oracle.sh cityscapes_to_foggy
bash scripts/exp_d_oracle.sh cityscapes_to_bdd100k
```

## Attention Map Generation

Before training the AWADA CycleGAN you need binary foreground attention masks for every source-domain image. `tools/generate_attention_maps.py` extracts the top-K RPN proposals from a trained Faster R-CNN checkpoint and saves one `.npy` mask per image (float32, 0 or 1, same H × W as the input image).

**Arguments**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--detector_checkpoint` | ✓ | — | Path to the trained Faster R-CNN checkpoint (`.pth`) |
| `--dataset` | ✓ | — | Source dataset: `sim10k`, `cityscapes`, `foggy_cityscapes`, or `bdd100k` |
| `--data_root` | ✓ | — | Root directory of the source dataset |
| `--output_dir` | ✓ | — | Directory where `.npy` attention maps are written |
| `--num_classes` | ✓ | — | Number of foreground classes (1 for sim10k, 8 for Cityscapes) |
| `--top_k` | | `10` | Number of top RPN proposals used to build each mask |
| `--split` | | `train` | Dataset split to process (`train` / `val`) |
| `--device` | | `cuda` | Compute device (`cuda` or `cpu`) |

**Examples**

```bash
# sim10k → Cityscapes (1 foreground class: car)
python tools/generate_attention_maps.py \
    --detector_checkpoint outputs/exp_a_sim10k2cs/detector_final.pth \
    --dataset            sim10k \
    --data_root          /data/sim10k \
    --output_dir         outputs/exp_c_sim10k2cs/attention_maps \
    --num_classes        1 \
    --top_k              10 \
    --split              train \
    --device             cuda

# Cityscapes → Foggy Cityscapes (8 foreground classes)
python tools/generate_attention_maps.py \
    --detector_checkpoint outputs/exp_a_cs2foggy/detector_final.pth \
    --dataset            cityscapes \
    --data_root          /data/cityscapes \
    --output_dir         outputs/exp_c_cs2foggy/attention_maps \
    --num_classes        8 \
    --top_k              10 \
    --split              train \
    --device             cuda

# Cityscapes → BDD100K (7 foreground classes, source attention maps)
python tools/generate_attention_maps.py \
    --detector_checkpoint outputs/exp_a_cs2bdd/detector_final.pth \
    --dataset            cityscapes \
    --data_root          /data/cityscapes \
    --output_dir         outputs/exp_c_cs2bdd/source_attention_maps \
    --num_classes        7 \
    --top_k              10 \
    --split              train \
    --device             cuda

# Cityscapes → BDD100K (7 foreground classes, target attention maps)
python tools/generate_attention_maps.py \
    --detector_checkpoint outputs/exp_c_cs2bdd/cycada_detector/detector_final.pth \
    --dataset            bdd100k \
    --data_root          /data/bdd100k \
    --output_dir         outputs/exp_c_cs2bdd/target_attention_maps \
    --num_classes        7 \
    --top_k              10 \
    --split              train \
    --device             cuda
```

The generated masks are automatically consumed by `tools/train_awada.py` via the `--attention_dir` flag. When running Experiment C end-to-end, `scripts/exp_c_awada.sh` calls this script automatically; the examples above are useful when you need to regenerate or inspect the masks independently.

## Project Structure

```
AWADA/
├── pyproject.toml
├── configs/
│   ├── cyclegan.yaml              # CycleGAN hyperparameter config
│   ├── cycada.yaml                # CyCada hyperparameter config (lambda_sem > 0)
│   └── awada.yaml                 # AWADA hyperparameter config
├── tools/
│   ├── train_detector.py          # Faster R-CNN training script
│   ├── evaluate_detector.py       # Faster R-CNN evaluation script
│   ├── train_cyclegan.py          # CycleGAN training (reads configs/cyclegan.yaml)
│   ├── train_cycada.py            # CyCada training (reads configs/cycada.yaml)
│   ├── train_awada.py             # AWADA training (reads configs/awada.yaml)
│   ├── generate_attention_maps.py # Generate RPN attention maps
│   ├── stylize_dataset.py         # Stylize images with trained generator
│   └── visualize_inference.py     # Side-by-side visualization of style transfer
├── scripts/
│   ├── exp_a_baseline.sh          # Experiment A: Baseline
│   ├── exp_b_cyclegan.sh          # Experiment B: CycleGAN
│   ├── exp_b_cycada.sh            # Experiment B2: CyCada
│   ├── exp_c_awada.sh             # Experiment C: AWADA
│   └── exp_d_oracle.sh            # Experiment D: Oracle
└── awada/
    ├── config.py                   # Centralised configuration constants
    ├── models/
    │   ├── generator.py            # ResNet-9 generator
    │   ├── discriminator.py        # PatchGAN discriminator
    │   ├── cyclegan.py             # CycleGAN with image replay buffer
    │   ├── cycada.py               # CyCada: CycleGAN + semantic consistency loss
    │   ├── awada.py                # AWADA: CyCada + attention-masked adversarial losses
    │   └── semantic_loss.py        # DeepLabV3-backed semantic consistency loss
    ├── datasets/
    │   ├── sim10k.py               # sim10k (Driving in the Matrix) detection dataset
    │   ├── cityscapes.py           # Cityscapes detection dataset
    │   ├── foggy_cityscapes.py     # Foggy Cityscapes detection dataset
    │   ├── bdd100k.py              # BDD100K detection dataset (7-class Cityscapes-aligned)
    │   ├── attention_dataset.py    # Paired dataset for AWADA GAN training
    │   └── unpaired_dataset.py     # Unpaired dataset for CycleGAN/CyCada training
    └── utils/
        ├── attention.py            # RPN attention map generation
        ├── metrics.py              # mAP computation (pycocotools)
        ├── train_utils.py          # Shared training utilities (seed, logging, LR schedule)
        └── transforms.py          # Image transforms and augmentations
```

## Evaluation

All experiments report:

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5 (PASCAL VOC metric).
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.50–0.95 in steps of 0.05 (COCO metric).

Evaluation is performed with [pycocotools](https://github.com/cocodataset/cocoapi), the standard COCO evaluation library.

Results are saved to `results.txt` in each experiment's output directory.

## Hyperparameter Configuration

Model-specific hyperparameters are stored in `configs/`:

| Config | Model | Key difference |
|---|---|---|
| `configs/cyclegan.yaml` | CycleGAN | `lambda_sem: 0.0` (no semantic loss) |
| `configs/cycada.yaml` | CyCada | `lambda_sem: 1.0` (semantic loss enabled) |
| `configs/awada.yaml` | AWADA | `lambda_sem: 0.0` by default (add `--lambda_sem 1.0` to enable) |

Example `configs/cycada.yaml`:

```yaml
epochs: 200
lr: 0.0002
betas: [0.5, 0.999]
lambda_cyc: 10.0
lambda_gan: 1.0
lambda_idt: 0.0
lambda_sem: 1.0
batch_size: 1
patch_size: 128
device: cuda
```

Each training script loads its own config file automatically. Any value can be overridden with the corresponding CLI flag, e.g.:

```bash
python tools/train_cycada.py --config configs/cycada.yaml --lr 0.0001 --epochs 100 ...
python tools/train_awada.py --config configs/awada.yaml --lambda_sem 1.0 ...
```

## Resuming Training

All three GAN training scripts support fully resumable training via the `--resume` flag. Checkpoints saved during training include not only the model weights but also the optimizer and learning-rate scheduler states, so training can be continued from any saved epoch without loss of training dynamics.

### Checkpoint format

Every `.pth` file produced by the training scripts contains:

| Key | Contents |
|---|---|
| `epoch` | Epoch number at which the checkpoint was saved (1-based) |
| `G_AB` | Generator A→B weights |
| `G_BA` | Generator B→A weights |
| `D_A` | Discriminator A weights |
| `D_B` | Discriminator B weights |
| `opt_G` | Generator Adam optimizer state (momentum buffers, step count) |
| `opt_D` | Discriminator Adam optimizer state |
| `sched_G` | Generator LR scheduler state |
| `sched_D` | Discriminator LR scheduler state |

### Usage

Pass the path of any previously saved checkpoint to `--resume`. Training will restore all states and continue from the next epoch:

```bash
# Resume CycleGAN from epoch 50 (continues at epoch 51)
python tools/train_cyclegan.py \
    --source_dir /data/sim10k/images \
    --target_dir /data/cityscapes/leftImg8bit/train \
    --output_dir outputs/cyclegan \
    --epochs     200 \
    --resume     outputs/cyclegan/cyclegan_epoch_50.pth

# Resume CyCada from epoch 100
python tools/train_cycada.py \
    --source_dir /data/sim10k/images \
    --target_dir /data/cityscapes/leftImg8bit/train \
    --output_dir outputs/cycada \
    --epochs     200 \
    --resume     outputs/cycada/cycada_epoch_100.pth

# Resume AWADA from epoch 75
python tools/train_awada.py \
    --source_dir         /data/sim10k/images \
    --target_dir         /data/cityscapes/leftImg8bit/train \
    --attention_dir      outputs/attention_maps \
    --output_dir         outputs/awada \
    --epochs             200 \
    --resume             outputs/awada/awada_epoch_75.pth
```

`--epochs` must still be set to the **total** number of epochs for the run (not the number of remaining epochs). Training will skip epochs already completed according to the checkpoint.

## Visualization

Use `tools/visualize_inference.py` to run style transfer on a set of images and produce
side-by-side comparison PNGs (original | translated):

```bash
# Visualize AWADA-translated images (source → target direction, A→B)
python tools/visualize_inference.py \
    --checkpoint outputs/awada_gan/awada_epoch_200.pth \
    --input_dir  /data/sim10k/images \
    --output_dir outputs/visualizations

# Visualize only the first 10 images
python tools/visualize_inference.py \
    --checkpoint outputs/cyclegan/cyclegan_epoch_200.pth \
    --input_dir  /data/sim10k/images \
    --output_dir outputs/visualizations \
    --num_images 10

# Use the inverse generator (B → A)
python tools/visualize_inference.py \
    --checkpoint outputs/awada_gan/awada_epoch_200.pth \
    --input_dir  /data/cityscapes/leftImg8bit/val/aachen \
    --output_dir outputs/visualizations \
    --direction  BA
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{menkeAWADA2024,
  title   = {AWADA: Foreground-focused adversarial learning for cross-domain object detection},
  journal = {Computer Vision and Image Understanding},
  volume  = {249},
  pages   = {104153},
  year    = {2024},
  issn    = {1077-3142},
  doi     = {10.1016/j.cviu.2024.104153},
  url     = {https://www.sciencedirect.com/science/article/pii/S1077314224002340},
  author  = {Maximilian Menke and Thomas Wenzel and Andreas Schwung},
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
