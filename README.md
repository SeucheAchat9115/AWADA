# AWADA: Foreground-focused adversarial learning for cross-domain object detection

AWADA is an unsupervised domain adaptation framework for object detection that leverages attention-weighted CycleGAN training to focus style transfer on semantically meaningful regions. By using RPN (Region Proposal Network) attention maps from a source-domain Faster R-CNN detector to bias the adversarial training, AWADA produces more faithful style-translated images for cross-domain object detection benchmarks including **sim10k → Cityscapes** and **Cityscapes → Foggy Cityscapes**.

## Method Overview

1. **Baseline detector**: Train Faster R-CNN on the source domain.
2. **Attention map generation**: Extract RPN proposals from the baseline detector to create binary foreground attention masks for each source image.
3. **AWADA CycleGAN**: Train a CycleGAN where adversarial losses are weighted by the attention masks — foreground regions receive weight 1, background regions weight 0 (masked out).
4. **Stylized training**: Stylize the source domain images using the trained generator, then train a new detector on the stylized images.

## Installation

**Requirements**: Python 3.8+, CUDA-capable GPU recommended.

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
requires-python = ">=3.8"
dependencies = [
    "torch>=1.13.0",
    "torchvision>=0.14.0",
    "numpy>=1.21.0",
    "Pillow>=9.0.0",
    "scipy>=1.7.0",
    "tqdm>=4.62.0",
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

## Quick Start

Set your data paths as environment variables, then run any of the four experiment scripts:

```bash
export SIM10K_ROOT=/data/sim10k
export CITYSCAPES_ROOT=/data/cityscapes
export FOGGY_ROOT=/data/foggy_cityscapes
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
```

### Experiment B: Standard CycleGAN

Train CycleGAN, stylize source images, train detector on stylized images:

```bash
bash scripts/exp_b_cyclegan.sh sim10k_to_cityscapes
bash scripts/exp_b_cyclegan.sh cityscapes_to_foggy
```

### Experiment C: AWADA (Attention-Weighted)

Requires Experiment A checkpoint. Generates attention maps, trains AWADA CycleGAN, stylizes and retrains detector:

```bash
bash scripts/exp_c_awada.sh sim10k_to_cityscapes
bash scripts/exp_c_awada.sh cityscapes_to_foggy
```

### Experiment D: Oracle (Upper Bound)

Train and evaluate directly on the target domain with labels:

```bash
bash scripts/exp_d_oracle.sh sim10k_to_cityscapes
bash scripts/exp_d_oracle.sh cityscapes_to_foggy
```

### Optional environment overrides

```bash
export EPOCHS=10          # detector training epochs
export GAN_EPOCHS=200     # GAN training epochs
export BATCH_SIZE=2       # detector batch size
export GAN_BATCH=1        # GAN batch size
export TOP_K=10           # top-k RPN proposals for attention maps
```

## Project Structure

```
AWADA/
├── pyproject.toml
├── configs/
│   └── awada.yaml                 # AWADA hyperparameter config
├── train_detector.py              # Faster R-CNN training script
├── train_cyclegan.py              # Standard CycleGAN training
├── train_awada.py                 # AWADA CycleGAN training (reads configs/awada.yaml)
├── generate_attention_maps.py     # Generate RPN attention maps
├── stylize_dataset.py             # Stylize images with trained generator
├── visualize_inference.py         # Side-by-side visualization of style transfer
├── scripts/
│   ├── exp_a_baseline.sh          # Experiment A: Baseline
│   ├── exp_b_cyclegan.sh          # Experiment B: CycleGAN
│   ├── exp_c_awada.sh             # Experiment C: AWADA
│   └── exp_d_oracle.sh            # Experiment D: Oracle
└── src/
    ├── models/
    │   ├── generator.py            # ResNet-9 generator
    │   ├── discriminator.py        # PatchGAN discriminator
    │   ├── cyclegan.py             # CycleGAN with image replay buffer
    │   └── awada_cyclegan.py       # AWADA variant with masked losses
    ├── datasets/
    │   ├── sim10k.py               # sim10k (Driving in the Matrix) detection dataset
    │   ├── cityscapes.py           # Cityscapes detection dataset
    │   ├── foggy_cityscapes.py     # Foggy Cityscapes detection dataset
    │   └── attention_dataset.py    # Paired dataset for AWADA GAN training
    └── utils/
        ├── attention.py            # RPN attention map generation
        └── metrics.py              # mAP computation (pycocotools)
```

## Evaluation

All experiments report:

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5 (PASCAL VOC metric).
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.50–0.95 in steps of 0.05 (COCO metric).

Evaluation is performed with [pycocotools](https://github.com/cocodataset/cocoapi), the standard COCO evaluation library.

Results are saved to `results.txt` in each experiment's output directory.

## Hyperparameter Configuration

AWADA-specific hyperparameters are stored in `configs/awada.yaml`:

```yaml
epochs: 200
lr: 0.0002
betas: [0.5, 0.999]
lambda_cyc: 10.0
lambda_idt: 5.0
batch_size: 1
patch_size: 128
device: cuda
```

`train_awada.py` loads this file automatically (`--config configs/awada.yaml`).
Any value can be overridden with the corresponding CLI flag, e.g.:

```bash
python train_awada.py --config configs/awada.yaml --lr 0.0001 --epochs 100 ...
```

## Visualization

Use `visualize_inference.py` to run style transfer on a set of images and produce
side-by-side comparison PNGs (original | translated):

```bash
# Visualize AWADA-translated images (source → target direction, A→B)
python visualize_inference.py \
    --checkpoint outputs/awada_gan/awada_epoch_200.pth \
    --input_dir  /data/sim10k/images \
    --output_dir outputs/visualizations

# Visualize only the first 10 images
python visualize_inference.py \
    --checkpoint outputs/cyclegan/cyclegan_epoch_200.pth \
    --input_dir  /data/sim10k/images \
    --output_dir outputs/visualizations \
    --num_images 10

# Use the inverse generator (B → A)
python visualize_inference.py \
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
