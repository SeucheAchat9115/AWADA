# AWADA: Attention-Weighted Domain Adaptation for Object Detection

AWADA is an unsupervised domain adaptation framework for object detection that leverages attention-weighted CycleGAN training to focus style transfer on semantically meaningful regions. By using RPN (Region Proposal Network) attention maps from a source-domain Faster R-CNN detector to bias the adversarial training, AWADA produces more faithful style-translated images for cross-domain object detection benchmarks including **GTA5 → Cityscapes** and **Cityscapes → Foggy Cityscapes**.

## Method Overview

1. **Baseline detector**: Train Faster R-CNN on the source domain.
2. **Attention map generation**: Extract RPN proposals from the baseline detector to create binary foreground attention masks for each source image.
3. **AWADA CycleGAN**: Train a CycleGAN where adversarial losses are weighted by the attention masks — foreground regions receive 2× weight, background 1×.
4. **Stylized training**: Stylize the source domain images using the trained generator, then train a new detector on the stylized images.

## Installation

**Requirements**: Python 3.8+, CUDA-capable GPU recommended.

```bash
pip install -r requirements.txt
```

### requirements.txt

```
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.21.0
Pillow>=9.0.0
scipy>=1.7.0
tqdm>=4.62.0
```

## Dataset Setup

### GTA5

Download from the [Playing for Data](https://download.visinf.tu-darmstadt.de/data/from_games/) project. Expected structure:

```
/data/gta5/
├── images/         # PNG images: 00001.png, 00002.png, ...
└── labels/         # PNG semantic label maps (same filenames)
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
export GTA5_ROOT=/data/gta5
export CITYSCAPES_ROOT=/data/cityscapes
export FOGGY_ROOT=/data/foggy_cityscapes
export OUTPUT_ROOT=./outputs
export DEVICE=cuda
```

### Experiment A: Non-Adaptive Baseline

Train on source domain, evaluate directly on target domain (no adaptation):

```bash
# GTA5 → Cityscapes
bash scripts/exp_a_baseline.sh gta5_to_cityscapes

# Cityscapes → Foggy Cityscapes
bash scripts/exp_a_baseline.sh cityscapes_to_foggy
```

### Experiment B: Standard CycleGAN

Train CycleGAN, stylize source images, train detector on stylized images:

```bash
bash scripts/exp_b_cyclegan.sh gta5_to_cityscapes
bash scripts/exp_b_cyclegan.sh cityscapes_to_foggy
```

### Experiment C: AWADA (Attention-Weighted)

Requires Experiment A checkpoint. Generates attention maps, trains AWADA CycleGAN, stylizes and retrains detector:

```bash
bash scripts/exp_c_awada.sh gta5_to_cityscapes
bash scripts/exp_c_awada.sh cityscapes_to_foggy
```

### Experiment D: Oracle (Upper Bound)

Train and evaluate directly on the target domain with labels:

```bash
bash scripts/exp_d_oracle.sh gta5_to_cityscapes
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
├── requirements.txt
├── train_detector.py          # Faster R-CNN training script
├── train_cyclegan.py          # Standard CycleGAN training
├── train_awada.py             # AWADA CycleGAN training
├── generate_attention_maps.py # Generate RPN attention maps
├── stylize_dataset.py         # Stylize images with trained generator
├── scripts/
│   ├── exp_a_baseline.sh      # Experiment A: Baseline
│   ├── exp_b_cyclegan.sh      # Experiment B: CycleGAN
│   ├── exp_c_awada.sh         # Experiment C: AWADA
│   └── exp_d_oracle.sh        # Experiment D: Oracle
└── src/
    ├── models/
    │   ├── generator.py        # ResNet-9 generator
    │   ├── discriminator.py    # PatchGAN discriminator
    │   ├── cyclegan.py         # CycleGAN with image replay buffer
    │   └── awada_cyclegan.py   # AWADA variant with masked losses
    ├── datasets/
    │   ├── gta5.py             # GTA5 detection dataset
    │   ├── cityscapes.py       # Cityscapes detection dataset
    │   ├── foggy_cityscapes.py # Foggy Cityscapes detection dataset
    │   └── attention_dataset.py# Paired dataset for AWADA GAN training
    └── utils/
        ├── attention.py        # RPN attention map generation
        └── metrics.py          # mAP computation
```

## Evaluation

All experiments report:

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5 (PASCAL VOC metric).
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.50–0.95 in steps of 0.05 (COCO metric).

Results are saved to `results.txt` in each experiment's output directory.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{awada2024,
  title     = {AWADA: Attention-Weighted Domain Adaptation for Object Detection},
  author    = {AWADA Authors},
  year      = {2024},
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
