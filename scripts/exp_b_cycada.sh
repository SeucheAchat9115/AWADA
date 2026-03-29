#!/bin/bash
# Experiment B (CyCada): CyCada Domain Adaptation
# Train CyCada (CycleGAN + semantic consistency), stylize source images,
# train detector on stylized images, evaluate on target domain.
# Usage: bash scripts/exp_b_cycada.sh [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]

set -euo pipefail

BENCHMARK=${1:-sim10k_to_cityscapes}
CONFIG_FILE="configs/benchmarks/${BENCHMARK}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Unknown benchmark: $BENCHMARK"
    echo "Usage: $0 [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]"
    exit 1
fi

# Read all settings from the benchmark YAML config
# shellcheck source=scripts/lib/config_helper.sh
source "$(dirname "$0")/lib/config_helper.sh"

SOURCE_DATASET=$(_cfg source_dataset)
SOURCE_ROOT=$(_cfg source_root)
SOURCE_IMAGES=$(_cfg source_images)
TARGET_DATASET=$(_cfg target_dataset)
TARGET_ROOT=$(_cfg target_root)
TARGET_IMAGES=$(_cfg target_images)
NUM_CLASSES=$(_cfg num_classes)
OUTPUT_SUFFIX=$(_cfg output_suffix)
DETECTOR_EPOCHS=$(_cfg detector_epochs)
DETECTOR_BATCH_SIZE=$(_cfg detector_batch_size)
DETECTOR_LR=$(_cfg detector_lr)
CLASSES=$(_cfg classes)
OUTPUT_BASE="./outputs/exp_b_cycada_${OUTPUT_SUFFIX}"

# Build optional --classes argument
CLASSES_ARG=""
[ -n "$CLASSES" ] && CLASSES_ARG="--classes $CLASSES"

GAN_OUTPUT="$OUTPUT_BASE/cycada_gan"
STYLIZED_DIR="$OUTPUT_BASE/stylized_images"
DETECTOR_OUTPUT="$OUTPUT_BASE/detector"

mkdir -p "$GAN_OUTPUT" "$STYLIZED_DIR" "$DETECTOR_OUTPUT"

echo "========================================"
echo "Experiment B (CyCada): CyCada Domain Adaptation"
echo "Benchmark: $BENCHMARK"
echo "========================================"

# Step 1: Train CyCada GAN (CycleGAN + semantic consistency loss)
echo "[Step 1] Training CyCada GAN..."
python tools/train_cycada.py \
    --source_dir "$SOURCE_IMAGES" \
    --target_dir "$TARGET_IMAGES" \
    --output_dir "$GAN_OUTPUT" \
    --config configs/cycada.yaml \
    --device cuda

# Step 2: Stylize source images using the CyCada generator
echo "[Step 2] Stylizing source images with CyCada generator..."
LATEST_GAN=$(ls -t "$GAN_OUTPUT"/cycada_epoch_*.pth | head -1)
python tools/stylize_dataset.py \
    --generator_checkpoint "$LATEST_GAN" \
    --source_dir "$SOURCE_IMAGES" \
    --output_dir "$STYLIZED_DIR" \
    --device cuda
# Images come from STYLIZED_DIR; annotations come from SOURCE_ROOT
echo "[Step 3] Training detector on CyCada-stylized source images..."
python tools/train_detector.py \
    --dataset "$SOURCE_DATASET" \
    --data_root "$SOURCE_ROOT" \
    --image_dir "$STYLIZED_DIR" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$DETECTOR_OUTPUT" \
    --epochs "$DETECTOR_EPOCHS" \
    --batch_size "$DETECTOR_BATCH_SIZE" \
    --lr "$DETECTOR_LR" \
    --device cuda \
    --pretrained

# Step 4: Evaluate on target domain
echo "[Step 4] Evaluating on target domain..."
python tools/evaluate_detector.py \
    --detector_checkpoint "$DETECTOR_OUTPUT/detector_final.pth" \
    --dataset "$TARGET_DATASET" \
    --data_root "$TARGET_ROOT" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$DETECTOR_OUTPUT" \
    --device cuda \
    --benchmark "$BENCHMARK" \
    $CLASSES_ARG

echo "Experiment B (CyCada) complete!"
