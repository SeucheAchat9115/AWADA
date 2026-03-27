#!/bin/bash
# Experiment B: Standard CycleGAN Domain Adaptation
# Train CycleGAN, stylize source images, train new detector on stylized data
# Usage: bash scripts/exp_b_cyclegan.sh [sim10k_to_cityscapes|cityscapes_to_foggy]

set -euo pipefail

BENCHMARK=${1:-sim10k_to_cityscapes}

if [ "$BENCHMARK" = "sim10k_to_cityscapes" ]; then
    SOURCE_DATASET="sim10k"
    SOURCE_ROOT="${SIM10K_ROOT:-/data/sim10k}"
    SOURCE_IMAGES="${SIM10K_ROOT:-/data/sim10k}/images"
    TARGET_DATASET="cityscapes"
    TARGET_ROOT="${CITYSCAPES_ROOT:-/data/cityscapes}"
    TARGET_IMAGES="${CITYSCAPES_ROOT:-/data/cityscapes}/leftImg8bit/train"
    NUM_CLASSES=1
    OUTPUT_BASE="${OUTPUT_ROOT:-./outputs}/exp_b_sim10k2cs"
elif [ "$BENCHMARK" = "cityscapes_to_foggy" ]; then
    SOURCE_DATASET="cityscapes"
    SOURCE_ROOT="${CITYSCAPES_ROOT:-/data/cityscapes}"
    SOURCE_IMAGES="${CITYSCAPES_ROOT:-/data/cityscapes}/leftImg8bit/train"
    TARGET_DATASET="foggy_cityscapes"
    TARGET_ROOT="${FOGGY_ROOT:-/data/foggy_cityscapes}"
    TARGET_IMAGES="${FOGGY_ROOT:-/data/foggy_cityscapes}/leftImg8bit_foggy/train"
    NUM_CLASSES=8
    OUTPUT_BASE="${OUTPUT_ROOT:-./outputs}/exp_b_cs2foggy"
elif [ "$BENCHMARK" = "cityscapes_to_bdd100k" ]; then
    SOURCE_DATASET="cityscapes"
    SOURCE_ROOT="${CITYSCAPES_ROOT:-/data/cityscapes}"
    SOURCE_IMAGES="${CITYSCAPES_ROOT:-/data/cityscapes}/leftImg8bit/train"
    TARGET_DATASET="bdd100k"
    TARGET_ROOT="${BDD100K_ROOT:-/data/bdd100k}"
    TARGET_IMAGES="${BDD100K_ROOT:-/data/bdd100k}/images/100k/train"
    NUM_CLASSES=7
    OUTPUT_BASE="${OUTPUT_ROOT:-./outputs}/exp_b_cs2bdd"
else
    echo "Unknown benchmark: $BENCHMARK"
    exit 1
fi

GAN_OUTPUT="$OUTPUT_BASE/cyclegan"
STYLIZED_DIR="$OUTPUT_BASE/stylized_images"
DETECTOR_OUTPUT="$OUTPUT_BASE/detector"

mkdir -p "$GAN_OUTPUT" "$STYLIZED_DIR" "$DETECTOR_OUTPUT"

echo "========================================"
echo "Experiment B: Standard CycleGAN"
echo "Benchmark: $BENCHMARK"
echo "========================================"

# Step 1: Train CycleGAN
echo "[Step 1] Training CycleGAN..."
python train_cyclegan.py \
    --source_dir "$SOURCE_IMAGES" \
    --target_dir "$TARGET_IMAGES" \
    --output_dir "$GAN_OUTPUT" \
    --epochs "${GAN_EPOCHS:-200}" \
    --batch_size "${GAN_BATCH:-1}" \
    --lr 0.0002 \
    --lambda_cyc 10.0 \
    --lambda_idt 5.0 \
    --device "${DEVICE:-cuda}"

# Step 2: Stylize source images
echo "[Step 2] Stylizing source images..."
LATEST_GAN=$(ls -t "$GAN_OUTPUT"/cyclegan_epoch_*.pth | head -1)
python stylize_dataset.py \
    --generator_checkpoint "$LATEST_GAN" \
    --source_dir "$SOURCE_IMAGES" \
    --output_dir "$STYLIZED_DIR" \
    --device "${DEVICE:-cuda}"

# Step 3: Train detector on stylized images
# Note: use original source labels with stylized images
echo "[Step 3] Training detector on stylized source images..."
python train_detector.py \
    --dataset "$SOURCE_DATASET" \
    --data_root "$SOURCE_ROOT" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$DETECTOR_OUTPUT" \
    --epochs "${DET_EPOCHS:-10}" \
    --batch_size "${BATCH_SIZE:-2}" \
    --lr 0.005 \
    --device "${DEVICE:-cuda}" \
    --pretrained

# Step 4: Evaluate on target domain
echo "[Step 4] Evaluating on target domain..."
python evaluate_detector.py \
    --detector_checkpoint "$DETECTOR_OUTPUT/detector_final.pth" \
    --dataset "$TARGET_DATASET" \
    --data_root "$TARGET_ROOT" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$DETECTOR_OUTPUT" \
    --device "${DEVICE:-cuda}" \
    --label "Experiment B: CycleGAN" \
    --benchmark "$BENCHMARK" \
    $([ "$BENCHMARK" = "sim10k_to_cityscapes" ] && echo "--classes car")

echo "Experiment B complete!"
