#!/bin/bash
# Experiment C: AWADA (Attention-Weighted Domain Adaptation)
# Requires Experiment A to be run first for the baseline source-domain detector checkpoint.
#
# Workflow:
#   1. Generate source attention maps using the Exp A (source-trained) detector.
#   2. Train CyCada to stylize source images into the target style.
#   3. Train a CyCada detector on the stylized images.
#   4. Generate target attention maps using the CyCada detector on real target images.
#   5. Train AWADA CycleGAN with both source and target attention-masked losses.
#   6. Stylize source images with the AWADA generator.
#   7. Train final detector on AWADA-stylized images.
#   8. Evaluate on target domain.
#
# Usage: bash scripts/exp_c_awada.sh [sim10k_to_cityscapes|cityscapes_to_foggy]

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
    OUTPUT_BASE="${OUTPUT_ROOT:-./outputs}/exp_c_sim10k2cs"
    BASELINE_CKPT="${OUTPUT_ROOT:-./outputs}/exp_a_sim10k2cs/detector_final.pth"
elif [ "$BENCHMARK" = "cityscapes_to_foggy" ]; then
    SOURCE_DATASET="cityscapes"
    SOURCE_ROOT="${CITYSCAPES_ROOT:-/data/cityscapes}"
    SOURCE_IMAGES="${CITYSCAPES_ROOT:-/data/cityscapes}/leftImg8bit/train"
    TARGET_DATASET="foggy_cityscapes"
    TARGET_ROOT="${FOGGY_ROOT:-/data/foggy_cityscapes}"
    TARGET_IMAGES="${FOGGY_ROOT:-/data/foggy_cityscapes}/leftImg8bit_foggy/train"
    NUM_CLASSES=8
    OUTPUT_BASE="${OUTPUT_ROOT:-./outputs}/exp_c_cs2foggy"
    BASELINE_CKPT="${OUTPUT_ROOT:-./outputs}/exp_a_cs2foggy/detector_final.pth"
else
    echo "Unknown benchmark: $BENCHMARK"
    exit 1
fi

SOURCE_ATTENTION_DIR="$OUTPUT_BASE/source_attention_maps"
CYCADA_GAN_OUTPUT="$OUTPUT_BASE/cycada_gan"
CYCADA_STYLIZED_DIR="$OUTPUT_BASE/cycada_stylized_images"
CYCADA_DETECTOR_OUTPUT="$OUTPUT_BASE/cycada_detector"
TARGET_ATTENTION_DIR="$OUTPUT_BASE/target_attention_maps"
AWADA_GAN_OUTPUT="$OUTPUT_BASE/awada_gan"
AWADA_STYLIZED_DIR="$OUTPUT_BASE/awada_stylized_images"
DETECTOR_OUTPUT="$OUTPUT_BASE/detector"

mkdir -p \
    "$SOURCE_ATTENTION_DIR" \
    "$CYCADA_GAN_OUTPUT" \
    "$CYCADA_STYLIZED_DIR" \
    "$CYCADA_DETECTOR_OUTPUT" \
    "$TARGET_ATTENTION_DIR" \
    "$AWADA_GAN_OUTPUT" \
    "$AWADA_STYLIZED_DIR" \
    "$DETECTOR_OUTPUT"

echo "========================================"
echo "Experiment C: AWADA"
echo "Benchmark: $BENCHMARK"
echo "Baseline checkpoint: $BASELINE_CKPT"
echo "========================================"

# Step 1: Generate source attention maps from the Exp A (source-trained) detector
echo "[Step 1] Generating source RPN attention maps from baseline detector..."
if [ ! -f "$BASELINE_CKPT" ]; then
    echo "ERROR: Baseline checkpoint not found: $BASELINE_CKPT"
    echo "Please run exp_a_baseline.sh first."
    exit 1
fi

python generate_attention_maps.py \
    --detector_checkpoint "$BASELINE_CKPT" \
    --dataset "$SOURCE_DATASET" \
    --data_root "$SOURCE_ROOT" \
    --output_dir "$SOURCE_ATTENTION_DIR" \
    --score_threshold "${SCORE_THRESHOLD:-0.5}" \
    --num_classes "$NUM_CLASSES" \
    --split train \
    --device "${DEVICE:-cuda}"

# Step 2: Train CyCada GAN to learn the source → target style mapping
echo "[Step 2] Training CyCada GAN for source→target style transfer..."
python train_cycada.py \
    --source_dir "$SOURCE_IMAGES" \
    --target_dir "$TARGET_IMAGES" \
    --output_dir "$CYCADA_GAN_OUTPUT" \
    --config configs/cycada.yaml \
    --epochs "${GAN_EPOCHS:-200}" \
    --batch_size "${GAN_BATCH:-1}" \
    --device "${DEVICE:-cuda}"

# Step 3: Stylize source images using the CyCada generator
echo "[Step 3] Stylizing source images with CyCada generator..."
LATEST_CYCADA_GAN=$(ls -t "$CYCADA_GAN_OUTPUT"/cycada_epoch_*.pth | head -1)
python stylize_dataset.py \
    --generator_checkpoint "$LATEST_CYCADA_GAN" \
    --source_dir "$SOURCE_IMAGES" \
    --output_dir "$CYCADA_STYLIZED_DIR" \
    --device "${DEVICE:-cuda}"

# Step 4: Train a detector on the CyCada-stylized images
# This detector has been exposed to the target visual style, making it suitable
# for generating attention maps on real target-domain images.
echo "[Step 4] Training CyCada detector on stylized source images..."
python train_detector.py \
    --dataset "$SOURCE_DATASET" \
    --data_root "$SOURCE_ROOT" \
    --image_dir "$CYCADA_STYLIZED_DIR" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$CYCADA_DETECTOR_OUTPUT" \
    --epochs "${DET_EPOCHS:-10}" \
    --batch_size "${BATCH_SIZE:-2}" \
    --lr 0.005 \
    --device "${DEVICE:-cuda}" \
    --pretrained

# Step 5: Generate target attention maps using the CyCada detector on real target images
echo "[Step 5] Generating target RPN attention maps from CyCada detector..."
python generate_attention_maps.py \
    --detector_checkpoint "$CYCADA_DETECTOR_OUTPUT/detector_final.pth" \
    --dataset "$TARGET_DATASET" \
    --data_root "$TARGET_ROOT" \
    --output_dir "$TARGET_ATTENTION_DIR" \
    --score_threshold "${SCORE_THRESHOLD:-0.5}" \
    --num_classes "$NUM_CLASSES" \
    --split train \
    --device "${DEVICE:-cuda}"

# Step 6: Train AWADA CycleGAN with attention-masked losses
# Source attention maps: from the source-trained detector (Step 1)
# Target attention maps: from the CyCada-trained detector (Step 5)
echo "[Step 6] Training AWADA CycleGAN with source and target attention maps..."
python train_awada.py \
    --source_dir "$SOURCE_IMAGES" \
    --target_dir "$TARGET_IMAGES" \
    --attention_dir "$SOURCE_ATTENTION_DIR" \
    --target_attention_dir "$TARGET_ATTENTION_DIR" \
    --output_dir "$AWADA_GAN_OUTPUT" \
    --config configs/awada.yaml \
    --epochs "${GAN_EPOCHS:-200}" \
    --batch_size "${GAN_BATCH:-1}" \
    --device "${DEVICE:-cuda}"

# Step 7: Stylize source images using the AWADA generator
echo "[Step 7] Stylizing source images with AWADA generator..."
LATEST_AWADA_GAN=$(ls -t "$AWADA_GAN_OUTPUT"/awada_epoch_*.pth | head -1)
python stylize_dataset.py \
    --generator_checkpoint "$LATEST_AWADA_GAN" \
    --source_dir "$SOURCE_IMAGES" \
    --output_dir "$AWADA_STYLIZED_DIR" \
    --device "${DEVICE:-cuda}"

# Step 8: Train final detector on AWADA-stylized images
echo "[Step 8] Training final detector on AWADA-stylized source images..."
python train_detector.py \
    --dataset "$SOURCE_DATASET" \
    --data_root "$SOURCE_ROOT" \
    --image_dir "$AWADA_STYLIZED_DIR" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$DETECTOR_OUTPUT" \
    --epochs "${DET_EPOCHS:-10}" \
    --batch_size "${BATCH_SIZE:-2}" \
    --lr 0.005 \
    --device "${DEVICE:-cuda}" \
    --pretrained

# Step 9: Evaluate on target domain
echo "[Step 9] Evaluating on target domain..."
python evaluate_detector.py \
    --detector_checkpoint "$DETECTOR_OUTPUT/detector_final.pth" \
    --dataset "$TARGET_DATASET" \
    --data_root "$TARGET_ROOT" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$DETECTOR_OUTPUT" \
    --device "${DEVICE:-cuda}" \
    --label "Experiment C: AWADA" \
    --benchmark "$BENCHMARK" \
    $([ "$BENCHMARK" = "sim10k_to_cityscapes" ] && echo "--classes car")

echo "Experiment C (AWADA) complete!"

