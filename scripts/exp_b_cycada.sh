#!/bin/bash
# Experiment B (CyCada): CyCada Domain Adaptation
# Train CyCada (CycleGAN + semantic consistency), stylize source images,
# train detector on stylized images, evaluate on target domain.
# Usage: bash scripts/exp_b_cycada.sh [sim10k_to_cityscapes|cityscapes_to_foggy]

set -euo pipefail

BENCHMARK=${1:-sim10k_to_cityscapes}

if [ "$BENCHMARK" = "sim10k_to_cityscapes" ]; then
    SOURCE_DATASET="sim10k"
    SOURCE_ROOT="./data/sim10k"
    SOURCE_IMAGES="./data/sim10k/images"
    TARGET_DATASET="cityscapes"
    TARGET_ROOT="./data/cityscapes"
    TARGET_IMAGES="./data/cityscapes/leftImg8bit/train"
    NUM_CLASSES=1
    OUTPUT_BASE="./outputs/exp_b_cycada_sim10k2cs"
elif [ "$BENCHMARK" = "cityscapes_to_foggy" ]; then
    SOURCE_DATASET="cityscapes"
    SOURCE_ROOT="./data/cityscapes"
    SOURCE_IMAGES="./data/cityscapes/leftImg8bit/train"
    TARGET_DATASET="foggy_cityscapes"
    TARGET_ROOT="./data/foggy_cityscapes"
    TARGET_IMAGES="./data/foggy_cityscapes/leftImg8bit_foggy/train"
    NUM_CLASSES=8
    OUTPUT_BASE="./outputs/exp_b_cycada_cs2foggy"
elif [ "$BENCHMARK" = "cityscapes_to_bdd100k" ]; then
    SOURCE_DATASET="cityscapes"
    SOURCE_ROOT="./data/cityscapes"
    SOURCE_IMAGES="./data/cityscapes/leftImg8bit/train"
    TARGET_DATASET="bdd100k"
    TARGET_ROOT="./data/bdd100k"
    TARGET_IMAGES="./data/bdd100k/images/100k/train"
    NUM_CLASSES=7
    OUTPUT_BASE="./outputs/exp_b_cycada_cs2bdd"
else
    echo "Unknown benchmark: $BENCHMARK"
    echo "Usage: $0 [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]"
    exit 1
fi

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
    --epochs 200 \
    --batch_size 1 \
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
    --epochs 10 \
    --batch_size 2 \
    --lr 0.005 \
    --device cuda \
    --pretrained \
    --val_dataset "$TARGET_DATASET" \
    --val_data_root "$TARGET_ROOT"

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
    $([ "$BENCHMARK" = "sim10k_to_cityscapes" ] && echo "--classes car")

echo "Experiment B (CyCada) complete!"
