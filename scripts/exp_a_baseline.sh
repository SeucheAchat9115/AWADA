#!/bin/bash
# Experiment A: Non-Adaptive Baseline
# Train Faster R-CNN on source domain, evaluate directly on target domain
# Usage: bash scripts/exp_a_baseline.sh [sim10k_to_cityscapes|cityscapes_to_foggy]

set -euo pipefail

BENCHMARK=${1:-sim10k_to_cityscapes}

if [ "$BENCHMARK" = "sim10k_to_cityscapes" ]; then
    SOURCE_DATASET="sim10k"
    SOURCE_ROOT="./data/sim10k"
    TARGET_DATASET="cityscapes"
    TARGET_ROOT="./data/cityscapes"
    NUM_CLASSES=1
    OUTPUT_DIR="./outputs/exp_a_sim10k2cs"
elif [ "$BENCHMARK" = "cityscapes_to_foggy" ]; then
    SOURCE_DATASET="cityscapes"
    SOURCE_ROOT="./data/cityscapes"
    TARGET_DATASET="foggy_cityscapes"
    TARGET_ROOT="./data/foggy_cityscapes"
    NUM_CLASSES=8
    OUTPUT_DIR="./outputs/exp_a_cs2foggy"
elif [ "$BENCHMARK" = "cityscapes_to_bdd100k" ]; then
    SOURCE_DATASET="cityscapes"
    SOURCE_ROOT="./data/cityscapes"
    TARGET_DATASET="bdd100k"
    TARGET_ROOT="./data/bdd100k"
    NUM_CLASSES=7
    OUTPUT_DIR="./outputs/exp_a_cs2bdd"
else
    echo "Unknown benchmark: $BENCHMARK"
    echo "Usage: $0 [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Experiment A: Non-Adaptive Baseline"
echo "Benchmark: $BENCHMARK"
echo "Source: $SOURCE_DATASET @ $SOURCE_ROOT"
echo "Target: $TARGET_DATASET @ $TARGET_ROOT"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Step 1: Train Faster R-CNN on source domain
echo "[Step 1] Training detector on source domain..."
python tools/train_detector.py \
    --dataset "$SOURCE_DATASET" \
    --data_root "$SOURCE_ROOT" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 10 \
    --batch_size 2 \
    --lr 0.005 \
    --device cuda \
    --pretrained \
    --val_dataset "$TARGET_DATASET" \
    --val_data_root "$TARGET_ROOT" \
    $([ "$BENCHMARK" = "sim10k_to_cityscapes" ] && echo "--val_classes car")

echo ""
echo "[Step 2] Evaluating on target domain (cross-domain, no adaptation)..."
python tools/evaluate_detector.py \
    --detector_checkpoint "$OUTPUT_DIR/detector_final.pth" \
    --dataset "$TARGET_DATASET" \
    --data_root "$TARGET_ROOT" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --label "Experiment A: Non-Adaptive Baseline" \
    --benchmark "$BENCHMARK" \
    $([ "$BENCHMARK" = "sim10k_to_cityscapes" ] && echo "--classes car")

echo "Experiment A complete!"
