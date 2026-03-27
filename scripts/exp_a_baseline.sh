#!/bin/bash
# Experiment A: Non-Adaptive Baseline
# Train Faster R-CNN on source domain, evaluate directly on target domain
# Usage: bash scripts/exp_a_baseline.sh [sim10k_to_cityscapes|cityscapes_to_foggy]
#
# Environment variable overrides:
#   EPOCHS (default 10) — number of detector training epochs; increase (e.g. 20)
#                         for higher accuracy at the cost of longer training time.

set -euo pipefail

BENCHMARK=${1:-sim10k_to_cityscapes}

if [ "$BENCHMARK" = "sim10k_to_cityscapes" ]; then
    SOURCE_DATASET="sim10k"
    SOURCE_ROOT="${SIM10K_ROOT:-/data/sim10k}"
    TARGET_DATASET="cityscapes"
    TARGET_ROOT="${CITYSCAPES_ROOT:-/data/cityscapes}"
    NUM_CLASSES=1
    OUTPUT_DIR="${OUTPUT_ROOT:-./outputs}/exp_a_sim10k2cs"
elif [ "$BENCHMARK" = "cityscapes_to_foggy" ]; then
    SOURCE_DATASET="cityscapes"
    SOURCE_ROOT="${CITYSCAPES_ROOT:-/data/cityscapes}"
    TARGET_DATASET="foggy_cityscapes"
    TARGET_ROOT="${FOGGY_ROOT:-/data/foggy_cityscapes}"
    NUM_CLASSES=8
    OUTPUT_DIR="${OUTPUT_ROOT:-./outputs}/exp_a_cs2foggy"
else
    echo "Unknown benchmark: $BENCHMARK"
    echo "Usage: $0 [sim10k_to_cityscapes|cityscapes_to_foggy]"
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
python train_detector.py \
    --dataset "$SOURCE_DATASET" \
    --data_root "$SOURCE_ROOT" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "${EPOCHS:-10}" \
    --batch_size "${BATCH_SIZE:-2}" \
    --lr 0.005 \
    --device "${DEVICE:-cuda}" \
    --pretrained

echo ""
echo "[Step 2] Evaluating on target domain (cross-domain, no adaptation)..."
python evaluate_detector.py \
    --detector_checkpoint "$OUTPUT_DIR/detector_final.pth" \
    --dataset "$TARGET_DATASET" \
    --data_root "$TARGET_ROOT" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$OUTPUT_DIR" \
    --device "${DEVICE:-cuda}" \
    --label "Experiment A: Non-Adaptive Baseline" \
    --benchmark "$BENCHMARK" \
    $([ "$BENCHMARK" = "sim10k_to_cityscapes" ] && echo "--classes car")

echo "Experiment A complete!"
