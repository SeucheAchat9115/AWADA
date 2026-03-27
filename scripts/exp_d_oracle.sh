#!/bin/bash
# Experiment D: Oracle (Upper Bound)
# Train Faster R-CNN on target domain with labels, evaluate on target domain
# Usage: bash scripts/exp_d_oracle.sh [sim10k_to_cityscapes|cityscapes_to_foggy]

set -euo pipefail

BENCHMARK=${1:-sim10k_to_cityscapes}

if [ "$BENCHMARK" = "sim10k_to_cityscapes" ]; then
    TARGET_DATASET="cityscapes"
    TARGET_ROOT="${CITYSCAPES_ROOT:-/data/cityscapes}"
    NUM_CLASSES=1
    OUTPUT_DIR="${OUTPUT_ROOT:-./outputs}/exp_d_sim10k2cs"
elif [ "$BENCHMARK" = "cityscapes_to_foggy" ]; then
    TARGET_DATASET="foggy_cityscapes"
    TARGET_ROOT="${FOGGY_ROOT:-/data/foggy_cityscapes}"
    NUM_CLASSES=8
    OUTPUT_DIR="${OUTPUT_ROOT:-./outputs}/exp_d_cs2foggy"
else
    echo "Unknown benchmark: $BENCHMARK"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Experiment D: Oracle (Upper Bound)"
echo "Benchmark: $BENCHMARK"
echo "Target: $TARGET_DATASET @ $TARGET_ROOT"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Train Faster R-CNN directly on target domain with labels
echo "[Step 1] Training detector on target domain (oracle, with labels)..."
python train_detector.py \
    --dataset "$TARGET_DATASET" \
    --data_root "$TARGET_ROOT" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "${EPOCHS:-10}" \
    --batch_size "${BATCH_SIZE:-2}" \
    --lr 0.005 \
    --device "${DEVICE:-cuda}" \
    --pretrained \
    $([ "$BENCHMARK" = "sim10k_to_cityscapes" ] && echo "--classes car")

echo "[Step 2] Evaluating on target domain validation set..."
python evaluate_detector.py \
    --detector_checkpoint "$OUTPUT_DIR/detector_final.pth" \
    --dataset "$TARGET_DATASET" \
    --data_root "$TARGET_ROOT" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$OUTPUT_DIR" \
    --device "${DEVICE:-cuda}" \
    --label "Experiment D: Oracle" \
    --benchmark "$BENCHMARK" \
    $([ "$BENCHMARK" = "sim10k_to_cityscapes" ] && echo "--classes car")

echo "Experiment D (Oracle) complete!"
