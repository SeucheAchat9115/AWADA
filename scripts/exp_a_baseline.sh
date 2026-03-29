#!/bin/bash
# Experiment A: Non-Adaptive Baseline
# Train Faster R-CNN on source domain, evaluate directly on target domain
# Usage: bash scripts/exp_a_baseline.sh [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]

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
TARGET_DATASET=$(_cfg target_dataset)
TARGET_ROOT=$(_cfg target_root)
NUM_CLASSES=$(_cfg num_classes)
OUTPUT_SUFFIX=$(_cfg output_suffix)
DETECTOR_EPOCHS=$(_cfg detector_epochs)
DETECTOR_BATCH_SIZE=$(_cfg detector_batch_size)
DETECTOR_LR=$(_cfg detector_lr)
CLASSES=$(_cfg classes)
OUTPUT_DIR="./outputs/exp_a_${OUTPUT_SUFFIX}"

# Build optional --classes argument
CLASSES_ARG=""
[ -n "$CLASSES" ] && CLASSES_ARG="--classes $CLASSES"

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
    --epochs "$DETECTOR_EPOCHS" \
    --batch_size "$DETECTOR_BATCH_SIZE" \
    --lr "$DETECTOR_LR" \
    --device cuda \
    --pretrained

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
    $CLASSES_ARG

echo "Experiment A complete!"
