#!/bin/bash
# Experiment D: Oracle (Upper Bound)
# Train Faster R-CNN on target domain with labels, evaluate on target domain
# Usage: bash scripts/exp_d_oracle.sh [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]

set -euo pipefail

BENCHMARK=${1:-sim10k_to_cityscapes}

if [ "$BENCHMARK" = "sim10k_to_cityscapes" ]; then
    TARGET_DATASET="cityscapes"
    TARGET_ROOT="./data/cityscapes"
    NUM_CLASSES=1
    OUTPUT_DIR="./outputs/exp_d_sim10k2cs"
elif [ "$BENCHMARK" = "cityscapes_to_foggy" ]; then
    TARGET_DATASET="foggy_cityscapes"
    TARGET_ROOT="./data/foggy_cityscapes"
    NUM_CLASSES=8
    OUTPUT_DIR="./outputs/exp_d_cs2foggy"
elif [ "$BENCHMARK" = "cityscapes_to_bdd100k" ]; then
    TARGET_DATASET="bdd100k"
    TARGET_ROOT="./data/bdd100k"
    NUM_CLASSES=7
    OUTPUT_DIR="./outputs/exp_d_cs2bdd"
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
python tools/train_detector.py \
    --dataset "$TARGET_DATASET" \
    --data_root "$TARGET_ROOT" \
    --num_classes "$NUM_CLASSES" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 10 \
    --batch_size 2 \
    --lr 0.005 \
    --device cuda \
    --pretrained \
    $([ "$BENCHMARK" = "sim10k_to_cityscapes" ] && echo "--classes car")

echo "Experiment D (Oracle) complete!"
