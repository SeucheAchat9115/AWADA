#!/bin/bash
# Experiment A: Non-Adaptive Baseline
# Train Faster R-CNN on source domain, evaluate directly on target domain
# Usage: bash scripts/exp_a_baseline.sh [gta5_to_cityscapes|cityscapes_to_foggy]
#
# Environment variable overrides:
#   EPOCHS (default 10) — number of detector training epochs; increase (e.g. 20)
#                         for higher accuracy at the cost of longer training time.

set -euo pipefail

BENCHMARK=${1:-gta5_to_cityscapes}

if [ "$BENCHMARK" = "gta5_to_cityscapes" ]; then
    SOURCE_DATASET="gta5"
    SOURCE_ROOT="${GTA5_ROOT:-/data/gta5}"
    TARGET_DATASET="cityscapes"
    TARGET_ROOT="${CITYSCAPES_ROOT:-/data/cityscapes}"
    NUM_CLASSES=7
    OUTPUT_DIR="${OUTPUT_ROOT:-./outputs}/exp_a_gta2cs"
elif [ "$BENCHMARK" = "cityscapes_to_foggy" ]; then
    SOURCE_DATASET="cityscapes"
    SOURCE_ROOT="${CITYSCAPES_ROOT:-/data/cityscapes}"
    TARGET_DATASET="foggy_cityscapes"
    TARGET_ROOT="${FOGGY_ROOT:-/data/foggy_cityscapes}"
    NUM_CLASSES=8
    OUTPUT_DIR="${OUTPUT_ROOT:-./outputs}/exp_a_cs2foggy"
else
    echo "Unknown benchmark: $BENCHMARK"
    echo "Usage: $0 [gta5_to_cityscapes|cityscapes_to_foggy]"
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
python - <<EOF
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import os
import sys
sys.path.insert(0, '.')

target_dataset_name = "$TARGET_DATASET"
target_root = "$TARGET_ROOT"
output_dir = "$OUTPUT_DIR"
num_classes = $NUM_CLASSES
device = torch.device("${DEVICE:-cuda}" if torch.cuda.is_available() else "cpu")

if target_dataset_name == "cityscapes":
    from src.datasets.cityscapes import CityscapesDetectionDataset
    dataset = CityscapesDetectionDataset(target_root, split='val')
elif target_dataset_name == "foggy_cityscapes":
    from src.datasets.foggy_cityscapes import FoggyCityscapesDataset
    dataset = FoggyCityscapesDataset(target_root, split='val')

def collate_fn(batch):
    return tuple(zip(*batch))

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

ckpt = torch.load(os.path.join(output_dir, 'detector_final.pth'), map_location=device)
model.load_state_dict(ckpt)
model.to(device).eval()

from src.utils.metrics import compute_map_range

predictions, targets_all = [], []
with torch.no_grad():
    for images, targets in tqdm(loader, desc='Evaluating on target domain'):
        imgs = [img.to(device) for img in images]
        outputs = model(imgs)
        for out in outputs:
            predictions.append({k: v.cpu() for k, v in out.items()})
        for t in targets:
            targets_all.append(t)

metrics = compute_map_range(predictions, targets_all, num_classes=num_classes)
print(f"Experiment A Results on {target_dataset_name}:")
print(f"  mAP@0.5      = {metrics['mAP@0.5']:.4f}")
print(f"  mAP@0.5:0.95 = {metrics['mAP@0.5:0.95']:.4f}")

results_path = os.path.join(output_dir, 'results.txt')
with open(results_path, 'w') as f:
    f.write(f"Experiment A: Non-Adaptive Baseline\n")
    f.write(f"Benchmark: $BENCHMARK\n")
    f.write(f"mAP@0.5: {metrics['mAP@0.5']:.4f}\n")
    f.write(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}\n")
print(f"Results saved to {results_path}")
EOF

echo "Experiment A complete!"
