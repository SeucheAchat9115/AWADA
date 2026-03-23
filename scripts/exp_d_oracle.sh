#!/bin/bash
# Experiment D: Oracle (Upper Bound)
# Train Faster R-CNN on target domain with labels, evaluate on target domain
# Usage: bash scripts/exp_d_oracle.sh [gta5_to_cityscapes|cityscapes_to_foggy]

set -euo pipefail

BENCHMARK=${1:-gta5_to_cityscapes}

if [ "$BENCHMARK" = "gta5_to_cityscapes" ]; then
    TARGET_DATASET="cityscapes"
    TARGET_ROOT="${CITYSCAPES_ROOT:-/data/cityscapes}"
    NUM_CLASSES=8
    OUTPUT_DIR="${OUTPUT_ROOT:-./outputs}/exp_d_gta2cs"
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
    --pretrained

echo "[Step 2] Evaluating on target domain validation set..."
python - <<EOF
import torch, os, sys
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from src.utils.metrics import compute_map_range

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

def collate_fn(batch): return tuple(zip(*batch))
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
ckpt = torch.load(os.path.join(output_dir, 'detector_final.pth'), map_location=device)
model.load_state_dict(ckpt)
model.to(device).eval()

predictions, targets_all = [], []
with torch.no_grad():
    for images, targets in tqdm(loader, desc='Evaluating oracle on target domain'):
        imgs = [img.to(device) for img in images]
        outputs = model(imgs)
        predictions.extend({k: v.cpu() for k, v in o.items()} for o in outputs)
        targets_all.extend(targets)

metrics = compute_map_range(predictions, targets_all, num_classes=num_classes)
print(f"Experiment D (Oracle) Results on {target_dataset_name}:")
print(f"  mAP@0.5      = {metrics['mAP@0.5']:.4f}")
print(f"  mAP@0.5:0.95 = {metrics['mAP@0.5:0.95']:.4f}")
with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
    f.write(f"Experiment D: Oracle\nBenchmark: $BENCHMARK\n")
    f.write(f"mAP@0.5: {metrics['mAP@0.5']:.4f}\nmAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}\n")
EOF

echo "Experiment D (Oracle) complete!"
