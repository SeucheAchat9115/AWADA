#!/bin/bash
# Experiment C: AWADA (Attention-Weighted Domain Adaptation)
# Requires Experiment A to be run first for baseline detector checkpoint
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

ATTENTION_DIR="$OUTPUT_BASE/attention_maps"
GAN_OUTPUT="$OUTPUT_BASE/awada_gan"
STYLIZED_DIR="$OUTPUT_BASE/stylized_images"
DETECTOR_OUTPUT="$OUTPUT_BASE/detector"

mkdir -p "$ATTENTION_DIR" "$GAN_OUTPUT" "$STYLIZED_DIR" "$DETECTOR_OUTPUT"

echo "========================================"
echo "Experiment C: AWADA"
echo "Benchmark: $BENCHMARK"
echo "Baseline checkpoint: $BASELINE_CKPT"
echo "========================================"

# Step 1: Generate attention maps from Experiment A detector
echo "[Step 1] Generating RPN attention maps from baseline detector..."
if [ ! -f "$BASELINE_CKPT" ]; then
    echo "ERROR: Baseline checkpoint not found: $BASELINE_CKPT"
    echo "Please run exp_a_baseline.sh first."
    exit 1
fi

python generate_attention_maps.py \
    --detector_checkpoint "$BASELINE_CKPT" \
    --dataset "$SOURCE_DATASET" \
    --data_root "$SOURCE_ROOT" \
    --output_dir "$ATTENTION_DIR" \
    --top_k "${TOP_K:-10}" \
    --num_classes "$NUM_CLASSES" \
    --split train \
    --device "${DEVICE:-cuda}"

# Step 2: Train AWADA CycleGAN with attention-masked losses
echo "[Step 2] Training AWADA CycleGAN..."
python train_awada.py \
    --source_dir "$SOURCE_IMAGES" \
    --target_dir "$TARGET_IMAGES" \
    --attention_dir "$ATTENTION_DIR" \
    --output_dir "$GAN_OUTPUT" \
    --epochs "${GAN_EPOCHS:-200}" \
    --batch_size "${GAN_BATCH:-1}" \
    --lr 0.0002 \
    --lambda_cyc 10.0 \
    --lambda_idt 5.0 \
    --device "${DEVICE:-cuda}"

# Step 3: Stylize source images using AWADA generator
echo "[Step 3] Stylizing source images with AWADA generator..."
LATEST_GAN=$(ls -t "$GAN_OUTPUT"/awada_epoch_*.pth | head -1)
python stylize_dataset.py \
    --generator_checkpoint "$LATEST_GAN" \
    --source_dir "$SOURCE_IMAGES" \
    --output_dir "$STYLIZED_DIR" \
    --device "${DEVICE:-cuda}"

# Step 4: Train detector on AWADA-stylized images
echo "[Step 4] Training detector on AWADA-stylized source images..."
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

# Step 5: Evaluate on target domain
echo "[Step 5] Evaluating on target domain..."
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
output_dir = "$DETECTOR_OUTPUT"
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
    for images, targets in tqdm(loader, desc='Evaluating'):
        imgs = [img.to(device) for img in images]
        outputs = model(imgs)
        predictions.extend({k: v.cpu() for k, v in o.items()} for o in outputs)
        targets_all.extend(targets)

metrics = compute_map_range(predictions, targets_all, num_classes=num_classes)
print(f"Experiment C (AWADA) Results on {target_dataset_name}:")
print(f"  mAP@0.5      = {metrics['mAP@0.5']:.4f}")
print(f"  mAP@0.5:0.95 = {metrics['mAP@0.5:0.95']:.4f}")
with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
    f.write(f"Experiment C: AWADA\nBenchmark: $BENCHMARK\n")
    f.write(f"mAP@0.5: {metrics['mAP@0.5']:.4f}\nmAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}\n")
EOF

echo "Experiment C (AWADA) complete!"
