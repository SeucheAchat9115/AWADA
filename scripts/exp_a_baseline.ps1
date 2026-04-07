# Experiment A: Non-Adaptive Baseline
# Train Faster R-CNN on source domain, evaluate directly on target domain
# Usage: .\scripts\exp_a_baseline.ps1 [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]

param(
    [string]$Benchmark = "sim10k_to_cityscapes"
)

$ErrorActionPreference = "Stop"

if ($Benchmark -eq "sim10k_to_cityscapes") {
    $SourceDataset = "sim10k"
    $SourceRoot    = "./data/sim10k"
    $TargetDataset = "cityscapes"
    $TargetRoot    = "./data/cityscapes"
    $NumClasses    = 1
    $OutputDir     = "./outputs/exp_a_sim10k2cs"
} elseif ($Benchmark -eq "cityscapes_to_foggy") {
    $SourceDataset = "cityscapes"
    $SourceRoot    = "./data/cityscapes"
    $TargetDataset = "foggy_cityscapes"
    $TargetRoot    = "./data/foggy_cityscapes"
    $NumClasses    = 8
    $OutputDir     = "./outputs/exp_a_cs2foggy"
} elseif ($Benchmark -eq "cityscapes_to_bdd100k") {
    $SourceDataset = "cityscapes"
    $SourceRoot    = "./data/cityscapes"
    $TargetDataset = "bdd100k"
    $TargetRoot    = "./data/bdd100k"
    $NumClasses    = 7
    $OutputDir     = "./outputs/exp_a_cs2bdd"
} else {
    Write-Error "Unknown benchmark: $Benchmark"
    Write-Host "Usage: .\scripts\exp_a_baseline.ps1 [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "========================================"
Write-Host "Experiment A: Non-Adaptive Baseline"
Write-Host "Benchmark: $Benchmark"
Write-Host "Source: $SourceDataset @ $SourceRoot"
Write-Host "Target: $TargetDataset @ $TargetRoot"
Write-Host "Output: $OutputDir"
Write-Host "========================================"

# Step 1: Train Faster R-CNN on source domain
Write-Host "[Step 1] Training detector on source domain..."
python tools/train_detector.py `
    --dataset $SourceDataset `
    --data_root $SourceRoot `
    --num_classes $NumClasses `
    --output_dir $OutputDir `
    --epochs 10 `
    --batch_size 2 `
    --lr 0.005 `
    --device cuda `
    --pretrained `
    --val_dataset $TargetDataset `
    --val_data_root $TargetRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Experiment A complete!"
