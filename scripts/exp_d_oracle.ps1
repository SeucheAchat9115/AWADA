# Experiment D: Oracle (Upper Bound)
# Train Faster R-CNN on target domain with labels, evaluate on target domain
# Usage: .\scripts\exp_d_oracle.ps1 [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]

param(
    [string]$Benchmark = "sim10k_to_cityscapes"
)

$ErrorActionPreference = "Stop"

if ($Benchmark -eq "sim10k_to_cityscapes") {
    $TargetDataset = "cityscapes"
    $TargetRoot    = "./data/cityscapes"
    $NumClasses    = 1
    $OutputDir     = "./outputs/exp_d_sim10k2cs"
} elseif ($Benchmark -eq "cityscapes_to_foggy") {
    $TargetDataset = "foggy_cityscapes"
    $TargetRoot    = "./data/foggy_cityscapes"
    $NumClasses    = 8
    $OutputDir     = "./outputs/exp_d_cs2foggy"
} elseif ($Benchmark -eq "cityscapes_to_bdd100k") {
    $TargetDataset = "bdd100k"
    $TargetRoot    = "./data/bdd100k"
    $NumClasses    = 7
    $OutputDir     = "./outputs/exp_d_cs2bdd"
} else {
    Write-Error "Unknown benchmark: $Benchmark"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "========================================"
Write-Host "Experiment D: Oracle (Upper Bound)"
Write-Host "Benchmark: $Benchmark"
Write-Host "Target: $TargetDataset @ $TargetRoot"
Write-Host "Output: $OutputDir"
Write-Host "========================================"

# Train Faster R-CNN directly on target domain with labels
Write-Host "[Step 1] Training detector on target domain (oracle, with labels)..."

$pythonArgs = @(
    "tools/train_detector.py",
    "--dataset", $TargetDataset,
    "--data_root", $TargetRoot,
    "--num_classes", $NumClasses,
    "--output_dir", $OutputDir,
    "--epochs", "10",
    "--batch_size", "2",
    "--lr", "0.005",
    "--device", "cuda",
    "--pretrained"
)

if ($Benchmark -eq "sim10k_to_cityscapes") {
    $pythonArgs += "--classes", "car"
}

python @pythonArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Experiment D (Oracle) complete!"
