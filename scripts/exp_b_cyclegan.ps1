# Experiment B: Standard CycleGAN Domain Adaptation
# Train CycleGAN, stylize source images, train new detector on stylized data
# Usage: .\scripts\exp_b_cyclegan.ps1 [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]

param(
    [string]$Benchmark = "sim10k_to_cityscapes"
)

$ErrorActionPreference = "Stop"

if ($Benchmark -eq "sim10k_to_cityscapes") {
    $SourceDataset = "sim10k"
    $SourceRoot    = "./data/sim10k"
    $SourceImages  = "./data/sim10k/images"
    $TargetDataset = "cityscapes"
    $TargetRoot    = "./data/cityscapes"
    $TargetImages  = "./data/cityscapes/leftImg8bit/train"
    $NumClasses    = 1
    $OutputBase    = "./outputs/exp_b_sim10k2cs"
} elseif ($Benchmark -eq "cityscapes_to_foggy") {
    $SourceDataset = "cityscapes"
    $SourceRoot    = "./data/cityscapes"
    $SourceImages  = "./data/cityscapes/leftImg8bit/train"
    $TargetDataset = "foggy_cityscapes"
    $TargetRoot    = "./data/foggy_cityscapes"
    $TargetImages  = "./data/foggy_cityscapes/leftImg8bit_foggy/train"
    $NumClasses    = 8
    $OutputBase    = "./outputs/exp_b_cs2foggy"
} elseif ($Benchmark -eq "cityscapes_to_bdd100k") {
    $SourceDataset = "cityscapes"
    $SourceRoot    = "./data/cityscapes"
    $SourceImages  = "./data/cityscapes/leftImg8bit/train"
    $TargetDataset = "bdd100k"
    $TargetRoot    = "./data/bdd100k"
    $TargetImages  = "./data/bdd100k/images/100k/train"
    $NumClasses    = 7
    $OutputBase    = "./outputs/exp_b_cs2bdd"
} else {
    Write-Error "Unknown benchmark: $Benchmark"
    exit 1
}

$GanOutput      = "$OutputBase/cyclegan"
$StylizedDir    = "$OutputBase/stylized_images"
$DetectorOutput = "$OutputBase/detector"

New-Item -ItemType Directory -Force -Path $GanOutput      | Out-Null
New-Item -ItemType Directory -Force -Path $StylizedDir    | Out-Null
New-Item -ItemType Directory -Force -Path $DetectorOutput | Out-Null

Write-Host "========================================"
Write-Host "Experiment B: Standard CycleGAN"
Write-Host "Benchmark: $Benchmark"
Write-Host "========================================"

# Step 1: Train CycleGAN
Write-Host "[Step 1] Training CycleGAN..."
python tools/train_cyclegan.py `
    --source_dir $SourceImages `
    --target_dir $TargetImages `
    --output_dir $GanOutput `
    --epochs 200 `
    --batch_size 1 `
    --lr 0.0002 `
    --lambda_cyc 10.0 `
    --lambda_idt 0.5 `
    --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 2: Stylize source images
Write-Host "[Step 2] Stylizing source images..."
$LatestGan = Get-ChildItem -Path $GanOutput -Filter "cyclegan_epoch_*.pth" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
python tools/stylize_dataset.py `
    --generator_checkpoint $LatestGan `
    --source_dir $SourceImages `
    --output_dir $StylizedDir `
    --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 3: Train detector on stylized images
# Note: use original source labels with stylized images
Write-Host "[Step 3] Training detector on stylized source images..."
python tools/train_detector.py `
    --dataset $SourceDataset `
    --data_root $SourceRoot `
    --num_classes $NumClasses `
    --output_dir $DetectorOutput `
    --epochs 10 `
    --batch_size 2 `
    --lr 0.005 `
    --device cuda `
    --pretrained `
    --val_dataset $TargetDataset `
    --val_data_root $TargetRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Experiment B complete!"
