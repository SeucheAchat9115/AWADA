# Experiment B (CyCada): CyCada Domain Adaptation
# Train CyCada (CycleGAN + semantic consistency), stylize source images,
# train detector on stylized images, evaluate on target domain.
# Usage: .\scripts\exp_b_cycada.ps1 [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]

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
    $OutputBase    = "./outputs/exp_b_cycada_sim10k2cs"
} elseif ($Benchmark -eq "cityscapes_to_foggy") {
    $SourceDataset = "cityscapes"
    $SourceRoot    = "./data/cityscapes"
    $SourceImages  = "./data/cityscapes/leftImg8bit/train"
    $TargetDataset = "foggy_cityscapes"
    $TargetRoot    = "./data/foggy_cityscapes"
    $TargetImages  = "./data/foggy_cityscapes/leftImg8bit_foggy/train"
    $NumClasses    = 8
    $OutputBase    = "./outputs/exp_b_cycada_cs2foggy"
} elseif ($Benchmark -eq "cityscapes_to_bdd100k") {
    $SourceDataset = "cityscapes"
    $SourceRoot    = "./data/cityscapes"
    $SourceImages  = "./data/cityscapes/leftImg8bit/train"
    $TargetDataset = "bdd100k"
    $TargetRoot    = "./data/bdd100k"
    $TargetImages  = "./data/bdd100k/images/100k/train"
    $NumClasses    = 7
    $OutputBase    = "./outputs/exp_b_cycada_cs2bdd"
} else {
    Write-Error "Unknown benchmark: $Benchmark"
    Write-Host "Usage: .\scripts\exp_b_cycada.ps1 [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]"
    exit 1
}

$GanOutput       = "$OutputBase/cycada_gan"
$StylizedDir     = "$OutputBase/stylized_images"
$DetectorOutput  = "$OutputBase/detector"

New-Item -ItemType Directory -Force -Path $GanOutput      | Out-Null
New-Item -ItemType Directory -Force -Path $StylizedDir    | Out-Null
New-Item -ItemType Directory -Force -Path $DetectorOutput | Out-Null

Write-Host "========================================"
Write-Host "Experiment B (CyCada): CyCada Domain Adaptation"
Write-Host "Benchmark: $Benchmark"
Write-Host "========================================"

# Step 1: Train CyCada GAN (CycleGAN + semantic consistency loss)
Write-Host "[Step 1] Training CyCada GAN..."
python tools/train_cycada.py `
    --source_dir $SourceImages `
    --target_dir $TargetImages `
    --output_dir $GanOutput `
    --config configs/cycada.yaml `
    --epochs 200 `
    --batch_size 1 `
    --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 2: Stylize source images using the CyCada generator
Write-Host "[Step 2] Stylizing source images with CyCada generator..."
$LatestGan = Get-ChildItem -Path $GanOutput -Filter "cycada_epoch_*.pth" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
python tools/stylize_dataset.py `
    --generator_checkpoint $LatestGan `
    --source_dir $SourceImages `
    --output_dir $StylizedDir `
    --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Images come from StylizedDir; annotations come from SourceRoot
Write-Host "[Step 3] Training detector on CyCada-stylized source images..."
python tools/train_detector.py `
    --dataset $SourceDataset `
    --data_root $SourceRoot `
    --image_dir $StylizedDir `
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

Write-Host "Experiment B (CyCada) complete!"
