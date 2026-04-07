# Experiment C: AWADA (Attention-Weighted Domain Adaptation)
# Requires Experiment A to be run first for the baseline source-domain detector checkpoint.
#
# Workflow:
#   1. Generate source attention maps using the Exp A (source-trained) detector.
#   2. Train CyCada to stylize source images into the target style.
#   3. Train a CyCada detector on the stylized images.
#   4. Generate target attention maps using the CyCada detector on real target images.
#   5. Train AWADA CycleGAN with both source and target attention-masked losses.
#   6. Stylize source images with the AWADA generator.
#   7. Train final detector on AWADA-stylized images.
#   8. Evaluate on target domain.
#
# Usage: .\scripts\exp_c_awada.ps1 [sim10k_to_cityscapes|cityscapes_to_foggy|cityscapes_to_bdd100k]

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
    $OutputBase    = "./outputs/exp_c_sim10k2cs"
    $BaselineCkpt  = "./outputs/exp_a_sim10k2cs/detector_final.pth"
} elseif ($Benchmark -eq "cityscapes_to_foggy") {
    $SourceDataset = "cityscapes"
    $SourceRoot    = "./data/cityscapes"
    $SourceImages  = "./data/cityscapes/leftImg8bit/train"
    $TargetDataset = "foggy_cityscapes"
    $TargetRoot    = "./data/foggy_cityscapes"
    $TargetImages  = "./data/foggy_cityscapes/leftImg8bit_foggy/train"
    $NumClasses    = 8
    $OutputBase    = "./outputs/exp_c_cs2foggy"
    $BaselineCkpt  = "./outputs/exp_a_cs2foggy/detector_final.pth"
} elseif ($Benchmark -eq "cityscapes_to_bdd100k") {
    $SourceDataset = "cityscapes"
    $SourceRoot    = "./data/cityscapes"
    $SourceImages  = "./data/cityscapes/leftImg8bit/train"
    $TargetDataset = "bdd100k"
    $TargetRoot    = "./data/bdd100k"
    $TargetImages  = "./data/bdd100k/images/100k/train"
    $NumClasses    = 7
    $OutputBase    = "./outputs/exp_c_cs2bdd"
    $BaselineCkpt  = "./outputs/exp_a_cs2bdd/detector_final.pth"
} else {
    Write-Error "Unknown benchmark: $Benchmark"
    exit 1
}

$SourceAttentionDir    = "$OutputBase/source_attention_maps"
$CycadaGanOutput       = "$OutputBase/cycada_gan"
$CycadaStylizedDir     = "$OutputBase/cycada_stylized_images"
$CycadaDetectorOutput  = "$OutputBase/cycada_detector"
$TargetAttentionDir    = "$OutputBase/target_attention_maps"
$AwadaGanOutput        = "$OutputBase/awada_gan"
$AwadaStylizedDir      = "$OutputBase/awada_stylized_images"
$DetectorOutput        = "$OutputBase/detector"

foreach ($dir in @(
    $SourceAttentionDir,
    $CycadaGanOutput,
    $CycadaStylizedDir,
    $CycadaDetectorOutput,
    $TargetAttentionDir,
    $AwadaGanOutput,
    $AwadaStylizedDir,
    $DetectorOutput
)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

Write-Host "========================================"
Write-Host "Experiment C: AWADA"
Write-Host "Benchmark: $Benchmark"
Write-Host "Baseline checkpoint: $BaselineCkpt"
Write-Host "========================================"

# Step 1: Generate source attention maps from the Exp A (source-trained) detector
Write-Host "[Step 1] Generating source RPN attention maps from baseline detector..."
if (-not (Test-Path $BaselineCkpt)) {
    Write-Error "ERROR: Baseline checkpoint not found: $BaselineCkpt"
    Write-Host "Please run exp_a_baseline.ps1 first."
    exit 1
}

python tools/generate_attention_maps.py `
    --detector_checkpoint $BaselineCkpt `
    --dataset $SourceDataset `
    --data_root $SourceRoot `
    --output_dir $SourceAttentionDir `
    --score_threshold 0.5 `
    --num_classes $NumClasses `
    --split train `
    --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 2: Train CyCada GAN to learn the source -> target style mapping
Write-Host "[Step 2] Training CyCada GAN for source->target style transfer..."
python tools/train_cycada.py `
    --source_dir $SourceImages `
    --target_dir $TargetImages `
    --output_dir $CycadaGanOutput `
    --config configs/cycada.yaml `
    --epochs 200 `
    --batch_size 1 `
    --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 3: Stylize source images using the CyCada generator
Write-Host "[Step 3] Stylizing source images with CyCada generator..."
$LatestCycadaGan = Get-ChildItem -Path $CycadaGanOutput -Filter "cycada_epoch_*.pth" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
python tools/stylize_dataset.py `
    --generator_checkpoint $LatestCycadaGan `
    --source_dir $SourceImages `
    --output_dir $CycadaStylizedDir `
    --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 4: Train a detector on the CyCada-stylized images
# This detector has been exposed to the target visual style, making it suitable
# for generating attention maps on real target-domain images.
Write-Host "[Step 4] Training CyCada detector on stylized source images..."
python tools/train_detector.py `
    --dataset $SourceDataset `
    --data_root $SourceRoot `
    --image_dir $CycadaStylizedDir `
    --num_classes $NumClasses `
    --output_dir $CycadaDetectorOutput `
    --epochs 10 `
    --batch_size 2 `
    --lr 0.005 `
    --device cuda `
    --pretrained `
    --val_dataset $TargetDataset `
    --val_data_root $TargetRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 5: Generate target attention maps using the CyCada detector on real target images
Write-Host "[Step 5] Generating target RPN attention maps from CyCada detector..."
python tools/generate_attention_maps.py `
    --detector_checkpoint "$CycadaDetectorOutput/detector_final.pth" `
    --dataset $TargetDataset `
    --data_root $TargetRoot `
    --output_dir $TargetAttentionDir `
    --score_threshold 0.5 `
    --num_classes $NumClasses `
    --split train `
    --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 6: Train AWADA CycleGAN with attention-masked losses
# Source attention maps: from the source-trained detector (Step 1)
# Target attention maps: from the CyCada-trained detector (Step 5)
Write-Host "[Step 6] Training AWADA CycleGAN with source and target attention maps..."
python tools/train_awada.py `
    --source_dir $SourceImages `
    --target_dir $TargetImages `
    --source_attention_dir $SourceAttentionDir `
    --target_attention_dir $TargetAttentionDir `
    --output_dir $AwadaGanOutput `
    --config configs/awada.yaml `
    --epochs 200 `
    --batch_size 1 `
    --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 7: Stylize source images using the AWADA generator
Write-Host "[Step 7] Stylizing source images with AWADA generator..."
$LatestAwadaGan = Get-ChildItem -Path $AwadaGanOutput -Filter "awada_epoch_*.pth" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
python tools/stylize_dataset.py `
    --generator_checkpoint $LatestAwadaGan `
    --source_dir $SourceImages `
    --output_dir $AwadaStylizedDir `
    --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 8: Train final detector on AWADA-stylized images
Write-Host "[Step 8] Training final detector on AWADA-stylized source images..."
python tools/train_detector.py `
    --dataset $SourceDataset `
    --data_root $SourceRoot `
    --image_dir $AwadaStylizedDir `
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

Write-Host "Experiment C (AWADA) complete!"
