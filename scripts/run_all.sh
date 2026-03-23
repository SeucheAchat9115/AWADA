#!/bin/bash
# Run all experiments end-to-end for a given benchmark.
# Executes Experiment A (baseline), B (CycleGAN), C (AWADA), and D (oracle)
# in the correct dependency order: A must finish before C begins.
#
# Usage: bash scripts/run_all.sh [sim10k_to_cityscapes|cityscapes_to_foggy]
#
# Environment variable overrides (optional):
#   SIM10K_ROOT      Path to sim10k dataset         (default: /data/sim10k)
#   CITYSCAPES_ROOT  Path to Cityscapes dataset      (default: /data/cityscapes)
#   FOGGY_ROOT       Path to Foggy Cityscapes        (default: /data/foggy_cityscapes)
#   OUTPUT_ROOT      Root directory for all outputs  (default: ./outputs)
#   DEVICE           Compute device                  (default: cuda)
#   EPOCHS           Detector training epochs        (default: 10)
#   GAN_EPOCHS       GAN training epochs             (default: 200)
#   BATCH_SIZE       Detector batch size             (default: 2)
#   GAN_BATCH        GAN batch size                  (default: 1)
#   TOP_K            Top-K RPN proposals for masks   (default: 10)

set -euo pipefail

BENCHMARK=${1:-sim10k_to_cityscapes}

if [ "$BENCHMARK" != "sim10k_to_cityscapes" ] && [ "$BENCHMARK" != "cityscapes_to_foggy" ]; then
    echo "Unknown benchmark: $BENCHMARK"
    echo "Usage: $0 [sim10k_to_cityscapes|cityscapes_to_foggy]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "AWADA End-to-End Pipeline"
echo "Benchmark: $BENCHMARK"
echo "========================================"
echo ""

# ------------------------------------------------------------------
# Experiment A: Non-Adaptive Baseline
# (must run before Experiment C, which needs the baseline checkpoint)
# ------------------------------------------------------------------
echo ">>> Running Experiment A: Non-Adaptive Baseline"
bash "$SCRIPT_DIR/exp_a_baseline.sh" "$BENCHMARK"
echo ""

# ------------------------------------------------------------------
# Experiment B: Standard CycleGAN
# ------------------------------------------------------------------
echo ">>> Running Experiment B: Standard CycleGAN"
bash "$SCRIPT_DIR/exp_b_cyclegan.sh" "$BENCHMARK"
echo ""

# ------------------------------------------------------------------
# Experiment C: AWADA (depends on Experiment A checkpoint)
# ------------------------------------------------------------------
echo ">>> Running Experiment C: AWADA"
bash "$SCRIPT_DIR/exp_c_awada.sh" "$BENCHMARK"
echo ""

# ------------------------------------------------------------------
# Experiment D: Oracle (Upper Bound)
# ------------------------------------------------------------------
echo ">>> Running Experiment D: Oracle"
bash "$SCRIPT_DIR/exp_d_oracle.sh" "$BENCHMARK"
echo ""

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
OUTPUT_ROOT_DIR="${OUTPUT_ROOT:-./outputs}"

echo "========================================"
echo "All experiments complete!"
echo "Results summary:"
echo "========================================"
for exp in A B C D; do
    case "$exp-$BENCHMARK" in
        A-sim10k_to_cityscapes)  results_file="$OUTPUT_ROOT_DIR/exp_a_sim10k2cs/results.txt" ;;
        A-cityscapes_to_foggy)   results_file="$OUTPUT_ROOT_DIR/exp_a_cs2foggy/results.txt" ;;
        B-sim10k_to_cityscapes)  results_file="$OUTPUT_ROOT_DIR/exp_b_sim10k2cs/detector/results.txt" ;;
        B-cityscapes_to_foggy)   results_file="$OUTPUT_ROOT_DIR/exp_b_cs2foggy/detector/results.txt" ;;
        C-sim10k_to_cityscapes)  results_file="$OUTPUT_ROOT_DIR/exp_c_sim10k2cs/detector/results.txt" ;;
        C-cityscapes_to_foggy)   results_file="$OUTPUT_ROOT_DIR/exp_c_cs2foggy/detector/results.txt" ;;
        D-sim10k_to_cityscapes)  results_file="$OUTPUT_ROOT_DIR/exp_d_sim10k2cs/results.txt" ;;
        D-cityscapes_to_foggy)   results_file="$OUTPUT_ROOT_DIR/exp_d_cs2foggy/results.txt" ;;
    esac
    if [ -f "$results_file" ]; then
        echo ""
        echo "--- Experiment $exp ---"
        cat "$results_file"
    fi
done
