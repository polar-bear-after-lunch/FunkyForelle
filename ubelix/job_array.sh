#!/bin/bash
# =============================================================================
# FunkyForelle: config x seed sweep as a Slurm array (CPU, epyc2).
# Each array task trains one (config, seed) pair; tasks run in parallel.
#
#     mkdir -p logs && sbatch ubelix/job_array.sh
# Edit CONFIGS / SEEDS below and keep --array=0-(N_CONFIGS*N_SEEDS-1).
#
# GPU variant (only worth it for Optuna, multi-gauge or TFT models): replace
# the partition/qos lines with
#   #SBATCH --partition=gpu-invest
#   #SBATCH --qos=job_gpu_preemptable
#   #SBATCH --gres=gpu:rtx4090:1
# =============================================================================
#SBATCH --job-name=ff_sweep
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

CONFIGS=(configs/exp2c_past_weather_day1.toml configs/exp3a_oracle_trained.toml
         configs/exp3b_noise_trained.toml configs/exp3c_noise_trained_no_precip.toml)
SEEDS=(42 1 2)

CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))]}

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate funkyforelle

echo "Task $SLURM_ARRAY_TASK_ID on $(hostname): $CFG seed $SEED"
funkyforelle train "$CFG" --seed "$SEED" \
    --ai-assistant "Perplexity Computer (Claude Opus 5.5) - UBELIX sweep" \
    --notes "UBELIX sweep $SLURM_ARRAY_JOB_ID: $(basename "$CFG" .toml) seed $SEED"
