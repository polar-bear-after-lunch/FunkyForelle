#!/bin/bash
# =============================================================================
# FunkyForelle: first UBELIX test (CPU, debug queue, max 20 min).
# Reproduces Exp 3b (ICON-CH2 noise-trained Seq2Seq, seed 42) and compares it
# with the reference run made in the Perplexity sandbox.
#
# Submit from the repo root on the login node:
#     mkdir -p logs && sbatch ubelix/job_first_test.sh
# =============================================================================
#SBATCH --job-name=ff_first_test
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_debug
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Use only the CPUs Slurm gave us (otherwise torch sees all 128 node cores)
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate funkyforelle

echo "=== FunkyForelle first UBELIX test ==="
echo "Job $SLURM_JOB_ID on $(hostname), partition $SLURM_JOB_PARTITION, $SLURM_CPUS_PER_TASK CPUs"
echo "Commit: $(git rev-parse --short HEAD)   $(date -Is)"
python -c "import torch; print('torch', torch.__version__, '| threads', torch.get_num_threads())"

echo; echo "=== [1/3] Offline tests ==="
python -m pytest -q

echo; echo "=== [2/3] Live API tests (FOEN, MeteoSwiss, Open-Meteo) ==="
# Non-fatal: tells us whether compute nodes can reach the APIs
python -m pytest -m live -v -rs || echo "WARNING: some live tests failed (see above)"

echo; echo "=== [3/3] Exp 3b, seed 42 ==="
START=$(date +%s)
funkyforelle train configs/exp3b_noise_trained.toml --seed 42 \
    --ai-assistant "Perplexity Computer (Claude Opus 5.5) - UBELIX first test" \
    --notes "UBELIX first test: reproduce Exp 3b seed 42 (job $SLURM_JOB_ID)"
echo "Training took $(( $(date +%s) - START )) s"

echo; echo "=== Comparison with sandbox reference (run 20261027-063422) ==="
python - <<'PY'
import json, pathlib
runs = sorted(p for p in pathlib.Path("runs").glob("*/run.json")
              if json.loads(p.read_text()).get("compute", {}).get("slurm_job_id"))
d = json.loads(runs[-1].read_text())
ref = {"do_mgl": 0.1529, "water_temp": 0.5156}
print(f"run {d['run_id']}  host {d['compute'].get('host')}")
ok = True
for t, r in ref.items():
    v = d["metrics"][t]["RMSE"]
    close = abs(v - r) < 0.01
    ok &= close
    print(f"  {t:11s} RMSE {v:.4f}   reference {r:.4f}   {'OK' if close else 'DIFFERENT'}")
print("RESULT:", "reproduced" if ok else "differs from sandbox - send me this log")
PY
echo "=== done $(date -Is) ==="
