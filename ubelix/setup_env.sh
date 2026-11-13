#!/bin/bash
# =============================================================================
# FunkyForelle: one-time UBELIX setup. Run on the LOGIN node, from the repo root:
#     bash ubelix/setup_env.sh
# Creates the conda env "funkyforelle" (Python 3.12, CPU PyTorch), installs the
# package, downloads CAMELS-CH-Chem and runs the offline tests.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/5] Loading Anaconda3..."
module load Anaconda3
eval "$(conda shell.bash hook)"

echo "[2/5] Conda env 'funkyforelle'..."
if conda env list | grep -q "^funkyforelle "; then
    echo "  exists, reusing"
else
    conda create -n funkyforelle python=3.12 -y -q
fi
conda activate funkyforelle

echo "[3/5] Installing PyTorch (CPU build) and FunkyForelle..."
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q -e ".[dev,ml]"

echo "[4/5] CAMELS-CH-Chem data..."
bash scripts/get_camels_ch_chem.sh

echo "[5/5] Offline tests..."
python -m pytest -q

python - <<'PY'
import torch, funkyforelle
print(f"\nOK  funkyforelle {getattr(funkyforelle, '__version__', '')}  torch {torch.__version__}")
PY
echo "Setup complete. Next: sbatch ubelix/job_first_test.sh"
