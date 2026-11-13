# Running FunkyForelle on UBELIX

UBELIX is the University of Bern HPC cluster. You can only reach it from the university network or VPN: `ssh <campus-id>@submit01.unibe.ch`.

| File | What it does | Where it runs |
|---|---|---|
| `setup_env.sh` | conda env `funkyforelle`, CPU PyTorch, package, data, offline tests | login node, once |
| `job_first_test.sh` | tests + live APIs + Exp 3b seed 42, compared with the sandbox reference | `sbatch`, debug queue (≤ 20 min) |
| `job_array.sh` | config × seed sweep as a Slurm array | `sbatch`, `job_gratis` |

FunkyForelle runs on CPU on purpose: the model is small (~105k parameters, about 1 min per run), and the LSTM is sequential. Use the array instead of a GPU to parallelise. GPUs only pay off for Optuna, multi-gauge or TFT runs (see the header of `job_array.sh`).

## 1. Clone and set up (login node, once)

```bash
cd $HOME
git clone https://github.com/polar-bear-after-lunch/FunkyForelle.git
cd FunkyForelle
bash ubelix/setup_env.sh          # ~5–10 min, ends with "Setup complete"
```

## 2. FOEN live test (login node)

```bash
module load Anaconda3 && eval "$(conda shell.bash hook)" && conda activate funkyforelle
python -m pytest -m live -v -rs
```

## 3. First test job

```bash
mkdir -p logs
sbatch ubelix/job_first_test.sh   # prints "Submitted batch job <JOBID>"
squeue --me                        # PD = waiting, R = running
tail -f logs/ff_first_test_<JOBID>.out
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
```

The log ends with a comparison block. `RESULT: reproduced` means DO RMSE ≈ 0.153 mg/L and temperature ≈ 0.516 °C, matching the sandbox run 20261027-063422.

## 4. Saving the run records

Jobs never push to git; the AareML auto-push caused rebase failures and wrong authors. Each run records its host and Slurm job ID in `runs/<id>/run.json` (`compute`). To publish runs, commit from the login node with your own identity:

```bash
git config user.name  "polar-bear-after-lunch"
git config user.email "85006740+polar-bear-after-lunch@users.noreply.github.com"
git pull --rebase
git add runs/ && git commit -m "UBELIX first test run" && git push
```

Pushing needs a GitHub SSH key or token on UBELIX. If you don't have one, paste the log's comparison block into the chat instead.

## 5. Sweeps (later)

Edit `CONFIGS` / `SEEDS` in `job_array.sh` and keep `--array=0-(N−1)`. Then run `sbatch ubelix/job_array.sh`. `sqos` shows the QoS limits.

## Troubleshooting

- **`module: command not found` in a job:** add `source /etc/profile` before `module load`.
- **Job stuck in `PD`:** check `squeue --me -o "%.10i %.9P %.8T %.20R"` for the reason. `job_debug` usually starts within minutes.
- **Live tests fail on the compute node but pass on the login node:** that's a network restriction. Fetch data on the login node; training doesn't need the internet.
- **Different RMSE (beyond ±0.01):** send the full log; a torch version or thread-count difference is the likely cause.
