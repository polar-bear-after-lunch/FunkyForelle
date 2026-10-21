"""Compare two groups of runs on the same test windows, with a block bootstrap.

Groups are selected by a substring of the run ``notes``. Errors are averaged
over seeds first. The confidence interval of the mean difference in squared
error uses a moving-block bootstrap, because daily river errors are
autocorrelated. A plain bootstrap would make the interval too narrow.

    python scripts/compare_runs.py "Exp 2b" "Exp 2c" --target do_mgl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_group(runs: Path, pattern: str):
    recs = [json.loads(p.read_text()) for p in sorted(runs.glob("*/run.json"))]
    g = [r for r in recs if pattern in r.get("notes", "")]
    if not g:
        raise SystemExit(f"no runs match {pattern!r}")
    pred = np.stack([np.load(runs / r["run_id"] / "test_pred.npy") for r in g])
    true = np.load(runs / g[0]["run_id"] / "test_true.npy")
    origins = (runs / g[0]["run_id"] / "test_origins.csv").read_text()
    for r in g[1:]:
        if (runs / r["run_id"] / "test_origins.csv").read_text() != origins:
            raise SystemExit("runs in a group use different test windows")
    return g, ((pred - true) ** 2).mean(0), origins, g[0]["targets"]


def block_bootstrap_ci(d: np.ndarray, block: int = 30, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(d)
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block, k)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        means[b] = d[idx].mean()
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="notes substring for the reference group")
    ap.add_argument("b", help="notes substring for the candidate group")
    ap.add_argument("--target", default="do_mgl")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--block", type=int, default=30, help="bootstrap block length in days")
    a = ap.parse_args()
    runs = Path(a.runs_dir)
    ga, sea, oa, ta = load_group(runs, a.a)
    gb, seb, ob, _ = load_group(runs, a.b)
    if oa != ob:
        raise SystemExit("groups use different test windows; comparison is not paired")
    ti = ta.index(a.target)
    print(f"{a.target}: {a.b!r} ({len(gb)} runs) vs {a.a!r} ({len(ga)} runs), "
          f"{sea.shape[0]} paired test windows")
    for h in range(sea.shape[1]):
        ra, rb = np.sqrt(sea[:, h, ti].mean()), np.sqrt(seb[:, h, ti].mean())
        lo, hi = block_bootstrap_ci(seb[:, h, ti] - sea[:, h, ti], a.block)
        sig = "significant" if hi < 0 or lo > 0 else "not significant"
        print(f"  day {h + 1}: RMSE {ra:.3f} -> {rb:.3f} ({(rb / ra - 1) * 100:+.1f}%)  "
              f"95% CI of dMSE [{lo:.4f}, {hi:.4f}]  {sig}")


if __name__ == "__main__":
    main()
