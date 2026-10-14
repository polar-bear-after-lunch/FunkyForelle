"""Command line entry point.

Examples
--------
Fetch recent weather for Altenrhein and save tidy CSV::

    funkyforelle fetch meteoswiss ARH --start 2026-06-01 --end 2026-08-31 -o data/arh.csv

Train and log a run (writes runs/<id>/run.json and appends runs/index.csv)::

    funkyforelle train configs/rhein_diepoldsau_2473.toml \\
        --ai-assistant "Perplexity Computer · Claude Opus 5.5"
"""
from __future__ import annotations

import argparse
import sys

from .connectors import REGISTRY


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="funkyforelle")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="fetch data from one connector into tidy CSV")
    f.add_argument("source", choices=sorted(REGISTRY))
    f.add_argument("stations", nargs="+")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--root", help="local data root (camels_ch_chem only)")
    f.add_argument("-o", "--out", help="output CSV (default: stdout)")

    t = sub.add_parser("train", help="run a config, evaluate vs baselines and log the run")
    t.add_argument("config")
    t.add_argument("--model", choices=["seq2seq_lstm", "persistence", "climatology"])
    t.add_argument("--ai-assistant", default="", help="AI assistant/model that helped, for the log")
    t.add_argument("--notes", default="")
    t.add_argument("--seed", type=int, help="override model.seed from the config")
    t.add_argument("--runs-dir", default="runs")

    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        kw = {"root": a.root} if a.root else {}
        df = REGISTRY[a.source](**kw).fetch(a.stations, a.start, a.end)
        df.to_csv(a.out or sys.stdout, index=False)
        if a.out:
            print(f"{len(df)} rows -> {a.out}")
    elif a.cmd == "train":
        from .pipeline import load_config, run
        cfg = load_config(a.config)
        if a.seed is not None:
            cfg.setdefault("model", {})["seed"] = a.seed
        rec = run(cfg, a.model, a.ai_assistant, a.notes, a.runs_dir)
        print(rec.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
