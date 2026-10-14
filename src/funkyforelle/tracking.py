"""Run records for the experiment log.

Every training run writes ``runs/<run_id>/run.json`` with everything needed to
reproduce it and to copy it into the team's Experiments table:

* ML model name and hyper-parameters
* stations, data sources, period and split dates, a hash of the training data
* metrics per target (and baselines on the same windows)
* git commit and whether the working tree was dirty
* who ran it and which AI assistant / model helped (``--ai-assistant``)

``runs/index.csv`` gets one summary line per run.
"""
from __future__ import annotations

import csv
import datetime as dt
import getpass
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


def git_info(cwd: str | Path = ".") -> dict[str, str | bool]:
    def run(*a):
        return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except FileNotFoundError:  # git not installed
        return {"commit": "", "dirty": False}


def data_hash(df: pd.DataFrame) -> str:
    """Short, stable fingerprint of a feature table (index + values)."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:12]


@dataclass
class RunRecord:
    model: str
    params: dict
    targets: list[str]
    features: list[str]
    stations: dict[str, list[str]]
    period: dict[str, str]
    data_hash: str
    metrics: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)
    notes: str = ""
    ai_assistant: str = ""
    run_id: str = field(default_factory=lambda: dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    created: str = field(default_factory=lambda: dt.datetime.now().astimezone().isoformat())
    user: str = field(default_factory=getpass.getuser)
    python: str = field(default_factory=platform.python_version)
    git: dict = field(default_factory=git_info)

    def save(self, runs_dir: str | Path = "runs") -> Path:
        d = Path(runs_dir) / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        path = d / "run.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        self._append_index(Path(runs_dir) / "index.csv")
        return path

    def _append_index(self, index: Path):
        first_target = self.targets[0]
        m = self.metrics.get(first_target, {})
        row = {
            "run_id": self.run_id,
            "created": self.created,
            "model": self.model,
            "target": first_target,
            "stations": ";".join(f"{k}:{','.join(v)}" for k, v in self.stations.items()),
            "RMSE": m.get("RMSE"),
            "NSE": m.get("NSE"),
            "KGE": m.get("KGE"),
            "persistence_RMSE": self.baselines.get("persistence", {}).get(first_target, {}).get("RMSE"),
            "commit": str(self.git.get("commit", ""))[:10],
            "ai_assistant": self.ai_assistant,
            "notes": self.notes,
        }
        new = not index.exists()
        with index.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def summary(self) -> str:
        """Markdown snippet to paste into the Experiments / Work Log tables."""
        lines = [f"**Run {self.run_id}** — {self.model}",
                 f"- Stations: {self.stations}",
                 f"- Period: {self.period}",
                 f"- Data hash: {self.data_hash} · commit {str(self.git.get('commit'))[:10]}"
                 + (" (dirty)" if self.git.get("dirty") else "")]
        for t, m in self.metrics.items():
            b = self.baselines.get("persistence", {}).get(t, {})
            lines.append(f"- {t}: RMSE {m['RMSE']:.3f} · NSE {m['NSE']:.3f} · KGE {m['KGE']:.3f}"
                         + (f" (persistence RMSE {b['RMSE']:.3f})" if b else ""))
        if self.ai_assistant:
            lines.append(f"- AI assistant: {self.ai_assistant}")
        return "\n".join(lines)
