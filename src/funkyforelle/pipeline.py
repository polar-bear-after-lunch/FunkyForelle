"""Config-driven pipeline: connectors -> daily feature table -> windows -> model -> run record.

A run is described by a TOML file in ``configs/`` (see ``configs/rhein_diepoldsau_2473.toml``).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .connectors import REGISTRY
from .metrics import evaluate, threshold_skill
from .models import Climatology, Persistence
from .schema import to_daily_wide
from .tracking import RunRecord, data_hash


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("lookback", 21)
    cfg.setdefault("horizon", 14)
    cfg.setdefault("max_gap", 7)
    return cfg


def build_table(cfg: dict) -> pd.DataFrame:
    """Fetch every configured source and join them into one daily table.

    The first source is the *primary* river station; its columns keep their
    canonical names (``do_mgl``, ``water_temp`` ...). Columns from later sources
    that would collide get a ``__<source>`` suffix (e.g. ``air_temp`` stays,
    but a second ``water_temp`` becomes ``water_temp__foen_hydro``).
    """
    start, end = cfg["start"], cfg["end"]
    table = None
    for src, scfg in cfg["sources"].items():
        opts = {k: v for k, v in scfg.items() if k not in ("stations", "variables")}
        conn = REGISTRY[src](**opts)
        tidy = conn.fetch(scfg["stations"], start, end)
        if "variables" in scfg:
            tidy = tidy[tidy["variable"].isin(scfg["variables"])]
        wide = to_daily_wide(tidy, prefix_station=len(scfg["stations"]) > 1)
        if table is None:
            table = wide
        else:
            wide = wide.rename(columns={c: f"{c}__{src}" for c in wide.columns if c in table})
            table = table.join(wide, how="left")
    table = table.loc[start:end]
    table = F.fill_short_gaps(table, cfg["max_gap"])
    table = F.add_do_features(table, cfg.get("elevation_m"))
    table = F.add_calendar(table)
    roll = cfg.get("rolling", [])
    if roll:
        table = F.add_rolling(table, roll)
    return table


def run(cfg: dict, model_override: str | None = None, ai_assistant: str = "",
        notes: str = "", runs_dir: str = "runs", verbose: bool = True) -> RunRecord:
    table = build_table(cfg)
    targets = cfg["targets"]
    feats = cfg.get("features") or [c for c in table.columns if table[c].notna().mean() > 0.5]
    for t in targets:  # autoregressive: past targets are always inputs
        if t not in feats:
            feats.insert(0, t)
    missing = [c for c in feats + targets if c not in table]
    if missing:
        raise KeyError(f"columns not in feature table: {missing}. Available: {list(table.columns)}")

    fut = cfg.get("future_features") or None
    missing = [c for c in (fut or []) if c not in table]
    if missing:
        raise KeyError(f"future features not in feature table: {missing}")
    L, H = cfg["lookback"], cfg["horizon"]
    tr_end, va_end = pd.Timestamp(cfg["train_end"]), pd.Timestamp(cfg["val_end"])
    parts = {
        "train": table.loc[:tr_end],
        "val": table.loc[tr_end + pd.Timedelta(days=1) - pd.Timedelta(days=L): va_end],
        "test": table.loc[va_end + pd.Timedelta(days=1) - pd.Timedelta(days=L):],
    }
    W = {k: F.make_windows(v, feats, targets, L, H, fut) for k, v in parts.items()}
    if not fut:  # uniform shape: (X, y, origins, Xf)
        W = {k: (*v, None) for k, v in W.items()}
    for k, (X, *_rest) in W.items():
        if verbose:
            print(f"[pipeline] {k}: {len(X)} windows")
        if len(X) == 0:
            raise ValueError(f"no complete windows in {k} split — check dates / gaps")
    (Xtr, ytr, _, Ftr), (Xva, yva, _, Fva), (Xte, yte, ote, Fte) = W["train"], W["val"], W["test"]

    tidx = [feats.index(t) for t in targets]
    baselines = {
        "persistence": evaluate(yte, Persistence().predict(Xte, tidx, H), targets),
        "climatology": evaluate(
            yte, Climatology().fit(parts["train"], targets).predict(ote, H), targets),
    }

    model_name = model_override or cfg.get("model", {}).get("type", "seq2seq_lstm")
    mparams = {k: v for k, v in cfg.get("model", {}).items() if k != "type"}
    if model_name == "persistence":
        pred, params = Persistence().predict(Xte, tidx, H), {}
    elif model_name == "climatology":
        pred, params = Climatology().fit(parts["train"], targets).predict(ote, H), {}
    elif model_name == "seq2seq_lstm":
        from .models.seq2seq import Seq2SeqForecaster
        m = Seq2SeqForecaster(horizon=H, verbose=verbose, **mparams).fit(
            Xtr, ytr, Xva, yva, Ftr, Fva)
        pred = m.predict(Xte, Fte)
        params = {**m.params, "best_epoch": m.best_epoch, "future_features": fut or []}
    else:
        raise ValueError(f"unknown model {model_name!r}")

    metrics = evaluate(yte, pred, targets)
    if "do_mgl" in targets and cfg.get("do_threshold_mgl"):
        i = targets.index("do_mgl")
        metrics["do_mgl"]["threshold"] = threshold_skill(
            yte[:, :, i], pred[:, :, i], cfg["do_threshold_mgl"])

    rec = RunRecord(
        model=model_name,
        params=params,
        targets=targets,
        features=feats,
        stations={s: list(map(str, c["stations"])) for s, c in cfg["sources"].items()},
        period={"start": str(cfg["start"]), "end": str(cfg["end"]),
                "train_end": str(cfg["train_end"]), "val_end": str(cfg["val_end"]),
                "lookback": L, "horizon": H,
                "n_windows": {k: int(len(v[0])) for k, v in W.items()}},
        data_hash=data_hash(table[feats]),
        metrics=metrics,
        baselines=baselines,
        notes=notes or cfg.get("name", ""),
        ai_assistant=ai_assistant,
    )
    rec.save(runs_dir)
    if model_name == "seq2seq_lstm":
        m.save(Path(runs_dir) / rec.run_id / "model.pt")
    out = Path(runs_dir) / rec.run_id
    np.save(out / "test_pred.npy", pred)
    np.save(out / "test_true.npy", yte)
    pd.Series(ote.strftime("%Y-%m-%d")).to_csv(out / "test_origins.csv", index=False, header=False)
    return rec
