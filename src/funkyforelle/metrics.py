"""Forecast metrics. Arrays are shaped ``[N, horizon, n_targets]``."""
from __future__ import annotations

import numpy as np


def _check(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")
    if y_true.ndim != 3:
        raise ValueError(f"expected [N, H, T], got {y_true.shape}")
    return y_true, y_pred


def rmse_per_step(y_true, y_pred) -> np.ndarray:
    """RMSE per forecast step -> ``[horizon, n_targets]``."""
    y_true, y_pred = _check(y_true, y_pred)
    return np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))


def nse(obs, sim) -> float:
    """Nash–Sutcliffe efficiency (1 = perfect, 0 = as good as the mean)."""
    obs, sim = np.ravel(obs), np.ravel(sim)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1 - np.sum((obs - sim) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def kge(obs, sim) -> float:
    """Kling–Gupta efficiency, Gupta et al. (2009): r, mean ratio, std ratio."""
    obs, sim = np.ravel(obs), np.ravel(sim)
    if obs.std() == 0 or sim.std() == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(obs, sim)[0, 1])
    beta = sim.mean() / obs.mean() if obs.mean() != 0 else 1.0
    alpha = sim.std() / obs.std() if obs.std() > 0 else 1.0
    return float(1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (alpha - 1) ** 2))


def evaluate(y_true, y_pred, targets: list[str]) -> dict[str, dict[str, float]]:
    """RMSE, MAE, NSE, KGE per target, plus RMSE at day 1 and at the last horizon day.

    RMSE is averaged over horizon steps (mean of per-step RMSE), matching AareML.
    """
    y_true, y_pred = _check(y_true, y_pred)
    per_step = rmse_per_step(y_true, y_pred)
    out = {}
    for i, t in enumerate(targets):
        o, s = y_true[:, :, i], y_pred[:, :, i]
        out[t] = {
            "RMSE": float(per_step[:, i].mean()),
            "MAE": float(np.abs(o - s).mean()),
            "NSE": nse(o, s),
            "KGE": kge(o, s),
            "RMSE_day1": float(per_step[0, i]),
            f"RMSE_day{per_step.shape[0]}": float(per_step[-1, i]),
        }
    return out


def threshold_skill(obs, sim, threshold: float) -> dict[str, float]:
    """Hit rate / false-alarm ratio for "DO below threshold" events (e.g. 6 mg/L).

    Fisheries trade off missed events against false alarms, so report both.
    """
    o, s = np.ravel(obs) < threshold, np.ravel(sim) < threshold
    hits, misses, fa = np.sum(o & s), np.sum(o & ~s), np.sum(~o & s)
    return {
        "events": int(o.sum()),
        "hit_rate": float(hits / (hits + misses)) if hits + misses else float("nan"),
        "false_alarm_ratio": float(fa / (hits + fa)) if hits + fa else float("nan"),
    }
