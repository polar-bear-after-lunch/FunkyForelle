"""Realistic weather-forecast errors for training and evaluation.

There is no archive of past ICON-CH2 forecasts for the years with river DO
data (1995–2020). Instead we measure real forecast errors where both exist
(Open-Meteo archive since 2025-07-06 vs MeteoSwiss observations), and add them
to observed weather.

Errors are resampled as whole **origin vectors**: all lead days and variables
from one real forecast run are drawn together. This keeps the correlation
between, say, a too-warm day 1 and a too-warm day 2, and between temperature
and radiation errors. Draws are matched by season, using origins within
``season_days`` of the same day of year.

To keep evaluation honest, the pool is split by ISO-week parity: even weeks
are used for training and validation, odd weeks for test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VARS = ["air_temp", "air_temp_max", "precip", "global_radiation"]
NONNEG = {"precip", "global_radiation"}


def build_error_pool(forecasts: pd.DataFrame, obs: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Error pool from archived forecasts and observations.

    ``forecasts``: long frame from ``OpenMeteoForecast.fetch_leads`` (valid_date,
    lead_day, variables). ``obs``: daily wide observations (date index).
    Returns one row per forecast origin t with columns ``<var>_d<L>`` =
    forecast(t+L, lead L) - observed(t+L), for L = 1..horizon.
    """
    obs = obs.copy()
    obs.index = pd.to_datetime(obs.index)
    cols = {}
    for L in range(1, horizon + 1):
        f = forecasts[forecasts["lead_day"] == L].set_index("valid_date")
        f.index = pd.to_datetime(f.index)
        for v in VARS:
            err = (f[v] - obs[v].reindex(f.index)).dropna()
            err.index = err.index - pd.Timedelta(days=L)  # valid date -> origin
            cols[f"{v}_d{L}"] = err
    pool = pd.DataFrame(cols).dropna().rename_axis("origin")
    return pool


class ForecastErrorSampler:
    """Draw season-matched real forecast-error vectors for given origins."""

    def __init__(self, pool: pd.DataFrame | str | Path, variables: list[str], horizon: int,
                 split: str = "train", season_days: int = 45):
        if not isinstance(pool, pd.DataFrame):
            pool = pd.read_csv(pool, index_col="origin", parse_dates=True)
        cols = [f"{v}_d{L}" for L in range(1, horizon + 1) for v in variables]
        missing = [c for c in cols if c not in pool.columns]
        if missing:
            raise KeyError(f"error pool has no columns {missing}")
        week = pool.index.isocalendar().week.to_numpy()
        if split == "train":
            pool = pool[week % 2 == 0]
        elif split == "test":
            pool = pool[week % 2 == 1]
        elif split != "all":
            raise ValueError("split must be train, test or all")
        self.E = pool[cols].to_numpy(np.float32).reshape(len(pool), horizon, len(variables))
        self.doy = pool.index.dayofyear.to_numpy()
        self.variables, self.horizon, self.season_days = variables, horizon, season_days
        self.split, self.n = split, len(pool)

    def sample(self, origins: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
        """Errors [N, horizon, F] for forecast origins (season-matched)."""
        doy = pd.DatetimeIndex(origins).dayofyear.to_numpy()
        out = np.empty((len(doy), self.horizon, len(self.variables)), np.float32)
        for d in np.unique(doy):
            dist = np.abs(self.doy - d)
            dist = np.minimum(dist, 366 - dist)
            cand = np.flatnonzero(dist <= self.season_days)
            if len(cand) == 0:
                cand = np.arange(self.n)
            sel = doy == d
            out[sel] = self.E[rng.choice(cand, sel.sum())]
        return out

    def apply(self, Xf: np.ndarray, origins, rng: np.random.Generator) -> np.ndarray:
        """Observed future weather + sampled forecast errors (clipped to physical range)."""
        noisy = Xf + self.sample(origins, rng)
        for j, v in enumerate(self.variables):
            if v in NONNEG:
                noisy[..., j] = np.maximum(noisy[..., j], 0.0)
        return noisy.astype(np.float32)
