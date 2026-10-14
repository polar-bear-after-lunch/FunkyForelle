"""Reference baselines. Every new model must beat these on the same test windows."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Persistence:
    """Tomorrow (and every later day) equals the last observed value."""

    name = "persistence"

    def fit(self, *_, **__):
        return self

    def predict(self, X: np.ndarray, target_idx: list[int], horizon: int) -> np.ndarray:
        last = X[:, -1, target_idx]  # [N, T]
        return np.repeat(last[:, None, :], horizon, axis=1)


class Climatology:
    """Day-of-year mean from the training period (smoothed with a 15-day window)."""

    name = "climatology"

    def fit(self, train: pd.DataFrame, targets: list[str]):
        doy = train.index.dayofyear
        clim = train[targets].groupby(doy).mean().reindex(range(1, 367))
        # circular smoothing so 31 Dec and 1 Jan are neighbours
        ext = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]])
        clim = ext.rolling(15, center=True, min_periods=1).mean().iloc[15:-15]
        self.table = clim.interpolate(limit_direction="both")
        self.targets = targets
        return self

    def predict(self, origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
        out = np.empty((len(origins), horizon, len(self.targets)), np.float32)
        for k in range(horizon):
            doy = (origins + pd.Timedelta(days=k + 1)).dayofyear
            out[:, k, :] = self.table.loc[doy].to_numpy()
        return out
