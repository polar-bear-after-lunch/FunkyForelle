"""Feature engineering shared by all models.

The physics feature here is the oxygen saturation concentration: the amount of
DO water can hold at a given temperature. It lets the model separate the
temperature effect (solubility) from biological effects (algae, decomposition),
which is exactly the part of DO that temperature-only warnings miss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def do_saturation_mgl(temp_c, pressure_hpa=1013.25):
    """Oxygen saturation concentration in fresh water [mg/L].

    Garcia & Gordon (1992) fit (Benson & Krause data), as recommended by
    USGS (2011) for fresh water, with a simple barometric pressure correction
    (ignores water-vapour pressure, error < 1 % below 30 °C).

    Valid for 0–40 °C.
    """
    t = np.asarray(temp_c, dtype=float)
    ts = np.log((298.15 - t) / (273.15 + t))
    a0, a1, a2, a3, a4, a5 = 2.00907, 3.22014, 4.0501, 4.94457, -0.256847, 3.88767
    ln_c = a0 + a1 * ts + a2 * ts**2 + a3 * ts**3 + a4 * ts**4 + a5 * ts**5
    c_ml_l = np.exp(ln_c)  # mL/L
    c_mg_l = c_ml_l * 1.42905  # mL O2 -> mg O2
    return c_mg_l * (np.asarray(pressure_hpa, dtype=float) / 1013.25)


def pressure_from_elevation(elevation_m: float) -> float:
    """Standard-atmosphere barometric pressure [hPa] at ``elevation_m``."""
    return 1013.25 * (1 - 2.25577e-5 * elevation_m) ** 5.25588


def add_do_features(df: pd.DataFrame, elevation_m: float | None = None) -> pd.DataFrame:
    """Add ``do_sat_mgl`` (from ``water_temp``) and, if DO is present, ``do_deficit``."""
    out = df.copy()
    if "water_temp" in out:
        p = pressure_from_elevation(elevation_m) if elevation_m is not None else 1013.25
        out["do_sat_mgl"] = do_saturation_mgl(out["water_temp"], p)
        if "do_mgl" in out:
            out["do_deficit"] = out["do_sat_mgl"] - out["do_mgl"]
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical day-of-year encoding (summer critical period is mid-June–August)."""
    out = df.copy()
    doy = out.index.dayofyear.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def add_rolling(df: pd.DataFrame, cols, windows=(3, 7)) -> pd.DataFrame:
    """Trailing rolling means (no look-ahead) for heat- and low-flow persistence."""
    out = df.copy()
    for c in cols:
        if c in out:
            for w in windows:
                out[f"{c}_mean{w}d"] = out[c].rolling(w, min_periods=1).mean()
    return out


def fill_short_gaps(df: pd.DataFrame, max_gap: int = 7) -> pd.DataFrame:
    """Linear interpolation of gaps up to ``max_gap`` days; longer gaps stay NaN."""
    return df.interpolate(method="time", limit=max_gap, limit_area="inside")


def make_windows(
    df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    lookback: int,
    horizon: int,
):
    """Sliding windows; windows with any NaN in inputs or targets are dropped.

    Returns ``X [N, lookback, F]``, ``y [N, horizon, T]`` (float32) and the
    forecast-origin dates (last day of each input window).
    """
    xa = df[features].to_numpy(dtype=np.float32)
    ya = df[targets].to_numpy(dtype=np.float32)
    n = len(df) - lookback - horizon + 1
    X, y, origins = [], [], []
    for i in range(max(n, 0)):
        xw = xa[i : i + lookback]
        yw = ya[i + lookback : i + lookback + horizon]
        if np.isnan(xw).any() or np.isnan(yw).any():
            continue
        X.append(xw)
        y.append(yw)
        origins.append(df.index[i + lookback - 1])
    if not X:
        return (
            np.empty((0, lookback, len(features)), np.float32),
            np.empty((0, horizon, len(targets)), np.float32),
            pd.DatetimeIndex([]),
        )
    return np.stack(X), np.stack(y), pd.DatetimeIndex(origins)
