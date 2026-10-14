"""Common tidy (long) schema shared by every connector.

Every connector returns a ``pandas.DataFrame`` with exactly these columns:

=========  ===================================================================
timestamp  timezone-aware UTC ``Timestamp``
station    station identifier as used by the provider (string)
variable   canonical variable name from ``VARIABLES`` below
value      float, in the canonical unit
unit       canonical unit string
source     connector name, e.g. ``"foen_hydro"``
=========  ===================================================================

Keeping one schema means new data sources can be added without touching the
model code: connectors -> ``to_daily_wide`` -> features -> model.
"""
from __future__ import annotations

import pandas as pd

COLUMNS = ["timestamp", "station", "variable", "value", "unit", "source"]

# canonical variable name -> canonical unit
VARIABLES: dict[str, str] = {
    "water_temp": "degC",
    "do_mgl": "mg/L",
    "do_sat_pct": "%",
    "discharge": "m3/s",
    "water_level": "m a.s.l.",
    "ph": "pH",
    "conductivity": "uS/cm",
    "air_temp": "degC",
    "air_temp_max": "degC",
    "precip": "mm",
    "global_radiation": "W/m2",
    "sunshine": "min",
}


def empty() -> pd.DataFrame:
    """An empty frame with the tidy schema."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS}).astype(
        {"value": "float64"}
    )


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check a connector result against the schema and normalise dtypes."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"tidy frame is missing columns: {missing}")
    unknown = sorted(set(df["variable"]) - set(VARIABLES))
    if unknown:
        raise ValueError(f"unknown variables {unknown}; add them to schema.VARIABLES first")
    out = df[COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["station"] = out["station"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"]).sort_values(["station", "variable", "timestamp"])


def to_daily_wide(df: pd.DataFrame, prefix_station: bool = False) -> pd.DataFrame:
    """Aggregate a tidy frame to daily means and pivot to one column per variable.

    Daily maxima (e.g. ``air_temp_max``) and sums (``precip``, ``sunshine``) are
    aggregated with max / sum respectively, everything else with the mean.
    The index is a timezone-naive daily ``DatetimeIndex`` (UTC dates).

    If ``prefix_station`` is true, columns are named ``<station>__<variable>`` so
    several stations can be merged into one feature table.
    """
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["date"] = d["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    agg = {"precip": "sum", "sunshine": "sum", "air_temp_max": "max"}
    parts = []
    for var, g in d.groupby("variable"):
        how = agg.get(var, "mean")
        s = g.groupby(["date", "station"])["value"].agg(how).unstack("station")
        if prefix_station:
            s.columns = [f"{st}__{var}" for st in s.columns]
        else:
            if s.shape[1] > 1:
                raise ValueError(
                    f"variable {var!r} has several stations; use prefix_station=True"
                )
            s.columns = [var]
        parts.append(s)
    wide = pd.concat(parts, axis=1).sort_index()
    wide.index.name = "date"
    return wide.asfreq("D")
