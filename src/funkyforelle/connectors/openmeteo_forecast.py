"""Weather *forecasts* (MeteoSwiss ICON-CH2 by default) through the Open-Meteo APIs.

Two modes:

* ``fetch(stations, start, end)`` reads the **archive of past forecasts** from
  the Previous Runs API. ``lead_day=N`` returns, for each valid day, the value
  forecast N days earlier, which is what a live system would have seen.
  Archive for ``meteoswiss_icon_ch2``: from 2025-07-06.
* ``latest(stations, days=3)`` reads the **current forecast** for the next days.

Stations are named sites from ``SITES`` (MeteoSwiss station codes and project
sites) or ``"lat,lon"`` strings.

Hourly values are aggregated to the daily definitions of MeteoSwiss SwissMetNet,
so forecasts and observations are comparable:

* ``air_temp``: mean, 00–24 UTC
* ``air_temp_max``: maximum, 00–24 UTC
* ``global_radiation``: mean, 00–24 UTC
* ``precip``: total, 06 UTC on the day to 06 UTC on the next day (as ``rre150d0``)

Licence: Open-Meteo data are CC BY 4.0, and the free API is for non-commercial
use only (fewer than 10,000 calls per day). The underlying model is from
MeteoSwiss. Attribution: "Weather data by Open-Meteo.com; ICON-CH2 model:
MeteoSwiss".
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .. import schema
from .base import Connector, as_date

PREVIOUS_RUNS_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

#: named sites -> (lat, lon). MeteoSwiss stations use their SwissMetNet codes.
SITES: dict[str, tuple[float, float]] = {
    "VAD": (47.1275, 9.5176),    # MeteoSwiss Vaduz
    "STG": (47.4253, 9.3985),    # MeteoSwiss St. Gallen
    "SMA": (47.3779, 8.5656),    # MeteoSwiss Zürich-Fluntern
    "BER": (46.9907, 7.4640),    # MeteoSwiss Bern-Zollikofen
    "2473": (47.3825, 9.6403),   # FOEN gauge Rhein - Diepoldsau, Rietbrücke
}

HOURLY = ["temperature_2m", "precipitation", "shortwave_radiation"]
MIN_HOURS = 20  # a day needs at least this many hourly values


def resolve_site(s: str) -> tuple[float, float]:
    s = str(s)
    if s in SITES:
        return SITES[s]
    try:
        lat, lon = (float(x) for x in s.split(","))
    except ValueError as e:
        raise KeyError(f"unknown site {s!r}; use one of {sorted(SITES)} or 'lat,lon'") from e
    return lat, lon


def hourly_to_daily(h: pd.DataFrame) -> pd.DataFrame:
    """Hourly frame (UTC index; temperature_2m, precipitation, shortwave_radiation) -> daily."""
    t = h["temperature_2m"]
    n_t = t.resample("D").count()
    d = pd.DataFrame({
        "air_temp": t.resample("D").mean(),
        "air_temp_max": t.resample("D").max(),
        "global_radiation": h["shortwave_radiation"].resample("D").mean(),
    })
    d.loc[n_t < MIN_HOURS] = np.nan
    # precipitation day d = 06 UTC d .. 06 UTC d+1 (SwissMetNet rre150d0)
    p = h["precipitation"]
    p_shift = p.copy()
    p_shift.index = p_shift.index - pd.Timedelta(hours=6)
    pd_ = p_shift.resample("D").sum(min_count=MIN_HOURS)
    d["precip"] = pd_.reindex(d.index)
    d.index = d.index.tz_localize(None).rename("date")  # same as schema.to_daily_wide
    return d


class OpenMeteoForecast(Connector):
    name = "openmeteo_forecast"
    licence = "CC BY 4.0 (Open-Meteo); free API for non-commercial use only"
    attribution = "Weather data by Open-Meteo.com; ICON-CH2 model: MeteoSwiss"

    def __init__(self, *args, model: str = "meteoswiss_icon_ch2", lead_day: int = 1, **kw):
        super().__init__(*args, **kw)
        if not 1 <= lead_day <= 7:
            raise ValueError("lead_day must be 1..7")
        self.model, self.lead_day = model, lead_day

    # archive ---------------------------------------------------------------
    def fetch(self, stations: Iterable[str], start, end) -> pd.DataFrame:
        start, end = as_date(start), as_date(end)
        frames = []
        for st in stations:
            frames.append(self._tidy(self.fetch_daily(st, start, end, self.lead_day), st))
        return schema.validate(pd.concat(frames)) if frames else schema.empty()

    def fetch_daily(self, station: str, start, end, lead_day: int) -> pd.DataFrame:
        """Daily forecast for valid days start..end, issued ``lead_day`` days earlier."""
        lat, lon = resolve_site(station)
        hv = ",".join(f"{v}_previous_day{lead_day}" for v in HOURLY)
        # precip day ``end`` needs hours up to 06 UTC on end + 1
        js = self._get(PREVIOUS_RUNS_API, latitude=lat, longitude=lon, hourly=hv,
                       models=self.model, timezone="UTC", start_date=str(as_date(start)),
                       end_date=str(as_date(end) + dt.timedelta(days=1))).json()
        h = self.parse_hourly(js, suffix=f"_previous_day{lead_day}")
        return hourly_to_daily(h).loc[str(start):str(end)]

    def fetch_leads(self, station: str, start, end, leads=(1, 2, 3)) -> pd.DataFrame:
        """Long frame: valid_date, lead_day, air_temp, air_temp_max, precip, global_radiation."""
        out = []
        for L in leads:
            d = self.fetch_daily(station, start, end, L)
            out.append(d.assign(lead_day=L).rename_axis("valid_date").reset_index())
        return pd.concat(out, ignore_index=True)

    # live ------------------------------------------------------------------
    def latest(self, stations: Iterable[str], days: int = 3) -> pd.DataFrame:
        """Current forecast for the next ``days`` full UTC days (tidy schema)."""
        frames = []
        today = dt.datetime.now(dt.UTC).date()
        for st in stations:
            lat, lon = resolve_site(st)
            js = self._get(FORECAST_API, latitude=lat, longitude=lon, hourly=",".join(HOURLY),
                           models=self.model, timezone="UTC", forecast_days=days + 2).json()
            d = hourly_to_daily(self.parse_hourly(js))
            d = d.loc[str(today + dt.timedelta(days=1)):str(today + dt.timedelta(days=days))]
            frames.append(self._tidy(d, st))
        return schema.validate(pd.concat(frames)) if frames else schema.empty()

    # parsing ---------------------------------------------------------------
    @staticmethod
    def parse_hourly(js: dict, suffix: str = "") -> pd.DataFrame:
        if "hourly" not in js:
            raise ValueError(f"unexpected Open-Meteo response: {str(js)[:200]}")
        h = pd.DataFrame(js["hourly"])
        h["time"] = pd.to_datetime(h["time"], utc=True)
        h = h.set_index("time").rename(columns=lambda c: c.removesuffix(suffix))
        return h[HOURLY].astype(float)

    def _tidy(self, daily: pd.DataFrame, station: str) -> pd.DataFrame:
        long = daily.rename_axis("timestamp").reset_index().melt(
            id_vars="timestamp", var_name="variable", value_name="value")
        long["timestamp"] = pd.to_datetime(long["timestamp"], utc=True)
        return long.assign(station=str(station), unit=long["variable"].map(schema.VARIABLES),
                           source=self.name)
