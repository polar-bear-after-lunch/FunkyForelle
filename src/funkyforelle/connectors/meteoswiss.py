"""MeteoSwiss open data (OGD) — SwissMetNet automatic weather stations.

Data are published as CSV files per station on the federal STAC catalogue
(https://data.geo.admin.ch/browser/#/collections/ch.meteoschweiz.ogd-smn).

File naming: ``ogd-smn_<abbr>_<granularity>_<period>.csv`` with

* granularity ``d`` (daily), ``h`` (hourly), ``t`` (10-minute)
* period ``historical`` (start of record until 31 Dec last year; hourly and
  10-minute files are split by decade, e.g. ``historical_2020-2029``),
  ``recent`` (1 Jan this year until yesterday) and ``now`` (last ~24 h, h/t only)

Timestamps are UTC, format ``dd.mm.yyyy HH:MM``. Licence: open use, source
attribution "MeteoSwiss" required.
"""
from __future__ import annotations

import datetime as dt
import io
from collections.abc import Iterable

import pandas as pd

from .. import schema
from .base import Connector, as_date

BASE = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"

# MeteoSwiss parameter short name -> canonical variable, per granularity
PARAMS: dict[str, dict[str, str]] = {
    "d": {
        "tre200d0": "air_temp",
        "tre200dx": "air_temp_max",
        "rre150d0": "precip",
        "gre000d0": "global_radiation",
        "sre000d0": "sunshine",
    },
    "h": {
        "tre200h0": "air_temp",
        "rre150h0": "precip",
        "gre000h0": "global_radiation",
        "sre000h0": "sunshine",
    },
}


class MeteoSwiss(Connector):
    name = "meteoswiss"
    licence = "Open Government Data (MeteoSwiss terms of use), attribution required"
    attribution = "Source: MeteoSwiss"

    def __init__(self, *args, granularity: str = "d", **kw):
        super().__init__(*args, **kw)
        if granularity not in PARAMS:
            raise ValueError("granularity must be 'd' or 'h'")
        self.granularity = granularity

    # file selection -----------------------------------------------------
    def files_for(self, abbr: str, start: dt.date, end: dt.date, today: dt.date | None = None):
        """Return the CSV URLs needed to cover ``start``..``end``."""
        today = today or dt.date.today()
        a = abbr.lower()
        g = self.granularity
        urls = []
        this_year = dt.date(today.year, 1, 1)
        if start < this_year:
            if g == "d":
                urls.append(f"{BASE}/{a}/ogd-smn_{a}_d_historical.csv")
            else:
                for dec in range(start.year // 10 * 10, min(end.year, today.year - 1) // 10 * 10 + 1, 10):
                    urls.append(f"{BASE}/{a}/ogd-smn_{a}_h_historical_{dec}-{dec + 9}.csv")
        if end >= this_year:
            urls.append(f"{BASE}/{a}/ogd-smn_{a}_{g}_recent.csv")
            if g == "h" and end >= today - dt.timedelta(days=1):
                urls.append(f"{BASE}/{a}/ogd-smn_{a}_h_now.csv")
        return urls

    def fetch(self, stations: Iterable[str], start, end) -> pd.DataFrame:
        start, end = as_date(start), as_date(end)
        frames = []
        for st in stations:
            for url in self.files_for(st, start, end):
                text = self._get(url).content.decode("latin-1")
                frames.append(self.parse_csv(text))
        if not frames:
            return schema.empty()
        df = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["timestamp", "station", "variable"], keep="last"
        )
        lo = pd.Timestamp(start, tz="UTC")
        hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        df = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)]
        return schema.validate(df)

    def parse_csv(self, text: str) -> pd.DataFrame:
        """Parse one OGD CSV file into the tidy schema."""
        raw = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
        mapping = {k: v for k, v in PARAMS[self.granularity].items() if k in raw.columns}
        if not mapping:
            return schema.empty()
        ts = pd.to_datetime(raw["reference_timestamp"], format="%d.%m.%Y %H:%M", utc=True)
        long = (
            raw[list(mapping)]
            .assign(timestamp=ts, station=raw["station_abbr"].str.upper())
            .melt(id_vars=["timestamp", "station"], var_name="param", value_name="value")
        )
        long["variable"] = long["param"].map(mapping)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long["unit"] = long["variable"].map(schema.VARIABLES)
        long["source"] = self.name
        return schema.validate(long.drop(columns="param"))
