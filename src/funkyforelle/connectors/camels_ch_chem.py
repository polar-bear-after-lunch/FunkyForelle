"""CAMELS-CH-Chem (Nascimento et al., 2025) — historical daily river chemistry.

Long daily records of DO, water temperature, pH and conductivity at Swiss
gauges; this is the training backbone carried over from AareML.

Download once (about 165 MB) from Zenodo, record 14980027:
https://zenodo.org/records/14980027 — then point ``root`` at the unzipped
``camels-ch-chem`` folder (default: ``data/camels-ch-chem``).

Licence: CC BY 4.0 — cite Nascimento et al. (2025).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .. import schema
from .base import Connector, as_date

ZENODO_URL = "https://zenodo.org/api/records/14980027/files/camels-ch-chem.zip/content"
DAILY_SUBDIR = Path("stream_water_chemistry/timeseries/daily")

# CAMELS-CH-Chem column -> canonical variable
COLUMNS: dict[str, str] = {
    "temp_sensor": "water_temp",
    "O2C_sensor": "do_mgl",
    "O2S_sensor": "do_sat_pct",
    "pH_sensor": "ph",
    "ec_sensor": "conductivity",
}


class CamelsChChem(Connector):
    name = "camels_ch_chem"
    licence = "CC BY 4.0"
    attribution = (
        "Nascimento, T. et al. (2025) CAMELS-CH-Chem, Zenodo, doi:10.5281/zenodo.14980027"
    )

    def __init__(self, root: str | Path = "data/camels-ch-chem", **kw):
        super().__init__(**kw)
        self.root = Path(root)

    def path_for(self, gauge: str) -> Path:
        return self.root / DAILY_SUBDIR / f"camels_ch_chem_daily_{gauge}.csv"

    def available_gauges(self) -> list[str]:
        d = self.root / DAILY_SUBDIR
        return sorted(p.stem.removeprefix("camels_ch_chem_daily_") for p in d.glob("*.csv"))

    def fetch(self, stations: Iterable[str], start=None, end=None) -> pd.DataFrame:
        frames = []
        for g in stations:
            p = self.path_for(str(g))
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found — download CAMELS-CH-Chem from {ZENODO_URL} "
                    f"and unzip into {self.root}"
                )
            frames.append(self.parse_csv(p, str(g)))
        df = pd.concat(frames, ignore_index=True) if frames else schema.empty()
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(as_date(start), tz="UTC")]
        if end is not None:
            df = df[df["timestamp"] < pd.Timestamp(as_date(end), tz="UTC") + pd.Timedelta(days=1)]
        return schema.validate(df)

    @classmethod
    def parse_csv(cls, path_or_buf, gauge: str) -> pd.DataFrame:
        raw = pd.read_csv(path_or_buf, parse_dates=["date"])
        cols = [c for c in COLUMNS if c in raw.columns]
        long = raw[["date", *cols]].melt(id_vars="date", var_name="col", value_name="value")
        long["timestamp"] = pd.to_datetime(long["date"]).dt.tz_localize("UTC")
        long["variable"] = long["col"].map(COLUMNS)
        long["unit"] = long["variable"].map(schema.VARIABLES)
        long["station"] = gauge
        long["source"] = cls.name
        return schema.validate(long)
