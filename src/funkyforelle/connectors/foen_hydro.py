"""FOEN / BAFU hydrology (real-time) via the existenz.ch API.

The Swiss Federal Office for the Environment (FOEN/BAFU) publishes river
discharge, water level and water temperature at ~10-minute resolution.
``api.existenz.ch`` is an unofficial, free JSON wrapper around that feed and is
the path already used by ``AareML/aareml_live_demo.py``.

Notes
-----
* The wrapper serves a *recent* window (days, not years). Historical training
  data come from CAMELS-CH-Chem (``connectors/camels_ch_chem.py``).
* The service has bot protection and may reject requests from cloud/CI IP
  ranges. It works from normal residential / university networks.
* Terms of use: FOEN hydrological data delivery and usage conditions; cite
  "Source: FOEN" in any publication or product.
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import pandas as pd

from .. import schema
from .base import Connector, as_date

API_BASE = "https://api.existenz.ch/apiv1/hydro"

# existenz.ch parameter -> (canonical variable, unit conversion factor)
PARAMS: dict[str, tuple[str, float]] = {
    "temperature": ("water_temp", 1.0),
    "flow": ("discharge", 1.0),
    "height": ("water_level", 1.0),
}


class FoenHydro(Connector):
    name = "foen_hydro"
    licence = "FOEN hydrological data: terms of delivery and use"
    attribution = "Source: FOEN (Federal Office for the Environment), via api.existenz.ch"

    def __init__(self, *args, parameters: Iterable[str] = ("temperature", "flow"), **kw):
        super().__init__(*args, **kw)
        bad = set(parameters) - set(PARAMS)
        if bad:
            raise ValueError(f"unsupported FOEN parameters: {sorted(bad)}")
        self.parameters = list(parameters)

    def fetch(self, stations, start, end) -> pd.DataFrame:
        stations = [str(s) for s in stations]
        payload = self._get(
            f"{API_BASE}/daterange",
            locations=",".join(stations),
            parameters=",".join(self.parameters),
            startDate=as_date(start).isoformat(),
            endDate=as_date(end).isoformat(),
            app="funkyforelle",
        ).json()
        return self.parse(payload)

    def latest(self, stations) -> pd.DataFrame:
        payload = self._get(
            f"{API_BASE}/latest",
            locations=",".join(str(s) for s in stations),
            parameters=",".join(self.parameters),
            app="funkyforelle",
        ).json()
        return self.parse(payload)

    @classmethod
    def parse(cls, payload: dict) -> pd.DataFrame:
        """Convert an existenz.ch JSON payload into the tidy schema."""
        rows = []
        for item in payload.get("payload", []):
            par = item.get("par")
            if par not in PARAMS or item.get("val") is None:
                continue
            var, factor = PARAMS[par]
            rows.append(
                {
                    "timestamp": dt.datetime.fromtimestamp(item["timestamp"], tz=dt.UTC),
                    "station": str(item["loc"]),
                    "variable": var,
                    "value": float(item["val"]) * factor,
                    "unit": schema.VARIABLES[var],
                    "source": cls.name,
                }
            )
        if not rows:
            return schema.empty()
        return schema.validate(pd.DataFrame(rows))
