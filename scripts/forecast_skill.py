"""Compare archived weather forecasts against MeteoSwiss station observations.

Uses the Open-Meteo Previous Runs API: ``<var>_previous_dayN`` is the value
forecast N days before the valid time. Hourly values are aggregated to the
daily inputs the model uses, and compared with SwissMetNet observations.

    python scripts/forecast_skill.py --station VAD --lat 47.1275 --lon 9.5176 \
        --start 2025-07-06 --end 2026-10-19 -o docs/forecast-skill-vaduz.csv

Open-Meteo data: CC BY 4.0. The free API is for non-commercial use only.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import requests

from funkyforelle.connectors import MeteoSwiss
from funkyforelle.schema import to_daily_wide

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODELS = ["meteoswiss_icon_ch2", "icon_seamless", "ecmwf_ifs025", "best_match"]
HOURLY = ["temperature_2m", "precipitation", "shortwave_radiation"]


def daily_forecast(model, lat, lon, start, end, lead):
    hv = ",".join(f"{v}_previous_day{lead}" for v in HOURLY)
    r = requests.get(API, params=dict(latitude=lat, longitude=lon, hourly=hv, models=model,
                                      start_date=start, end_date=end, timezone="UTC"), timeout=120)
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    h = h.set_index("time")
    t = h[f"temperature_2m_previous_day{lead}"]
    d = pd.DataFrame({
        "air_temp": t.resample("D").mean(),
        "air_temp_max": t.resample("D").max(),
        "precip": h[f"precipitation_previous_day{lead}"].resample("D").sum(min_count=20),
        "global_radiation": h[f"shortwave_radiation_previous_day{lead}"].resample("D").mean(),
    })
    d.loc[t.resample("D").count() < 20] = np.nan
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="VAD")
    ap.add_argument("--lat", type=float, default=47.1275)
    ap.add_argument("--lon", type=float, default=9.5176)
    ap.add_argument("--start", default="2025-07-06")
    ap.add_argument("--end", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    obs = to_daily_wide(MeteoSwiss().fetch([a.station], a.start, a.end))
    rows = []
    for m in a.models:
        for lead in (1, 2, 3):
            j = daily_forecast(m, a.lat, a.lon, a.start, a.end, lead).join(obs, rsuffix="_obs").dropna()
            row = {"model": m, "lead_day": lead, "n_days": len(j)}
            for v in ["air_temp", "air_temp_max", "precip", "global_radiation"]:
                e = j[v] - j[f"{v}_obs"]
                row[f"{v}_RMSE"] = round(float(np.sqrt((e**2).mean())), 2)
                row[f"{v}_bias"] = round(float(e.mean()), 2)
            rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
