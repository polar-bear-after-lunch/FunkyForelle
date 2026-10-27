"""Build a real forecast-error pool: archived forecasts minus MeteoSwiss observations.

    python scripts/build_error_pool.py --station VAD --start 2025-07-06 --end 2026-10-25 \
        -o resources/forecast_errors/icon_ch2_VAD.csv

Data: Open-Meteo Previous Runs API (CC BY 4.0; ICON-CH2 model by MeteoSwiss)
and MeteoSwiss SwissMetNet observations (open data, "Source: MeteoSwiss").
"""
from __future__ import annotations

import argparse

from funkyforelle.connectors import MeteoSwiss, OpenMeteoForecast
from funkyforelle.forecast_noise import build_error_pool
from funkyforelle.schema import to_daily_wide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="VAD")
    ap.add_argument("--model", default="meteoswiss_icon_ch2")
    ap.add_argument("--start", default="2025-07-06")
    ap.add_argument("--end", required=True)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    fc = OpenMeteoForecast(model=a.model).fetch_leads(a.station, a.start, a.end,
                                                      range(1, a.horizon + 1))
    obs = to_daily_wide(MeteoSwiss().fetch([a.station], a.start, a.end))
    pool = build_error_pool(fc, obs, a.horizon)
    pool.round(3).to_csv(a.out)
    print(f"{len(pool)} forecast origins {pool.index.min().date()}..{pool.index.max().date()}")
    print("RMSE by column:")
    print(((pool**2).mean() ** 0.5).round(2).to_string())


if __name__ == "__main__":
    main()
