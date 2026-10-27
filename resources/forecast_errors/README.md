# Forecast-error pools

`icon_ch2_VAD.csv` contains MeteoSwiss ICON-CH2 forecast errors at MeteoSwiss
Vaduz (VAD), one row per forecast origin from 2025-07-05 to 2026-10-22.
Each `<variable>_d<L>` column is the forecast minus the observation for the
day origin + L, issued L days earlier. Daily definitions match SwissMetNet:
precipitation is summed from 06 to 06 UTC; temperature and radiation use 00–24 UTC.

Rebuild with:

```bash
python scripts/build_error_pool.py --station VAD --end <date> -o resources/forecast_errors/icon_ch2_VAD.csv
```

Sources: forecasts from [Open-Meteo](https://open-meteo.com) (CC BY 4.0; ICON-CH2
model by MeteoSwiss); observations from MeteoSwiss SwissMetNet open data
("Source: MeteoSwiss"). The errors are a derived product, shared under CC BY 4.0
with the same attribution.
