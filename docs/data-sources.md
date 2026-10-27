# Data sources

Status of every data source considered for FunkyForelle. **Integrated** means a
tested connector exists in `src/funkyforelle/connectors/`.

| Priority | Source | Operator | Variables | Frequency | Access | Status |
|---|---|---|---|---|---|---|
| P1 | [CAMELS-CH-Chem](https://zenodo.org/records/14980027) | Eawag / research (Nascimento et al. 2025) | DO, water temp, pH, conductivity (daily) | historical, 1981–2020 | open, CC BY 4.0 | **Integrated** (`camels_ch_chem`) |
| P1 | FOEN hydrology via [api.existenz.ch](https://api.existenz.ch) | BAFU / FOEN | discharge, water level, water temp (10 min) | real-time, recent window | open; FOEN terms of use | **Integrated** (`foen_hydro`), live check pending (see note) |
| P1 | [MeteoSwiss SwissMetNet OGD](https://data.geo.admin.ch/browser/#/collections/ch.meteoschweiz.ogd-smn) | MeteoSwiss | air temp (mean, max), precipitation, global radiation, sunshine | daily / hourly, historical + real-time | open, attribution | **Integrated** (`meteoswiss`), live-tested |
| P1 | AWEL hydrometry & water monitoring | Canton Zürich (AWEL) | water temp, discharge, DO, pH, conductivity | real-time (some stations) | agreement needed | Access to request |
| P1 | FunkyForelle field sensors | FunkyForelle / Fischereiverein See + Gaster | DO, water temp | real-time | own data | Planned (5 sites) |
| **P1 (top)** | **MeteoSwiss ICON-CH2 forecasts via [Open-Meteo](https://open-meteo.com/en/docs/previous-runs-api)** | MeteoSwiss model, archived by Open-Meteo | forecast air temp, precipitation, radiation, 1–5 days | live + archive from 2025-07-06 | free for non-commercial use, CC BY 4.0 | **Integrated** (`openmeteo_forecast`, 2026-10-27); used in experiment 3 |
| P2 | MeteoSwiss OGD ICON-CH1/CH2-EPS (GRIB) | MeteoSwiss | full model fields, 11/21 ensemble members | live, kept only 24 h | open, attribution | Self-archive daily from now on, as the long-term official source |
| P2 | MeteoSwiss local forecasts (point, post-processed) | MeteoSwiss | temp incl. 10/90% quantiles, precipitation, radiation, 9 days | hourly updates; past forecasts not documented | open, attribution | Self-archive and compare with ICON-CH2 after one season |
| P3 | ECMWF reforecasts (MARS archive) | ECMWF | 20-year ensemble hindcasts | archive | CC BY 4.0; research fee waiver needed for access | Backup if long forecast training data are needed |
| — | ECMWF IFS 0.25° open data, DWD ICON | ECMWF / DWD | forecasts | live + Open-Meteo archive from 2024 | open | Rejected for the Alpine valleys: about 2 °C temperature error from day 2 |
| P2 | FOEN river temperature network | BAFU / FOEN | water temp | real-time | open | Candidate |
| P2 | NAWA TREND / NAWA FRACHT | BAFU + cantons | nutrients, DO, conductivity, pH | periodic samples | on request | Candidate |
| P2 | Eawag Urban Water Observatory | Eawag | DO, water temp, conductivity | high-frequency | on request | Candidate |
| P2 | CAMELS-CH (base) | Eawag / research | catchment attributes, discharge, meteo | historical | open | Candidate (static features, EA-LSTM) |

## Notes per source

### CAMELS-CH-Chem
- Daily files per gauge: `stream_water_chemistry/timeseries/daily/camels_ch_chem_daily_<id>.csv`
  with columns `temp_sensor, pH_sensor, ec_sensor, O2C_sensor`.
- Gauge 2473 = Rhein – Diepoldsau, Rietbrücke (station moved from Schmitter
  to Diepoldsau in 1984).
- Cite: Nascimento, T. et al. (2025), doi:10.5281/zenodo.14980027.

### FOEN hydrology (existenz.ch)
- Unofficial JSON wrapper around the FOEN real-time feed; the same path as
  `AareML/aareml_live_demo.py`.
- Bot protection blocks many cloud and CI IP ranges, so the live test
  (`pytest -m live`) needs a normal network; see [foen-live-test.md](foen-live-test.md).
  The parser is tested offline.
- Serves a recent window, not long histories. Use CAMELS-CH-Chem for training.
- Attribution: "Source: FOEN".

### MeteoSwiss OGD
- Files: `ogd-smn_<abbr>_<d|h|t>_<historical|recent|now>.csv`, semicolon
  separated, Latin-1 encoded, UTC timestamps `dd.mm.yyyy HH:MM`.
- Daily precipitation (`rre150d0`) is the 06–06 UTC total.
- Coverage varies by station. Altenrhein (ARH) only has precipitation and
  radiation from 2014, so experiment 1 uses Vaduz (VAD), which is complete
  back to 1995.
- Weather stations for the Fischereiverein See + Gaster waters are still to be
  chosen once the field sites are fixed.

## Weather forecasts — choice and plan

**Decision (2026-10-21):** use **MeteoSwiss ICON-CH2** forecasts as the
future-weather input. Read them as point time series through the Open-Meteo
APIs: the Forecast API for live use, and the Previous Runs API for archived
forecasts at fixed lead times of 1–3 days.

**Why.** In experiment 2, a perfect 3-day weather forecast cut day-3 DO error
by 28% and roughly halved temperature error, so forecast quality matters
directly. We compared archived forecasts with observations at MeteoSwiss
Vaduz (VAD) from 2025-07-06 to 2026-10-19 (468 days, `scripts/forecast_skill.py`,
results in [forecast-skill-vaduz.csv](forecast-skill-vaduz.csv)):

| Model (via Open-Meteo) | Daily mean temp RMSE, day 1 / 2 / 3 (°C) | Daily max temp RMSE, day 1 / 3 | Radiation RMSE, day 1 / 3 (W/m²) | Archive since |
|---|---|---|---|---|
| **MeteoSwiss ICON-CH2** | **1.22 / 1.35 / 1.50** | **1.72 / 2.08** | 30.5 / 37.1 | 2025-07-06 |
| DWD ICON seamless (= "best match" here) | 1.37 / 1.98 / 2.04 | 1.82 / 2.99 | 30.1 / 36.3 | 2024-01-22 |
| ECMWF IFS 0.25° | 2.04 / 2.13 / 2.20 | 3.04 / 3.08 | 34.9 / 41.7 | 2024-02-06 |

- ICON-CH2 (2 km) resolves the Alpine Rhine valley. The coarser global model
  runs about 1 °C too cold there, and DWD ICON loses accuracy after day 1,
  when its 2 km model hands over to a coarser one.
- Precipitation errors are similar across models (about 5 mm/day RMSE).
  Daily rain is hard to forecast for any model.
- MeteoSwiss's own open data only keeps ICON forecasts for **24 hours**
  ([MeteoSwiss open data docs](https://opendatadocs.meteoswiss.ch/e-forecast-data/e2-e3-numerical-weather-forecasting-model)),
  so an archive has to come from Open-Meteo or from our own daily downloads.
- The Open-Meteo archive starts mid-2025. That doesn't overlap with
  CAMELS-CH-Chem (ends 2020), so we can't train directly on archived forecasts.

**Status (2026-10-27):** steps 1 and 2 are done. Experiment 3 shows that
realistic forecasts keep about 86% of the perfect-forecast gain in day-3 DO
error ([experiments.md](experiments.md)). Steps 3 and 4 are open.

**Plan**
1. **Connector `openmeteo_forecast`** (done). Point forecasts per site and lead day,
   ICON-CH2 by default, aggregated to the daily schema (`air_temp`,
   `air_temp_max`, `precip`, `global_radiation`). Covers live forecasts plus
   the archive.
2. **Training: observed weather plus realistic forecast noise** (done: real error pool, `src/funkyforelle/forecast_noise.py`). Train on
   observed weather for 1995–2020 (as in 2d), but add noise with the measured
   ICON-CH2 error by lead day and variable (e.g. temperature: σ ≈ 1.2 / 1.35 /
   1.5 °C, bias about +0.3 °C). The model then learns not to over-trust the
   forecast.
3. **Test with real forecasts.** Evaluate on 2025-07 onwards with archived
   ICON-CH2 forecasts. For now this tests temperature only, at FOEN gauges;
   DO testing needs the FSG field sensors or recent DO data from AWEL/Eawag.
4. **Self-archive from now on.** Store MeteoSwiss local forecasts and
   ICON-CH2 values for each field site every day, so the project builds its own
   archive under MeteoSwiss's unrestricted open-data terms.

**Licences.** Open-Meteo's free API is for non-commercial use (public research
and non-profit apps qualify) under CC BY 4.0, with limits of under 10,000 calls
per day ([Open-Meteo terms](https://open-meteo.com/en/terms)). A paid or
commercial FunkyForelle service would need an Open-Meteo plan, or would have to
switch to the self-archived MeteoSwiss open data, which is free without
restriction (attribution "Source: MeteoSwiss").

## Adding a source

See [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-data-source).
