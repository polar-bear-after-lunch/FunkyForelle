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
| P1 (top) | MeteoSwiss local forecasts / ICON-CH | MeteoSwiss | forecast air temp, precipitation, radiation | forecasts | open | Candidate. Experiment 2: a perfect forecast cuts day-3 DO error by 28% |
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

## Adding a source

See [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-data-source).
