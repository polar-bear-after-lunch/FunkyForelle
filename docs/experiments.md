# Experiments

Summary of notable runs. Full records: `runs/<run_id>/run.json` and `runs/index.csv`.

## Experiment 1 — Rhein–Diepoldsau (gauge 2473): does weather help?

**Question.** Starting from the AareML single-site setup, do MeteoSwiss weather
inputs improve DO and water-temperature forecasts?

**Setup.** CAMELS-CH-Chem daily data, 1995–2020. Time-based split: train ≤ 2014,
validation 2015–2016, test 2017–2020 (1,304 test windows). The model is a seq2seq
LSTM (64 hidden units, 2 layers), with a 21-day lookback and a 14-day horizon.
Both variants include a new physics feature, DO saturation from water
temperature (Garcia & Gordon 1992).

- **1a:** river inputs only (DO, water temp, pH, conductivity, DO saturation, day of year).
- **1b:** 1a plus MeteoSwiss Vaduz (air temp mean and max, precipitation, global radiation).

Three seeds each (42, 1, 2). Code: `configs/rhein_diepoldsau_2473*.toml` at commit `a106813` (clean tree).
Run on 2026-10-14 (CPU), with Perplexity Computer · Claude Opus 5.5. Run IDs: `20261014-090945` … `20261014-091559`.

| Model | DO RMSE (mg/L), mean over 14 days | DO RMSE day 1 | DO RMSE day 14 | DO KGE | Temp RMSE (°C) | Temp RMSE day 1 |
|---|---|---|---|---|---|---|
| Persistence | 0.343 | — | — | — | 1.368 | — |
| Climatology | 0.335 | — | — | — | 1.371 | — |
| 1a LSTM, no weather | 0.285 ± 0.005 | 0.162 | 0.326 | 0.953 | 1.142 ± 0.035 | 0.618 |
| 1b LSTM + Vaduz weather | 0.280 ± 0.003 | **0.146** | 0.325 | 0.919 | 1.145 ± 0.012 | **0.576** |

(± = standard deviation over 3 seeds.)

**Findings**
- Both LSTM variants beat persistence (−17% DO RMSE) and climatology on the
  same test windows.
- Weather inputs reduce **next-day** error: about −10% for DO and −7% for
  temperature. Next-day accuracy is what the fisheries' 24-hour rescue
  decision depends on.
- Averaged over 14 days, the weather gain (0.005 mg/L) is within seed noise,
  and KGE drops (0.953 → 0.919). This is no clear improvement yet.
- The test period contains **no DO values below 6 mg/L** at this large river,
  so alert skill can't be evaluated here. This is why the small-stream field
  sites matter.
- The results are not directly comparable with the AareML report (0.303 mg/L),
  because the period, feature set and tuning differ.

**Next**
- Use a 1–3 day horizon (the decision window) and weight the loss towards day 1.
- Use weather *forecasts* rather than observations as future inputs (ICON-CH /
  local forecasts), so the setup is operationally honest.
- Run a multi-gauge evaluation across all CAMELS-CH-Chem DO gauges, and add
  discharge from FOEN.
- Test significance (paired, block bootstrap) before claiming any gain.

**Discarded run.** An earlier attempt trained on 2010–2014 only (5 years). It
underperformed persistence (DO RMSE 0.392), so the training start was moved to
1995. This run isn't kept in `runs/`.

## Experiment 2 — 1–3 day forecasts at Rhein–Diepoldsau (gauge 2473)

**Question.** For the 24–72 hour window that fish-rescue decisions depend on:
1. Does weighting the loss towards day 1 help?
2. Does past weather help?
3. How much could weather *forecasts* add?

**Setup.** The same data, split and gauge as experiment 1, with a 21-day
lookback and a 3-day horizon. All four variants are evaluated on the **same
1,336 test windows** (2017–2020), with three seeds each (42, 1, 2). Code: `configs/exp2*.toml` at commit
`0a88a50` (clean tree). Run on 2026-10-21 (CPU) with Perplexity Computer ·
Claude Opus 5.5. Run IDs: `20261021-084917` … `20261021-085842`
(see `runs/index.csv`).

| Variant | Inputs | Loss |
|---|---|---|
| 2a | river (DO, temperature, pH, conductivity, DO saturation, day of year) | uniform |
| 2b | river | day-weighted 0.5 / 0.3 / 0.2 |
| 2c | river + past weather (MeteoSwiss Vaduz) | day-weighted |
| 2d | 2c + weather on the 3 forecast days (**observed weather as a perfect forecast**) | day-weighted |

**DO (mg/L)**: RMSE, mean ± standard deviation over 3 seeds

| Model | 3-day mean | Day 1 | Day 3 | KGE |
|---|---|---|---|---|
| Persistence | 0.230 | 0.159 | 0.286 | — |
| Climatology | 0.340 | — | — | — |
| 2a river, uniform | 0.202 ± 0.001 | 0.144 | 0.247 | 0.967 |
| 2b river, day-weighted | 0.200 ± 0.001 | 0.143 | 0.246 | 0.961 |
| 2c + past weather | **0.184 ± 0.002** | **0.125** | **0.233** | 0.960 |
| 2d + perfect weather forecast (upper bound) | 0.146 ± 0.002 | 0.118 | 0.167 | 0.986 |

**Water temperature (°C)**: RMSE

| Model | 3-day mean | Day 1 | Day 3 |
|---|---|---|---|
| Persistence | 0.868 | 0.573 | 1.114 |
| 2a | 0.800 | 0.553 | 0.998 |
| 2b | 0.794 | 0.538 | 1.000 |
| 2c | **0.736** | **0.477** | **0.956** |
| 2d (upper bound) | 0.443 | 0.374 | 0.494 |

**Paired significance.** We used a moving-block bootstrap (30-day blocks) on
seed-averaged squared errors, run with `scripts/compare_runs.py`. "Significant"
means the 95% interval of the error difference excludes zero.

| Comparison | DO day 1 | DO day 3 | Temp day 1 | Temp day 3 |
|---|---|---|---|---|
| 2b vs 2a (day-weighted loss) | −1.1%, significant | −0.3%, not significant | −2.7%, significant | +0.1%, not significant |
| 2c vs 2b (past weather) | −12.1%, significant | −5.3%, significant | −11.4%, significant | −4.4%, significant |
| 2d vs 2c (perfect forecast) | −6.1%, significant | −28.4%, significant | −21.5%, significant | −48.3%, significant |

**Findings**
- With a 3-day horizon, past weather now gives a **clear, significant gain**
  at every horizon day (DO day 1: −12%). In experiment 1 (14-day horizon) the
  gain was within seed noise. The best operational model (2c) beats
  persistence on next-day DO by 21% (0.125 vs 0.159 mg/L).
- The day-weighted loss helps only slightly: −1% for DO and −3% for
  temperature on day 1. It's cheap, so we keep it.
- **Weather forecasts are the biggest opportunity.** With perfect knowledge
  of the next 3 days' weather, day-3 DO error falls another 28% and day-3
  temperature error halves. Real forecasts will land somewhere between 2c and
  2d. Integrating MeteoSwiss forecasts (ICON-CH) is therefore the top
  data-source priority.
- 2d is an upper bound, not an operational result: it uses *observed*
  weather for the forecast days.
- As in experiment 1, the test period has **no DO below 6 mg/L** at this
  river, so alert skill (hit rate, false alarms) can't be evaluated yet.

**Next**
- Real forecast inputs: an archive of MeteoSwiss ICON-CH (or ECMWF open data)
  forecasts for the forecast days, then re-run 2d with forecasts instead of
  observations.
- Use gauges with low-DO events (smaller rivers in CAMELS-CH-Chem), and later
  the field sites, so alert skill can be measured.
