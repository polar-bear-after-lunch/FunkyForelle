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
