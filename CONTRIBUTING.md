# Contributing to FunkyForelle

Thanks for helping protect fish and rivers. This guide covers how we work, how to
add a data source, and how to log experiments so results stay reproducible.

## How the work is organised

We work in four streams:

| Stream | Scope |
|---|---|
| Model & Data (Funky) | data connectors, features, models, evaluation |
| Funding & Dissemination | grants, conferences, posters, talks |
| Sensors (Forelle) | DO/temperature field stations at the Fischereiverein See + Gaster waters |
| Networking | researchers, agencies and practitioners |

The team's status board, contacts and grant drafts are in a private workspace.
Anything public — code, data documentation, results — lives in this repository.
If you're a new collaborator, ask a maintainer for access to the status board.

## Getting started

```bash
git clone https://github.com/polar-bear-after-lunch/FunkyForelle.git
cd FunkyForelle
python -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -e ".[ml,dev]"
pytest                    # offline tests, a few seconds
pytest -m live            # optional: hits the real APIs
```

Historical training data (CAMELS-CH-Chem) is not in git. Download it from
[Zenodo record 14980027](https://zenodo.org/records/14980027) and unzip it into
`data/camels-ch-chem/`.

## Workflow

1. Pick or open a GitHub issue and assign yourself.
2. Branch from `main`: `feat/<short-name>`, `fix/<short-name>` or `data/<source>`.
3. Keep pull requests small and focused; include tests.
4. `pytest` and `ruff check src tests` must pass.
5. At least one other person reviews before merging.

Commit messages: imperative and short, e.g. `connectors: add AWEL hydrometry`.

## Adding a data source

Every connector returns the same tidy table (`src/funkyforelle/schema.py`):
`timestamp, station, variable, value, unit, source`. Models never see
provider-specific formats.

1. Create `src/funkyforelle/connectors/<name>.py` subclassing `Connector`.
2. Map provider variables onto the canonical names in `schema.VARIABLES`. Add a
   new variable there only if none fits, and convert to the canonical unit.
3. Register it in `connectors/__init__.py`.
4. Add an offline test with a small recorded response in `tests/fixtures/`.
   Mark any test that calls the real API with `@pytest.mark.live`.
5. Document it in `docs/data-sources.md`: operator, licence, attribution, access
   and known gaps.
6. Never commit credentials or data under restricted licences. Read API keys
   from environment variables.

## Logging experiments

Every training run must be reproducible and logged:

```bash
funkyforelle train configs/<config>.toml --seed 0 \
    --ai-assistant "<assistant · model, or 'none'>" \
    --notes "what you changed and why"
```

This writes `runs/<run_id>/run.json` (config, features, data hash, git commit,
metrics, baselines) and appends a line to `runs/index.csv`.

- **Commit your code first.** A run from a dirty working tree is flagged
  `dirty` and can't be fully reproduced.
- **Always compare against the baselines.** Persistence and climatology are
  evaluated automatically on the same test windows. A model that doesn't beat
  persistence isn't an improvement.
- **Use at least 3 seeds** before claiming one setup beats another. Report the
  mean and spread.
- **Record the AI assistant.** If an AI assistant wrote or changed code,
  configs or analysis for a run, name the assistant and model in
  `--ai-assistant`. This keeps the work transparent for funders and reviewers.
- Copy the printed summary into the team's Experiments table and add a line
  to the Work Log.
- Commit `run.json`, `index.csv` and the small `test_*.npy` / `test_origins.csv`
  files. Model weights (`*.pt`) are git-ignored.
- To compare two setups, use the paired block-bootstrap test:
  `python scripts/compare_runs.py "<notes A>" "<notes B>" --target do_mgl`.
- Add notable results to `docs/experiments.md`, with negative results as well.

## Scientific conventions

- Gauge 2473 is **Rhein – Diepoldsau (Rietbrücke)**, not "Aare at Bern".
- Split by time (train / validation / test), never randomly. Fit scalers on
  the training period only.
- Report RMSE, NSE and KGE per target, plus RMSE at day 1 and the last horizon
  day. For DO alerts, also report hit rate and false-alarm ratio at the
  fish-stress threshold.
- Alerts are recommendations to fisheries experts, never automatic triggers.
  Show uncertainty wherever forecasts are displayed.

## Data licences and attribution

Respect each provider's terms (see `docs/data-sources.md`). Attribute FOEN,
MeteoSwiss and CAMELS-CH-Chem in any publication, poster or app that uses
their data.

## Contact

polar.bear.after.lunch [at] gmail [dot] com
