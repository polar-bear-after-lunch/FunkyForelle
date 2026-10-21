# FOEN live test

The FOEN hydrology connector uses `api.existenz.ch`. That service's bot
protection blocks many cloud and CI IP ranges, so the live check must be run
from a normal home or university network. It takes about 5 minutes.

## 1. Setup (once)

macOS / Linux, Python 3.11 or newer:

```bash
git clone https://github.com/polar-bear-after-lunch/FunkyForelle.git
cd FunkyForelle
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows, use `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

If you already have the repo, run `git pull` and then `pip install -e ".[dev]"`.

## 2. Quick check that the API is reachable

```bash
curl -s "https://api.existenz.ch/apiv1/hydro/latest?locations=2473&parameters=temperature,flow&app=funkyforelle"
```

If it works, you'll see JSON containing `"payload"` with temperature and flow
values. If the response is HTML ("415 Unsupported Media Type") or says "Access
denied by Imunify360 bot-protection", that network is blocked as well. Try
another network, e.g. a phone hotspot.

## 3. Run the live tests

```bash
pytest -m live -v
```

You should see:

```
tests/test_connectors.py::test_meteoswiss_live PASSED
tests/test_connectors.py::test_foen_live PASSED
tests/test_connectors.py::test_foen_live_daterange PASSED
```

`test_foen_live` checks the latest values for gauge 2473 (Rhein – Diepoldsau).
`test_foen_live_daterange` fetches the last 2 days and checks the temperature
and discharge values are plausible.

## 4. Look at the data (optional)

```bash
funkyforelle fetch foen_hydro 2473 --start 2026-10-19 --end 2026-10-21 -o foen_2473.csv
head foen_2473.csv
```

Replace the dates with the last 2–3 days.

## 5. Report back

Post the output of `pytest -m live -v` in a GitHub issue, or in the Work Log,
including any error text. If it passes, mark the FOEN data source as
**Integrated** on the status board.

| Symptom | Meaning |
|---|---|
| `HTTPError: 415` or "Imunify360" | Network blocked by bot protection; try another network |
| `test_foen_live` passes, `daterange` fails | The date-range endpoint changed; open an issue with the error |
| `ConnectionError` / timeout | Service down or no internet; try again later |
| `ModuleNotFoundError: funkyforelle` | Virtual environment not activated, or `pip install -e .` not run |
