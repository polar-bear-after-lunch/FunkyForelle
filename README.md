<p align="center">
  <img src="assets/funkyforelle-fish.svg" alt="FunkyForelle fish" width="100%">
</p>

# FunkyForelle

An open-source early warning system predicting dissolved oxygen in Swiss rivers and lakes — protecting fish before heatwaves strike.

## Code

The `funkyforelle` Python package connects public Swiss river and weather data to DO forecasting models:

- **Data connectors**: CAMELS-CH-Chem (historical river chemistry), FOEN hydrology (real-time) and MeteoSwiss weather (open data). They all share one tidy schema.
- **Models**: persistence and climatology baselines, plus a seq2seq LSTM carried over from [AareML](https://github.com/polar-bear-after-lunch/AareML).
- **Experiment log**: every run records its data, configuration, git commit, metrics and the AI assistant used.

```bash
pip install -e ".[ml,dev]"
funkyforelle train configs/rhein_diepoldsau_2473.toml --ai-assistant "none"
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [docs/data-sources.md](docs/data-sources.md) and [docs/experiments.md](docs/experiments.md).

## We are looking for collaborators

A big thank you to all the new friends we met at the [AI + Environment Summit 2026](https://ai-environment-summit.com/) at Innovation Park Zurich! We are growing the FunkyForelle community and would love to hear from:

- **AI experts** — time-series forecasting, hydrological ML, MLOps
- **Scientists** — freshwater ecology, hydrology, water chemistry, sensor calibration
- **Funding angels** — people and organisations who want to back open environmental technology
- **Data providers** — agencies sharing river monitoring data
- **Anyone curious** — everyone interested in protecting fish and rivers

Get in touch: **polar.bear.after.lunch [at] gmail [dot] com**

## Funding status

We have applied to two funding programmes and are waiting for both decisions:

- [UniBE Digital Venture Fellowship](https://www.unibe.ch/research/funding_and_prizes/unibe_funding_schemes/digital_venture_fellowship/index_eng.html) — University of Bern
- [Prototype Fund Switzerland](https://prototypefund.opendata.ch/en/) — Opendata.ch

## Internship Positions

If the funding is successful, we will offer **two paid internships** (November 2026 – February 2027):

- **Machine Learning Engineer** — build the universal DO prediction model
- **IoT & Sensor Systems Engineer** — design and deploy the field sensor station

See the [Internship](./Internship/) folder for full specifications.

## Contact

polar.bear.after.lunch [at] gmail [dot] com

## Licence

Code is released under the [MIT Licence](LICENSE). Data from FOEN, MeteoSwiss and CAMELS-CH-Chem stay under their providers' terms; see [docs/data-sources.md](docs/data-sources.md).
