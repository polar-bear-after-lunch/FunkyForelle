"""FunkyForelle — open dissolved-oxygen (DO) early warning for Swiss rivers.

Package layout
--------------
connectors/  one module per data source; every connector returns the tidy schema in ``schema.py``
features.py  physics and calendar features (e.g. DO saturation from water temperature)
models/      baselines and the seq2seq LSTM carried over from AareML
metrics.py   RMSE, MAE, NSE, KGE
tracking.py  run records for the experiment log
cli.py       ``funkyforelle fetch`` / ``funkyforelle train``
"""

__version__ = "0.1.0"
