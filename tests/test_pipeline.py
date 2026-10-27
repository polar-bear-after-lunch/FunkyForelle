"""End-to-end pipeline on a small synthetic CAMELS-style gauge (no network)."""
import json

import numpy as np
import pandas as pd
import pytest

from funkyforelle.pipeline import run


@pytest.fixture
def synthetic_root(tmp_path):
    d = tmp_path / "camels/stream_water_chemistry/timeseries/daily"
    d.mkdir(parents=True)
    idx = pd.date_range("2010-01-01", "2013-12-31")
    rng = np.random.default_rng(1)
    temp = 11 + 7 * np.sin(2 * np.pi * (idx.dayofyear - 110) / 365.25) + rng.normal(0, 0.5, len(idx))
    do = 14.6 - 0.25 * temp + rng.normal(0, 0.2, len(idx))
    pd.DataFrame({"date": idx, "temp_sensor": temp, "O2C_sensor": do,
                  "pH_sensor": 8.1, "ec_sensor": 300.0}).to_csv(
        d / "camels_ch_chem_daily_9000.csv", index=False)
    return tmp_path / "camels"


def _cfg(root):
    return {
        "name": "synthetic", "start": "2010-01-01", "end": "2013-12-31",
        "train_end": "2011-12-31", "val_end": "2012-12-31",
        "lookback": 10, "horizon": 3, "max_gap": 7, "do_threshold_mgl": 10.5,
        "targets": ["do_mgl", "water_temp"],
        "features": ["do_mgl", "water_temp", "do_sat_mgl", "doy_sin", "doy_cos"],
        "sources": {"camels_ch_chem": {"stations": ["9000"], "root": str(root)}},
        "model": {"type": "seq2seq_lstm", "hidden": 8, "n_layers": 1, "epochs": 3,
                  "patience": 2, "batch_size": 128},
    }


def test_pipeline_persistence(synthetic_root, tmp_path):
    rec = run(_cfg(synthetic_root), model_override="persistence", ai_assistant="test",
              runs_dir=str(tmp_path / "runs"), verbose=False)
    saved = json.loads((tmp_path / "runs" / rec.run_id / "run.json").read_text())
    assert saved["model"] == "persistence" and saved["ai_assistant"] == "test"
    assert saved["metrics"]["do_mgl"]["RMSE"] > 0
    assert "threshold" in saved["metrics"]["do_mgl"]
    assert (tmp_path / "runs" / "index.csv").exists()


def test_pipeline_seq2seq(synthetic_root, tmp_path):
    pytest.importorskip("torch")
    rec = run(_cfg(synthetic_root), runs_dir=str(tmp_path / "runs"), verbose=False)
    assert rec.model == "seq2seq_lstm"
    assert np.isfinite(rec.metrics["water_temp"]["RMSE"])
    assert (tmp_path / "runs" / rec.run_id / "model.pt").exists()


def test_pipeline_future_features_and_step_weights(synthetic_root, tmp_path):
    pytest.importorskip("torch")
    cfg = _cfg(synthetic_root)
    cfg["future_features"] = ["doy_sin", "doy_cos"]
    cfg["model"]["step_weights"] = [0.5, 0.3, 0.2]
    rec = run(cfg, runs_dir=str(tmp_path / "runs"), verbose=False)
    assert rec.params["future_features"] == ["doy_sin", "doy_cos"]
    assert rec.params["step_weights"] == [0.5, 0.3, 0.2]
    assert np.isfinite(rec.metrics["do_mgl"]["RMSE_day1"])


def test_step_weights_length_checked():
    pytest.importorskip("torch")
    from funkyforelle.models.seq2seq import Seq2SeqForecaster
    with pytest.raises(ValueError):
        Seq2SeqForecaster(horizon=3, step_weights=[1, 1])


def test_pipeline_forecast_noise(synthetic_root, tmp_path):
    pytest.importorskip("torch")
    idx = pd.date_range("2024-01-01", periods=400).rename("origin")
    rng = np.random.default_rng(0)
    pool = pd.DataFrame({f"{v}_d{L}": rng.normal(0, 0.1, 400)
                         for L in (1, 2, 3) for v in ["doy_sin", "doy_cos"]}, idx)
    pool.to_csv(tmp_path / "pool.csv")
    cfg = _cfg(synthetic_root)
    cfg["future_features"] = ["doy_sin", "doy_cos"]
    cfg["forecast_noise"] = {"pool": str(tmp_path / "pool.csv"), "train": True, "test": True}
    rec = run(cfg, runs_dir=str(tmp_path / "runs"), verbose=False)
    fn = rec.params["forecast_noise"]
    assert fn["train"] and fn["test"] and fn["pool_sizes"]["train"] > 0
    assert np.isfinite(rec.metrics["do_mgl"]["RMSE"])
    cfg.pop("future_features")
    with pytest.raises(ValueError):
        run(cfg, runs_dir=str(tmp_path / "runs"), verbose=False)
