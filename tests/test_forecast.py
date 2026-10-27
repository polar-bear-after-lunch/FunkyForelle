"""Offline tests for the Open-Meteo forecast connector and forecast-error sampling."""
import numpy as np
import pandas as pd
import pytest

from funkyforelle.connectors import REGISTRY, OpenMeteoForecast
from funkyforelle.connectors.openmeteo_forecast import hourly_to_daily, resolve_site
from funkyforelle.forecast_noise import ForecastErrorSampler, build_error_pool


def _hourly_json(start="2025-08-01", days=3, suffix=""):
    t = pd.date_range(start, periods=24 * days, freq="h", tz="UTC")
    return {"hourly": {
        "time": t.strftime("%Y-%m-%dT%H:%M").tolist(),
        f"temperature_2m{suffix}": (10 + np.arange(len(t)) % 24 / 2).tolist(),
        f"precipitation{suffix}": [1.0 if 6 <= ts.hour < 12 and ts.day == 1 else 0.0 for ts in t],
        f"shortwave_radiation{suffix}": [100.0] * len(t),
    }}


def test_registry_and_sites():
    assert REGISTRY["openmeteo_forecast"] is OpenMeteoForecast
    assert resolve_site("VAD") == (47.1275, 9.5176)
    assert resolve_site("47.0,8.5") == (47.0, 8.5)
    with pytest.raises(KeyError):
        resolve_site("nowhere")
    with pytest.raises(ValueError):
        OpenMeteoForecast(lead_day=0)


def test_parse_and_daily_aggregation():
    h = OpenMeteoForecast.parse_hourly(_hourly_json(suffix="_previous_day2"), "_previous_day2")
    d = hourly_to_daily(h)
    assert list(d.columns) == ["air_temp", "air_temp_max", "global_radiation", "precip"]
    assert d.index[0] == pd.Timestamp("2025-08-01") and d.index.tz is None
    assert d["air_temp"].iloc[0] == pytest.approx(10 + 11.5 / 2)
    assert d["air_temp_max"].iloc[0] == pytest.approx(21.5)
    # 6 mm fell 06-12 UTC on day 1 -> belongs to day 1 (06-06 UTC window)
    assert d["precip"].iloc[0] == pytest.approx(6.0)
    # last day has no 06 UTC next-day hours -> incomplete precip window
    assert np.isnan(d["precip"].iloc[-1])


def test_tidy_schema():
    c = OpenMeteoForecast()
    d = hourly_to_daily(OpenMeteoForecast.parse_hourly(_hourly_json()))
    tidy = c._tidy(d, "VAD")
    assert set(tidy["variable"]) == {"air_temp", "air_temp_max", "precip", "global_radiation"}
    assert (tidy["source"] == "openmeteo_forecast").all()


def _pool(n=400, horizon=3):
    idx = pd.date_range("2025-07-01", periods=n).rename("origin")
    rng = np.random.default_rng(0)
    cols = {f"{v}_d{L}": rng.normal(L, 0.1, n)
            for L in range(1, horizon + 1) for v in ["air_temp", "precip"]}
    return pd.DataFrame(cols, idx)


def test_build_error_pool_alignment():
    obs = pd.DataFrame({v: 0.0 for v in ["air_temp", "air_temp_max", "precip", "global_radiation"]},
                       index=pd.date_range("2025-07-01", "2025-07-10"))
    rows = []
    for L in (1, 2):
        for day in pd.date_range("2025-07-01", "2025-07-10"):
            rows.append({"valid_date": day, "lead_day": L, "air_temp": 10 * L + day.day,
                         "air_temp_max": 0.0, "precip": 0.0, "global_radiation": 0.0})
    pool = build_error_pool(pd.DataFrame(rows), obs, horizon=2)
    # origin t: lead L error is the forecast valid at t+L
    assert pool.loc["2025-07-03", "air_temp_d1"] == 10 + 4
    assert pool.loc["2025-07-03", "air_temp_d2"] == 20 + 5


def test_sampler_split_season_and_clip():
    pool = _pool()
    tr = ForecastErrorSampler(pool, ["air_temp", "precip"], 3, split="train")
    te = ForecastErrorSampler(pool, ["air_temp", "precip"], 3, split="test")
    assert tr.n + te.n == len(pool) and min(tr.n, te.n) > 0
    rng = np.random.default_rng(1)
    origins = pd.date_range("2018-01-01", periods=50)
    e = tr.sample(origins, rng)
    assert e.shape == (50, 3, 2)
    assert np.allclose(e[:, :, 0].mean(0), [1, 2, 3], atol=0.1)  # lead structure kept
    Xf = np.zeros((50, 3, 2), np.float32)
    Xf[..., 1] = -10  # precip error can't push below zero
    out = tr.apply(Xf, origins, rng)
    assert (out[..., 1] >= 0).all()
    with pytest.raises(KeyError):
        ForecastErrorSampler(pool, ["sunshine"], 3)


def test_sampler_is_deterministic_for_seed():
    s = ForecastErrorSampler(_pool(), ["air_temp"], 3, split="test")
    o = pd.date_range("2019-06-01", periods=20)
    a = s.sample(o, np.random.default_rng(5))
    b = s.sample(o, np.random.default_rng(5))
    assert np.array_equal(a, b)
