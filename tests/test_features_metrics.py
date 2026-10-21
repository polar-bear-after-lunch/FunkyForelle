import numpy as np
import pandas as pd
import pytest

from funkyforelle import features as F
from funkyforelle.metrics import evaluate, kge, nse, threshold_skill
from funkyforelle.models import Climatology, Persistence


@pytest.mark.parametrize("t,expected", [(0, 14.62), (10, 11.29), (20, 9.09), (30, 7.56)])
def test_do_saturation_matches_usgs_table(t, expected):
    # USGS DOTABLES, 760 mm Hg, salinity 0
    assert F.do_saturation_mgl(t) == pytest.approx(expected, abs=0.03)


def test_pressure_correction_lowers_saturation():
    p = F.pressure_from_elevation(1000)
    assert 890 < p < 905
    assert F.do_saturation_mgl(15, p) < F.do_saturation_mgl(15)


def test_make_windows_drops_nan_windows():
    idx = pd.date_range("2020-01-01", periods=30)
    df = pd.DataFrame({"a": np.arange(30, dtype=float), "b": np.arange(30, dtype=float)}, idx)
    df.iloc[10, 0] = np.nan
    X, y, o = F.make_windows(df, ["a", "b"], ["a"], lookback=5, horizon=3)
    assert X.shape[1:] == (5, 2) and y.shape[1:] == (3, 1)
    assert not np.isnan(X).any() and not np.isnan(y).any()
    # target equals the next values after the window
    assert y[0, 0, 0] == X[0, -1, 0] + 1
    assert o[0] == idx[4]


def test_make_windows_future_features_are_forecast_days():
    idx = pd.date_range("2020-01-01", periods=20)
    df = pd.DataFrame({"a": np.arange(20.0), "w": 100 + np.arange(20.0)}, idx)
    X, y, o, Xf = F.make_windows(df, ["a"], ["a"], lookback=4, horizon=3, future_features=["w"])
    assert Xf.shape == (len(X), 3, 1)
    # future covariates are the values on the same days as the targets
    assert (Xf[:, :, 0] - 100 == y[:, :, 0]).all()
    df.iloc[6, 1] = np.nan  # a gap in the future covariate drops affected windows
    X2, *_ = F.make_windows(df, ["a"], ["a"], 4, 3, ["w"])
    assert len(X2) == len(X) - 3


def test_fill_short_gaps_respects_limit():
    s = pd.Series([1.0, np.nan, np.nan, 4.0] + [np.nan] * 10 + [5.0],
                  index=pd.date_range("2020-01-01", periods=15))
    out = F.fill_short_gaps(s.to_frame("x"), max_gap=3)["x"]
    assert out.iloc[1] == pytest.approx(2.0)
    assert out.iloc[5:13].isna().any()


def test_metrics_perfect_and_mean():
    y = np.random.default_rng(0).normal(size=(50, 4, 2))
    m = evaluate(y, y, ["a", "b"])
    assert m["a"]["RMSE"] == 0 and m["a"]["NSE"] == 1 and m["a"]["KGE"] == pytest.approx(1)
    assert nse(y[..., 0], np.full_like(y[..., 0], y[..., 0].mean())) == pytest.approx(0)
    assert kge(y[..., 0], y[..., 0]) == pytest.approx(1)


def test_threshold_skill():
    obs = np.array([5, 7, 5.5, 8])
    sim = np.array([5.5, 5, 7, 8])
    s = threshold_skill(obs, sim, 6)
    assert s["events"] == 2 and s["hit_rate"] == 0.5 and s["false_alarm_ratio"] == 0.5


def test_baselines():
    X = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    p = Persistence().predict(X, [1], horizon=4)
    assert p.shape == (2, 4, 1) and (p[0, :, 0] == X[0, -1, 1]).all()

    idx = pd.date_range("2018-01-01", "2019-12-31")
    train = pd.DataFrame({"t": 10 + 5 * np.sin(2 * np.pi * idx.dayofyear / 365.25)}, idx)
    c = Climatology().fit(train, ["t"]).predict(pd.DatetimeIndex(["2020-06-01"]), 3)
    assert c.shape == (1, 3, 1) and np.isfinite(c).all()
