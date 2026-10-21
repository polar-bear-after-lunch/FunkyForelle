"""Offline connector tests using recorded / synthetic responses.

Live smoke tests are marked ``live`` and skipped by default:
``pytest -m live``.
"""
from pathlib import Path

import pandas as pd
import pytest

from funkyforelle import schema
from funkyforelle.connectors import CamelsChChem, FoenHydro, MeteoSwiss
from funkyforelle.connectors.meteoswiss import BASE

FIX = Path(__file__).parent / "fixtures"


def test_foen_parse_payload():
    payload = {
        "payload": [
            {"timestamp": 1760313600, "loc": "2473", "par": "temperature", "val": 12.4},
            {"timestamp": 1760313600, "loc": "2473", "par": "flow", "val": 210.0},
            {"timestamp": 1760314200, "loc": "2473", "par": "temperature", "val": 12.6},
            {"timestamp": 1760314200, "loc": "2473", "par": "unknown", "val": 1.0},
            {"timestamp": 1760314200, "loc": "2473", "par": "flow", "val": None},
        ]
    }
    df = FoenHydro.parse(payload)
    assert list(df.columns) == schema.COLUMNS
    assert set(df["variable"]) == {"water_temp", "discharge"}
    assert len(df) == 3
    assert str(df["timestamp"].dt.tz) == "UTC"
    wide = schema.to_daily_wide(df)
    assert wide.loc["2025-10-13", "water_temp"] == pytest.approx(12.5)


def test_foen_rejects_unknown_parameter():
    with pytest.raises(ValueError):
        FoenHydro(parameters=["oxygen"])


def test_meteoswiss_parse_real_sample():
    text = (FIX / "meteoswiss_stg_d_sample.csv").read_text(encoding="latin-1")
    df = MeteoSwiss(granularity="d").parse_csv(text)
    assert set(df["variable"]) >= {"air_temp", "precip", "global_radiation"}
    assert (df["station"] == "STG").all()
    first = df[(df["variable"] == "air_temp")].iloc[0]
    assert first["timestamp"] == pd.Timestamp("2026-01-01", tz="UTC")
    assert first["value"] == pytest.approx(-2.5)


def test_meteoswiss_file_selection():
    import datetime as dt

    m = MeteoSwiss(granularity="d")
    today = dt.date(2026, 10, 14)
    urls = m.files_for("ARH", dt.date(2015, 1, 1), dt.date(2026, 6, 1), today=today)
    assert urls == [f"{BASE}/arh/ogd-smn_arh_d_historical.csv", f"{BASE}/arh/ogd-smn_arh_d_recent.csv"]
    assert m.files_for("ARH", dt.date(2015, 1, 1), dt.date(2016, 1, 1), today=today) == [
        f"{BASE}/arh/ogd-smn_arh_d_historical.csv"]
    h = MeteoSwiss(granularity="h").files_for("arh", dt.date(2019, 6, 1), dt.date(2026, 10, 14), today)
    assert f"{BASE}/arh/ogd-smn_arh_h_historical_2010-2019.csv" in h
    assert f"{BASE}/arh/ogd-smn_arh_h_historical_2020-2029.csv" in h
    assert h[-1].endswith("_h_now.csv")


def test_camels_parse(tmp_path):
    d = tmp_path / "stream_water_chemistry/timeseries/daily"
    d.mkdir(parents=True)
    pd.DataFrame({
        "date": pd.date_range("2015-07-01", periods=5),
        "temp_sensor": [18.0, 18.5, None, 19.5, 20.0],
        "O2C_sensor": [9.1, 9.0, 8.9, 8.7, 8.6],
        "pH_sensor": 8.1,
        "ec_sensor": 320.0,
        "other_col": 1,
    }).to_csv(d / "camels_ch_chem_daily_2473.csv", index=False)
    c = CamelsChChem(root=tmp_path)
    assert c.available_gauges() == ["2473"]
    df = c.fetch(["2473"], "2015-07-02", "2015-07-04")
    assert set(df["variable"]) == {"water_temp", "do_mgl", "ph", "conductivity"}
    assert df["timestamp"].min() == pd.Timestamp("2015-07-02", tz="UTC")
    assert df["timestamp"].max() == pd.Timestamp("2015-07-04", tz="UTC")
    with pytest.raises(FileNotFoundError):
        c.fetch(["9999"])


def test_schema_rejects_unknown_variable():
    df = pd.DataFrame([{"timestamp": "2026-01-01", "station": "x", "variable": "foo",
                        "value": 1, "unit": "", "source": "t"}])
    with pytest.raises(ValueError):
        schema.validate(df)


@pytest.mark.live
def test_meteoswiss_live():
    import datetime as dt

    end = dt.date.today() - dt.timedelta(days=2)
    df = MeteoSwiss().fetch(["ARH"], end - dt.timedelta(days=10), end)
    assert not df.empty


@pytest.mark.live
def test_foen_live():
    df = FoenHydro().latest(["2473"])
    assert not df.empty
    assert set(df["variable"]) <= {"water_temp", "discharge"}
    assert (df["station"] == "2473").all()


@pytest.mark.live
def test_foen_live_daterange():
    import datetime as dt

    end = dt.date.today()
    df = FoenHydro().fetch(["2473"], end - dt.timedelta(days=2), end)
    assert not df.empty
    wide = schema.to_daily_wide(df)
    assert {"water_temp", "discharge"} <= set(wide.columns)
    # plausibility: Rhine water temperature and discharge
    assert wide["water_temp"].dropna().between(0, 30).all()
    assert (wide["discharge"].dropna() > 0).all()
