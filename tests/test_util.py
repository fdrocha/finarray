"""Ingestion helpers and assertions."""

import datetime as dt

import pandas as pd
import pytest
import xarray as xr

import finarray as fr
from finarray import util


def test_dcast(dates):
    assert util.dcast("2024-11-27") == dt.date(2024, 11, 27)
    assert util.dcast(dt.date(2024, 11, 27)) == dt.date(2024, 11, 27)
    assert util.dcast(["2024-11-27", "2024-12-02"]) == dates[:2]


def test_check():
    util.check(True, "fine")
    with pytest.raises(ValueError, match="not fine"):
        util.check(False, "not fine")


def test_error_raises_rather_than_exiting():
    with pytest.raises(fr.FinArrayError, match="bad"):
        util.error("bad")


def test_dim_assertions():
    da = xr.DataArray(
        [[1.0, 2.0]], dims=["time", "ticker"], coords={"time": [0], "ticker": ["A", "B"]}
    )
    util.check_exact_dim_names(da, "time", "ticker")
    util.check_first_dims(da, "time")
    util.check_has_dims(da, "ticker")
    with pytest.raises(ValueError):
        util.check_exact_dim_names(da, "ticker")
    with pytest.raises(ValueError):
        util.check_has_dims(da, "nope")


def test_check_has_vars():
    ds = xr.Dataset({"a": ("x", [1]), "b": ("x", [2])})
    util.check_has_vars(ds, "a", "b")
    with pytest.raises(ValueError, match="missing variables"):
        util.check_has_vars(ds, "c")


def test_intersect():
    assert util.intersect(["a", "b", "c"], ["c", "a"]) == ["a", "c"]


def test_level_time_to_date():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-11-27 15:54:00"]), ["AAA"]], names=["time", "ticker"]
    )
    s = pd.Series([1.0], index=idx)
    util.level_time_to_date(s)
    assert s.index.names == ["date", "ticker"]
    assert s.index.get_level_values("date")[0] == dt.date(2024, 11, 27)


def _write_csv(path, date):
    rows = []
    for t in ("15:50:00", "15:50:01"):
        for ticker in ("AAA", "BBB"):
            rows.append({"time": t, "ticker": ticker, "px": 1.5, "sz": 100})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_csv(tmp_path):
    csv = tmp_path / "2024-11-27.csv"
    _write_csv(csv, "2024-11-27")
    ds = util.load_csv(str(csv), date="2024-11-27")
    assert set(ds.dims) == {"time", "ticker"}
    assert ds["px"].dtype == "float32"
    assert ds["sz"].dtype == "int32"
    assert ds.attrs["date"] == dt.date(2024, 11, 27)


def test_load_csv_rename_func(tmp_path):
    csv = tmp_path / "2024-11-27.csv"
    _write_csv(csv, "2024-11-27")
    ds = util.load_csv(str(csv), date="2024-11-27", rename_func=lambda c: f"x_{c}")
    assert "x_px" in ds.data_vars


def test_load_csv_rejects_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="time_col"):
        util.load_csv(str(csv))


def test_csv_to_bars_roundtrip(tmp_path):
    """load_csv -> to_frdir -> BarsSet is the ingestion path the frcsv CLI uses."""
    csv = tmp_path / "2024-11-27.csv"
    _write_csv(csv, "2024-11-27")
    base = tmp_path / "base"
    base.mkdir()

    ds = util.load_csv(str(csv), date="2024-11-27")
    util.to_frdir(ds, str(base))

    bars = fr.BarsSet(str(base), restrict_dates=["2024-11-27"])
    bd = bars["2024-11-27"]
    assert sorted(bd.get_vars_available()) == ["px", "sz"]
    assert bd.get_var("px").dims == ("time", "ticker")
    assert float(bd.get_var("px").max()) == pytest.approx(1.5)


def test_to_frdir_refuses_to_overwrite(tmp_path):
    csv = tmp_path / "2024-11-27.csv"
    _write_csv(csv, "2024-11-27")
    base = tmp_path / "base"
    base.mkdir()
    ds = util.load_csv(str(csv), date="2024-11-27")
    util.to_frdir(ds, str(base))
    with pytest.raises(FileExistsError):
        util.to_frdir(ds, str(base))


def test_to_frdir_needs_a_date(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    ds = xr.Dataset(
        {"px": (("time", "ticker"), [[1.0]])},
        coords={"time": pd.to_datetime(["2024-11-27 15:50:00"]), "ticker": ["AAA"]},
    )
    with pytest.raises(ValueError, match="date not provided"):
        util.to_frdir(ds, str(base))
