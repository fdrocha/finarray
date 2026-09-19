"""Single-date behaviour: lazy loading, autoloading eval, and persistence."""

import datetime as dt
import os

import numpy as np
import pytest
import xarray as xr

import finarray as fr


def test_vars_are_discovered_but_not_loaded(bd):
    assert "mid" in bd.get_vars_available()
    assert bd.live_vars() == []


def test_load_var_is_idempotent(bd):
    bd.load_var("mid")
    bd.load_var("mid")
    assert bd.live_vars() == ["mid"]


def test_get_var_autoloads(bd):
    assert "mid" not in bd.live_vars()
    assert bd.get_var("mid").dims == ("time", "ticker")
    assert "mid" in bd.live_vars()


def test_eval_autoloads_named_variables(bd):
    """The signature trick: an expression pulls in the data it mentions."""
    assert bd.live_vars() == []
    result = bd.eval("(bid + ask) * 0.5")
    assert set(bd.live_vars()) >= {"bid", "ask"}
    xr.testing.assert_allclose(result, bd.get_var("mid"), atol=1e-5)


def test_eval_raises_for_genuinely_unknown_names(bd):
    with pytest.raises((NameError, KeyError, ValueError)):
        bd.eval("no_such_variable * 2")


def test_attribute_access_forwards_to_dataset(bd):
    bd.load_var("mid")
    assert bd.mid.dims == ("time", "ticker")
    assert list(bd.ticker.values) == ["AAA", "BBB", "CCC", "SPY"]


def test_unknown_attribute_raises(bd):
    with pytest.raises(AttributeError):
        _ = bd.definitely_not_a_thing


def test_get_value_requires_a_scalar(bd):
    scalar = bd.sel_ticker("SPY").sel_time("15:54:00")
    assert isinstance(scalar.get_value("mid"), float)
    with pytest.raises(ValueError):
        bd.get_value("mid")


def test_tcast_promotes_times_using_the_bars_date(bd):
    assert bd.tcast("15:54:00") == dt.datetime.combine(bd.date, dt.time(15, 54))
    assert bd.tcast(dt.time(15, 54)) == dt.datetime.combine(bd.date, dt.time(15, 54))
    assert bd.tcast(["15:54:00"]) == [dt.datetime.combine(bd.date, dt.time(15, 54))]


def test_save_and_reload_a_created_var(bars_rw, bars_path):
    bd = bars_rw["2024-11-27"]
    bd.assign(spread="ask - bid")
    assert bd.get_created_vars() == ["spread"]
    bd.save_created_vars()
    assert os.path.exists(bars_path / "2024-11-27" / "spread.nc")

    fresh = fr.BarsSet(str(bars_path))["2024-11-27"]
    assert "spread" in fresh.get_vars_available()
    xr.testing.assert_allclose(fresh.get_var("spread"), bd["spread"])


def test_sliced_bars_cannot_be_saved(bars_rw):
    bd = bars_rw["2024-11-27"].sel_ticker("SPY")
    bd["spread"] = bd.eval("ask - bid")
    with pytest.raises(ValueError):
        bd.save_var("spread")


def test_create_var_matches_the_grid(bd):
    da = bd.create_var(dtype="float32", fill_value=np.nan)
    assert da.dims == ("time", "ticker")
    assert da.shape == (bd.dataset.sizes["time"], bd.dataset.sizes["ticker"])


def test_set_alias(bars_rw, bars_path):
    bd = bars_rw["2024-11-27"]
    bd.set_alias("midpoint", "mid")
    fresh = fr.BarsSet(str(bars_path))["2024-11-27"]
    xr.testing.assert_allclose(fresh.get_var("midpoint"), fresh.get_var("mid"))


def test_delitem_forgets_the_on_disk_link(bd):
    bd.load_var("mid")
    del bd["mid"]
    assert "mid" not in bd.live_vars()
    assert "mid" not in bd.loaded_vars()


def test_has_var(bd):
    assert bd.has_var("mid")
    assert not bd.has_var("nope")


def test_add_daily_var(bd):
    daily = xr.DataArray(
        np.arange(4, dtype="float32"),
        dims=["ticker"],
        coords={"ticker": ["AAA", "BBB", "CCC", "SPY"]},
        name="rank",
    )
    bd.add_daily_var(daily)
    assert bd["rank"].dims == ("ticker",)


def test_merge_df_asof(bd):
    import pandas as pd

    rows = pd.DataFrame(
        {
            "signal": [1.0, 2.0],
            "ticker": ["AAA", "AAA"],
            "time": [
                dt.datetime.combine(bd.date, dt.time(15, 51, 30)),
                dt.datetime.combine(bd.date, dt.time(15, 55, 0)),
            ],
        }
    ).set_index(["ticker", "time"])
    bd.merge_df_asof(rows)
    signal = bd["signal"].sel(ticker="AAA")
    assert np.isnan(signal.sel(time=bd.tcast("15:50:00")).item())
    assert signal.sel(time=bd.tcast("15:52:00")).item() == 1.0
    assert signal.sel(time=bd.tcast("15:56:00")).item() == 2.0
