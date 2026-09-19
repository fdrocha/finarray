"""The selection/scalar-shape contract.

These assertions are ported from `notebooks/fr_tests.ipynb` in the private
monorepo this library was extracted from -- they were the de-facto spec for how
`sel_time`/`sel_ticker` interact with `cat_var` across dates.
"""

import datetime as dt

import pytest

import finarray as fr


def test_no_selection_keeps_both_dims(bars):
    r = bars.cat_var("mid")
    assert r.index.names == ["time", "ticker"]


def test_ticker_list_keeps_ticker_dim(bars):
    r = bars.sel_ticker(["SPY", "AAA"]).cat_var("mid")
    assert r.index.names == ["time", "ticker"]


def test_single_element_ticker_list_keeps_ticker_dim(bars):
    r = bars.sel_ticker(ticker=["SPY"]).cat_var("mid")
    assert r.index.names == ["time", "ticker"]


def test_scalar_ticker_drops_ticker_dim(bars):
    r = bars.sel_ticker(ticker="SPY").cat_var("mid")
    assert r.index.names == ["time"]


def test_time_list_keeps_time_dim(bars):
    r = bars.sel_time(time=["15:54:00"]).cat_var("mid")
    assert r.index.names == ["time", "ticker"]


def test_scalar_time_becomes_a_date_level(bars):
    r = bars.sel_time(time="15:54:00").cat_var("mid")
    assert r.index.names == ["date", "ticker"]


def test_both_scalar_leaves_only_date(bars):
    r = bars.sel_ticker("SPY").sel_time("15:54:00").cat_var("mid")
    assert r.index.names == ["date"]


def test_selection_order_does_not_matter(bars):
    a = bars.sel_ticker("SPY").sel_time("15:54:00").cat_var("mid")
    b = bars.sel_time(time="15:54:00").sel_ticker(ticker="SPY").cat_var("mid")
    assert a.index.names == b.index.names == ["date"]
    assert a.to_list() == b.to_list()


def test_cat_vars_multiple(bars):
    r = bars.sel_ticker("AAA").sel_time("15:54:00").cat_vars(["mid", "bid", "ask"])
    assert r.index.names == ["date"]
    assert r.columns.to_list() == ["mid", "bid", "ask"]


def test_selection_is_immutable(bars):
    """Each sel_* returns a copy; the original is untouched."""
    narrowed = bars.sel_ticker("SPY")
    assert narrowed.cat_var("mid").index.names == ["time"]
    assert bars.cat_var("mid").index.names == ["time", "ticker"]


def test_sel_accepts_time_strings_and_time_objects(bd):
    a = bd.sel_time("15:54:00").get_var("mid")
    b = bd.sel_time(dt.time(15, 54, 0)).get_var("mid")
    assert a.values.tolist() == b.values.tolist()


def test_time_slice(bd):
    sliced = bd.sel_time_slice("15:51:00", "15:52:00")
    times = sliced.get_var("mid").indexes["time"]
    assert times[0].time() == dt.time(15, 51, 0)
    assert times[-1].time() == dt.time(15, 52, 0)


def test_sel_on_bars_is_rejected(bd):
    with pytest.raises(NotImplementedError):
        bd.sel(time="15:54:00")


def test_selections_are_replayed_onto_lazily_loaded_vars(bd):
    """A variable loaded *after* a selection still gets the selection applied."""
    narrowed = bd.sel_ticker("SPY")
    assert "volume" not in narrowed.live_vars()
    assert narrowed.get_var("volume").dims == ("time",)


def test_getitem_list_returns_bars(bd):
    bd.load_vars(["mid", "bid"])
    subset = bd[["mid", "bid"]]
    assert isinstance(subset, fr.Bars)
    assert set(subset.live_vars()) == {"mid", "bid"}
