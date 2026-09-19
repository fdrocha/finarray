"""The variable-agnostic ticker filter that replaced the monorepo's StockUniverse."""

import pytest

import finarray as fr


def test_range_constraint(bd):
    # adv is base_price * 2e5, so: AAA 1.0e7, BBB 2.4e7, CCC 1.6e6, SPY 1.0e8
    out = bd.sel_where(adv=(2e7, None))
    assert out.ticker.values.tolist() == ["BBB", "SPY"]


def test_range_with_both_bounds(bd):
    out = bd.sel_where(adv=(5e6, 5e7))
    assert out.ticker.values.tolist() == ["AAA", "BBB"]


def test_equality_constraint(bd):
    out = bd.sel_where(listing_exchange=1)
    assert out.ticker.values.tolist() == ["AAA", "BBB"]


def test_constraints_are_anded(bd):
    out = bd.sel_where(adv=(5e6, None), listing_exchange=1)
    assert out.ticker.values.tolist() == ["AAA", "BBB"]


def test_extra_tickers_are_kept_regardless(bd):
    out = bd.sel_where(adv=(5e7, None), extra_tickers=["CCC"])
    assert out.ticker.values.tolist() == ["CCC", "SPY"]


def test_constraint_variables_are_autoloaded(bd):
    assert "adv" not in bd.live_vars()
    bd.sel_where(adv=(1e6, None))
    assert "adv" in bd.live_vars()


def test_time_varying_variables_are_rejected(bd):
    with pytest.raises(ValueError, match="ticker-only"):
        bd.sel_where(mid=(0, 1000))


def test_unknown_variable_is_rejected(bd):
    with pytest.raises(ValueError, match="not available"):
        bd.sel_where(no_such_var=(0, 1))


def test_malformed_range_is_rejected():
    with pytest.raises(ValueError, match="must be"):
        fr.Where(constraints={"adv": (1, 2, 3)})


def test_where_applies_to_every_date_in_a_set(bars):
    filtered = bars.sel_where(adv=(2e7, None))
    for _, day in filtered.iter_bars():
        assert day.ticker.values.tolist() == ["BBB", "SPY"]


def test_where_is_remembered_for_dates_loaded_later(sample_tree, dates):
    bars = fr.BarsSet(str(sample_tree)).sel_where(adv=(2e7, None))
    assert bars.dates_loaded() == []
    assert bars[dates[0]].ticker.values.tolist() == ["BBB", "SPY"]


def test_where_via_the_constructor(sample_tree, dates):
    bars = fr.BarsSet(
        str(sample_tree),
        restrict_dates=[dates[0].isoformat()],
        where=fr.Where(constraints={"adv": (2e7, None)}),
    )
    assert bars[dates[0]].ticker.values.tolist() == ["BBB", "SPY"]


def test_where_composes_with_sel(bars):
    r = bars.sel_where(adv=(2e7, None)).sel_time("15:54:00").cat_var("mid")
    assert r.index.names == ["date", "ticker"]
    assert r.index.get_level_values("ticker").unique().to_list() == ["BBB", "SPY"]


def test_where_does_not_mutate_the_original(bars):
    bars.sel_where(adv=(2e7, None))
    assert bars["2024-11-27"].ticker.values.tolist() == ["AAA", "BBB", "CCC", "SPY"]
