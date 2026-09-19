"""Signal return profiles."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from finarray import profile


@pytest.fixture
def signs(bd):
    return (
        pd.Series(
            [1.0, -1.0, 1.0, 1.0],
            index=pd.Index(["AAA", "BBB", "CCC", "SPY"], name="ticker"),
        )
        .to_xarray()
        .rename("s")
    )


def test_single_date_unhedged(bd, signs):
    ds = profile.profile_fixed_single_date(
        bd.sel_time_slice("15:50:00", "15:52:00"), signs, beta=None
    )
    assert set(ds.data_vars) == {"ret", "n"}
    assert ds.ret.dims == ("time",)
    # returns are measured from the reference time, so they start at zero
    assert ds.ret.isel(time=0).item() == pytest.approx(0.0)


def test_single_date_hedged(bd, signs):
    ds = profile.profile_fixed_single_date(
        bd.sel_time_slice("15:50:00", "15:52:00"), signs, beta="beta", hedge_ticker="SPY"
    )
    assert set(ds.data_vars) == {"ret", "ret_hedged", "n"}


def test_per_ticker_keeps_the_ticker_dim(bd, signs):
    ds = profile.profile_fixed_single_date(
        bd.sel_time_slice("15:50:00", "15:51:00"), signs, beta=None, per_ticker=True
    )
    assert set(ds.ret.dims) == {"time", "ticker"}


def test_cps_mode(bd, signs):
    ds = profile.profile_fixed_single_date(
        bd.sel_time_slice("15:50:00", "15:51:00"), signs, beta=None, mode="cps"
    )
    assert set(ds.data_vars) == {"cps", "n"}


def test_invalid_mode(bd, signs):
    with pytest.raises(ValueError, match="Invalid mode"):
        profile.profile_fixed_single_date(bd, signs, mode="nope")


def test_missing_hedge_ticker(bd, signs):
    with pytest.raises(ValueError, match="hedge_ticker"):
        profile.profile_fixed_single_date(
            bd.sel_ticker(["AAA", "BBB"]), signs, beta="beta", hedge_ticker="SPY"
        )


@pytest.mark.parametrize("t_kind", ["time", "timedelta", "seconds", "smidnight"])
def test_t_kind_variants(bd, signs, t_kind):
    ds = profile.profile_fixed_single_date(
        bd.sel_time_slice("15:50:00", "15:51:00"), signs, beta=None, t_kind=t_kind
    )
    assert "time" in ds.dims


def test_profile_fixed_across_dates(bars, dates, tickers):
    alphas = pd.DataFrame(
        [
            {"date": d, "ticker": t, "alpha": 1.0 if t != "BBB" else -1.0}
            for d in dates
            for t in tickers
        ]
    ).set_index(["date", "ticker"])

    ds = profile.profile_fixed(
        bars.sel_time_slice("15:50:00", "15:53:00"),
        alphas,
        alpha_col="alpha",
        beta_col="beta",
        hedge_ticker="SPY",
        t_kind="seconds",
        do_progress=False,
    )
    assert "date" in ds.dims
    assert ds.sizes["date"] == len(dates)
    assert set(ds.data_vars) >= {"ret", "ret_hedged", "n"}
    assert np.isfinite(ds.ret.values).any()


def test_profile_fixed_accepts_a_series(bars, dates, tickers):
    alphas = pd.Series(
        1.0,
        index=pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"]),
    )
    ds = profile.profile_fixed(
        bars.sel_time_slice("15:50:00", "15:52:00"),
        alphas,
        beta_col=None,
        t_kind="seconds",
        do_progress=False,
    )
    assert ds.sizes["date"] == len(dates)


def test_profile_fixed_rejects_datetime_t_kind(bars, dates, tickers):
    alphas = pd.Series(
        1.0,
        index=pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"]),
    )
    with pytest.raises(ValueError, match="t_kind=datetime"):
        profile.profile_fixed(bars, alphas, t_kind="datetime", do_progress=False)


def test_extra_price_extends_the_profile(bd, signs):
    extra = (
        pd.Series(
            [50.0, 120.0, 8.0, 500.0],
            index=pd.Index(["AAA", "BBB", "CCC", "SPY"], name="ticker"),
        )
        .to_xarray()
        .rename("price")
    )
    sliced = bd.sel_time_slice("15:50:00", "15:51:00")
    base = profile.profile_fixed_single_date(sliced, signs, beta=None)
    extended = profile.profile_fixed_single_date(
        sliced,
        signs,
        beta=None,
        extra_price=extra,
        extra_price_time=dt.time(16, 1),
    )
    assert extended.sizes["time"] == base.sizes["time"] + 1
