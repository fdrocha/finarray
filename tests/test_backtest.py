"""Backtest engine and summary statistics.

The bars here are tiny and deterministic so every expected number below is
computed by hand in the comments, not copied from a previous run.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import finarray as fr
import finarray.backtest.engine  # noqa: F401
from finarray import backtest
from finarray.backtest import Fees

TIMES = [dt.time(10, 0, s) for s in range(4)]
# AAA rises 10 -> 13 over four bars and closes at 14; BBB is flat at 20, closes 19.
PRICES = {"AAA": [10.0, 11.0, 12.0, 13.0], "BBB": [20.0, 20.0, 20.0, 20.0]}
CLOSES = {"AAA": 14.0, "BBB": 19.0}
DATES = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]


@pytest.fixture
def tiny_bars(tmp_path):
    """Two dates, two tickers, four bars, all prices known exactly."""
    base = tmp_path / "tiny"
    base.mkdir()
    tickers = list(PRICES)
    for date in DATES:
        times = pd.DatetimeIndex([dt.datetime.combine(date, t) for t in TIMES], name="time")
        ds = xr.Dataset(
            {
                "mid": (("time", "ticker"), np.array([PRICES[k] for k in tickers]).T),
                "close_price": ("ticker", np.array([CLOSES[k] for k in tickers])),
                "target": (
                    ("time", "ticker"),
                    np.tile(np.array([10.0, 5.0]), (len(TIMES), 1)),
                ),
                "cap": ("ticker", np.array([3.0, 3.0])),
            },
            coords={"time": times, "ticker": pd.Index(tickers, name="ticker")},
        )
        fr.to_frdir(ds, str(base), date=date)
    bars = fr.BarsSet(str(base))
    bars.load_dates(all_available=True)
    return bars


def test_buy_and_hold_price_pnl(tiny_bars):
    """Buy at the first bar, hold, no unwind: pnl is the mark-to-market move."""
    r = backtest.run_backtest(tiny_bars, "target", prices="mid")
    rows = r.rows.xs(DATES[0], level="date")
    # AAA: 10 shares * (13 - 10) = 30. BBB: 5 shares * (20 - 20) = 0.
    assert rows.loc["AAA", "pnl"] == pytest.approx(30.0)
    assert rows.loc["BBB", "pnl"] == pytest.approx(0.0)
    # Only the opening trade happened.
    assert rows.loc["AAA", "volume"] == pytest.approx(10.0)
    assert rows.loc["AAA", "dvolume"] == pytest.approx(100.0)
    # Nothing unwound them, so the position is still open.
    assert rows.loc["AAA", "eod_pos"] == pytest.approx(10.0)


def test_unwind_closes_positions_and_adds_pnl(tiny_bars):
    r = backtest.run_backtest(tiny_bars, "target", prices="mid", unwind_price="close_price")
    rows = r.rows.xs(DATES[0], level="date")
    # AAA: 30 from the bars + 10 * (14 - 13) = 40. BBB: 5 * (19 - 20) = -5.
    assert rows.loc["AAA", "pnl"] == pytest.approx(40.0)
    assert rows.loc["BBB", "pnl"] == pytest.approx(-5.0)
    # Entry 10 shares + unwind 10 shares.
    assert rows.loc["AAA", "volume"] == pytest.approx(20.0)
    # 10 @ 10 entry + 10 @ 14 unwind.
    assert rows.loc["AAA", "dvolume"] == pytest.approx(240.0)
    assert (r.daily["open_pos"] == 0).all()
    # but the position carried into the close is still reported
    assert r.daily.loc[DATES[0], "close_pos"] == pytest.approx(15.0)  # 10 + 5


def test_per_share_fees(tiny_bars):
    """A mils fee is tenths of a cent per share: 2.18 mils = $2.18e-4/share."""
    free = backtest.run_backtest(tiny_bars, "target", prices="mid")
    charged = backtest.run_backtest(tiny_bars, "target", prices="mid", fees=Fees.mils(2.18))
    aaa_free = free.rows.xs(DATES[0], level="date").loc["AAA", "pnl"]
    aaa_paid = charged.rows.xs(DATES[0], level="date").loc["AAA", "pnl"]
    assert aaa_free - aaa_paid == pytest.approx(10 * 2.18e-4)


def test_per_dollar_fees(tiny_bars):
    """A bips fee is hundredths of a percent of notional."""
    free = backtest.run_backtest(tiny_bars, "target", prices="mid")
    charged = backtest.run_backtest(tiny_bars, "target", prices="mid", fees=Fees.bips(1))
    aaa_free = free.rows.xs(DATES[0], level="date").loc["AAA", "pnl"]
    aaa_paid = charged.rows.xs(DATES[0], level="date").loc["AAA", "pnl"]
    # 10 shares at $10 = $100 of notional.
    assert aaa_free - aaa_paid == pytest.approx(100 * 1e-4)


def test_unwind_fees_are_charged_separately(tiny_bars):
    base = backtest.run_backtest(tiny_bars, "target", prices="mid", unwind_price="close_price")
    with_unwind_fee = backtest.run_backtest(
        tiny_bars,
        "target",
        prices="mid",
        unwind_price="close_price",
        fees_unwind=Fees.mils(10),
    )
    a = base.rows.xs(DATES[0], level="date").loc["AAA", "pnl"]
    b = with_unwind_fee.rows.xs(DATES[0], level="date").loc["AAA", "pnl"]
    assert a - b == pytest.approx(10 * 10e-4)  # 10 shares unwound


def test_sell_only_fees_are_halved():
    assert Fees.mils(4, sell_only=True).as_two_sided() == Fees.mils(2)
    assert Fees.mils(4).calculate(shares=100, dollars=0) == pytest.approx(4 * 100e-4)
    assert Fees.mils(4, sell_only=True).calculate(100, 0) == pytest.approx(2 * 100e-4)


def test_fee_arithmetic():
    combined = Fees.mils(2) + Fees.bips(1)
    assert combined.per_share_mils == 2
    assert combined.per_dollar_bips == 1
    assert (Fees.mils(4) * 0.5).per_share_mils == 2
    assert (Fees.mils(4) / 2).per_share_mils == 2


def test_vol_cap_limits_shares_per_bar(tiny_bars):
    """With a cap of 3/bar, a target of 10 takes four bars to reach."""
    r = backtest.run_backtest(tiny_bars, "target", prices="mid", vol_cap_per_bar="cap")
    rows = r.rows.xs(DATES[0], level="date")
    # 3 + 3 + 3 + 1 = 10 shares over the four bars.
    assert rows.loc["AAA", "volume"] == pytest.approx(10.0)
    # Position is 3, 6, 9, 10 entering bars 1..3 and after bar 3:
    # pnl = 3*(11-10) + 6*(12-11) + 9*(13-12) = 3 + 6 + 9 = 18
    assert rows.loc["AAA", "pnl"] == pytest.approx(18.0)


def test_callable_target(tiny_bars):
    by_expr = backtest.run_backtest(tiny_bars, "target", prices="mid")
    by_func = backtest.run_backtest(tiny_bars, lambda bd: bd.get_var("target"), prices="mid")
    pd.testing.assert_frame_equal(by_expr.rows, by_func.rows)


def test_target_expression_autoloads(tiny_bars):
    """The expression goes through Bars.eval, so it pulls in what it names."""
    r = backtest.run_backtest(tiny_bars, "target * 2", prices="mid")
    rows = r.rows.xs(DATES[0], level="date")
    assert rows.loc["AAA", "volume"] == pytest.approx(20.0)


def test_nan_prices_are_skipped(tiny_bars):
    bd = tiny_bars[DATES[0]]
    prices = bd.get_var("mid").copy()
    prices[0, :] = np.nan  # no trading in the first bar
    rows = backtest.run_date(bd, bd.get_var("target"), prices)
    # First trade now happens at bar 1 (price 11), so pnl is 10 * (13 - 11) = 20.
    assert rows.loc["AAA", "pnl"] == pytest.approx(20.0)
    assert rows.loc["AAA", "dvolume"] == pytest.approx(110.0)


def test_untraded_tickers_are_dropped(tiny_bars):
    r = backtest.run_backtest(tiny_bars, "target * 0", prices="mid")
    assert r.rows.empty


# --------------------------------------------------------------------------
# summary statistics
# --------------------------------------------------------------------------


def _rows_from_daily(pnls):
    """Build a minimal (date, ticker) frame with one ticker and given daily pnl."""
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(pnls))]
    return pd.DataFrame(
        {
            "pnl": pnls,
            "eod_pos": [0.0] * len(pnls),
            "close_pos": [0.0] * len(pnls),
            "volume": [100.0] * len(pnls),
            "dvolume": [1000.0] * len(pnls),
        },
        index=pd.MultiIndex.from_arrays([dates, ["AAA"] * len(pnls)], names=["date", "ticker"]),
    )


def test_daily_stats_aggregates_across_tickers(tiny_bars):
    r = backtest.run_backtest(tiny_bars, "target", prices="mid", unwind_price="close_price")
    daily = r.daily
    assert daily.index.name == "date"
    assert len(daily) == 2
    # AAA 40 + BBB -5
    assert daily.loc[DATES[0], "pnl"] == pytest.approx(35.0)
    assert daily.loc[DATES[0], "ntickers"] == 2
    assert daily.loc[DATES[0], "shares"] == pytest.approx(30.0)  # (10 + 5) * 2


def test_summary_core_metrics():
    pnls = [10.0, -5.0, 20.0, 5.0]
    s = backtest.summarize(_rows_from_daily(pnls))
    mean, std = np.mean(pnls), np.std(pnls, ddof=1)

    assert s["ndays"] == 4
    assert s["total_pnl"] == pytest.approx(30.0)
    assert s["pnl"] == pytest.approx(mean)
    assert s["pnl_std"] == pytest.approx(std)
    assert s["sharpe"] == pytest.approx(np.sqrt(252) * mean / std)
    assert s["tstat"] == pytest.approx(mean / (std / np.sqrt(4)))
    assert s["win_rate"] == pytest.approx(75.0)  # 3 of 4 days positive
    assert s["best_day"] == pytest.approx(20.0)
    assert s["worst_day"] == pytest.approx(-5.0)


def test_summary_trading_metrics():
    s = backtest.summarize(_rows_from_daily([10.0, -5.0, 20.0, 5.0]))
    # 4 days * 100 shares = 400 shares, $30 pnl -> 7.5 cents per share
    assert s["cps"] == pytest.approx(100 * 30.0 / 400)
    # 4 days * $1000 = $4000 notional -> 75 bps
    assert s["margin_bps"] == pytest.approx(1e4 * 30.0 / 4000)
    assert s["shares"] == pytest.approx(100.0)
    assert s["dollars"] == pytest.approx(1000.0)
    assert s["ntickers"] == pytest.approx(1.0)


def test_summary_drawdown():
    # cumulative: 10, 5, 25, 30 -> one dip of 5 from the peak at day 1
    s = backtest.summarize(_rows_from_daily([10.0, -5.0, 20.0, 5.0]))
    assert s["max_drawdown"] == pytest.approx(5.0)


def test_summary_of_a_flat_strategy():
    s = backtest.summarize(_rows_from_daily([0.0, 0.0, 0.0]))
    assert s["total_pnl"] == 0.0
    assert np.isnan(s["sharpe"])  # no dispersion, so no Sharpe
    assert s["win_rate"] == 0.0


def test_summary_of_nothing():
    empty = _rows_from_daily([]).iloc[:0]
    s = backtest.summarize(empty)
    assert s["ndays"] == 0


def test_drawdowns_detail():
    rows = _rows_from_daily([10.0, -5.0, -5.0, 20.0, -30.0])
    dd = backtest.drawdowns(backtest.daily_stats(rows))
    # cumulative: 10, 5, 0, 20, -10. Two episodes; the second never recovers.
    assert len(dd) == 2
    worst = dd.iloc[0]
    assert worst["depth"] == pytest.approx(30.0)
    assert pd.isna(worst["end"])
    assert dd.iloc[1]["depth"] == pytest.approx(10.0)


def test_cumulative_pnl(tiny_bars):
    r = backtest.run_backtest(tiny_bars, "target", prices="mid")
    cumulative = r.cumulative_pnl()
    assert cumulative.iloc[-1] == pytest.approx(r.daily["pnl"].sum())


def test_by_ticker(tiny_bars):
    r = backtest.run_backtest(tiny_bars, "target", prices="mid", unwind_price="close_price")
    per_ticker = r.by_ticker()
    assert per_ticker.index.to_list() == ["AAA", "BBB"]  # sorted by pnl
    assert per_ticker.loc["AAA", "pnl"] == pytest.approx(80.0)  # 40 on each of 2 days
    assert per_ticker.loc["AAA", "ndays"] == 2


def test_repr_mentions_key_metrics(tiny_bars):
    r = backtest.run_backtest(tiny_bars, "target", prices="mid")
    text = repr(r)
    for key in ("ndays", "total_pnl", "sharpe", "win_rate", "cps"):
        assert key in text


# --------------------------------------------------------------------------
# the numba-free fallback
# --------------------------------------------------------------------------


def test_pure_python_kernel_matches_the_compiled_one():
    """The fallback used when numba is absent must agree with the compiled path."""
    walk = backtest.engine._walk
    if not backtest.HAVE_NUMBA:
        pytest.skip("numba not installed; the fallback is already what ran")

    rng = np.random.default_rng(0)
    n_time, n_tick = 50, 7
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, (n_time, n_tick)), axis=0))
    prices[rng.random(prices.shape) < 0.1] = np.nan
    target = np.round(rng.normal(0, 500, (n_time, n_tick)))
    target[rng.random(target.shape) < 0.2] = np.nan
    unwind = prices[-1].copy()
    cap = np.full(n_tick, 120.0)

    for vol_cap in (None, cap):
        compiled = walk(target, prices, unwind, vol_cap, 1e-4, 2e-4, 0.0, 0.0)
        plain = walk.py_func(target, prices, unwind, vol_cap, 1e-4, 2e-4, 0.0, 0.0)
        for a, b in zip(compiled, plain):
            np.testing.assert_allclose(a, b, rtol=1e-9, equal_nan=True)


def test_mapcat_reports_when_every_date_fails(tiny_bars):
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="every date"):
        backtest.run_backtest(tiny_bars, "no_such_variable", prices="mid")
