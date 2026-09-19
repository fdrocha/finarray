"""Multi-date behaviour: aggregation, mapcat, and joining external tables."""

import datetime as dt

import pandas as pd
import pytest

import finarray as fr


def test_dates_available_and_loaded(bars, dates):
    assert bars.dates_available() == dates
    assert bars.dates_loaded() == dates


def test_indexing_by_date_string_and_position(bars, dates):
    assert bars[dates[0]].date == dates[0]
    assert bars["2024-11-27"].date == dates[0]
    assert bars[-1].date == dates[-1]
    assert bars[0].date == dates[0]


def test_restrict_dates_warns_for_missing_dates(sample_tree):
    with pytest.warns(UserWarning, match="No data available"):
        bars = fr.BarsSet(str(sample_tree), restrict_dates=["2024-11-27", "1999-01-04"])
    assert bars.dates_available() == [dt.date(2024, 11, 27)]


def test_preload_vars(sample_tree):
    bars = fr.BarsSet(str(sample_tree), restrict_dates=["2024-11-27"], preload_vars=["mid", "bid"])
    assert set(bars["2024-11-27"].live_vars()) == {"mid", "bid"}


def test_has_date(bars):
    assert bars.has_date(dt.date(2024, 11, 27))
    assert not bars.has_date(dt.date(1999, 1, 4))


def test_mapcat_concatenates_per_date_frames(bars, dates):
    df = bars.mapcat(lambda bd: bd.get_var("mid").to_pandas().head(3), add_date=True)
    assert df.index.names[0] == "date"
    assert df.index.get_level_values("date").unique().to_list() == dates
    assert len(df) == 3 * len(dates)


def test_mapcat_split_datetime(bars):
    df = bars.mapcat(lambda bd: bd.get_var("mid").to_pandas().head(2), split_datetime=True)
    assert df.index.names[:2] == ["date", "time"]


def test_mapcat_warns_and_skips_failing_dates(bars):
    def fail_on_first(bd):
        if bd.date == dt.date(2024, 11, 27):
            raise RuntimeError("boom")
        return bd.get_var("mid").to_pandas().head(1)

    with pytest.warns(UserWarning, match="boom"):
        df = bars.mapcat(fail_on_first, add_date=True)
    assert dt.date(2024, 11, 27) not in df.index.get_level_values("date")


def test_mapcat_can_reraise(bars):
    def always_fail(bd):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        bars.mapcat(always_fail, on_errors="raise")


def test_assign_expr_defines_a_var_on_every_date(bars, dates):
    b = fr.BarsSet(bars.base_path, restrict_dates=[d.isoformat() for d in dates])
    b.assign_expr("spread", "ask - bid")
    for _, day in b.iter_bars():
        assert "spread" in day.live_vars()


def test_assign_func(bars, dates):
    b = fr.BarsSet(bars.base_path, restrict_dates=[d.isoformat() for d in dates])
    b.assign_func("double_mid", lambda day: day.get_var("mid") * 2)
    assert "double_mid" in b[dates[0]].live_vars()


def _event_frame(dates, tickers):
    rows = [
        {"date": d, "ticker": t, "alpha": 1.0 if i % 2 else -1.0}
        for i, d in enumerate(dates)
        for t in tickers
    ]
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_attach_to_df_by_tickers(bars, dates, tickers):
    df = _event_frame(dates, tickers)
    out = bars.sel_time("15:54:00").attach_to_df_by_tickers(df, ["mid", "bid"])
    assert out.index.names == ["date", "ticker"]
    assert set(out.columns) >= {"alpha", "mid", "bid"}
    assert len(out) == len(df)


def test_attach_to_df_by_tickers_needs_a_scalar_time(bars, dates, tickers):
    df = _event_frame(dates, tickers)
    with pytest.raises(ValueError, match="single time"):
        bars.attach_to_df_by_tickers(df, ["mid"])


def test_attach_to_df_by_ticker_time(bars, dates, tickers):
    df = _event_frame(dates, tickers)
    df["event_time"] = [
        dt.datetime.combine(d, dt.time(15, 54, 30)) for d in df.index.get_level_values("date")
    ]
    out = bars.attach_to_df_by_ticker_time(df, ["mid"], time_column="event_time", method="ffill")
    assert out.index.names == ["date", "ticker"]
    assert out["mid"].notna().all()


def test_attach_to_df_by_ticker_time_ffill_picks_the_prior_bar(bars, dates):
    """A timestamp between bars resolves back to the most recent one."""
    date = dates[0]
    df = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": "SPY",
                "event_time": dt.datetime.combine(date, dt.time(15, 54, 0, 500000)),
            }
        ]
    ).set_index(["date", "ticker"])
    out = bars.attach_to_df_by_ticker_time(df, ["mid"], time_column="event_time", method="ffill")
    expected = bars[date].sel_ticker("SPY").sel_time("15:54:00").get_value("mid")
    assert out["mid"].iloc[0] == pytest.approx(expected)


def test_restrict_other_drops_unknown_tickers(bars, dates, tickers):
    df = _event_frame(dates, tickers + ["NOT_A_TICKER"])
    out = bars.restrict_other(df)
    assert "NOT_A_TICKER" not in out.index.get_level_values("ticker")
    assert len(out) == len(dates) * len(tickers)


def test_bars_getter(sample_tree, dates):
    getter = fr.bars_getter(str(sample_tree))
    bars = getter(restrict_dates=["2024-11-27"], preload_vars=["mid"])
    assert isinstance(bars, fr.BarsSet)
    assert bars.dates_loaded() == [dates[0]]


def test_copy_is_independent(bars):
    other = bars.copy()
    other.assign_expr("spread", "ask - bid")
    assert "spread" not in bars["2024-11-27"].live_vars()


def test_tilde_in_paths_is_expanded(monkeypatch, sample_tree):
    monkeypatch.setenv("HOME", str(sample_tree.parent))
    bars = fr.BarsSet(f"~/{sample_tree.name}", restrict_dates=["2024-11-27"])
    assert bars.dates_loaded() == [dt.date(2024, 11, 27)]
