"""Aggregation and summary statistics for a backtest run."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def daily_stats(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse a (date, ticker) frame of fills into one row per date.

    Columns: ``pnl``, ``shares`` and ``dollars`` traded, ``ntickers`` traded,
    ``close_pos`` -- the gross share position carried into the close -- and
    ``open_pos``, what is still open afterwards, which is zero for every date if
    the run used an ``unwind_price``.
    """
    if rows.empty:
        return pd.DataFrame(
            columns=["pnl", "shares", "dollars", "ntickers", "close_pos", "open_pos"],
            index=pd.Index([], name="date"),
        )
    grouped = rows.groupby(level="date")
    return pd.DataFrame(
        {
            "pnl": grouped.pnl.sum(),
            "shares": grouped.volume.sum(),
            "dollars": grouped.dvolume.sum(),
            "ntickers": grouped.size(),
            "close_pos": grouped.close_pos.apply(lambda s: s.abs().sum()),
            "open_pos": grouped.eod_pos.apply(lambda s: s.abs().sum()),
        }
    )


def drawdowns(daily: pd.DataFrame, pnl_column: str = "pnl") -> pd.DataFrame:
    """Every peak-to-trough episode in cumulative pnl, worst first.

    Columns: ``start`` (the peak), ``trough``, ``end`` (recovery, or NaT if the
    episode never recovered), ``depth`` in currency, and ``days`` from peak to
    recovery.
    """
    if daily.empty:
        return pd.DataFrame(columns=["start", "trough", "end", "depth", "days"])

    cumulative = daily[pnl_column].cumsum()
    peak = cumulative.cummax()
    underwater = cumulative < peak

    episodes = []
    dates = cumulative.index
    i = 0
    while i < len(dates):
        if not underwater.iloc[i]:
            i += 1
            continue
        # The peak that started this episode is the bar before it went under.
        start = dates[i - 1] if i > 0 else dates[i]
        j = i
        while j < len(dates) and underwater.iloc[j]:
            j += 1
        segment = cumulative.iloc[i:j]
        trough = segment.idxmin()
        recovered = j < len(dates)
        episodes.append(
            {
                "start": start,
                "trough": trough,
                "end": dates[j] if recovered else pd.NaT,
                "depth": float(peak.iloc[i] - segment.min()),
                "days": (j - i + 1) if recovered else (j - i),
            }
        )
        i = j

    if not episodes:
        return pd.DataFrame(columns=["start", "trough", "end", "depth", "days"])
    out = pd.DataFrame(episodes)
    return out.sort_values("depth", ascending=False).reset_index(drop=True)


def summarize(rows: pd.DataFrame, name: str | None = None) -> pd.Series:
    """One-line summary of a backtest run.

    Takes the (date, ticker) frame. Pnl is in whatever currency the prices were
    quoted in; since the engine has no notion of deployed capital, the risk
    figures are expressed on daily pnl rather than on returns.
    """
    daily = daily_stats(rows)
    ndays = len(daily)
    stats: dict[str, float] = {"ndays": ndays}

    if ndays == 0:
        return pd.Series(stats, dtype="object", name=name)

    pnl = daily["pnl"]
    total_pnl = float(pnl.sum())
    total_shares = float(daily["shares"].sum())
    total_dollars = float(daily["dollars"].sum())
    # ddof=1: these are a sample of the strategy's daily outcomes.
    std = float(pnl.std())

    stats["total_pnl"] = total_pnl
    stats["pnl"] = float(pnl.mean())
    stats["pnl_median"] = float(pnl.median())
    stats["pnl_std"] = std
    # Sharpe on daily pnl, annualized. No risk-free rate: this is a dollar pnl
    # series, not a return series.
    stats["sharpe"] = (
        float(np.sqrt(TRADING_DAYS_PER_YEAR) * pnl.mean() / std) if std > 0 else np.nan
    )
    # Whether the mean daily pnl is distinguishable from zero.
    stats["tstat"] = float(pnl.mean() / (std / np.sqrt(ndays))) if std > 0 and ndays > 1 else np.nan
    stats["win_rate"] = float(100.0 * (pnl > 0).mean())
    stats["best_day"] = float(pnl.max())
    stats["worst_day"] = float(pnl.min())

    dd = drawdowns(daily)
    stats["max_drawdown"] = float(dd["depth"].iloc[0]) if not dd.empty else 0.0
    stats["max_dd_days"] = float(dd["days"].iloc[0]) if not dd.empty else 0.0

    stats["shares"] = float(daily["shares"].mean())
    stats["dollars"] = float(daily["dollars"].mean())
    stats["ntickers"] = float(daily["ntickers"].mean())
    stats["close_pos"] = float(daily["close_pos"].mean())
    # Profitability per unit of trading -- the two ways a trading desk quotes it.
    stats["cps"] = 100.0 * total_pnl / total_shares if total_shares > 0 else np.nan
    stats["margin_bps"] = 1e4 * total_pnl / total_dollars if total_dollars > 0 else np.nan

    return pd.Series(stats, name=name)


class BacktestResult:
    """The output of `run_backtest`: fills, daily aggregates, and a summary.

    ``rows`` is the raw (date, ticker) frame with ``pnl``, ``eod_pos``,
    ``close_pos``, ``volume`` and ``dvolume``. Everything else is derived from it.
    """

    def __init__(self, rows: pd.DataFrame, name: str | None = None):
        self.rows = rows
        self.name = name

    @property
    def daily(self) -> pd.DataFrame:
        """One row per date."""
        return daily_stats(self.rows)

    def summary(self) -> pd.Series:
        """One-line summary; see `summarize`."""
        return summarize(self.rows, name=self.name)

    def drawdowns(self) -> pd.DataFrame:
        """Peak-to-trough episodes in cumulative pnl, worst first."""
        return drawdowns(self.daily)

    def cumulative_pnl(self) -> pd.Series:
        """Cumulative pnl by date, ready to plot."""
        return self.daily["pnl"].cumsum()

    def by_ticker(self) -> pd.DataFrame:
        """Totals per ticker across every date, most profitable first."""
        grouped = self.rows.groupby(level="ticker")
        out = pd.DataFrame(
            {
                "pnl": grouped.pnl.sum(),
                "shares": grouped.volume.sum(),
                "dollars": grouped.dvolume.sum(),
                "ndays": grouped.size(),
            }
        )
        return out.sort_values("pnl", ascending=False)

    def __repr__(self) -> str:
        summary = self.summary()
        label = f" {self.name}" if self.name else ""
        body = "\n".join(
            f"  {k:<13} {v:,.4g}" if isinstance(v, float) else f"  {k:<13} {v}"
            for k, v in summary.items()
        )
        return f"<BacktestResult{label}>\n{body}"
