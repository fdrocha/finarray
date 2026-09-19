"""Walk a path of target positions through a day of bars and account for it.

The model is deliberately simple: at each bar you state how many shares you want
to hold, and the engine trades the difference at that bar's price, charging
linear fees. There is no market-impact or queue model -- what it gives you is a
clean upper bound plus an explicit, tunable cost.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..bars import Bars
from ..bars_set import BarsSet
from .fees import ZERO, Fees
from .result import BacktestResult

try:  # pragma: no cover - exercised by whichever branch the environment takes
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        """No-op stand-in so the kernels still run (slowly) without numba."""

        def wrap(func):
            return func

        return wrap(args[0]) if args and callable(args[0]) else wrap


_warned_no_numba = False


def _warn_if_slow() -> None:
    global _warned_no_numba
    if not HAVE_NUMBA and not _warned_no_numba:
        _warned_no_numba = True
        warnings.warn(
            "numba is not installed, so the backtest kernels run as plain Python "
            "and will be orders of magnitude slower. Install it with "
            "`pip install 'finarray[backtest]'`.",
            RuntimeWarning,
            stacklevel=3,
        )


@njit(cache=True)
def _walk(
    tgt_pos: np.ndarray,
    prices: np.ndarray,
    unwind_prices: np.ndarray | None,
    vol_cap: np.ndarray | None,
    fee_dollars: float,
    fee_shares: float,
    fee_unwind_dollars: float,
    fee_unwind_shares: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate pnl, positions, share volume and dollar volume per ticker.

    Returns ``(pnl, eod_pos, close_pos, volume, dvolume)``. ``close_pos`` is the
    position carried into the unwind -- the size you had to liquidate -- while
    ``eod_pos`` is what is left afterwards, and so is zero whenever the run
    unwinds.

    `tgt_pos` and `prices` are (time, ticker). A NaN price means the ticker is
    not trading in that bar and is skipped; a NaN target means hold. `vol_cap`,
    if given, limits the shares traded per bar per ticker.
    """
    n_time, n_tick = prices.shape

    pos = np.zeros(n_tick, dtype=np.float64)
    pnl = np.zeros(n_tick, dtype=np.float64)
    volume = np.zeros(n_tick, dtype=np.float64)
    dvolume = np.zeros(n_tick, dtype=np.float64)
    last_price = np.zeros(n_tick, dtype=np.float64)
    seen = np.zeros(n_tick, dtype=np.bool_)

    for ti in range(n_time):
        for ki in range(n_tick):
            price = prices[ti, ki]
            if np.isnan(price):
                continue

            # Mark the existing position to the new price before trading it.
            if seen[ki]:
                pnl[ki] += (price - last_price[ki]) * pos[ki]
            last_price[ki] = price
            seen[ki] = True

            target = tgt_pos[ti, ki]
            if np.isnan(target):
                continue

            diff = target - pos[ki]
            if vol_cap is None:
                traded = float(int(abs(diff)))
            else:
                traded = min(abs(diff), vol_cap[ki])
            if traded == 0.0:
                continue

            pos[ki] += math.copysign(traded, diff)
            volume[ki] += traded
            dvolume[ki] += traded * price
            pnl[ki] -= fee_dollars * traded * price + fee_shares * traded

    close_pos = pos.copy()

    if unwind_prices is not None:
        for ki in range(n_tick):
            price = unwind_prices[ki]
            if np.isnan(price) or not seen[ki] or pos[ki] == 0.0:
                continue
            size = abs(pos[ki])
            pnl[ki] += pos[ki] * (price - last_price[ki])
            pnl[ki] -= fee_unwind_dollars * size * price + fee_unwind_shares * size
            volume[ki] += size
            dvolume[ki] += size * price
            pos[ki] = 0.0

    return pnl, pos, close_pos, volume, dvolume


def run_date(
    bars: Bars,
    tgt_pos: xr.DataArray,
    prices: xr.DataArray,
    unwind_prices: xr.DataArray | None = None,
    vol_cap_per_bar: xr.DataArray | None = None,
    fees: Fees = ZERO,
    fees_unwind: Fees = ZERO,
) -> pd.DataFrame:
    """Run one date; returns a ticker-indexed frame of the per-ticker totals."""
    _warn_if_slow()
    tgt_pos, prices = xr.align(tgt_pos, prices, join="right")
    tgt = tgt_pos.transpose("time", "ticker").to_numpy().astype("float64")
    px = prices.transpose("time", "ticker").to_numpy().astype("float64")

    f, fu = fees.as_two_sided(), fees_unwind.as_two_sided()
    pnl, pos, close_pos, volume, dvolume = _walk(
        tgt,
        px,
        unwind_prices.to_numpy().astype("float64") if unwind_prices is not None else None,
        vol_cap_per_bar.to_numpy().astype("float64") if vol_cap_per_bar is not None else None,
        f.per_dollar_bips * 1e-4,
        f.per_share_mils * 1e-4,
        fu.per_dollar_bips * 1e-4,
        fu.per_share_mils * 1e-4,
    )
    return pd.DataFrame(
        {
            "pnl": pnl,
            "eod_pos": pos,
            "close_pos": close_pos,
            "volume": volume,
            "dvolume": dvolume,
        },
        index=pd.Index(prices.ticker.values, name="ticker"),
    )


def run_backtest(
    bars_set: BarsSet,
    tgt_pos: str | Callable[[Bars], xr.DataArray],
    prices: str = "mid",
    unwind_price: str | None = None,
    vol_cap_per_bar: str | None = None,
    fees: Fees = ZERO,
    fees_unwind: Fees = ZERO,
    on_errors: Literal["raise", "warn", "ignore"] = "warn",
    progress: bool = False,
) -> BacktestResult:
    """Run a target-position strategy across every loaded date.

    Args:
        bars_set: the dates to run over. Narrow it with `sel_time_slice` first
            to restrict trading to part of the day.
        tgt_pos: the desired position in shares at each bar, as an expression
            evaluated against the day's bars (autoloading whatever it names) or
            as a callable taking a `Bars` and returning a `(time, ticker)` array.
        prices: variable to trade at.
        unwind_price: if given, flatten every position at this price at the end
            of the day (e.g. a closing price). Defaults to None, which leaves
            end-of-day positions open and excludes their unwind from pnl.
        vol_cap_per_bar: a ticker-only variable capping shares traded per bar.
            Without it, the full difference to target is traded every bar.
        fees: trading costs; `fees_unwind` applies to the closing unwind only.

    Returns:
        A `BacktestResult` wrapping a (date, ticker) frame. Tickers that never
        traded are dropped.
    """

    def one_date(bd: Bars) -> pd.DataFrame:
        target = bd.eval(tgt_pos) if isinstance(tgt_pos, str) else tgt_pos(bd)
        rows = run_date(
            bd,
            target,  # type: ignore[arg-type]
            bd.get_var(prices),
            bd.get_var(unwind_price) if unwind_price is not None else None,
            bd.get_var(vol_cap_per_bar) if vol_cap_per_bar is not None else None,
            fees=fees,
            fees_unwind=fees_unwind,
        )
        return rows[rows.volume != 0]

    frame = bars_set.mapcat(one_date, add_date=True, on_errors=on_errors, progress=progress)
    return BacktestResult(frame)
