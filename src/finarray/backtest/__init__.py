"""Target-position backtesting over finarray bars.

Optional extra -- install with ``pip install 'finarray[backtest]'`` to get the
numba acceleration. Without numba the kernels still run, just far slower.

    from finarray import backtest

    result = backtest.run_backtest(
        bars.sel_time_slice("15:50:00", "15:59:59"),
        "signal * 1000",              # target shares at each bar
        prices="mid",
        unwind_price="close_price",
        fees=backtest.Fees.mils(2.18),
    )
    result.summary()
"""

from .engine import HAVE_NUMBA, run_backtest, run_date
from .fees import ZERO, Fees
from .result import BacktestResult, daily_stats, drawdowns, summarize

__all__ = [
    "ZERO",
    "BacktestResult",
    "Fees",
    "HAVE_NUMBA",
    "daily_stats",
    "drawdowns",
    "run_backtest",
    "run_date",
    "summarize",
]
