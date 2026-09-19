"""Generate a small, entirely synthetic bars tree.

Nothing here is real market data -- prices are a random walk. The point is to
give the tests and the README something runnable to point at.

    python examples/make_sample_bars.py /tmp/sample-bars
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_TICKERS = ("AAA", "BBB", "CCC", "SPY")
DEFAULT_DATES = ("2024-11-27", "2024-12-02", "2024-12-03", "2024-12-04")


def make_date_dir(
    path: str,
    date: dt.date,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    t_start: dt.time = dt.time(15, 50, 0),
    n_seconds: int = 600,
    seed: int = 0,
) -> None:
    """Write one date directory: the two coordinate CSVs plus a few variables."""
    rng = np.random.default_rng(seed)
    start = dt.datetime.combine(date, t_start)
    times = pd.date_range(start, periods=n_seconds, freq="1s", name="time")
    n_t, n_k = len(times), len(tickers)

    # A driftless random walk per ticker, starting from a per-ticker base price.
    base = np.array([50.0, 120.0, 8.0, 500.0][:n_k])
    steps = rng.normal(0.0, 1e-4, size=(n_t, n_k))
    mid = base * np.exp(np.cumsum(steps, axis=0))
    half_spread = np.where(np.asarray(tickers) == "SPY", 0.005, 0.01)

    coords = {"time": times, "ticker": pd.Index(tickers, name="ticker")}
    ds = xr.Dataset(
        {
            "mid": (("time", "ticker"), mid.astype("float32")),
            "bid": (("time", "ticker"), (mid - half_spread).astype("float32")),
            "ask": (("time", "ticker"), (mid + half_spread).astype("float32")),
            "volume": (
                ("time", "ticker"),
                rng.integers(0, 5000, size=(n_t, n_k)).astype("int32"),
            ),
            # ticker-only ("daily") variables: one value per ticker for the day
            "adv": ("ticker", (base * 2e5).astype("float32")),
            "ref_price": ("ticker", mid[0].astype("float32")),
            "beta": ("ticker", np.array([1.1, 0.8, 1.4, 1.0][:n_k], dtype="float32")),
            "listing_exchange": ("ticker", np.array([1, 1, 2, 3][:n_k], dtype="int32")),
        },
        coords=coords,
    )

    os.makedirs(path, exist_ok=True)
    ds.time.to_series().reset_index(drop=True).to_csv(os.path.join(path, "time.csv"), index=False)
    ds.ticker.to_series().reset_index(drop=True).to_csv(
        os.path.join(path, "ticker.csv"), index=False
    )
    for var in ds.data_vars:
        ds[var].to_netcdf(os.path.join(path, f"{var}.nc"))


def make_sample_bars(
    base_path: str,
    dates: tuple[str, ...] = DEFAULT_DATES,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    **kwargs,
) -> list[dt.date]:
    """Write a whole bars base directory, one sub-directory per date."""
    out = []
    for i, date_str in enumerate(dates):
        date = dt.date.fromisoformat(date_str)
        make_date_dir(
            os.path.join(base_path, date.isoformat()),
            date,
            tickers=tickers,
            seed=i,
            **kwargs,
        )
        out.append(date)
    return out


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "sample-bars"
    dates = make_sample_bars(target)
    print(f"wrote {len(dates)} dates to {target}/")
