"""
Functions to do return profiles of an alpha or signal across single or multiple dates.
"""

import datetime as dt
from typing import Literal
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr

from . import util
from .bars import Bars
from .bars_set import BarsSet

ret_unit = 100.0  # Set this to 1e4 to get returns in bips for example
cps_unit = 100.0  # Set this to 1e4 to get cps in mils


def _adjust_time_helper(
    ds: xr.Dataset,
    t_kind: Literal["datetime", "time", "timedelta", "seconds", "smidnight"],
    t_ref: dt.time | dt.datetime | None,
    t_start: dt.time,
    date: dt.date,
) -> None:
    if t_ref is not None and t_kind in ["datetime"]:
        raise ValueError(f"t_ref is not supported when t_kind is {t_kind}")
    if t_kind == "datetime":
        return
    if t_kind == "time":
        ds["time"] = ds.time.dt.time
        return

    if t_ref is None:
        if t_kind == "smidnight":
            t_ref = dt.time(0, 0)
        else:
            t_ref = t_start

    if isinstance(t_ref, dt.time):
        dt_ref = dt.datetime.combine(date, t_ref)
    else:
        dt_ref = t_ref
    ds["time"] = ds.time.to_index() - dt_ref
    if t_kind in ("timedelta", "smidnight"):
        return
    if t_kind == "seconds":
        ds["time"] = ds.time.dt.total_seconds()
        return

    raise ValueError(f"Invalid t_kind: {t_kind}")


# TODO: all of this could be simplified and improved using the new sel().get_var() interface


def profile_fixed_single_date(
    bars: Bars,
    signs: xr.DataArray,
    weights: xr.DataArray | None = None,
    price_var: str = "mid",
    per_ticker: bool = False,
    t_kind: Literal["datetime", "time", "timedelta", "seconds", "smidnight"] = "seconds",
    t_ref: dt.datetime | dt.time | None = None,
    extra_price: xr.DataArray | None = None,
    extra_price_time: dt.time | None = None,
    beta: str | None = "beta_spy",
    hedge_ticker: str = "SPY",
    mode: Literal["ret", "cps"] = "ret",
) -> xr.Dataset:
    """Returns a return profile based on signals for a single date."""
    # NOTE: the extra_prices logic only gives something sensible if extra_prices_time is after the end of the bars...
    if mode not in ["ret", "cps"]:
        raise ValueError(f"Invalid mode: {mode}")

    prices = bars.get_var(price_var).rename("price")

    if beta is not None:
        if hedge_ticker not in bars.ticker:
            raise ValueError(f"Bars do not have data for hedge_ticker {hedge_ticker}.")
        bars.load_var(beta)

    if t_ref is None:
        dt_ref = prices.indexes["time"][0]
    elif isinstance(t_ref, dt.time):
        assert bars.date is not None
        dt_ref = dt.datetime.combine(bars.date, t_ref)
    else:
        dt_ref = t_ref
    ref_prices = prices.sel(time=dt_ref).rename("ref_price")

    if extra_price is not None:
        assert extra_price_time is not None
        extra_dt = dt.datetime.combine(bars.date, extra_price_time)
        prices = xr.concat(
            [prices, extra_price.expand_dims({"time": [extra_dt]})], dim="time", join="left"
        )

    ds = xr.merge((prices, signs.rename("s"), ref_prices), join="inner")

    if mode == "ret":
        ds["ret"] = ds.eval(f"{ret_unit} * s * (price/ref_price - 1)")
        if beta is not None:
            mkt_rets = (
                prices.sel(ticker=hedge_ticker, drop=True)
                / ref_prices.sel(ticker=hedge_ticker, drop=True)
                - 1
            ).rename("mkt_ret")
            betas = bars.get_var(beta).rename("beta")
            ds = xr.merge((ds, betas, mkt_rets), join="inner")
            ds["ret_hedged"] = ds.eval(f"ret - {ret_unit} * s * beta * mkt_ret")
            ds = ds[["ret", "ret_hedged"]]
        else:
            ds = ds[["ret"]]
    else:
        ds["cps"] = ds.eval(f"{cps_unit} * s * (price - ref_price)")
        if beta is not None:
            mkt_rets = (
                prices.sel(ticker=hedge_ticker, drop=True)
                / ref_prices.sel(ticker=hedge_ticker, drop=True)
                - 1
            ).rename("mkt_ret")
            betas = bars.get_var(beta).rename("beta")
            ds = xr.merge((ds, betas, mkt_rets), join="inner")
            ds["cps_hedged"] = ds.eval(f"cps - {cps_unit} * ref_price * s * beta * mkt_ret")
            ds = ds[["cps", "cps_hedged"]]
        else:
            ds = ds[["cps"]]

    _adjust_time_helper(ds, t_kind, t_ref, ds.indexes["time"][0], bars.date)

    if per_ticker:
        return ds

    if weights is not None:
        mean_ret = ds.weighted(weights).mean(dim="ticker")
    else:
        mean_ret = ds.mean(dim="ticker")

    ret_count = ds[mode].count(dim="ticker").rename("n")

    return xr.merge((mean_ret, ret_count), join="exact")


def profile_fixed(
    bars_set: BarsSet,
    df: pd.DataFrame | pd.Series,
    alpha_col: str = "alpha",
    weight_col: str | None = None,
    beta_col: str | None = None,
    hedge_ticker: str = "SPY",
    mode: Literal["ret", "cps"] = "ret",
    extra_price: pd.Series | None = None,
    extra_price_time: dt.time | None = None,
    do_progress=True,
    **kvargs,
) -> xr.Dataset:
    """TODO

    df should have date and ticker index.

    The sign of alpha_col is used to determine the direction of the return for that ticker.
    If df is a Series then it is assumed to be the alpha, and alpha_col is ignored.

    If beta_col is given, then it is used to hedge the returns using the prices of hedge_ticker.

    If extra_price is given then it is used to add a final return to the profile, at extra_price_time.
    (Use for e.g. to add return to close at 16:05). It is provided as a separate Series because df
    might not have data for the hedge_ticker.

    All other arguments are passed as is to profile_fixed_single_date.
    """

    if isinstance(df, pd.Series):
        df = df.to_frame(alpha_col)

    dates = df.index.get_level_values("date").unique()

    if kvargs.get("t_kind") == "datetime":
        raise ValueError("t_kind=datetime not supported in return_profile_fixed for BarsBaseDir.")

    if mode not in ["ret", "cps"]:
        raise ValueError(f"Invalid mode: {mode}")

    dss = []
    extra_prices_date = None
    weights = None
    actual_dates = []
    for date in util.progress_iter(dates, desc="profile", disable=not do_progress):
        df_date = df.loc[date]
        signs = np.sign(df_date[alpha_col]).to_xarray()
        if weight_col is not None:
            weights = df_date[weight_col].to_xarray()
        if extra_price is not None:
            extra_prices_date = extra_price.loc[date].to_xarray()

        try:
            ds = profile_fixed_single_date(
                bars_set[date],
                signs,
                weights=weights,
                extra_price=extra_prices_date,
                extra_price_time=extra_price_time,
                beta=beta_col,
                mode=mode,
                hedge_ticker=hedge_ticker,
                **kvargs,
            )
        except Exception as e:
            warn(f"profile_fixed: Error on date {date}, will skip it. Error: {e}", stacklevel=2)
            continue
        actual_dates.append(date)
        dss.append(ds)
    ds = xr.concat(dss, dim=pd.Index(actual_dates, name="date"))
    return ds
