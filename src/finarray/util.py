import datetime as dt
import os
import sys
import warnings
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import (
    Any,
    NoReturn,
    TypeVar,
    cast,
    overload,
)

import pandas as pd
import xarray as xr

T = TypeVar("T")


class FinArrayError(Exception):
    """Base class for errors raised explicitly by finarray."""


BARS_TICKER_FILENAME = "ticker.csv"
BARS_TIME_FILENAME = "time.csv"

TimeLike = str | dt.time | dt.datetime
DateLike = str | dt.date


RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _get_warnings_formatter(prefix, use_colors):
    if use_colors:
        prefix = f"{RED}{BOLD}{prefix}"
        suffix = RESET
    else:
        suffix = ""

    def formatter(message, category, filename, lineno, line=None):
        return f"{prefix}: {message} [{os.path.basename(filename)}:{lineno}]{suffix}\n"

    return formatter


def set_warnings_format(prefix="finarray"):
    warnings.formatwarning = _get_warnings_formatter(prefix, sys.stdout.isatty())


def warn(msg) -> None:
    warnings.warn(msg, stacklevel=2)


def error(msg: str) -> NoReturn:
    """Raise a FinArrayError.

    Note for callers ported from the original in-monorepo version: this used to
    warn and call ``sys.exit(1)``. Exiting the interpreter is not something a
    library should do, so it now raises instead. CLI callers that relied on the
    old behaviour should catch FinArrayError and exit themselves.
    """
    raise FinArrayError(msg)


@overload
def dcast(date: DateLike) -> dt.date: ...  # type: ignore[overload-overlap]


@overload
def dcast(date: Iterable[DateLike]) -> list[dt.date]: ...


def dcast(date):
    """Promote a date-like (ISO string, date, or iterable of those) to dt.date."""
    if isinstance(date, str):
        return dt.date.fromisoformat(date)
    elif isinstance(date, Iterable):
        return [dcast(d) for d in date]
    else:
        return date


def _simple_progress(
    iterable: Iterable[T], desc: str | None = None, total: int | None = None
) -> Iterable[T]:
    """A counter on stderr, for when tqdm is not installed."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    show = sys.stderr.isatty()
    label = f"{desc}: " if desc else ""
    for i, item in enumerate(iterable, 1):
        if show:
            sys.stderr.write(f"\r{label}{i}/{total}" if total else f"\r{label}{i}")
            sys.stderr.flush()
        yield item
    if show:
        sys.stderr.write("\n")
        sys.stderr.flush()


def progress_iter(
    iterable: Iterable[T],
    desc: str | None = None,
    disable: bool = False,
    total: int | None = None,
) -> Iterable[T]:
    """Wrap an iterable in a progress display.

    Uses tqdm when it is installed -- which is what you want in a notebook --
    and falls back to a plain stderr counter otherwise. tqdm is not a dependency
    of finarray.
    """
    if disable:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _simple_progress(iterable, desc, total)
    return tqdm(iterable, desc=desc, total=total)


def load_csv(
    path: str,
    time_col: str = "time",
    ticker_col: str = "ticker",
    date: str | dt.date | None = None,
    use32: bool = True,
    rename_func: Callable[[str], str] | None = None,
) -> xr.Dataset:
    df = pd.read_csv(path)

    check(
        time_col in df.columns,
        f"time_col {time_col} not in {path}: columns are {df.columns}",
    )
    check(
        ticker_col in df.columns,
        f"ticker_col {ticker_col} not in {path}: columns are {df.columns}",
    )

    if date is not None:
        # TODO : Only do this when it's necessary (i.e., time col is not already a datetime)
        if isinstance(date, dt.date):
            date = date.isoformat()
        df[time_col] = pd.to_datetime(date + " " + df[time_col])

    df = df.set_index([time_col, ticker_col])
    if use32:
        for col in df.columns:
            if df[col].dtype == "float64":
                df[col] = df[col].astype("float32")
            elif df[col].dtype == "int64":
                df[col] = df[col].astype("int32")
    df.sort_index(inplace=True)

    if rename_func is not None:
        df.columns = [rename_func(col) for col in df.columns]

    ds = df.to_xarray()

    if date is not None:
        ds.attrs["date"] = dt.date.fromisoformat(date)

    return ds


def to_frdir(ds: xr.Dataset, base_path: str, date: None | dt.date = None) -> None:
    # TODO: the order here seems weird, and probably shouldn't matter...
    if tuple(ds.coords) != ("time", "ticker"):
        raise ValueError(f"Expected coords to be 'time' and 'ticker', got {tuple(ds.coords)}")

    if date is None:
        check("date" in ds.attrs, "date not provided and not in ds.attrs")
        date_: dt.date = ds.attrs["date"]
    else:
        date_ = date

    path = os.path.join(base_path, date_.isoformat())
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists!")
        # TODO option to delete and overwrite
    os.mkdir(path)

    time_fn = os.path.join(path, BARS_TIME_FILENAME)
    ticker_fn = os.path.join(path, BARS_TICKER_FILENAME)

    ds.time.to_series().reset_index(drop=True).to_csv(time_fn, index=False)
    ds.ticker.to_series().reset_index(drop=True).to_csv(ticker_fn, index=False)

    for var in ds.data_vars:
        ds[var].to_netcdf(os.path.join(path, f"{var}.nc"))


def intersect(xs: Sequence[T], ys: Sequence[T]) -> list[T]:
    y_set = set(ys)
    return [x for x in xs if x in y_set]


def get_dims(
    da: xr.DataArray | xr.Dataset | pd.DataFrame | pd.Series | pd.Index,
) -> tuple[Hashable, ...]:
    if isinstance(da, pd.Index):
        return tuple(da.names)
    elif isinstance(da, (pd.DataFrame, pd.Series)):
        return tuple(da.index.names)
    else:
        return tuple(da.dims)


def check(condition: bool, msg: str) -> None:
    if not condition:
        raise ValueError(msg)


def check_first_dims(
    da: xr.DataArray | xr.Dataset | pd.DataFrame | pd.Series | pd.Index,
    *expected_dims: str,
) -> None:
    dims = get_dims(da)[: len(expected_dims)]
    if dims != expected_dims:
        raise ValueError(f"Expected first dimensions to be {expected_dims}, got {dims}")


def check_exact_dim_names(
    da: xr.DataArray | xr.Dataset | pd.DataFrame | pd.Series | pd.Index,
    *expected_dims: str,
) -> None:
    dims = get_dims(da)
    if dims != expected_dims:
        raise ValueError(f"Expected dimensions to be {expected_dims}, got {dims}")


def check_has_dims(
    da: xr.DataArray | xr.Dataset | pd.DataFrame | pd.Series | pd.Index,
    *expected_dims: str,
) -> None:
    dims = set(get_dims(da))
    edims = set(expected_dims)
    missing_dims = edims - dims  # type: ignore
    if missing_dims:
        raise ValueError(f"Expected dimensions to contain {expected_dims}, missing {missing_dims}")


def check_has_vars(da: xr.Dataset, *expected_vars: str) -> None:
    missing_vars = set(expected_vars) - set(da.data_vars)
    if missing_vars:
        raise ValueError(f"Dataset missing variables: {missing_vars}")


def index_loc(idx: pd.Index, key: Any) -> pd.Index:
    loc = idx.get_loc(key)
    if isinstance(loc, int):
        # Annoyingly, if there is a single element, get_loc returns an int instead of a slice
        # This makes it more consistent
        loc = slice(loc, loc + 1)
    return idx[loc]


def get_empty_bars_dataset(path: str) -> xr.Dataset:
    tickers = pd.Index(
        pd.read_csv(filepath_or_buffer=os.path.join(path, BARS_TICKER_FILENAME)).ticker
    )
    times = pd.Index(pd.to_datetime(pd.read_csv(os.path.join(path, BARS_TIME_FILENAME)).time))

    dataset = xr.Dataset(coords={"time": times, "ticker": tickers})
    return dataset


def level_time_to_date(df: pd.Series | pd.DataFrame):
    if isinstance(df.index, pd.MultiIndex):
        i = df.index.names.index("time")
        time_level = cast(pd.DatetimeIndex, df.index.levels[i])
        df.index = df.index.set_levels(list(time_level.date), level=i)  # type: ignore[call-overload]
        df.index = df.index.rename("date", level=i)  # type: ignore[call-overload]
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.date  # type: ignore
        df.index = df.index.rename("date")  # type: ignore
    else:
        raise ValueError(f"Expected a MultiIndex or DatetimeIndex, got {type(df.index)}")
