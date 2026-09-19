import datetime as dt
import os
from collections.abc import Callable, Iterable, Iterator
from typing import Literal, Self, overload
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr

from . import util
from .bars import Bars
from .sel_array import BarsSel, SelMixin, _apply_sels_bars
from .where import Where, make_where

_PARENT_LINK_NAME = "PARENT"


def _resolve_parent_chain(base_path: str) -> list[str]:
    link = os.path.join(base_path, _PARENT_LINK_NAME)
    if not os.path.exists(link):
        return []
    parent = os.path.realpath(link)
    return [parent] + _resolve_parent_chain(parent)


def create_child_bars(child_path: str, parent_path: str) -> None:
    """
    Create a child bars directory that links to a parent directory.
    """
    child_path = os.path.expanduser(child_path)
    parent_path = os.path.expanduser(parent_path)
    util.check(not os.path.exists(child_path), f"Child path {child_path} already exists")
    util.check(
        os.path.isdir(parent_path),
        f"Parent path {parent_path} does not exist or is not a directory",
    )
    os.makedirs(child_path, exist_ok=False)
    os.symlink(parent_path, os.path.join(child_path, _PARENT_LINK_NAME))


class BarsSet(SelMixin):
    _restrict_dates: tuple[dt.date, ...] | None = None

    def __init__(
        self,
        base_path: str,
        restrict_dates: None | Iterable[util.DateLike] = None,
        preload_vars: None | Iterable[str] = None,
        where: None | Where = None,
    ):
        super().__init__()
        self.base_path = os.path.expanduser(base_path)
        self._parent_base_paths: list[str] = _resolve_parent_chain(self.base_path)
        self._wheres: list[Where] = [where] if where is not None else []

        self._loaded_dates: dict[dt.date, Bars] = {}
        self._preload_vars: list[str] = list(preload_vars) if preload_vars is not None else []

        if restrict_dates is not None:
            dates_available = self.dates_available()
            dates = []
            for date in util.dcast(restrict_dates):
                if date not in dates_available:
                    warn(
                        f"No data available for date={date} in {self.base_path}, will skip",
                        stacklevel=2,
                    )
                    continue
                self.load_date(date)
                dates.append(date)
            self._restrict_dates = tuple(sorted(dates))

    def dates_available(self, check_parents: bool = True) -> list[dt.date]:
        """Dates this set can load, inherited dates included.

        Pass ``check_parents=False`` for only the dates with their own directory
        in ``base_path``.
        """
        if self._restrict_dates is not None:
            return list(self._restrict_dates)
        dates: set[dt.date] = set()
        if not check_parents:
            bases = [self.base_path]
        else:
            bases = [self.base_path] + self._parent_base_paths
        for base in bases:
            for dirname in os.listdir(base):
                try:
                    dates.add(dt.date.fromisoformat(dirname))
                except ValueError:
                    continue
        return sorted(d for d in dates if self.has_date(d, check_parents))

    def copy(self) -> Self:
        o = type(self)(self.base_path)
        # This is hacky, but it will do for now
        SelMixin._copy_to(self, o)
        o._loaded_dates = {date: bars.copy() for date, bars in self._loaded_dates.items()}
        o._restrict_dates = self._restrict_dates
        o._preload_vars = self._preload_vars
        o._wheres = list(self._wheres)
        return o

    def __copy__(self) -> Self:
        return self.copy()

    def dates_loaded(self) -> list[dt.date]:
        return list(self._loaded_dates.keys())

    def _get_date_dir(self, date: dt.date) -> str:
        return os.path.join(self.base_path, date.isoformat())

    def has_date(self, date: dt.date, check_parents: bool = True) -> bool:
        if self._restrict_dates is not None:
            return date in self._restrict_dates
        if not check_parents:
            bases = [self.base_path]
        else:
            bases = [self.base_path] + self._parent_base_paths
        for base in bases:
            path = os.path.join(base, date.isoformat())
            if os.path.exists(path) and os.path.isdir(path):
                return True
        return False

    def _sel_impl(self, sel: BarsSel) -> None:
        self._loaded_dates = {date: bars._add_sel(sel) for date, bars in self._loaded_dates.items()}

    def load_date(self, date: dt.date, reload: bool = False) -> None:
        if self._restrict_dates:
            if date not in self._restrict_dates:
                raise ValueError(f"Date {date} not in restrict_dates")
            return
        if not reload and date in self._loaded_dates:
            return
        util.check(
            self.has_date(date),
            f"No data available for date={date} in {self.base_path} or its parents",
        )
        parent_date_dirs = [
            os.path.join(p, date.isoformat())
            for p in self._parent_base_paths
            if os.path.isdir(os.path.join(p, date.isoformat()))
        ]
        bars_dir = Bars(self._get_date_dir(date), date, parent_date_paths=parent_date_dirs)

        bars_dir = _apply_sels_bars(self._active_sels, bars_dir)
        for where in self._wheres:
            bars_dir = bars_dir.apply_where(where)

        self._loaded_dates[date] = bars_dir
        if self._preload_vars:
            bars_dir.load_vars(self._preload_vars)

    def load_dates(
        self,
        dates: Iterable[dt.date] | None = None,
        all_available: bool = False,
        reload: bool = False,
        errors: Literal["raise", "warn", "ignore"] = "warn",
    ):
        if all_available:
            if dates is not None:
                raise ValueError("Cannot specify dates if all_available is True")
            dates = self.dates_available()
        else:
            if dates is None:
                raise ValueError("Must specify dates if all_available is False")
        for date in dates:
            try:
                self.load_date(date, reload)
            except Exception as e:
                if errors == "raise":
                    raise e
                elif errors == "warn":
                    warn(
                        f"BarSet.load_dates: Error in load_date for date={date}: {e}", stacklevel=2
                    )

    def reload(self):
        for date in self.dates_loaded():
            self.load_date(date, reload=True)

    def load_var(self, var: str, reload: bool = False) -> None:
        for bars in self._loaded_dates.values():
            bars.load_var(var, reload)

    def load_vars(self, vars: Iterable[str] | str, reload: bool = False) -> None:
        for bars in self._loaded_dates.values():
            bars.load_vars(vars, reload)

    def delete_var(self, var: str, dates: Iterable[dt.date] | None = None) -> list[dt.date]:
        """Delete a variable from every given date (default: all available).

        Only deletes from this base directory, never from a parent. Returns the
        dates a file was actually removed from.
        """
        removed = []
        for date in self._dates_or_available(dates):
            if self[date].delete_var(var, missing_ok=True):
                removed.append(date)
        return removed

    def _dates_or_available(self, dates: Iterable[dt.date] | None) -> list[dt.date]:
        return list(dates) if dates is not None else self.dates_available()

    def _parse_date_or_index(self, date_or_index: dt.date | str | int) -> dt.date:
        if isinstance(date_or_index, str):
            return dt.date.fromisoformat(date_or_index)
        elif isinstance(date_or_index, int):
            return self.dates_available()[date_or_index]
        elif isinstance(date_or_index, dt.date):
            return date_or_index
        else:
            raise ValueError(f"Invalid type for date_or_index: {type(date_or_index)}")

    def __getitem__(self, date_or_index: dt.date | str | int) -> Bars:
        date = self._parse_date_or_index(date_or_index)
        if date not in self._loaded_dates:
            self.load_date(date)
        return self._loaded_dates[date]

    def bars_valid_dates(
        self, dates: Iterable[dt.date], only_loaded: bool = False
    ) -> Iterable[tuple[dt.date, Bars]]:
        if only_loaded:
            return ((date, self[date]) for date in dates if date in self._loaded_dates)
        else:
            return ((date, self[date]) for date in dates if self.has_date(date))

    @overload
    def restrict_other(self, other: pd.Series) -> pd.Series: ...

    @overload
    def restrict_other(self, other: pd.DataFrame) -> pd.DataFrame: ...

    def restrict_other(self, other):
        """Expects a DataFrame or Series with a ('date', 'ticker') index.
        Returns its restriction to the ones that are in BarsSet."""
        # We need this because we want to use have_data.loc[date, ticker]
        util.check_first_dims(other, "date", "ticker")

        # We will first define a boolean Series to use to index "other", set to False by default
        have_data = pd.Series(False, index=other.index)
        # Then we populate it with True where appropriate
        other_dates = other.index.get_level_values("date").unique()
        for date, day_bars in self.bars_valid_dates(other_dates):
            other_tickers = other.loc[date].index.get_level_values("ticker")
            bars_tickers = day_bars.ticker.values
            common_tickers = np.intersect1d(other_tickers, bars_tickers)
            if common_tickers.size > 0:
                have_data.loc[date, common_tickers] = True

        return other.loc[have_data]

    def apply_where(self, where: Where) -> Self:
        """Restrict every date to the tickers matching `where`.

        The filter is remembered, so dates loaded later are filtered too.
        """
        o = self.copy()
        o._wheres.append(where)
        o._loaded_dates = {date: bars.apply_where(where) for date, bars in o._loaded_dates.items()}
        return o

    def sel_where(self, extra_tickers: Iterable[str] | None = None, **constraints) -> Self:
        """Restrict every date to tickers matching constraints on ticker-only variables.

        See `Bars.sel_where`.
        """
        return self.apply_where(make_where(extra_tickers, **constraints))

    def _get_var_list(self, var: str):
        return [self[date].get_var(var) for date in self.dates_loaded()]

    def _fixup_index(self, xs: pd.Series | pd.DataFrame) -> None:
        # Turns a "time" level into a "date" level when appropriate
        if self._time_is_scalar:
            util.level_time_to_date(xs)

    def _dates(self, dates: Iterable[dt.date] | None) -> list[dt.date]:
        if dates is None:
            return self.dates_loaded()
        else:
            return list(dates)

    # TODO: should accept not just vars but also expressions. new name?
    def cat_var(self, var: str, dates: Iterable[dt.date] | None = None) -> pd.Series:
        """Concatenate the variable across all dates and returns a Series with date in the index.
        Takes into account active sels."""
        dates = self._dates(dates)
        if self._time_is_scalar and self._ticker_is_scalar:
            return self.cat_vars([var], dates=dates)[var]
        series_list = [self[date].get_var(var).to_series() for date in dates]
        if self._time_is_scalar:
            series = pd.concat(series_list, keys=dates, names=["date"])
        else:
            series = pd.concat(series_list)
        return series
        # TODO: removed this, not sure if it breaks some use case
        # self._fixup_index(series)

    def cat_vars(self, vars: list[str], dates: Iterable[dt.date] | None = None) -> pd.DataFrame:
        """Concatenate multiple variables and return a DataFrame with date in the index."""
        dates = self._dates(dates)
        if self._time_is_scalar and self._ticker_is_scalar:
            # FIXME: ugly special case logic, there must be a better way
            dfs = [self[date].get_vars(vars).to_pandas().to_frame().T for date in dates]  # type: ignore
            df = pd.concat(dfs, keys=dates, names=["date"])
            df.index = df.index.droplevel(1)
        else:
            dfs = [self[date].get_vars(vars).to_pandas() for date in dates]
            if self._time_is_scalar:
                df = pd.concat(dfs, keys=dates, names=["date"])
            else:
                df = pd.concat(dfs)
        # TODO: same as above
        # self._fixup_index(df)

        # For some annoying reason, if ticker is a scalar, xarray will return a DataFrame with a "ticker" column
        # (otherwise it will be a level in the index)
        # We don't want that column, so we drop it if it's there
        if "ticker" in df.columns:
            df = df.drop("ticker", axis=1)
        # similarly for time:
        if "time" in df.columns:
            df = df.drop("time", axis=1)
        return df

    def attach_to_df_by_tickers(self, df: pd.DataFrame, vars: str | Iterable[str]) -> pd.DataFrame:
        """
        Adds data from bars to df by looking it up based on "date" and "ticker" levels in df.index.
        This only works if sel_time has been used to select a single time.

        Note that rows that for tickers that are not in the corresponding day's bars will be dropped silently.
        """
        util.check_has_dims(df, "date", "ticker")
        util.check(
            self._time_is_scalar,
            "Only supported for now when a single time is selected (so time is a scalar)",
        )

        # This computes a bunch of dfs for each date and then concatenates them, could be done in a more efficient way
        var_list = [vars] if isinstance(vars, str) else list(vars)
        dates = df.index.get_level_values("date").unique()
        day_dfs: list[pd.DataFrame] = []

        for date, day_bars in self.bars_valid_dates(dates):
            day_bars.load_vars(var_list)
            day_df_in = df.xs(date, level="date", drop_level=False)
            day_df_in_r = day_bars._restrict_other(day_df_in)
            tickers = day_df_in_r.index.get_level_values("ticker")
            day_df_new = day_bars.sel_ticker(ticker=tickers)[var_list].to_dataframe()
            day_df_new.index = day_df_in_r.index
            day_df_out = pd.concat((day_df_in_r, day_df_new), axis=1)
            day_dfs.append(day_df_out)
        df_out = pd.concat(day_dfs, axis=0)
        return df_out

    def attach_to_df_by_ticker_time(
        self,
        df: pd.DataFrame,
        vars: str | list[str],
        time_column: str = "time",
        method: Literal["bfill", "ffill", "nearest"] | None = None,
    ) -> pd.DataFrame:
        """
        Adds data from bars to df by looking it up based on "date" and "ticker" levels in df.index and the times in the `time_column` column.

        Note that rows that for tickers that are not in the corresponding day's bars will be dropped silently.
        """
        util.check_has_dims(df, "date", "ticker")
        util.check(time_column in df.columns, f"df has no '{time_column}' column")

        dates = df.index.get_level_values("date").unique()
        day_dfs: list[pd.DataFrame] = []
        for date, day_bars in self.bars_valid_dates(dates):
            day_bars.load_vars(vars)
            day_df_in = df.xs(date, level="date", drop_level=False)
            # It's important to do this restriction before calling _lookup_indirect!
            day_df_in_r = day_bars._restrict_other(day_df_in)
            ds_r = day_bars.lookup_indirect(day_df_in_r[time_column], method)
            df_new = ds_r[vars].to_dataframe()
            df_new.index = day_df_in_r.index
            day_df_out = pd.concat((day_df_in_r, df_new), axis=1)
            day_dfs.append(day_df_out)
        df_out = pd.concat(day_dfs)
        return df_out

    def mapcat(
        self,
        func: Callable[..., pd.DataFrame | None],
        add_date: bool = False,
        split_datetime: bool = False,
        dates: None | Iterable[dt.date] = None,
        on_errors: Literal["raise", "warn", "ignore"] = "warn",
        progress: bool = False,
        *func_args,
        **func_kwargs,
    ) -> pd.DataFrame:
        """
        func should be a function that takes a Bars object and returns a pd.DataFrame or None (meaning the corresponding date is skipped). If an exception is raised, it is caught and date is skipped.

        mapcat runs func on each Bars object (on all loaded dates) and concatenates the results together.

        If add_date is True, the output will have a "date" level in the index.
        If split_datetime is True, it is assumed output of func has a "time" level in the index of datetime type,
        and this is split into separate "date" and "time" levels.
        """
        if split_datetime and add_date:
            raise ValueError("Cannot use both add_time and split_datetime")
        dfs: list[pd.DataFrame] = []
        dates_ = self._dates(dates)
        good_dates = []
        # Keep `dates_` as a list: the progress wrapper may be a generator, and
        # the error below needs the count after it has been consumed.
        iter_dates: Iterable[dt.date] = dates_
        if progress:
            iter_dates = util.progress_iter(dates_, desc="mapcat")
        for date in iter_dates:
            try:
                day_bars = self[date]
                df = func(day_bars, *func_args, **func_kwargs)
            except Exception as e:
                if on_errors == "raise":
                    raise e
                elif on_errors == "warn":
                    warn(f"BarSet.map_cat: Error in func for date={date}: {e}", stacklevel=2)
                continue
            if df is not None:
                dfs.append(df)
                good_dates.append(date)
            elif on_errors == "warn":
                warn(f"BarSet.map_cat: Skipping date={date}, func returned None.", stacklevel=2)
        if not dfs:
            raise ValueError(
                f"mapcat produced nothing from {len(dates_)} date(s): every "
                "date either raised or returned None. Re-run with on_errors='raise' "
                "to see the underlying error."
            )
        if add_date:
            df_out = pd.concat(dfs, keys=good_dates, names=["date"])
        else:
            df_out = pd.concat(dfs)
        if split_datetime:
            levels = list(df_out.index.names)
            assert "time" in levels
            df_out = df_out.reset_index()
            df_out["date"] = df_out["time"].dt.date
            df_out["time"] = df_out["time"].dt.time
            df_out = df_out.set_index(["date"] + levels)
        return df_out

    def iter_bars(self) -> Iterator[tuple[dt.date, Bars]]:
        for date in self.dates_loaded():
            yield date, self[date]

    def assign_expr(self, name: str, expr: str):
        """
        Assign an expression to a variable.
        """
        for _, bd in self.iter_bars():
            bd[name] = bd.eval(expr)

    def assign_func(self, name: str, func: Callable[[Bars], pd.Series], *args, **kwargs):
        """
        Assign a function to a variable.
        """
        for _, bd in self.iter_bars():
            bd[name] = func(bd, *args, **kwargs)

    def assign(self, **kwargs: str | Callable[[Bars], xr.DataArray]) -> None:
        for _, bd in self.iter_bars():
            bd.assign(**kwargs)


def bars_getter(base_path: str) -> Callable[..., BarsSet]:

    def getter(
        restrict_dates: None | Iterable[util.DateLike] = None,
        preload_vars: None | Iterable[str] = None,
        *args,
        **kwargs,
    ) -> BarsSet:
        return BarsSet(base_path, restrict_dates, preload_vars, *args, **kwargs)

    return getter
