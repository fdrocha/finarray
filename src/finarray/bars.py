import datetime as dt
import os
import re
from collections.abc import Callable, Iterable, Sequence
from typing import (
    Any,
    Literal,
    Self,
    cast,
    overload,
)

import numpy as np
import pandas as pd
import xarray as xr

from . import util
from .sel_array import BarsSel, SelMixin, _apply_sels_dataset
from .where import Where, make_where

_re_undefined_var: re.Pattern[str] = re.compile(r"name '(?P<varname>.+)' is not defined")


class Bars(SelMixin):
    # List of variables that have a matching file on disk that we are in sync with
    _can_be_saved: bool = True

    def __init__(
        self,
        path: str,
        date: dt.date,
        _assume_dataset: None | xr.Dataset = None,
        parent_date_paths: list[str] | None = None,
    ):
        super().__init__()
        self.path = os.path.expanduser(path)
        self._parent_date_paths: list[str] = parent_date_paths or []
        self.date = date
        self._live_vars_on_disk: set[str] = set()
        self.dataset: xr.Dataset = (
            _assume_dataset.copy()
            if _assume_dataset is not None
            else util.get_empty_bars_dataset(self._find_coord_path())
        )

    def copy(self) -> Self:
        o = type(self)(
            self.path,
            self.date,
            _assume_dataset=self.dataset,
            parent_date_paths=self._parent_date_paths,
        )
        SelMixin._copy_to(self, o)
        o._live_vars_on_disk = self._live_vars_on_disk.copy()
        o._can_be_saved = self._can_be_saved
        return o

    @overload
    def tcast(self, t: util.TimeLike) -> dt.datetime: ...  # type: ignore[overload-overlap]

    @overload
    def tcast(self, t: Sequence[util.TimeLike]) -> list[dt.datetime]: ...

    def tcast(self, t):
        """If necessary, promote t to a datetime by adding the date of the bars object. If it is a str, try to parse it as a time first."""
        if hasattr(t, "dtype"):
            # We don't mess with numpy/pandas/xarray objects, just return them as is
            return t
        if isinstance(t, str):
            t = dt.time.fromisoformat(t)
        if isinstance(t, dt.time):
            return dt.datetime.combine(self.date, t)
        if isinstance(t, Iterable):
            return [self.tcast(t_) for t_ in t]
        return t

    def _sel_impl(self, sel: BarsSel) -> None:
        self.dataset = cast(xr.Dataset, sel.apply_dataset(self, self.dataset))
        self._can_be_saved = False

    def sel(self, *args, **kwargs):
        raise NotImplementedError(
            "Cannot use sel directly on Bars object. Use dataset.sel() if you really need it..."
        )

    def _all_paths(self) -> list[str]:
        return [self.path] + self._parent_date_paths

    def _find_coord_path(self) -> str:
        for p in self._all_paths():
            if all(
                os.path.exists(os.path.join(p, fn))
                for fn in (util.BARS_TICKER_FILENAME, util.BARS_TIME_FILENAME)
            ):
                return p
        raise FileNotFoundError(f"No ticker/time coordinate files found in chain for {self.path}")

    def _find_var_path(self, var: str) -> str | None:
        for p in self._all_paths():
            candidate = os.path.join(p, f"{var}.nc")
            if os.path.exists(candidate):
                return candidate
        return None

    def _get_var_path(self, var: str) -> str:
        return os.path.join(self.path, f"{var}.nc")

    def get_vars_available(self, local_only: bool = False) -> list[str]:
        vars: set[str] = set()
        paths = [self.path] if local_only else self._all_paths()
        for p in paths:
            if os.path.isdir(p):
                vars.update(fn.removesuffix(".nc") for fn in os.listdir(p) if fn.endswith(".nc"))
        return sorted(vars)

    def load_var(self, var: str, reload: bool = False) -> None:
        if not reload and var in self._live_vars_on_disk:
            return
        util.check(
            var in self.get_vars_available(),
            f"Variable {var} not available in chain for {self.path}",
        )
        var_path = self._find_var_path(var)
        assert var_path is not None  # guaranteed by the get_vars_available check above
        da = xr.load_dataarray(var_path)
        # An alias (see set_alias) is a symlink, so the array inside carries the
        # name of the variable it points at. Name it after what was asked for.
        da = da.rename(var)
        da = _apply_sels_dataset(self._active_sels, self, da)
        self.dataset = xr.merge([self.dataset, da], join="left")
        self._live_vars_on_disk.add(var)

    def save_var(self, var: str):
        # TODO better logic for checking if it can be saved
        util.check(self._can_be_saved, "BarsDir cannot be saved!")
        util.check(var in self.dataset.data_vars, f"Variable {var} not defined in dataset.")
        os.makedirs(self.path, exist_ok=True)
        self._live_vars_on_disk = self._live_vars_on_disk | {var}
        # For some reason the encoding part is not being used...
        self[var].to_netcdf(self._get_var_path(var), encoding={var: {"dtype": self[var].dtype}})

    def set_alias(self, alias_name: str, var: str, remove_old: bool = False):
        var_path = self._get_var_path(var)
        alias_path = self._get_var_path(alias_name)
        if remove_old and os.path.exists(alias_path):
            os.remove(alias_path)
        os.symlink(os.path.basename(var_path), alias_path)

    def create_var(self, dtype="float32", fill_value=np.nan):
        data = np.full(
            shape=(
                len(self.dataset.time),
                len(self.dataset.ticker),
            ),
            dtype=dtype,
            fill_value=fill_value,
        )
        da = xr.DataArray(
            data,
            dims=["time", "ticker"],
            coords=dict(time=self.dataset.time, ticker=self.dataset.ticker),
        )
        return da

    def load_vars(self, vars: Iterable[str] | str, reload: bool = False):
        if isinstance(vars, str):
            vars = [vars]
        for var in vars:
            self.load_var(var, reload)

    def save_vars(self, vars: Iterable[str]):
        for var in vars:
            self.save_var(var)

    def assign(self, **kwargs: str | Callable[["Bars"], xr.DataArray]) -> None:
        for name, value in kwargs.items():
            if isinstance(value, str):
                self[name] = self.eval(value)
            else:
                self[name] = value(self)

    @overload
    def __getitem__(self, v: str) -> xr.DataArray: ...
    @overload
    def __getitem__(self, v: list[str]) -> Self: ...

    def __getitem__(self, v):
        if isinstance(v, list):
            o = self.copy()
            o.dataset = self.dataset[v]
            o._live_vars_on_disk = o._live_vars_on_disk & set(v)
            return o
        else:
            return self.dataset[v]

    def __setitem__(self, key: Any, value):
        self.dataset[key] = value

    def __delitem__(self, key: Any):
        self._live_vars_on_disk.discard(key)
        del self.dataset[key]

    def __getattr__(self, v: str):
        ds = self.__dict__.get("dataset", None)
        if ds is not None and hasattr(ds, v):
            return getattr(ds, v)
        raise AttributeError(f"{self} has no '{v}' attribute.")

    # def __delattr__(self, name: str) -> None:
    #     if not self._is_initialized or hasattr(self, name):
    #         super().__delattr__(name)
    #     else:
    #         delattr(self.dataset, name)

    # def __setattr__(self, name: str, value):
    #     if not self._is_initialized or hasattr(self, name):
    #         super().__setattr__(name, value)
    #     else:
    #         setattr(self.dataset, name, value)

    def _load_var_from_exception(self, e):
        # The error looks like "name 'varname' is not defined", we need to extract varname

        m = _re_undefined_var.match(e.args[0])
        if not m:
            raise

        varname = m.group("varname")
        if varname not in self.get_vars_available():
            raise
        self.load_var(varname)

    def _run_with_autoload(self, callable):
        while True:
            try:
                return callable()
            except pd.errors.UndefinedVariableError as e:
                self._load_var_from_exception(e)
            except NameError as e:
                self._load_var_from_exception(e)

    def eval(self, expr: str, autoload: bool = True) -> xr.Dataset | xr.DataArray:
        if autoload:
            r = self._run_with_autoload(lambda: self.dataset.eval(expr))
        else:
            r = self.dataset.eval(expr)
        if isinstance(r, xr.Dataset):
            # This is hacky. if expr is just an expression, r is a DataArray and we don't want to overwrite the dataset
            self.dataset = r
        return r

    def get_var(self, var: str, autoload: bool = True) -> xr.DataArray:
        if autoload and var not in self.dataset.data_vars:
            self.load_var(var)
        return self[var]

    def get_vars(self, vars: list[str], autoload: bool = True) -> xr.Dataset:
        if autoload:
            for var in vars:
                if var not in self.dataset.data_vars:
                    self.load_var(var)
        return self.dataset[vars]  # type: ignore

    def get_value(self, var: str, autoload: bool = True) -> Any:
        """Returns the value of the variable, assuming it is a scalar.
        Throws an error if it is not a scalar."""
        return self.get_var(var, autoload).item()

    def get_values(self, vars: list[str], autoload: bool = True) -> list[Any]:
        return [self.get_value(var, autoload) for var in vars]

    def live_vars(self) -> list[str]:
        """Variables live in the dataset."""
        return [str(v) for v in self.dataset.data_vars]

    def loaded_vars(self) -> list[str]:
        return list(self._live_vars_on_disk)

    def get_created_vars(self) -> list[str]:
        """Live variables that were not loaded from disk."""
        return [v for v in self.live_vars() if v not in self.loaded_vars()]

    def save_created_vars(self) -> None:
        self.save_vars(self.get_created_vars())

    def add_daily_var(self, da: xr.DataArray) -> None:
        util.check_exact_dim_names(da, "ticker")
        fill_value = da.attrs.get("_FillValue", np.nan)
        da = da.reindex_like(self.dataset.ticker, fill_value=fill_value).astype(da.dtype)
        self.dataset[da.name] = da

    def lookup_indirect(
        self,
        ticker_times_r: pd.Series,
        method: Literal["bfill", "ffill", "nearest"] | None = None,
    ) -> xr.Dataset:
        # ticker_times is a Series with a 'ticker' index and a 'time' value
        # IMPORTANT: this assumes that ticker_times is already restricted to the tickers we have data for, or things might break silently
        tickers = ticker_times_r.reset_index().ticker

        # We use indexing by DataArray to select the values
        da_tickers = xr.DataArray(data=tickers, dims=["ticker2"], coords=dict(ticker2=tickers))
        da_times = xr.DataArray(data=ticker_times_r, dims=["ticker2"], coords=dict(ticker2=tickers))

        ds = (
            self.dataset.sel(ticker=da_tickers, time=da_times, method=method)
            .drop_vars(["ticker", "time"])
            .rename(dict(ticker2="ticker"))
        )
        return ds

    def _restrict_other(self, other: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        # restricts 'other' to the tickers we have data for
        mask = other.index.isin(self.dataset.ticker.values, level="ticker")
        return other[mask]

    def apply_where(self, where: Where) -> Self:
        """Restrict to the tickers matching `where`, loading what it needs."""
        return self.sel_ticker(where.mask(self))

    def sel_where(self, extra_tickers: Iterable[str] | None = None, **constraints) -> Self:
        """Restrict to tickers matching constraints on ticker-only variables.

        Each keyword names a bars variable; the value is either a `(min, max)`
        range (either bound `None` for unbounded) or a single value to compare
        for equality. Tickers in `extra_tickers` are kept regardless.

            bars.sel_where(price=(2, 2500), adv=(1e6, None), extra_tickers=["SPY"])
        """
        return self.apply_where(make_where(extra_tickers, **constraints))

    def reindex_array(self, da: xr.DataArray) -> xr.DataArray:
        return da.reindex({"ticker": self.ticker, "time": self.time})

    def has_var(self, var: str, local_only: bool = False) -> bool:
        return var in self.get_vars_available(local_only)

    def merge_df_asof(self, df: pd.DataFrame) -> None:
        """
        Merges a dataframe with the bars dataset using merge_asof.
        The dataframe is assumed to have a MultiIndex with ticker and time.
        """
        target_tickers = self.dataset.ticker.values
        new_vars = df.columns
        df = df.loc[df.index.get_level_values("ticker").isin(target_tickers)].reset_index()
        df.sort_values("time", inplace=True)

        target = pd.MultiIndex.from_product(
            [target_tickers, self.dataset.time.to_index()], names=["ticker", "time"]
        ).to_frame(index=False)

        target.sort_values("time", inplace=True)

        joined = pd.merge_asof(
            target,
            df,
            on="time",
            by="ticker",
            direction="backward",
        )

        new_ds = joined.set_index(["ticker", "time"]).to_xarray()
        for var in new_vars:
            self[var] = new_ds[var]
