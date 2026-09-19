from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Self, TypeVar

import xarray as xr

from . import util

if TYPE_CHECKING:
    from .bars import Bars


DS_TYPE = TypeVar("DS_TYPE", bound=xr.Dataset | xr.DataArray)


class BarsSel(ABC):
    @abstractmethod
    def apply_dataset(self, bars: Bars, ds: DS_TYPE) -> DS_TYPE:
        pass


class TimeSel(BarsSel):
    time: util.TimeLike | Sequence[util.TimeLike]
    method: Literal["nearest", "bfill", "ffill"] | None
    tolerance: float | None
    drop: bool

    def __init__(
        self,
        time: util.TimeLike | Sequence[util.TimeLike],
        method: Literal["nearest", "bfill", "ffill"] | None = None,
        tolerance: float | None = None,
        drop: None | bool = None,
    ):
        self.time = time
        self.method = method
        self.tolerance = tolerance
        if drop is None:
            self.drop = isinstance(time, util.TimeLike)
        else:
            self.drop = drop

    def __repr__(self):
        return f"TimeSel(time={self.time}, method={self.method}, tolerance={self.tolerance}, drop={self.drop})"

    def apply_dataset(self, bars: Bars, ds: DS_TYPE) -> DS_TYPE:
        if "time" not in ds.dims:
            # In case we are called on ticker only
            return ds
        time = bars.tcast(self.time)
        return ds.sel(time=time, method=self.method, tolerance=self.tolerance, drop=self.drop)  # type: ignore


class TickerSel(BarsSel):
    ticker: str | Sequence[str]
    drop: bool

    def __init__(self, ticker: str | Sequence[str], drop: None | bool = None):
        self.ticker = ticker
        if drop is None:
            # We drop the dimension if we select a single ticker
            self.drop = isinstance(ticker, str)
        else:
            self.drop = drop

    def __repr__(self):
        return f"TickerSel(ticker={self.ticker}, drop={self.drop})"

    def apply_dataset(self, bars: Bars, ds: DS_TYPE) -> DS_TYPE:
        return ds.sel(ticker=self.ticker, drop=self.drop)  # type: ignore


class TimeSliceSel(BarsSel):
    t_start: util.TimeLike | None
    t_end: util.TimeLike | None

    def __init__(
        self,
        t_start: util.TimeLike | None = None,
        t_end: util.TimeLike | None = None,
    ):
        self.t_start = t_start
        self.t_end = t_end

    def __repr__(self):
        return f"TimeSliceSel(t_start={self.t_start}, t_end={self.t_end})"

    def apply_dataset(self, bars: Bars, ds: DS_TYPE) -> DS_TYPE:
        if "time" not in ds.dims:
            # In case we are called on ticker only variable
            return ds
        start = bars.tcast(self.t_start) if self.t_start is not None else None
        end = bars.tcast(self.t_end) if self.t_end is not None else None
        return ds.sel(time=slice(start, end))  # type: ignore


class SelMixin(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._active_sels: list[BarsSel] = []
        self._time_is_scalar: bool = False
        self._ticker_is_scalar: bool = False

    def _copy_to(self, obj: SelMixin) -> None:
        obj._active_sels = self._active_sels.copy()
        obj._time_is_scalar = self._time_is_scalar
        obj._ticker_is_scalar = self._ticker_is_scalar

    @abstractmethod
    def copy(self) -> Self:
        pass

    @abstractmethod
    def _sel_impl(self, sel: BarsSel) -> None:
        pass

    def _add_sel(self, sel: BarsSel) -> Self:
        o = self.copy()
        o._active_sels.append(sel)
        o._sel_impl(sel)
        return o

    def sel_time(
        self,
        time: util.TimeLike | Sequence[util.TimeLike],
        method=None,
        tolerance=None,
        drop: bool | None = None,
    ) -> Self:
        # If we get a scalar time, we promote it to a list of one element and make a note of it
        # This is helpful to handle the scalar time case in a nicer way when concatenating stuff from different dates
        if isinstance(time, util.TimeLike):
            tscalar = True
            # time_arg = [time]
        else:
            tscalar = self._time_is_scalar
            # time_arg = time
        o = self._add_sel(TimeSel(time, method, tolerance, drop))
        o._time_is_scalar = tscalar
        return o

    def sel_ticker(self, ticker) -> Self:
        o = self._add_sel(TickerSel(ticker))
        if isinstance(ticker, str):
            o._ticker_is_scalar = True
        else:
            o._ticker_is_scalar = self._ticker_is_scalar
        return o

    def sel_time_slice(
        self, t_start: util.TimeLike | None = None, t_end: util.TimeLike | None = None
    ) -> Self:
        return self._add_sel(TimeSliceSel(t_start, t_end))


def _apply_sels_dataset(sels: list[BarsSel], bars: Bars, ds: DS_TYPE) -> DS_TYPE:
    for sel in sels:
        ds = sel.apply_dataset(bars, ds)
    return ds


def _apply_sels_bars(sels: list[BarsSel], bars: Bars) -> Bars:
    for sel in sels:
        bars = bars._add_sel(sel)
    return bars
