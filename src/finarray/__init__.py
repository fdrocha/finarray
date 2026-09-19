"""finarray: xarray-backed intraday bars.

A "bars" directory holds one sub-directory per date. Each date directory has
``ticker.csv`` and ``time.csv`` giving the two coordinates, plus one netCDF file
per variable. `Bars` is one date, `BarsSet` is the whole directory; variables are
loaded from disk lazily, as expressions and selections ask for them.
"""

from . import profile, util
from .bars import Bars
from .bars_set import BarsSet, bars_getter, create_child_bars
from .sel_array import TickerSel, TimeSel, TimeSliceSel
from .util import DateLike, FinArrayError, TimeLike, load_csv, to_frdir
from .where import Where, make_where

__all__ = [
    "Bars",
    "BarsSet",
    "DateLike",
    "FinArrayError",
    "TickerSel",
    "TimeLike",
    "TimeSel",
    "TimeSliceSel",
    "Where",
    "bars_getter",
    "create_child_bars",
    "load_csv",
    "make_where",
    "profile",
    "to_frdir",
    "util",
]

__version__ = "0.3.0"
