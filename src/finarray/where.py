"""Declarative ticker filters.

A :class:`Where` describes a subset of tickers in terms of *daily* (ticker-only)
bars variables: ``price`` between 2 and 2500, ``adv`` of at least a million, a
particular listing exchange, and so on.  It deliberately knows nothing about
which variables you happen to have -- the names are whatever is on disk.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import xarray as xr

from . import util

if TYPE_CHECKING:
    from .bars import Bars


Bound = float | int | None
#: Either a ``(min, max)`` range -- with either bound ``None`` for unbounded --
#: or a single value to test for equality.
Constraint = tuple[Bound, Bound] | float | int | str | bool


def _is_range(constraint: Constraint) -> bool:
    return isinstance(constraint, tuple)


@dataclass(frozen=True)
class Where:
    """A ticker filter built from constraints on ticker-only bars variables.

    Every variable named in ``constraints`` must exist in the bars (it is loaded
    on demand) and must have ``ticker`` as its only dimension.  Constraints are
    combined with AND; ``extra_tickers`` are then OR-ed back in, so they survive
    the filter whether or not they satisfy it.
    """

    constraints: Mapping[str, Constraint] = field(default_factory=dict)
    extra_tickers: tuple[str, ...] | None = None

    def __post_init__(self):
        for var, constraint in self.constraints.items():
            if _is_range(constraint) and len(constraint) != 2:  # type: ignore[arg-type]
                raise ValueError(
                    f"Constraint for '{var}' is a tuple of length "
                    f"{len(constraint)}; a range must be (min, max)."  # type: ignore[arg-type]
                )

    def mask(self, bars: Bars) -> xr.DataArray:
        """Build the boolean ticker mask this filter describes for ``bars``."""
        mask = xr.ones_like(bars.dataset.ticker, dtype=bool)
        for var, constraint in self.constraints.items():
            values = bars.get_var(var)
            util.check(
                util.get_dims(values) == ("ticker",),
                f"Variable '{var}' has dims {util.get_dims(values)}; Where only "
                "supports ticker-only (daily) variables.",
            )
            if isinstance(constraint, tuple):
                low, high = constraint
                if low is not None:
                    mask &= values >= low
                if high is not None:
                    mask &= values <= high
            else:
                mask &= values == constraint
        if self.extra_tickers is not None:
            mask |= bars.dataset.ticker.isin(list(self.extra_tickers))
        return mask


def make_where(extra_tickers: Iterable[str] | None = None, **constraints: Constraint) -> Where:
    """Build a :class:`Where` from keyword constraints.

    This is what ``sel_where`` calls; the one wart is that ``extra_tickers`` is a
    reserved name, so a bars variable called ``extra_tickers`` has to be filtered
    by constructing :class:`Where` directly.
    """
    return Where(
        constraints=constraints,
        extra_tickers=tuple(extra_tickers) if extra_tickers is not None else None,
    )
