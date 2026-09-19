"""Shared plumbing for the subcommands."""

from __future__ import annotations

import datetime as dt
import os
import sys
from contextlib import contextmanager

from ..bars_set import BarsSet
from ..util import FinArrayError


@contextmanager
def report_errors(msg: str):
    """Turn an exception into a one-line message, unless PY_TRACEBACK is set."""
    if os.environ.get("PY_TRACEBACK", "0") != "0":
        yield
        return
    try:
        yield
    except Exception as exc:
        text = str(exc)
        if len(text) > 150:
            text = text[:147] + "[...]"
        print(f"finarray: {msg % text}", file=sys.stderr)
        raise SystemExit(1) from exc


def date_from_dirname(path: str) -> dt.date:
    """A date directory is named for its date: bars/eod/2025-01-13."""
    name = os.path.basename(os.path.normpath(path))
    try:
        return dt.date.fromisoformat(name)
    except ValueError:
        raise FinArrayError(
            f"Cannot tell the date from directory name {name!r}; a bars date "
            "directory must be named YYYY-MM-DD."
        ) from None


def split_list(arg: str | None) -> list[str]:
    """Accept comma- or whitespace-separated lists, as the original scripts did."""
    if not arg:
        return []
    if "," in arg:
        return [p.strip() for p in arg.split(",") if p.strip()]
    return arg.split()


def resolve_dates(bars: BarsSet, spec: str | None) -> list[dt.date]:
    """Turn a --dates specification into dates the set actually has.

    Accepts a list (``2025-01-13,2025-01-15``), an inclusive range
    (``2025-01-13:2025-01-17``, either end optional), or nothing for every
    available date.
    """
    available = bars.dates_available()
    if not spec:
        return available

    if ":" in spec:
        start_s, _, end_s = spec.partition(":")
        start = dt.date.fromisoformat(start_s) if start_s.strip() else None
        end = dt.date.fromisoformat(end_s) if end_s.strip() else None
        return [d for d in available if (start is None or d >= start) and (end is None or d <= end)]

    wanted = [dt.date.fromisoformat(d) for d in split_list(spec)]
    have = set(available)
    missing = [d for d in wanted if d not in have]
    if missing:
        raise FinArrayError(
            f"No data for {', '.join(d.isoformat() for d in missing)} in {bars.base_path}"
        )
    return wanted


def open_bars(base_path: str) -> BarsSet:
    if not os.path.isdir(os.path.expanduser(base_path)):
        raise FinArrayError(f"Directory {base_path!r} not found.")
    return BarsSet(base_path)


def add_dates_argument(parser, help_extra: str = "") -> None:
    parser.add_argument(
        "-D",
        "--dates",
        metavar="SPEC",
        help=(
            "dates to use: a list (2025-01-13,2025-01-15) or an inclusive range "
            "(2025-01-13:2025-01-17, either end optional). Default: all." + help_extra
        ),
    )
