"""Getting data into a bars directory: import-csv, import-parquet, link."""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import sys
from argparse import Namespace

import pandas as pd

from ..bars_set import BarsSet, create_child_bars
from ..util import FinArrayError, load_csv, progress_iter, to_frdir
from ._common import add_dates_argument, open_bars, resolve_dates, split_list

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _date_from_filename(filename: str) -> dt.date:
    match = _DATE_RE.search(filename)
    if match is None:
        raise FinArrayError(f"No YYYY-MM-DD date found in filename {filename!r}.")
    return dt.date.fromisoformat(match.group(1))


def _run_over_files(args: Namespace, handler) -> None:
    """Apply `handler` to each input file, reporting failures without stopping."""
    failures = 0
    files = progress_iter(args.FILE, desc=args.command, disable=args.quiet or len(args.FILE) == 1)
    for filename in files:
        try:
            handler(filename)
        except Exception as exc:
            if os.environ.get("PY_TRACEBACK", "0") != "0":
                raise
            print(f"finarray: {filename}: {exc}", file=sys.stderr)
            failures += 1
    if failures:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# import-csv
# ---------------------------------------------------------------------------


def add_import_csv_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "import-csv",
        help="build date directories from CSV files",
        description=(
            "Create bars date directories from CSV files. Each file must have "
            "time and ticker columns and a YYYY-MM-DD date somewhere in its name."
        ),
    )
    parser.add_argument("BASEDIR", help="the bars base directory")
    parser.add_argument("FILE", nargs="+", help="one CSV per date")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="replace a date directory that already exists",
    )
    mode.add_argument(
        "--add",
        action="store_true",
        help=(
            "add variables to an existing date directory, keeping its ticker and time coordinates"
        ),
    )
    parser.add_argument(
        "--rename", metavar="MASK", help="rename columns through this mask, e.g. 'raw_%%s'"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    parser.set_defaults(func=run_import_csv)


def _import_csv_one(args: Namespace, bars: BarsSet, filename: str) -> None:
    date = _date_from_filename(filename)
    rename_func = None if args.rename is None else lambda col: args.rename % col

    if args.add:
        # Looked up through the BarsSet so an inherited date counts as present.
        if not bars.has_date(date):
            raise FinArrayError(f"No existing bars for {date} in {args.BASEDIR}, and --add is set.")
        ds = load_csv(filename, date=date, rename_func=rename_func)
        bd = bars[date]
        for variable in ds.variables:
            if variable in ("time", "ticker"):
                continue
            bd[variable] = ds[variable]
            bd.save_var(str(variable))
        return

    path = os.path.join(args.BASEDIR, date.isoformat())
    if os.path.exists(path):
        if not args.force:
            raise FinArrayError(f"{path} already exists; pass --force to replace it.")
        print(f"finarray: replacing {path}", file=sys.stderr)
        shutil.rmtree(path)
    ds = load_csv(filename, date=date, rename_func=rename_func)
    to_frdir(ds, args.BASEDIR)


def run_import_csv(args: Namespace) -> None:
    bars = open_bars(args.BASEDIR)
    _run_over_files(args, lambda fn: _import_csv_one(args, bars, fn))


# ---------------------------------------------------------------------------
# import-parquet
# ---------------------------------------------------------------------------


def add_import_parquet_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "import-parquet",
        help="add daily (ticker-only) variables from a parquet file",
        description=(
            "Read a parquet file indexed by (date, ticker) and write each of its "
            "columns into the matching date directories as a daily, ticker-only "
            "variable. Dates in the file that the bars do not have are skipped."
        ),
    )
    parser.add_argument("BASEDIR", help="the bars base directory")
    parser.add_argument("FILE", nargs="+", help="parquet files indexed by (date, ticker)")
    parser.add_argument(
        "-v", "--vars", metavar="VARS", help="columns to import (default: all of them)"
    )
    add_dates_argument(parser, " Restricted further to dates present in the file.")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    parser.set_defaults(func=run_import_parquet)


def _load_daily_frame(path: str) -> pd.DataFrame:
    if not path.endswith((".parquet", ".pq")):
        raise FinArrayError(f"Expected a .parquet file, got {path!r}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex) or tuple(df.index.names) != (
        "date",
        "ticker",
    ):
        raise FinArrayError(
            f"{path}: expected a MultiIndex of (date, ticker), got {df.index.names}"
        )
    return df.sort_index()


def _import_parquet_one(args: Namespace, bars: BarsSet, filename: str) -> None:
    df = _load_daily_frame(filename)
    columns = split_list(args.vars) or df.columns.to_list()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise FinArrayError(f"{filename} has no column(s) {', '.join(missing)}")

    wanted = set(resolve_dates(bars, args.dates))
    dates = [d for d in df.index.get_level_values("date").unique() if d in wanted]
    if not dates:
        raise FinArrayError(f"{filename} has no dates in common with {args.BASEDIR}")

    for date in progress_iter(
        dates, desc=os.path.basename(filename), disable=args.quiet or len(dates) == 1
    ):
        bd = bars[date]
        for column in columns:
            bd.add_daily_var(df.loc[date][column].to_xarray())
            bd.save_var(column)
    print(
        f"finarray: wrote {', '.join(columns)} for {len(dates)} date(s)",
        file=sys.stderr,
    )


def run_import_parquet(args: Namespace) -> None:
    bars = open_bars(args.BASEDIR)
    _run_over_files(args, lambda fn: _import_parquet_one(args, bars, fn))


# ---------------------------------------------------------------------------
# link
# ---------------------------------------------------------------------------


def add_link_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "link",
        help="create a child bars directory inheriting from a parent",
        description=(
            "Create CHILD as a bars directory that inherits PARENT's dates and "
            "variables through a PARENT symlink. Variables written to the child "
            "shadow the parent's without modifying or copying it."
        ),
    )
    parser.add_argument("CHILD", help="directory to create (must not exist)")
    parser.add_argument("PARENT", help="existing bars base directory to inherit from")
    parser.set_defaults(func=run_link)


def run_link(args: Namespace) -> None:
    child = os.path.expanduser(args.CHILD)
    parent = os.path.abspath(os.path.expanduser(args.PARENT))
    create_child_bars(child, parent)
    n_dates = len(BarsSet(child).dates_available())
    print(f"finarray: {child} -> {parent} ({n_dates} inherited dates)", file=sys.stderr)
