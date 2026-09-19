"""Command-line tools for building and extending bars directories.

Two subcommands, which together cover getting data in and deriving things from
it once it is there:

    finarray import-csv BASEDIR FILE...    # CSVs -> date directories
    finarray eval --dir DIR EXPR...        # derive variables, save them

Both are ports of scripts from the research repository finarray came out of,
where they ran as a pipeline:

    finarray import-csv -f bars/eod quotes-2025-01-13.csv
    finarray eval --dir bars/eod/2025-01-13 \\
        -f '(abs(ask-bid) <= ask*0.01) | (abs(ask-bid) <= 0.05)' 'mid=(bid+ask)/2'
"""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import sys
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager

from . import Bars, BarsSet
from .util import FinArrayError, load_csv, set_warnings_format, to_frdir, warn

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@contextmanager
def _report_errors(msg: str):
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


def _date_from_dirname(path: str) -> dt.date:
    """A date directory is named for its date: bars/eod/2025-01-13."""
    name = os.path.basename(os.path.normpath(path))
    try:
        return dt.date.fromisoformat(name)
    except ValueError:
        raise FinArrayError(
            f"Cannot tell the date from directory name {name!r}; a bars date "
            "directory must be named YYYY-MM-DD."
        ) from None


# ---------------------------------------------------------------------------
# finarray eval
# ---------------------------------------------------------------------------


def add_eval_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "eval",
        help="evaluate expressions against a date directory and save the results",
        description=(
            "Evaluate one or more expressions against a single date directory and "
            "save every variable they create. Expressions autoload whatever "
            "variables they name, so nothing has to be listed up front."
        ),
    )
    parser.add_argument("-d", "--dir", default=".", help="the date directory (default: .)")
    parser.add_argument(
        "-f",
        "--filter",
        default="",
        metavar="EXPR",
        help="boolean expression; created variables are masked to where it holds",
    )
    parser.add_argument("EXPR", nargs="+", help="assignments such as 'mid=(bid+ask)/2'")
    parser.set_defaults(func=run_eval)


def run_eval(args: Namespace) -> None:
    date = _date_from_dirname(args.dir)
    with _report_errors(f"could not open bars directory {args.dir}: %s"):
        bars = Bars(args.dir, date)

    for expr in args.EXPR:
        bars.eval(expr)

    new_vars = bars.get_created_vars()
    if not new_vars:
        warn("No new variables were created; nothing to save.")
        return

    if args.filter:
        mask = bars.eval(args.filter)
        for var in new_vars:
            bars[var] = bars[var].where(mask)

    bars.save_vars(new_vars)
    print(f"finarray: wrote {', '.join(new_vars)} to {args.dir}", file=sys.stderr)


# ---------------------------------------------------------------------------
# finarray import-csv
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
        "--rename",
        metavar="MASK",
        help="rename columns through this mask, e.g. 'raw_%%s'",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    parser.set_defaults(func=run_import_csv)


def _import_one(args: Namespace, bars: BarsSet, filename: str) -> None:
    match = _DATE_RE.search(filename)
    if match is None:
        raise FinArrayError(f"No YYYY-MM-DD date found in filename {filename!r}.")
    date = dt.date.fromisoformat(match.group(1))
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
    from .util import progress_iter

    if not os.path.isdir(args.BASEDIR):
        raise FinArrayError(f"Directory {args.BASEDIR!r} not found.")

    bars = BarsSet(args.BASEDIR)
    files = progress_iter(
        args.FILE,
        desc="import-csv",
        disable=args.quiet or len(args.FILE) == 1,
    )
    failures = 0
    for filename in files:
        # One bad file should not abandon the rest of the batch.
        try:
            _import_one(args, bars, filename)
        except Exception as exc:
            if os.environ.get("PY_TRACEBACK", "0") != "0":
                raise
            print(f"finarray: {filename}: {exc}", file=sys.stderr)
            failures += 1
    if failures:
        raise SystemExit(1)


# ---------------------------------------------------------------------------


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="finarray", description="Tools for building and extending bars directories."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_eval_parser(subparsers)
    add_import_csv_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_warnings_format(f"finarray {args.command}")
    try:
        args.func(args)
    except SystemExit as exc:
        return int(exc.code or 0)
    except FinArrayError as exc:
        print(f"finarray: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
