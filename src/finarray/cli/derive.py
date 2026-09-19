"""Computing and removing variables: eval, rm."""

from __future__ import annotations

import sys
from argparse import Namespace

import pandas as pd

from ..bars import Bars
from ..util import FinArrayError, progress_iter, warn
from ._common import add_dates_argument, date_from_dirname, open_bars, report_errors, resolve_dates

# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def add_eval_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "eval",
        help="evaluate expressions against bars and save the results",
        description=(
            "Evaluate one or more expressions and save every variable they "
            "create. Expressions autoload whatever variables they name, so "
            "nothing has to be listed up front. By default --dir is a single "
            "date directory; with --all-dates or --dates it is a base directory "
            "and the expressions run on each date in turn."
        ),
    )
    parser.add_argument(
        "-d",
        "--dir",
        default=".",
        help="a date directory, or a base directory with --all-dates/--dates",
    )
    parser.add_argument(
        "-f",
        "--filter",
        default="",
        metavar="EXPR",
        help="boolean expression; created variables are masked to where it holds",
    )
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="treat --dir as a base directory and run on every date",
    )
    add_dates_argument(parser, " Implies that --dir is a base directory.")
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="warn and continue when a date fails, instead of stopping",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    parser.add_argument("EXPR", nargs="+", help="assignments such as 'mid=(bid+ask)/2'")
    parser.set_defaults(func=run_eval)


def _evaluate(bars: Bars, expr: str):
    """Evaluate one expression, reporting an unknown name as a user error."""
    try:
        return bars.eval(expr)
    except (NameError, pd.errors.UndefinedVariableError) as exc:
        raise FinArrayError(
            f"{bars.date}: cannot evaluate {expr!r}: {exc}. "
            f"Variables in {bars.path}: {', '.join(bars.get_vars_available())}"
        ) from exc


def _eval_one(bars: Bars, exprs: list[str], filter_expr: str) -> list[str]:
    for expr in exprs:
        _evaluate(bars, expr)
    new_vars = bars.get_created_vars()
    if not new_vars:
        return []
    if filter_expr:
        mask = _evaluate(bars, filter_expr)
        for var in new_vars:
            bars[var] = bars[var].where(mask)
    bars.save_vars(new_vars)
    return new_vars


def run_eval(args: Namespace) -> None:
    if not args.all_dates and not args.dates:
        _run_eval_single(args)
        return

    bars_set = open_bars(args.dir)
    dates = resolve_dates(bars_set, args.dates)
    if not dates:
        raise FinArrayError(f"No dates found in {bars_set.base_path}")

    written: set[str] = set()
    failed = 0
    for date in progress_iter(dates, desc="eval", disable=args.quiet or len(dates) == 1):
        try:
            written.update(_eval_one(bars_set[date], args.EXPR, args.filter))
        except Exception as exc:
            if not args.skip_errors:
                raise
            warn(f"{date}: {exc}")
            failed += 1

    if written:
        print(
            f"finarray: wrote {', '.join(sorted(written))} for {len(dates) - failed} date(s)",
            file=sys.stderr,
        )
    else:
        warn("No new variables were created; nothing to save.")
    if failed:
        raise SystemExit(1)


def _run_eval_single(args: Namespace) -> None:
    date = date_from_dirname(args.dir)
    with report_errors(f"could not open bars directory {args.dir}: %s"):
        bars = Bars(args.dir, date)

    new_vars = _eval_one(bars, args.EXPR, args.filter)
    if not new_vars:
        warn("No new variables were created; nothing to save.")
        return
    print(f"finarray: wrote {', '.join(new_vars)} to {args.dir}", file=sys.stderr)


# ---------------------------------------------------------------------------
# rm
# ---------------------------------------------------------------------------


def add_rm_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "rm",
        help="delete variables from a bars directory",
        description=(
            "Delete one or more variables from every selected date. Only ever "
            "deletes from BASEDIR itself: a variable inherited from a parent is "
            "left alone and reported. Asks for confirmation unless -y is given."
        ),
    )
    parser.add_argument("BASEDIR", help="the bars base directory")
    parser.add_argument("VAR", nargs="+", help="variables to delete")
    add_dates_argument(parser)
    parser.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="report what would be deleted and stop",
    )
    parser.set_defaults(func=run_rm)


def run_rm(args: Namespace) -> None:
    bars = open_bars(args.BASEDIR)
    dates = resolve_dates(bars, args.dates)
    if not dates:
        raise FinArrayError(f"No dates found in {bars.base_path}")

    # Work out what is actually there before touching anything.
    targets: dict[str, list] = {}
    inherited: dict[str, int] = {}
    for var in args.VAR:
        local_dates, inherited_count = [], 0
        for date in dates:
            bd = bars[date]
            if var in bd.get_vars_available(local_only=True):
                local_dates.append(date)
            elif bd.has_var(var):
                inherited_count += 1
        targets[var] = local_dates
        if inherited_count:
            inherited[var] = inherited_count

    for var, count in inherited.items():
        warn(
            f"{var}: inherited from a parent on {count} date(s); "
            "leaving those alone (delete them from the parent directory itself)."
        )

    total = sum(len(d) for d in targets.values())
    for var, local_dates in targets.items():
        print(f"{var}: {len(local_dates)} file(s) in {bars.base_path}")
    if total == 0:
        print("finarray: nothing to delete")
        return

    if args.dry_run:
        print(f"finarray: would delete {total} file(s) (dry run)")
        return

    if not args.yes:
        if not sys.stdin.isatty():
            raise FinArrayError(
                "Refusing to delete without confirmation; pass -y when not on a terminal."
            )
        answer = input(f"Delete {total} file(s)? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("finarray: aborted")
            return

    deleted = 0
    for var, local_dates in targets.items():
        deleted += len(bars.delete_var(var, dates=local_dates))
    print(f"finarray: deleted {deleted} file(s)")
