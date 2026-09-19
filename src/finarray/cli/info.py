"""Looking at a bars directory without opening Python: ls, check."""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from collections import Counter

import xarray as xr

from ..bars_set import BarsSet
from ..util import BARS_TICKER_FILENAME, BARS_TIME_FILENAME, FinArrayError
from ._common import add_dates_argument, open_bars, resolve_dates

# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


def add_ls_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "ls",
        help="summarize a bars directory: dates, variables, coordinates",
        description=(
            "Show what a bars base directory contains. With --dates-only or "
            "--vars-only, prints one item per line for use in shell loops."
        ),
    )
    parser.add_argument("BASEDIR", help="the bars base directory")
    parser.add_argument(
        "--date",
        metavar="DATE",
        help="the date to report coordinates and variables for (default: the last)",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--dates-only", action="store_true", help="print just the dates, one per line"
    )
    output.add_argument(
        "--vars-only", action="store_true", help="print just the variables, one per line"
    )
    parser.set_defaults(func=run_ls)


def _pick_date(bars: BarsSet, spec: str | None):
    dates = bars.dates_available()
    if not dates:
        raise FinArrayError(f"No dates found in {bars.base_path}")
    if spec is None:
        return dates[-1]
    return resolve_dates(bars, spec)[-1]


def run_ls(args: Namespace) -> None:
    bars = open_bars(args.BASEDIR)
    dates = bars.dates_available()

    if args.dates_only:
        for date in dates:
            print(date.isoformat())
        return

    if not dates:
        raise FinArrayError(f"No dates found in {bars.base_path}")

    date = _pick_date(bars, args.date)
    bd = bars[date]
    all_vars = bd.get_vars_available()

    if args.vars_only:
        for var in all_vars:
            print(var)
        return

    local = set(bd.get_vars_available(local_only=True))
    inherited = [v for v in all_vars if v not in local]

    print(f"{bars.base_path}")
    for i, parent in enumerate(bars._parent_base_paths, 1):
        print(f"  {'  ' * i}PARENT: {parent}")
    if not bars._parent_base_paths:
        print("  PARENT: (none)")

    if len(dates) == 1:
        print(f"  1 date: {dates[0]}")
    else:
        print(f"  {len(dates)} dates: {dates[0]} .. {dates[-1]}")

    sizes = bd.dataset.sizes
    times = bd.dataset.time
    span = ""
    if len(times):
        span = f" ({times.values[0]} .. {times.values[-1]})"
    print(f"  at {date}: {sizes.get('ticker', 0)} tickers x {sizes.get('time', 0)} times{span}")

    counts = f"{len(all_vars)} variables"
    if inherited:
        counts += f" ({len(local)} local, {len(inherited)} inherited)"
    print(f"  {counts}:")
    for var in all_vars:
        mark = " " if var in local else "^"
        print(f"    {mark} {var}")
    if inherited:
        print("  (^ = inherited from a parent)")


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def add_check_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "check",
        help="look for missing coordinate files and variables across dates",
        description=(
            "Validate a bars base directory. By default this is a cheap check: "
            "every date has its coordinate files, and every variable that any "
            "date has is present on all of them. --shapes additionally opens "
            "each variable file and checks it against the date's coordinates, "
            "which is thorough but slow. Exits non-zero if anything is wrong."
        ),
    )
    parser.add_argument("BASEDIR", help="the bars base directory")
    add_dates_argument(parser)
    parser.add_argument(
        "--shapes",
        action="store_true",
        help="also open every variable file and check its dimensions (slow)",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=10,
        metavar="N",
        help="how many dates to name per problem (default: 10)",
    )
    parser.set_defaults(func=run_check)


def _names(dates, limit: int) -> str:
    shown = ", ".join(d.isoformat() for d in dates[:limit])
    if len(dates) > limit:
        shown += f", ... (+{len(dates) - limit} more)"
    return shown


def _check_shapes(bd, problems: list[str]) -> None:
    expected = {"time": bd.dataset.sizes["time"], "ticker": bd.dataset.sizes["ticker"]}
    for var in bd.get_vars_available(local_only=True):
        path = bd._get_var_path(var)
        try:
            da = xr.load_dataarray(path)
        except Exception as exc:
            problems.append(f"{bd.date}: cannot read {var}.nc: {exc}")
            continue
        for dim, size in zip(da.dims, da.shape):
            if dim not in expected:
                problems.append(f"{bd.date}: {var} has unexpected dimension {dim!r}")
            elif expected[dim] != size:
                problems.append(
                    f"{bd.date}: {var} has {dim}={size}, coordinates say {expected[dim]}"
                )


def run_check(args: Namespace) -> None:
    bars = open_bars(args.BASEDIR)
    dates = resolve_dates(bars, args.dates)
    if not dates:
        raise FinArrayError(f"No dates found in {bars.base_path}")

    problems: list[str] = []
    vars_by_date: dict = {}
    counts: Counter = Counter()

    for date in dates:
        date_dir = os.path.join(bars.base_path, date.isoformat())
        try:
            bd = bars[date]
        except Exception as exc:
            problems.append(f"{date}: cannot open: {exc}")
            continue
        if not any(
            os.path.exists(os.path.join(p, fn))
            for p in bd._all_paths()
            for fn in (BARS_TICKER_FILENAME, BARS_TIME_FILENAME)
        ):  # pragma: no cover - Bars() would already have raised
            problems.append(f"{date}: no coordinate files in {date_dir}")
            continue
        present = bd.get_vars_available()
        vars_by_date[date] = set(present)
        counts.update(present)
        if args.shapes:
            _check_shapes(bd, problems)

    print(f"{len(dates)} date(s), {len(counts)} variable(s) in {bars.base_path}")

    incomplete = {v: n for v, n in counts.items() if n < len(vars_by_date)}
    if incomplete:
        print(f"\n{len(incomplete)} variable(s) missing on some dates:")
        for var in sorted(incomplete, key=lambda v: counts[v]):
            missing = [d for d, have in vars_by_date.items() if var not in have]
            print(
                f"  {var:<24} missing on {len(missing)} date(s): {_names(missing, args.max_report)}"
            )

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems[: args.max_report]:
            print(f"  {problem}")
        if len(problems) > args.max_report:
            print(f"  ... (+{len(problems) - args.max_report} more)")

    if problems or incomplete:
        print("\nfinarray check: problems found", file=sys.stderr)
        raise SystemExit(1)
    print("\nfinarray check: ok")
