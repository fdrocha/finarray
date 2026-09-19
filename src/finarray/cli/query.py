"""Pull data out of bars and write it somewhere: query.

Output is produced date by date and written as it is collected, so a query with
a limit reads only the dates -- and, within a date, only the time steps -- it
actually needs. Nothing builds the whole result first.
"""

from __future__ import annotations

import math
import sys
from argparse import Namespace
from typing import Any

import pandas as pd

from ..bars import Bars
from ..bars_set import BarsSet
from ..util import FinArrayError
from ..where import Constraint, Where
from ._common import add_dates_argument, open_bars, resolve_dates, split_list

DEFAULT_LIMIT = 50
_FORMATS = ("csv", "tsv", "parquet")
_EXTENSIONS = {".csv": "csv", ".tsv": "tsv", ".parquet": "parquet", ".pq": "parquet"}


def add_query_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "query",
        help="read variables out of bars, to stdout or a file",
        description=(
            "Select variables, tickers, times and dates, and write the result as "
            f"CSV to stdout (at most {DEFAULT_LIMIT} rows by default) or to a "
            "file with --output. Selecting a single ticker or time drops that "
            "column, exactly as sel_ticker/sel_time do in Python; the date is "
            "always the first index level."
        ),
    )
    parser.add_argument("BASEDIR", help="the bars base directory")
    parser.add_argument("-v", "--vars", required=True, metavar="VARS", help="variables to read")

    tickers = parser.add_mutually_exclusive_group()
    tickers.add_argument("--ticker", metavar="TICKER", help="a single ticker")
    tickers.add_argument("--tickers", metavar="LIST", help="several tickers")

    times = parser.add_mutually_exclusive_group()
    times.add_argument("--time", metavar="TIME", help="a single time, e.g. 15:54:00")
    times.add_argument("--times", metavar="LIST", help="several times")
    times.add_argument(
        "--time-slice",
        metavar="START,END",
        help="an inclusive time range, e.g. 15:50:00,15:59:59",
    )

    parser.add_argument(
        "--where",
        action="append",
        metavar="EXPR",
        help=(
            "restrict tickers by a daily (ticker-only) variable: VAR=MIN:MAX for "
            "an inclusive range with either end optional, or VAR=VALUE for "
            "equality. Repeatable; conditions are combined with AND."
        ),
    )
    parser.add_argument(
        "--extra-tickers",
        metavar="LIST",
        help="tickers to keep regardless of --where",
    )
    add_dates_argument(parser)
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        metavar="N",
        help=(
            f"stop after N rows (default: {DEFAULT_LIMIT} to stdout, unlimited "
            "with --output). 0 means unlimited."
        ),
    )
    parser.add_argument(
        "-o", "--output", metavar="FILE", help="write to this file instead of stdout"
    )
    parser.add_argument(
        "--format",
        choices=_FORMATS,
        help="output format (default: from the --output extension, else csv)",
    )
    parser.add_argument("--no-header", action="store_true", help="omit the header row")
    parser.set_defaults(func=run_query)


# ---------------------------------------------------------------------------
# selections
# ---------------------------------------------------------------------------


def _bound(text: str, spec: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise FinArrayError(f"--where {spec!r}: {text!r} is not a number") from None


def _value(text: str, spec: str) -> Constraint:
    text = text.strip()
    if not text:
        raise FinArrayError(f"--where {spec!r}: no value given")
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _parse_where(specs: list[str]) -> dict[str, Constraint]:
    """Turn --where strings into Where constraints.

    ``adv=1e7:`` is a range with no upper bound, ``price=2:2500`` a closed one,
    ``listing_exchange=1`` an equality test.
    """
    constraints: dict[str, Constraint] = {}
    for spec in specs:
        name, sep, rhs = spec.partition("=")
        name = name.strip()
        if not sep or not name:
            raise FinArrayError(f"--where takes VAR=MIN:MAX or VAR=VALUE; got {spec!r}")
        if name in constraints:
            raise FinArrayError(f"--where names {name!r} more than once")
        if ":" in rhs:
            low, _, high = rhs.partition(":")
            constraints[name] = (_bound(low, spec), _bound(high, spec))
        else:
            constraints[name] = _value(rhs, spec)
    return constraints


def _apply_selections(bars: BarsSet, args: Namespace) -> BarsSet:
    if args.where or args.extra_tickers:
        # Built directly rather than through sel_where, whose `extra_tickers`
        # keyword would collide with a variable of that name.
        bars = bars.apply_where(
            Where(
                constraints=_parse_where(args.where or []),
                extra_tickers=tuple(split_list(args.extra_tickers)) or None,
            )
        )

    if args.ticker:
        bars = bars.sel_ticker(args.ticker)
    elif args.tickers:
        bars = bars.sel_ticker(split_list(args.tickers))

    if args.time:
        bars = bars.sel_time(args.time)
    elif args.times:
        bars = bars.sel_time(split_list(args.times))
    elif args.time_slice:
        parts = split_list(args.time_slice)
        if len(parts) != 2:
            raise FinArrayError(f"--time-slice takes START,END; got {args.time_slice!r}")
        bars = bars.sel_time_slice(parts[0], parts[1])
    return bars


# ---------------------------------------------------------------------------
# one date's worth of rows
# ---------------------------------------------------------------------------


def _date_frame(bd: Bars, var_list: list[str], max_rows: int | None) -> pd.DataFrame:
    """The rows for one date, reading no more of the grid than `max_rows` needs.

    Rows come out in (time, ticker) order, so bounding the time dimension takes
    a true prefix of the result rather than an arbitrary subset.
    """
    ds = bd.get_vars(var_list)

    if not ds.sizes:  # both ticker and time were selected down to scalars
        scalars = ds.to_pandas()
        assert isinstance(scalars, pd.Series)
        frame = scalars.to_frame().T
        frame.index = pd.Index([bd.date], name="date")
        return frame[[v for v in var_list if v in frame.columns]]

    if max_rows is not None and "time" in ds.dims:
        n_tickers = int(ds.sizes.get("ticker", 1)) or 1
        n_times = max(1, math.ceil(max_rows / n_tickers))
        if n_times < ds.sizes["time"]:
            ds = ds.isel(time=slice(0, n_times))
        # Only safe once a single time step is left: then the row prefix is a
        # ticker prefix. With more than one step it spans whole ticker blocks.
        if n_times == 1 and "ticker" in ds.dims and n_tickers > max_rows:
            ds = ds.isel(ticker=slice(0, max_rows))

    frame = ds.to_dataframe()
    # xarray keeps a dropped scalar coordinate around as a column.
    frame = frame.drop(columns=[c for c in ("time", "ticker") if c in frame.columns])
    frame = frame[[v for v in var_list if v in frame.columns]]
    return pd.concat([frame], keys=[bd.date], names=["date"])


# ---------------------------------------------------------------------------
# sinks
# ---------------------------------------------------------------------------


class _TextSink:
    def __init__(self, stream, sep: str, header: bool):
        self.stream = stream
        self.sep = sep
        self.header = header

    def _open(self):
        return self.stream

    def write(self, frame: pd.DataFrame) -> None:
        stream = self._open()
        frame.to_csv(stream, sep=self.sep, header=self.header, lineterminator="\n")
        self.header = False  # only once, however many chunks follow
        stream.flush()

    def close(self) -> None:
        pass


class _FileTextSink(_TextSink):
    """Opens on the first write, so a query matching nothing leaves no file."""

    def __init__(self, path: str, sep: str, header: bool):
        super().__init__(None, sep, header)
        self.path = path

    def _open(self):
        if self.stream is None:
            # Held open across chunks; run_query closes it in a finally.
            self.stream = open(self.path, "w", encoding="utf-8", newline="")  # noqa: SIM115
        return self.stream

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()


class _ParquetSink:
    def __init__(self, path: str):
        self.path = path
        self._writer: Any = None

    def write(self, frame: pd.DataFrame) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(frame.reset_index(), preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def _resolve_format(args: Namespace) -> str:
    if args.format:
        return args.format
    if args.output:
        for ext, fmt in _EXTENSIONS.items():
            if args.output.endswith(ext):
                return fmt
    return "csv"


def _make_sink(args: Namespace, fmt: str) -> _TextSink | _ParquetSink:
    header = not args.no_header
    sep = "\t" if fmt == "tsv" else ","
    if args.output is None:
        return _TextSink(sys.stdout, sep, header)
    if fmt == "parquet":
        return _ParquetSink(args.output)
    return _FileTextSink(args.output, sep, header)


# ---------------------------------------------------------------------------


def run_query(args: Namespace) -> None:
    fmt = _resolve_format(args)
    if fmt == "parquet" and args.output is None:
        raise FinArrayError("--format parquet needs --output FILE.")

    var_list = split_list(args.vars)
    if not var_list:
        raise FinArrayError("--vars must name at least one variable.")

    bars = _apply_selections(open_bars(args.BASEDIR), args)
    dates = resolve_dates(bars, args.dates)
    if not dates:
        raise FinArrayError(f"No dates found in {bars.base_path}")

    # An explicit --limit always wins; otherwise stdout is capped and a file is not.
    if args.limit is not None:
        limit = None if args.limit <= 0 else args.limit
    else:
        limit = None if args.output else DEFAULT_LIMIT

    sink = _make_sink(args, fmt)
    emitted = 0
    truncated = False
    try:
        for date in dates:
            # One more than we can use, so that filling up is distinguishable
            # from running out of data.
            want = None if limit is None else (limit - emitted) + 1
            frame = _date_frame(bars[date], var_list, want)
            if frame.empty:
                continue
            if limit is not None and emitted + len(frame) > limit:
                sink.write(frame.iloc[: limit - emitted])
                emitted = limit
                truncated = True
                break
            sink.write(frame)
            emitted += len(frame)
    finally:
        sink.close()

    if emitted == 0:
        message = "# no rows matched"
        if args.output is None:
            print(message, flush=True)
        else:
            print(f"finarray:{message[1:]}; {args.output} not written", file=sys.stderr)
        return

    if truncated:
        print(
            f"# ... stopped at {limit} rows. Use --limit N for more, "
            "--limit 0 for all, or --output FILE to export.",
            flush=True,
        )
    elif args.output is not None:
        print(f"finarray: wrote {emitted} row(s) to {args.output}", file=sys.stderr)
