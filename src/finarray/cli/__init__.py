"""Command-line tools for building, inspecting and querying bars directories.

    finarray ls BASEDIR                     what a directory holds
    finarray check BASEDIR                  look for gaps and inconsistencies
    finarray query BASEDIR -v mid           read data out, to stdout or a file
    finarray import-csv BASEDIR FILE...     CSVs -> date directories
    finarray import-parquet BASEDIR FILE    daily (date, ticker) values
    finarray eval --dir DIR EXPR...         derive variables and save them
    finarray link CHILD PARENT              a child directory inheriting a parent
    finarray rm BASEDIR VAR...              delete variables

Set PY_TRACEBACK=1 for full tracebacks instead of one-line error messages.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser

from ..util import FinArrayError, set_warnings_format
from .derive import add_eval_parser, add_rm_parser
from .info import add_check_parser, add_ls_parser
from .ingest import add_import_csv_parser, add_import_parquet_parser, add_link_parser
from .query import add_query_parser

__all__ = ["build_parser", "main"]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="finarray",
        description="Tools for building, inspecting and querying bars directories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for add in (
        add_ls_parser,
        add_check_parser,
        add_query_parser,
        add_import_csv_parser,
        add_import_parquet_parser,
        add_eval_parser,
        add_link_parser,
        add_rm_parser,
    ):
        add(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_warnings_format(f"finarray {args.command}")
    try:
        args.func(args)
    except SystemExit as exc:
        return int(exc.code or 0)
    except BrokenPipeError:  # e.g. `finarray query ... | head`
        return 0
    except FinArrayError as exc:
        print(f"finarray: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
