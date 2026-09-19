"""The `finarray query` subcommand, including its streaming behaviour."""

import datetime as dt

import pandas as pd
import pytest

from finarray.cli import main


def run(*argv) -> int:
    return main([str(a) for a in argv])


@pytest.fixture
def base(sample_tree):
    return str(sample_tree)


def rows_of(capsys) -> list[str]:
    out = capsys.readouterr().out.strip().splitlines()
    return [line for line in out if not line.startswith("#")]


# ---------------------------------------------------------------------------
# shape of the output
# ---------------------------------------------------------------------------


def test_scalar_ticker_and_time_gives_one_row_per_date(base, capsys):
    assert run("query", base, "-v", "mid", "--ticker", "SPY", "--time", "15:54:00") == 0
    lines = rows_of(capsys)
    assert lines[0] == "date,mid"
    assert len(lines) == 5  # header + 4 dates
    assert lines[1].startswith("2024-11-27,")


def test_scalar_time_keeps_ticker(base, capsys):
    assert run("query", base, "-v", "mid", "--time", "15:54:00", "--limit", "0") == 0
    lines = rows_of(capsys)
    assert lines[0] == "date,ticker,mid"
    assert len(lines) == 1 + 4 * 4  # 4 dates x 4 tickers


def test_full_grid_keeps_time_and_ticker(base, capsys):
    assert run("query", base, "-v", "mid", "-D", "2024-11-27", "--limit", "0") == 0
    lines = rows_of(capsys)
    assert lines[0] == "date,time,ticker,mid"
    assert len(lines) == 1 + 600 * 4


def test_several_variables_keep_their_order(base, capsys):
    assert run("query", base, "-v", "mid,bid,ask", "--ticker", "AAA", "--time", "15:54:00") == 0
    assert rows_of(capsys)[0] == "date,mid,bid,ask"


def test_ticker_list_keeps_the_column(base, capsys):
    assert run("query", base, "-v", "mid", "--tickers", "SPY", "--time", "15:54:00") == 0
    assert rows_of(capsys)[0] == "date,ticker,mid"


def test_time_slice(base, capsys):
    assert (
        run(
            "query",
            base,
            "-v",
            "mid",
            "--ticker",
            "AAA",
            "--time-slice",
            "15:50:00,15:50:09",
            "-D",
            "2024-11-27",
            "--limit",
            "0",
        )
        == 0
    )
    assert len(rows_of(capsys)) == 1 + 10


def test_date_range(base, capsys):
    assert (
        run(
            "query",
            base,
            "-v",
            "mid",
            "--ticker",
            "SPY",
            "--time",
            "15:54:00",
            "-D",
            "2024-12-02:2024-12-03",
        )
        == 0
    )
    lines = rows_of(capsys)
    assert [line.split(",")[0] for line in lines[1:]] == ["2024-12-02", "2024-12-03"]


def test_date_list(base, capsys):
    assert (
        run(
            "query",
            base,
            "-v",
            "mid",
            "--ticker",
            "SPY",
            "--time",
            "15:54:00",
            "-D",
            "2024-11-27,2024-12-04",
        )
        == 0
    )
    assert [line.split(",")[0] for line in rows_of(capsys)[1:]] == [
        "2024-11-27",
        "2024-12-04",
    ]


# ---------------------------------------------------------------------------
# the limit, and that it is honoured by reading less rather than truncating
# ---------------------------------------------------------------------------


def test_default_limit_is_fifty(base, capsys):
    assert run("query", base, "-v", "mid", "-D", "2024-11-27") == 0
    captured = capsys.readouterr().out
    data = [line for line in captured.strip().splitlines() if not line.startswith("#")]
    assert len(data) == 1 + 50
    assert "stopped at 50 rows" in captured


def test_notice_only_when_truncated(base, capsys):
    assert run("query", base, "-v", "mid", "--ticker", "SPY", "--time", "15:54:00") == 0
    assert "stopped at" not in capsys.readouterr().out


def test_limit_zero_means_everything(base, capsys):
    assert run("query", base, "-v", "mid", "-D", "2024-11-27", "--limit", "0") == 0
    captured = capsys.readouterr().out
    assert "stopped at" not in captured
    assert len(captured.strip().splitlines()) == 1 + 600 * 4


@pytest.mark.parametrize("limit", [1, 3, 4, 5, 17, 50, 123])
def test_limited_output_is_a_true_prefix_of_the_full_output(base, capsys, limit):
    """The bounded read must return the same rows the full query starts with.

    This is what makes the streaming optimisation safe: `query` narrows the time
    (and, at one time step, the ticker) dimension before converting to pandas,
    which is only correct if rows come out in (time, ticker) order.
    """
    assert run("query", base, "-v", "mid", "-D", "2024-11-27,2024-12-02", "--limit", limit) == 0
    limited = rows_of(capsys)

    assert run("query", base, "-v", "mid", "-D", "2024-11-27,2024-12-02", "--limit", "0") == 0
    full = rows_of(capsys)

    assert limited == full[: limit + 1]  # +1 for the header


def test_limit_spanning_a_date_boundary(base, capsys):
    """A limit larger than one date's worth of rows must roll onto the next."""
    assert (
        run(
            "query",
            base,
            "-v",
            "mid",
            "--ticker",
            "SPY",
            "--time",
            "15:54:00",
            "-D",
            "2024-11-27,2024-12-02,2024-12-03",
            "--limit",
            "2",
        )
        == 0
    )
    lines = rows_of(capsys)
    assert [line.split(",")[0] for line in lines[1:]] == ["2024-11-27", "2024-12-02"]


def test_streaming_does_not_read_every_date(base, capsys, monkeypatch):
    """A small limit must touch only the dates it needs."""
    import finarray.cli.query as query_mod

    seen: list[dt.date] = []
    original = query_mod._date_frame

    def spy(bd, var_list, max_rows):
        seen.append(bd.date)
        return original(bd, var_list, max_rows)

    monkeypatch.setattr(query_mod, "_date_frame", spy)
    assert run("query", base, "-v", "mid", "--limit", "3") == 0
    assert seen == [dt.date(2024, 11, 27)]  # not all four dates


def test_streaming_bounds_the_time_dimension(base, capsys, monkeypatch):
    """Within a date it must not convert the whole grid to pandas."""
    import finarray.cli.query as query_mod

    sizes: list[int] = []
    original = query_mod.pd.concat

    def spy(objs, *args, **kwargs):
        if objs and isinstance(objs[0], pd.DataFrame):
            sizes.append(len(objs[0]))
        return original(objs, *args, **kwargs)

    monkeypatch.setattr(query_mod.pd, "concat", spy)
    assert run("query", base, "-v", "mid", "--limit", "3") == 0
    # 600 times x 4 tickers = 2400 rows available; one time step is enough.
    assert sizes and max(sizes) <= 8


# ---------------------------------------------------------------------------
# output targets
# ---------------------------------------------------------------------------


def test_output_to_csv_has_no_limit_by_default(base, tmp_path, capsys):
    out = tmp_path / "out.csv"
    assert run("query", base, "-v", "mid", "-D", "2024-11-27", "-o", out) == 0
    assert capsys.readouterr().out == ""  # nothing on stdout
    assert len(out.read_text().strip().splitlines()) == 1 + 600 * 4


def test_output_respects_an_explicit_limit(base, tmp_path):
    out = tmp_path / "out.csv"
    assert run("query", base, "-v", "mid", "-D", "2024-11-27", "-o", out, "--limit", "9") == 0
    assert len(out.read_text().strip().splitlines()) == 1 + 9


def test_output_to_parquet(base, tmp_path):
    out = tmp_path / "out.parquet"
    assert run("query", base, "-v", "mid,bid", "--time", "15:54:00", "-o", out) == 0
    df = pd.read_parquet(out)
    assert list(df.columns) == ["date", "ticker", "mid", "bid"]
    assert len(df) == 4 * 4


def test_parquet_spans_dates_written_in_separate_chunks(base, tmp_path):
    out = tmp_path / "out.parquet"
    assert run("query", base, "-v", "mid", "--ticker", "SPY", "-o", out) == 0
    df = pd.read_parquet(out)
    assert sorted(df["date"].unique().tolist()) == [
        dt.date(2024, 11, 27),
        dt.date(2024, 12, 2),
        dt.date(2024, 12, 3),
        dt.date(2024, 12, 4),
    ]


def test_tsv_and_no_header(base, capsys):
    assert (
        run(
            "query",
            base,
            "-v",
            "mid",
            "--ticker",
            "SPY",
            "--time",
            "15:54:00",
            "--format",
            "tsv",
            "--no-header",
        )
        == 0
    )
    lines = rows_of(capsys)
    assert "\t" in lines[0]
    assert not lines[0].startswith("date")


def test_parquet_to_stdout_is_rejected(base, capsys):
    assert run("query", base, "-v", "mid", "--format", "parquet") == 1
    assert "needs --output" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def test_unknown_variable(base, capsys):
    assert run("query", base, "-v", "nope") == 1
    assert "not available" in capsys.readouterr().err


def test_unknown_date(base, capsys):
    assert run("query", base, "-v", "mid", "-D", "1999-01-04") == 1
    assert "No data for 1999-01-04" in capsys.readouterr().err


def test_missing_basedir(tmp_path, capsys):
    assert run("query", tmp_path / "nope", "-v", "mid") == 1
    assert "not found" in capsys.readouterr().err


def test_bad_time_slice(base, capsys):
    assert run("query", base, "-v", "mid", "--time-slice", "15:50:00") == 1
    assert "START,END" in capsys.readouterr().err
