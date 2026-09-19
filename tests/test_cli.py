"""The `finarray` command-line tools."""

import datetime as dt
import sys

import pandas as pd
import pytest

import finarray as fr
from finarray import util
from finarray.cli import main


@pytest.fixture
def csv_dir(tmp_path):
    """Two days of quotes as CSVs, plus an empty base directory to import into."""
    src = tmp_path / "csv"
    src.mkdir()
    for date in ("2024-11-27", "2024-12-02"):
        rows = []
        for second, (bid, ask) in enumerate([(10.0, 10.2), (10.1, 10.3)]):
            for ticker, offset in (("AAA", 0.0), ("BBB", 5.0)):
                rows.append(
                    {
                        "time": f"15:50:0{second}",
                        "ticker": ticker,
                        "bid": bid + offset,
                        "ask": ask + offset,
                    }
                )
        pd.DataFrame(rows).to_csv(src / f"quotes-{date}.csv", index=False)
    base = tmp_path / "bars"
    base.mkdir()
    return src, base


def run(*argv) -> int:
    return main([str(a) for a in argv])


# ---------------------------------------------------------------------------
# import-csv
# ---------------------------------------------------------------------------


def test_import_csv_creates_date_directories(csv_dir):
    src, base = csv_dir
    assert run("import-csv", base, *sorted(src.glob("*.csv"))) == 0

    bars = fr.BarsSet(str(base))
    assert bars.dates_available() == [dt.date(2024, 11, 27), dt.date(2024, 12, 2)]
    bd = bars["2024-11-27"]
    assert sorted(bd.get_vars_available()) == ["ask", "bid"]
    assert bd.get_var("bid").dims == ("time", "ticker")
    assert (base / "2024-11-27" / "ticker.csv").exists()
    assert (base / "2024-11-27" / "time.csv").exists()


def test_import_csv_refuses_to_overwrite(csv_dir, capsys):
    src, base = csv_dir
    csv = src / "quotes-2024-11-27.csv"
    assert run("import-csv", base, csv) == 0
    assert run("import-csv", base, csv) == 1
    assert "already exists" in capsys.readouterr().err


def test_import_csv_force_replaces(csv_dir):
    src, base = csv_dir
    csv = src / "quotes-2024-11-27.csv"
    assert run("import-csv", base, csv) == 0
    assert run("import-csv", "--force", base, csv) == 0
    assert fr.BarsSet(str(base))["2024-11-27"].get_var("bid").dims == ("time", "ticker")


def test_import_csv_add_extends_an_existing_date(csv_dir):
    src, base = csv_dir
    csv = src / "quotes-2024-11-27.csv"
    assert run("import-csv", base, csv) == 0

    extra = src / "extra-2024-11-27.csv"
    pd.DataFrame(
        [
            {"time": "15:50:00", "ticker": "AAA", "volume": 100},
            {"time": "15:50:01", "ticker": "AAA", "volume": 200},
            {"time": "15:50:00", "ticker": "BBB", "volume": 300},
            {"time": "15:50:01", "ticker": "BBB", "volume": 400},
        ]
    ).to_csv(extra, index=False)
    assert run("import-csv", "--add", base, extra) == 0

    bd = fr.BarsSet(str(base))["2024-11-27"]
    assert sorted(bd.get_vars_available()) == ["ask", "bid", "volume"]
    assert bd.get_var("volume").sel(ticker="AAA").values.tolist() == [100, 200]


def test_import_csv_add_requires_an_existing_date(csv_dir, capsys):
    src, base = csv_dir
    assert run("import-csv", "--add", base, src / "quotes-2024-11-27.csv") == 1
    assert "--add is set" in capsys.readouterr().err


def test_import_csv_rename_mask(csv_dir):
    src, base = csv_dir
    assert run("import-csv", "--rename", "raw_%s", base, src / "quotes-2024-11-27.csv") == 0
    assert sorted(fr.BarsSet(str(base))["2024-11-27"].get_vars_available()) == [
        "raw_ask",
        "raw_bid",
    ]


def test_import_csv_needs_a_date_in_the_filename(csv_dir, capsys):
    src, base = csv_dir
    nameless = src / "quotes.csv"
    nameless.write_text("time,ticker,bid\n15:50:00,AAA,1.0\n")
    assert run("import-csv", base, nameless) == 1
    assert "No YYYY-MM-DD date" in capsys.readouterr().err


def test_import_csv_missing_basedir(tmp_path, capsys):
    assert run("import-csv", tmp_path / "nope", tmp_path / "x.csv") == 1
    assert "not found" in capsys.readouterr().err


def test_import_csv_continues_past_a_bad_file(csv_dir, capsys):
    """One unreadable file must not abandon the rest of the batch."""
    src, base = csv_dir
    bad = src / "broken-2024-12-03.csv"
    bad.write_text("no_time_column\n1\n")
    assert run("import-csv", base, src / "quotes-2024-11-27.csv", bad) == 1
    # the good file still landed
    assert fr.BarsSet(str(base)).dates_available() == [dt.date(2024, 11, 27)]
    assert "broken-2024-12-03.csv" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@pytest.fixture
def imported(csv_dir):
    src, base = csv_dir
    assert run("import-csv", base, *sorted(src.glob("quotes-*.csv"))) == 0
    return base


def test_eval_creates_and_saves_a_variable(imported):
    date_dir = imported / "2024-11-27"
    assert run("eval", "--dir", date_dir, "mid=(bid+ask)/2") == 0

    bd = fr.BarsSet(str(imported))["2024-11-27"]
    assert "mid" in bd.get_vars_available()
    # AAA bid 10.0 ask 10.2 -> 10.1
    assert bd.get_var("mid").sel(ticker="AAA").values[0] == pytest.approx(10.1)


def test_eval_autoloads_what_the_expression_names(imported):
    """Nothing is listed up front; the expression pulls in bid and ask itself."""
    assert run("eval", "--dir", imported / "2024-11-27", "spread=ask-bid") == 0
    bd = fr.BarsSet(str(imported))["2024-11-27"]
    assert bd.get_var("spread").values.ravel().tolist() == pytest.approx([0.2] * 4)


def test_eval_multiple_expressions(imported):
    assert run("eval", "--dir", imported / "2024-11-27", "mid=(bid+ask)/2", "spread=ask-bid") == 0
    assert set(fr.BarsSet(str(imported))["2024-11-27"].get_vars_available()) >= {
        "mid",
        "spread",
    }


def test_eval_filter_masks_created_variables(imported):
    """The filter from the original pipeline: drop absurdly wide quotes."""
    date_dir = imported / "2024-11-27"
    # BBB trades at ~15 with a 0.2 spread; AAA at ~10. Keep only spreads < 0.02 * ask.
    assert run("eval", "--dir", date_dir, "-f", "(ask-bid) < 0.015*ask", "mid=(bid+ask)/2") == 0

    mid = fr.BarsSet(str(imported))["2024-11-27"].get_var("mid")
    # AAA: 0.2 < 0.153? no -> masked out. BBB: 0.2 < 0.2295? yes -> kept.
    assert bool(mid.sel(ticker="AAA").isnull().all())
    assert bool(mid.sel(ticker="BBB").notnull().all())


def test_eval_rejects_a_non_date_directory(tmp_path, capsys):
    (tmp_path / "notadate").mkdir()
    assert run("eval", "--dir", tmp_path / "notadate", "x=1") == 1
    assert "named YYYY-MM-DD" in capsys.readouterr().err


def test_eval_missing_directory(tmp_path, capsys):
    assert run("eval", "--dir", tmp_path / "2024-11-27", "x=1") == 1
    assert "could not open bars directory" in capsys.readouterr().err


def test_eval_with_nothing_to_save_warns(imported):
    with pytest.warns(UserWarning, match="No new variables"):
        assert run("eval", "--dir", imported / "2024-11-27", "bid+ask") == 0


def test_eval_trailing_slash_on_dir(imported):
    assert run("eval", "--dir", str(imported / "2024-11-27") + "/", "mid=(bid+ask)/2") == 0


# ---------------------------------------------------------------------------


def test_no_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        main([])


def test_pipeline_end_to_end(csv_dir):
    """import-csv then eval, the way the original shell pipeline ran."""
    src, base = csv_dir
    assert run("import-csv", "-f", base, *sorted(src.glob("quotes-*.csv"))) == 0
    for date in ("2024-11-27", "2024-12-02"):
        assert run("eval", "--dir", base / date, "mid=(bid+ask)/2") == 0

    bars = fr.BarsSet(str(base), restrict_dates=["2024-11-27", "2024-12-02"])
    series = bars.sel_ticker("AAA").sel_time("15:50:00").cat_var("mid")
    assert series.index.names == ["date"]
    assert series.to_list() == pytest.approx([10.1, 10.1])


# ---------------------------------------------------------------------------
# progress, without tqdm
# ---------------------------------------------------------------------------


def test_progress_iter_falls_back_without_tqdm(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    out = list(util.progress_iter(range(3), desc="work"))
    assert out == [0, 1, 2]
    assert "work: 3/3" in capsys.readouterr().err


def test_progress_iter_can_be_disabled(capsys):
    assert list(util.progress_iter(range(3), disable=True)) == [0, 1, 2]
    assert capsys.readouterr().err == ""


def test_progress_iter_uses_tqdm_when_present():
    pytest.importorskip("tqdm")
    assert list(util.progress_iter(range(3), desc="work")) == [0, 1, 2]
