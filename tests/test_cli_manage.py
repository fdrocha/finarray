"""ls, check, link, rm, import-parquet, and eval over a whole base directory."""

import datetime as dt
import os

import pandas as pd
import pytest

import finarray as fr
from finarray.cli import main


def run(*argv) -> int:
    return main([str(a) for a in argv])


@pytest.fixture
def base(bars_path):
    """A writable copy of the sample tree."""
    return bars_path


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


def test_ls_summary(base, capsys):
    assert run("ls", base) == 0
    out = capsys.readouterr().out
    assert "PARENT: (none)" in out
    assert "4 dates: 2024-11-27 .. 2024-12-04" in out
    assert "4 tickers x 600 times" in out
    assert "8 variables" in out
    assert "mid" in out


def test_ls_dates_only_is_scriptable(base, capsys):
    assert run("ls", base, "--dates-only") == 0
    assert capsys.readouterr().out.split() == [
        "2024-11-27",
        "2024-12-02",
        "2024-12-03",
        "2024-12-04",
    ]


def test_ls_vars_only(base, capsys):
    assert run("ls", base, "--vars-only") == 0
    assert "mid" in capsys.readouterr().out.split()


def test_ls_marks_inherited_variables(base, tmp_path, capsys):
    child = tmp_path / "child"
    assert run("link", child, base) == 0
    capsys.readouterr()
    assert run("ls", child) == 0
    out = capsys.readouterr().out
    assert "PARENT:" in out and str(base) in out
    assert "0 local, 8 inherited" in out
    assert "^ mid" in out


def test_ls_specific_date(base, capsys):
    assert run("ls", base, "--date", "2024-11-27") == 0
    assert "at 2024-11-27" in capsys.readouterr().out


def test_ls_empty_directory(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert run("ls", tmp_path / "empty") == 1
    assert "No dates found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def test_check_clean_tree(base, capsys):
    assert run("check", base) == 0
    assert "check: ok" in capsys.readouterr().out


def test_check_reports_a_missing_variable(base, capsys):
    (base / "2024-12-02" / "mid.nc").unlink()
    assert run("check", base) == 1
    out = capsys.readouterr().out
    assert "1 variable(s) missing on some dates" in out
    assert "mid" in out and "2024-12-02" in out


def test_check_names_several_missing_dates(base, capsys):
    for date in ("2024-12-02", "2024-12-03"):
        (base / date / "mid.nc").unlink()
    assert run("check", base) == 1
    assert "missing on 2 date(s)" in capsys.readouterr().out


def test_check_shapes_catches_a_mismatched_variable(base, capsys):
    """A variable whose grid no longer matches the coordinate files."""
    import numpy as np
    import xarray as xr

    bad = xr.DataArray(
        np.zeros((3, 4), dtype="float32"),
        dims=["time", "ticker"],
        name="mid",
    )
    bad.to_netcdf(base / "2024-11-27" / "mid.nc")
    assert run("check", base, "--shapes", "-D", "2024-11-27") == 1
    out = capsys.readouterr().out
    assert "mid has time=3" in out


def test_check_without_shapes_does_not_open_files(base):
    """The cheap check must stay cheap: presence only."""
    (base / "2024-11-27" / "mid.nc").write_text("not netcdf at all")
    assert run("check", base) == 0


def test_check_restricted_to_dates(base, capsys):
    (base / "2024-12-02" / "mid.nc").unlink()
    assert run("check", base, "-D", "2024-11-27") == 0
    assert "1 date(s)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# link
# ---------------------------------------------------------------------------


def test_link_creates_an_inheriting_child(base, tmp_path, capsys):
    child = tmp_path / "child"
    assert run("link", child, base) == 0
    assert (child / "PARENT").is_symlink()
    assert "4 inherited dates" in capsys.readouterr().err
    assert fr.BarsSet(str(child)).dates_available() == [
        dt.date(2024, 11, 27),
        dt.date(2024, 12, 2),
        dt.date(2024, 12, 3),
        dt.date(2024, 12, 4),
    ]


def test_link_refuses_an_existing_child(base, tmp_path, capsys):
    child = tmp_path / "child"
    child.mkdir()
    assert run("link", child, base) == 1
    assert "already exists" in capsys.readouterr().err


def test_link_requires_a_real_parent(tmp_path, capsys):
    assert run("link", tmp_path / "child", tmp_path / "nope") == 1
    assert "does not exist" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# rm
# ---------------------------------------------------------------------------


def test_rm_deletes_across_dates(base, capsys):
    assert run("rm", base, "mid", "-y") == 0
    assert "deleted 4 file(s)" in capsys.readouterr().out
    assert "mid" not in fr.BarsSet(str(base))["2024-11-27"].get_vars_available()


def test_rm_several_variables(base):
    assert run("rm", base, "mid", "bid", "-y") == 0
    available = fr.BarsSet(str(base))["2024-11-27"].get_vars_available()
    assert "mid" not in available and "bid" not in available


def test_rm_restricted_to_dates(base):
    assert run("rm", base, "mid", "-D", "2024-11-27", "-y") == 0
    bars = fr.BarsSet(str(base))
    assert "mid" not in bars["2024-11-27"].get_vars_available()
    assert "mid" in bars["2024-12-02"].get_vars_available()


def test_rm_dry_run_changes_nothing(base, capsys):
    assert run("rm", base, "mid", "--dry-run") == 0
    assert "would delete 4 file(s)" in capsys.readouterr().out
    assert "mid" in fr.BarsSet(str(base))["2024-11-27"].get_vars_available()


def test_rm_will_not_delete_from_a_parent(base, tmp_path, capsys):
    """The whole point of a child directory is that it cannot damage the parent."""
    child = tmp_path / "child"
    assert run("link", child, base) == 0
    capsys.readouterr()

    with pytest.warns(UserWarning, match="inherited from a parent"):
        assert run("rm", child, "mid", "-y") == 0
    assert "nothing to delete" in capsys.readouterr().out
    # untouched upstream
    assert "mid" in fr.BarsSet(str(base))["2024-11-27"].get_vars_available()


def test_rm_unknown_variable_is_harmless(base, capsys):
    assert run("rm", base, "not_a_variable", "-y") == 0
    assert "nothing to delete" in capsys.readouterr().out


def test_rm_needs_confirmation_when_not_a_tty(base, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert run("rm", base, "mid") == 1
    assert "Refusing to delete without confirmation" in capsys.readouterr().err
    assert "mid" in fr.BarsSet(str(base))["2024-11-27"].get_vars_available()


def test_rm_aborts_on_a_negative_answer(base, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert run("rm", base, "mid") == 0
    assert "aborted" in capsys.readouterr().out
    assert "mid" in fr.BarsSet(str(base))["2024-11-27"].get_vars_available()


def test_rm_deletes_an_alias_symlink(base):
    bars = fr.BarsSet(str(base))
    bars["2024-11-27"].set_alias("midpoint", "mid")
    assert (base / "2024-11-27" / "midpoint.nc").is_symlink()
    assert run("rm", base, "midpoint", "-D", "2024-11-27", "-y") == 0
    assert not os.path.lexists(base / "2024-11-27" / "midpoint.nc")
    # the target survives
    assert (base / "2024-11-27" / "mid.nc").exists()


# ---------------------------------------------------------------------------
# eval over a base directory
# ---------------------------------------------------------------------------


def test_eval_all_dates(base, capsys):
    assert run("eval", "--dir", base, "--all-dates", "spread=ask-bid") == 0
    assert "for 4 date(s)" in capsys.readouterr().err
    bars = fr.BarsSet(str(base))
    for date in bars.dates_available():
        assert "spread" in bars[date].get_vars_available()


def test_eval_dates_range_implies_a_base_directory(base):
    assert run("eval", "--dir", base, "-D", "2024-11-27:2024-12-02", "spread=ask-bid") == 0
    bars = fr.BarsSet(str(base))
    assert "spread" in bars["2024-11-27"].get_vars_available()
    assert "spread" not in bars["2024-12-04"].get_vars_available()


def test_eval_all_dates_with_a_filter(base):
    assert run("eval", "--dir", base, "--all-dates", "-f", "mid > 1e9", "capped=mid*2") == 0
    capped = fr.BarsSet(str(base))["2024-11-27"].get_var("capped")
    assert bool(capped.isnull().all())


def test_eval_single_date_still_works(base, capsys):
    """The original freval invocation must keep working unchanged."""
    assert run("eval", "--dir", base / "2024-11-27", "spread=ask-bid") == 0
    assert "wrote spread" in capsys.readouterr().err


def test_eval_all_dates_stops_on_error_by_default(base, capsys):
    (base / "2024-12-02" / "bid.nc").unlink()
    assert run("eval", "--dir", base, "--all-dates", "spread=ask-bid") == 1
    err = capsys.readouterr().err
    assert "cannot evaluate" in err and "bid" in err


def test_eval_all_dates_can_skip_failures(base, capsys):
    (base / "2024-12-02" / "bid.nc").unlink()
    with pytest.warns(UserWarning):
        assert run("eval", "--dir", base, "--all-dates", "--skip-errors", "spread=ask-bid") == 1
    bars = fr.BarsSet(str(base))
    assert "spread" in bars["2024-11-27"].get_vars_available()


# ---------------------------------------------------------------------------
# import-parquet
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_parquet(tmp_path):
    dates = [dt.date(2024, 11, 27), dt.date(2024, 12, 2)]
    index = pd.MultiIndex.from_product(
        [dates, ["AAA", "BBB", "CCC", "SPY"]], names=["date", "ticker"]
    )
    path = tmp_path / "daily.parquet"
    pd.DataFrame({"sector": range(8), "weight": [0.125] * 8}, index=index).to_parquet(path)
    return path


def test_import_parquet(base, daily_parquet, capsys):
    assert run("import-parquet", base, daily_parquet) == 0
    assert "wrote sector, weight for 2 date(s)" in capsys.readouterr().err

    bd = fr.BarsSet(str(base))["2024-11-27"]
    assert bd.get_var("sector").dims == ("ticker",)
    assert bd.get_var("weight").sel(ticker="AAA").item() == pytest.approx(0.125)
    # dates not in the file are untouched
    assert "sector" not in fr.BarsSet(str(base))["2024-12-04"].get_vars_available()


def test_import_parquet_selected_columns(base, daily_parquet):
    assert run("import-parquet", base, daily_parquet, "-v", "weight") == 0
    available = fr.BarsSet(str(base))["2024-11-27"].get_vars_available()
    assert "weight" in available and "sector" not in available


def test_import_parquet_selected_dates(base, daily_parquet):
    assert run("import-parquet", base, daily_parquet, "-D", "2024-11-27") == 0
    bars = fr.BarsSet(str(base))
    assert "weight" in bars["2024-11-27"].get_vars_available()
    assert "weight" not in bars["2024-12-02"].get_vars_available()


def test_import_parquet_unknown_column(base, daily_parquet, capsys):
    assert run("import-parquet", base, daily_parquet, "-v", "nope") == 1
    assert "no column(s) nope" in capsys.readouterr().err


def test_import_parquet_rejects_a_bad_index(base, tmp_path, capsys):
    path = tmp_path / "flat.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(path)
    assert run("import-parquet", base, path) == 1
    assert "expected a MultiIndex of (date, ticker)" in capsys.readouterr().err


def test_import_parquet_rejects_a_non_parquet_file(base, tmp_path, capsys):
    path = tmp_path / "data.csv"
    path.write_text("date,ticker,a\n")
    assert run("import-parquet", base, path) == 1
    assert "Expected a .parquet file" in capsys.readouterr().err


def test_import_parquet_with_no_overlapping_dates(base, tmp_path, capsys):
    index = pd.MultiIndex.from_product([[dt.date(1999, 1, 4)], ["AAA"]], names=["date", "ticker"])
    path = tmp_path / "old.parquet"
    pd.DataFrame({"a": [1.0]}, index=index).to_parquet(path)
    assert run("import-parquet", base, path) == 1
    assert "no dates in common" in capsys.readouterr().err


def test_import_parquet_then_query(base, daily_parquet, capsys):
    """The two halves of the pipeline meet: daily values come back out."""
    assert run("import-parquet", base, daily_parquet) == 0
    capsys.readouterr()
    assert run("query", base, "-v", "weight", "--ticker", "AAA", "-D", "2024-11-27") == 0
    lines = [
        line for line in capsys.readouterr().out.strip().splitlines() if not line.startswith("#")
    ]
    assert lines[0] == "date,weight"
    assert lines[1] == "2024-11-27,0.125"
