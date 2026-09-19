"""PARENT-symlink inheritance between bars base directories."""

import os

import pytest
import xarray as xr

import finarray as fr


@pytest.fixture
def child_path(bars_path, tmp_path):
    child = tmp_path / "child"
    fr.create_child_bars(str(child), str(bars_path))
    return child


def test_create_child_bars_makes_the_link(child_path, bars_path):
    link = child_path / "PARENT"
    assert os.path.islink(link)
    assert os.path.realpath(link) == os.path.realpath(bars_path)


def test_create_child_bars_refuses_an_existing_path(bars_path, tmp_path):
    existing = tmp_path / "already-here"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        fr.create_child_bars(str(existing), str(bars_path))


def test_create_child_bars_requires_a_real_parent(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        fr.create_child_bars(str(tmp_path / "child"), str(tmp_path / "nope"))


def test_child_sees_parent_dates(child_path, dates):
    bars = fr.BarsSet(str(child_path))
    assert bars.dates_available(check_parents=True) == dates


def test_child_reads_parent_variables(child_path, bars_path, dates):
    child = fr.BarsSet(str(child_path), restrict_dates=[dates[0].isoformat()])
    parent = fr.BarsSet(str(bars_path), restrict_dates=[dates[0].isoformat()])
    xr.testing.assert_allclose(child[dates[0]].get_var("mid"), parent[dates[0]].get_var("mid"))


def test_child_writes_do_not_touch_the_parent(child_path, bars_path, dates):
    date = dates[0]
    child = fr.BarsSet(str(child_path), restrict_dates=[date.isoformat()])
    bd = child[date]
    bd.assign(spread="ask - bid")
    bd.save_created_vars()

    assert (child_path / date.isoformat() / "spread.nc").exists()
    assert not (bars_path / date.isoformat() / "spread.nc").exists()

    parent = fr.BarsSet(str(bars_path), restrict_dates=[date.isoformat()])
    assert "spread" not in parent[date].get_vars_available()


def test_local_only_hides_inherited_vars(child_path, dates):
    date = dates[0]
    child = fr.BarsSet(str(child_path), restrict_dates=[date.isoformat()])
    bd = child[date]
    bd.assign(spread="ask - bid")
    bd.save_created_vars()
    assert bd.get_vars_available(local_only=True) == ["spread"]
    assert "mid" in bd.get_vars_available()


def test_child_var_shadows_the_parent(child_path, dates):
    """Lookup walks [self] + parents, so a local file wins."""
    date = dates[0]
    child = fr.BarsSet(str(child_path), restrict_dates=[date.isoformat()])
    bd = child[date]
    bd["mid"] = bd.get_var("mid") * 0 + 42.0
    bd.save_var("mid")

    fresh = fr.BarsSet(str(child_path), restrict_dates=[date.isoformat()])[date]
    assert float(fresh.get_var("mid").max()) == 42.0
