import datetime as dt
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from make_sample_bars import (  # noqa: E402
    DEFAULT_DATES,
    DEFAULT_TICKERS,
    make_sample_bars,
)

import finarray as fr  # noqa: E402


@pytest.fixture(scope="session")
def sample_tree(tmp_path_factory) -> Path:
    """A read-only bars tree shared by every test that doesn't write."""
    path = tmp_path_factory.mktemp("sample-bars")
    make_sample_bars(str(path))
    return path


@pytest.fixture
def bars_path(sample_tree, tmp_path) -> Path:
    """A private copy of the sample tree, safe to write to."""
    path = tmp_path / "bars"
    shutil.copytree(sample_tree, path)
    return path


@pytest.fixture
def dates() -> list[dt.date]:
    return [dt.date.fromisoformat(d) for d in DEFAULT_DATES]


@pytest.fixture
def tickers() -> list[str]:
    return list(DEFAULT_TICKERS)


@pytest.fixture
def bars(sample_tree) -> fr.BarsSet:
    """A BarsSet over every sample date, all loaded."""
    return fr.BarsSet(str(sample_tree), restrict_dates=list(DEFAULT_DATES))


@pytest.fixture
def bars_rw(bars_path) -> fr.BarsSet:
    return fr.BarsSet(str(bars_path), restrict_dates=list(DEFAULT_DATES))


@pytest.fixture
def bd(bars) -> fr.Bars:
    """A single date."""
    return bars["2024-11-27"]
