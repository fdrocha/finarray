# finarray

**xarray-backed intraday bars.** A day of market data is a directory; each variable in it is a
file. `finarray` gives you a lazy, chainable way to slice that data by ticker and time, derive new
variables from it, write them back, and collapse hundreds of days into a single pandas DataFrame.

It is a thin layer — about 1500 lines — over [xarray](https://xarray.dev). Everything it writes is
plain netCDF and CSV, readable without this library.

```python
import finarray as fr

bars = fr.BarsSet("~/bars/eod", restrict_dates=dates)

# One number per day: SPY's mid price at 15:54, across every date.
bars.sel_ticker("SPY").sel_time("15:54:00").cat_var("mid")
```

---

## Why

If you keep intraday panel data as one file per day, you end up re-reading a whole day to get at one
column. If you keep it in a database, you give up the array semantics that make this kind of
analysis pleasant. `finarray` splits the difference: the day is a directory, each variable is its own
netCDF file, and nothing is read until something asks for it.

That layout buys three things:

- **You can have hundreds of variables per day** and pay only for the ones you touch.
- **Adding a variable is writing a file.** No migration, no schema, no rewrite of the day.
- **Layering is a symlink.** A scratch directory can inherit from your real data, and anything you
  write to it shadows the original without copying or endangering it.

## Install

```bash
uv add finarray          # or: pip install finarray
```

Requires Python 3.11+.

## The layout on disk

```
bars/eod/                    <- a BarsSet: the base directory
├── 2025-01-13/              <- a Bars: one date
│   ├── ticker.csv           <- coordinate labels, one column named "ticker"
│   ├── time.csv             <- coordinate labels, one column named "time" (datetimes)
│   ├── mid.nc               <- one xr.DataArray per file, dims (time, ticker)
│   ├── bid.nc
│   ├── adv.nc               <- some variables are (ticker,)-only "daily" values
│   └── ...
├── 2025-01-14/
└── ...
```

`ticker.csv` and `time.csv` define the two coordinates for the whole date directory; every `.nc`
file in it is shape-consistent with them. To build such a directory from a CSV, see
[Ingesting data](#ingesting-data).

## Quickstart

Everything below runs against synthetic data — `examples/make_sample_bars.py` in this repo writes a
small tree of random-walk prices for four tickers over four dates:

```bash
python examples/make_sample_bars.py /tmp/sample-bars
```

```python
import finarray as fr

bars = fr.BarsSet("/tmp/sample-bars", restrict_dates=["2024-11-27", "2024-12-02"])

bars.dates_available()
# [datetime.date(2024, 11, 27), datetime.date(2024, 12, 2)]

bd = bars["2024-11-27"]        # a single date
bd.get_vars_available()
# ['adv', 'ask', 'beta', 'bid', 'listing_exchange', 'mid', 'ref_price', 'volume']
bd.live_vars()                 # nothing read from disk yet
# []
```

### Expressions load their own data

`eval` runs against the day's dataset. When an expression names a variable that isn't in memory,
`finarray` finds it on disk, loads it, and retries:

```python
bd.eval("(bid + ask) * 0.5")   # reads bid.nc and ask.nc, on demand
bd.live_vars()
# ['bid', 'ask']

bd.assign(spread="ask - bid")  # define a new variable from an expression
bd.save_created_vars()         # writes spread.nc next to the others
```

A `Bars` forwards unknown attributes to the underlying `xr.Dataset`, so once something is loaded it
behaves the way you'd expect:

```python
bd.mid.sel(ticker="SPY").to_pandas().plot()
bd[["mid", "bid"]]             # a list gives you back a Bars, not a DataArray
```

### Selections are lazy and immutable

`sel_ticker`, `sel_time`, and `sel_time_slice` each return a *new* object with the selection queued
up. Queued selections are applied to variables as they load, so narrowing before loading means you
never read the full array:

```python
spy = bd.sel_ticker("SPY")     # bd is unchanged
spy.get_var("volume").dims     # volume.nc is read and sliced in one step
# ('time',)

window = bars.sel_time_slice("15:50:00", "15:52:00")
```

Times can be strings, `datetime.time`, or `datetime.datetime` — a bare time is combined with the
date of the bars it's applied to.

### Scalar selections change the shape you get back

This is the one piece of behaviour worth reading twice. Selecting a **scalar** drops that dimension;
selecting a **one-element list** keeps it. `BarsSet.cat_var` uses that to decide the index of the
Series it builds across dates:

| selection | `cat_var("mid")` index |
| --- | --- |
| *(none)* | `(time, ticker)` |
| `.sel_ticker(["SPY", "AAA"])` | `(time, ticker)` |
| `.sel_ticker(["SPY"])` | `(time, ticker)` |
| `.sel_ticker("SPY")` | `(time,)` |
| `.sel_time(["15:54:00"])` | `(time, ticker)` |
| `.sel_time("15:54:00")` | `(date, ticker)` |
| `.sel_ticker("SPY").sel_time("15:54:00")` | `(date,)` |

The order you chain them in doesn't matter. The intuition: once time is pinned to a single value,
"time" stops being the interesting axis and "date" takes its place.

```python
bars.sel_ticker("SPY").sel_time("15:54:00").cat_var("mid")
# date
# 2024-11-27    500.543304
# 2024-12-02    499.584534
# Name: mid, dtype: float64

bars.sel_ticker("AAA").sel_time("15:54:00").cat_vars(["mid", "bid", "ask"])
# a DataFrame indexed by date
```

### Working across dates

`mapcat` runs a function per date and concatenates the results, skipping (and warning about) dates
that raise:

```python
import numpy as np

bars.mapcat(
    lambda bd: np.abs(bd.get_var("mid")).quantile([0.1, 0.5, 0.9]).to_pandas(),
    add_date=True,
)
```

`assign_expr` / `assign_func` define a variable on every loaded date:

```python
bars.assign_expr("spread", "ask - bid")
bars.assign_func("norm_vol", lambda bd: bd.get_var("volume") / bd.get_var("adv"))
```

### Joining bars onto an event table

Given a DataFrame indexed by `(date, ticker)`, pull bars variables onto it. With a single time
selected, every row gets that time's value:

```python
enriched = bars.sel_time("15:54:00").attach_to_df_by_tickers(events, ["mid", "bid"])
```

Or let each row look up **its own** timestamp — `method="ffill"` takes the most recent bar at or
before each event:

```python
enriched = bars.attach_to_df_by_ticker_time(
    events, ["mid"], time_column="event_time", method="ffill"
)
```

Rows whose ticker has no data that day are dropped silently. Use `bars.restrict_other(df)` first if
you want to know which those are.

### Filtering the ticker universe

`sel_where` filters on *daily* (ticker-only) variables. The names are yours — the library has no
opinion about what your columns are called. A `(min, max)` tuple is an inclusive range, with either
bound `None` for unbounded; anything else is an equality test:

```python
liquid = bars.sel_where(
    adv=(1e7, None),          # at least $10m average daily volume
    ref_price=(2, 2500),      # price between 2 and 2500
    listing_exchange=1,       # equality
    extra_tickers=["SPY"],    # kept regardless of the above
)
```

The filter is remembered, so dates loaded later are filtered too. Constraint variables are loaded on
demand, and must have `ticker` as their only dimension.

### Layering with `PARENT`

A `PARENT` symlink chains one base directory to another. The child sees the parent's dates and
variables; anything written to the child shadows the parent without touching it:

```python
fr.create_child_bars("/tmp/scratch-bars", "/tmp/sample-bars")

scratch = fr.BarsSet("/tmp/scratch-bars", restrict_dates=["2024-11-27"])
bd = scratch["2024-11-27"]
bd.assign(spread="ask - bid")
bd.save_created_vars()              # writes into /tmp/scratch-bars only

bd.get_vars_available(local_only=True)
# ['spread']
bd.get_vars_available()             # inherited variables are visible too
# ['adv', 'ask', 'beta', 'bid', 'listing_exchange', 'mid', 'ref_price', 'spread', 'volume']
```

This is how you run experiments — derived variables, alternative fills, backtest outputs — against
real data without copying it or being able to corrupt it.

### Ingesting data

`load_csv` reads a `(time, ticker)` CSV into a Dataset (downcasting to 32-bit by default), and
`to_frdir` writes a Dataset out as a new date directory:

```python
from finarray import util

ds = util.load_csv("quotes-2024-11-27.csv", date="2024-11-27")
util.to_frdir(ds, "/tmp/sample-bars")     # creates /tmp/sample-bars/2024-11-27/
```

### Return profiles

`finarray.profile` builds return profiles for a signal: take the sign of an alpha per ticker,
measure forward returns from a reference time across the bar grid, optionally beta-hedge against a
hedge ticker, and average across tickers and dates.

```python
from finarray import profile

ds = profile.profile_fixed(
    bars.sel_time_slice("15:50:00", "15:59:59"),
    alphas,                  # (date, ticker)-indexed DataFrame or Series
    alpha_col="alpha",
    beta_col="beta",
    hedge_ticker="SPY",
    t_kind="seconds",        # time axis as seconds from the reference
)
ds.ret.mean(dim="date").to_pandas().plot()
```

`mode="cps"` gives cents per share instead of percent returns.

## API summary

**`Bars`** — one date.
`get_var` · `get_vars` · `get_value` · `load_var(s)` · `eval` · `assign` · `save_var(s)` ·
`save_created_vars` · `create_var` · `set_alias` · `get_vars_available` · `has_var` · `live_vars` ·
`sel_ticker` · `sel_time` · `sel_time_slice` · `sel_where` · `add_daily_var` · `merge_df_asof` ·
`lookup_indirect` · `reindex_array` · `tcast`

**`BarsSet`** — many dates.
`dates_available` · `dates_loaded` · `has_date` · `load_date(s)` · `reload` · `iter_bars` ·
`cat_var` · `cat_vars` · `mapcat` · `attach_to_df_by_tickers` · `attach_to_df_by_ticker_time` ·
`restrict_other` · `assign` · `assign_expr` · `assign_func` · the same `sel_*` methods

**Module level.**
`bars_getter(base_path)` returns a preconfigured `BarsSet` factory · `create_child_bars` ·
`Where` · `load_csv` · `to_frdir` · `finarray.profile`

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests examples
uv run mypy
```

## History

`finarray` was written for, and extracted from, a private single-person research repository focused
on the NYSE closing auction. It has been generalized where that repository's assumptions had leaked
into it, but its shape still reflects that origin: it is built for panels of ticker-by-time data at
second granularity, over a window of a day rather than a whole session.

## License

MIT — see [LICENSE](LICENSE).
