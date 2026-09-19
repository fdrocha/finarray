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

Requires Python 3.11+. This also installs the [`finarray` command](#command-line).

[tqdm](https://github.com/tqdm/tqdm) is not a dependency, but progress bars use it when it happens
to be installed, and fall back to a plain stderr counter when it isn't.

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

For a directory of files, the [`finarray` command](#command-line) does this from the shell.

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

### Backtesting

`finarray.backtest` turns a path of target positions into pnl. It is an optional extra, because it
pulls in numba:

```bash
uv add "finarray[backtest]"      # or: pip install "finarray[backtest]"
```

State how many shares you want to hold at each bar, and the engine trades the difference at that
bar's price, charging linear fees:

```python
from finarray import backtest

result = backtest.run_backtest(
    bars.sel_time_slice("15:50:00", "15:59:00"),
    "1e4 / mid",                   # target shares per bar; an expression or a callable
    prices="mid",                  # what you trade at
    unwind_price="close_price",    # flatten everything at the close
    vol_cap_per_bar="volume_cap",  # optional: max shares per bar per ticker
    fees=backtest.Fees.mils(2.18) + backtest.Fees.bips(1),
)
```

`Fees` is quoted the way venues quote it — mils (tenths of a cent) per share, bips (hundredths of a
percent) of notional — and composes with `+`, `*` and `/`. A `sell_only=True` fee is halved, since
the engine doesn't track side.

The result summarizes itself:

```python
>>> result
<BacktestResult>
  ndays         40
  total_pnl     -8.402e+04
  pnl           -2,101
  pnl_median    -327.8
  pnl_std       8,078
  sharpe        -4.128
  tstat         -1.645
  win_rate      42.5
  best_day      2.101e+04
  worst_day     -2.008e+04
  max_drawdown  1.187e+05
  max_dd_days   39
  shares        2.398e+05
  dollars       1.712e+07
  ntickers      429.1
  close_pos     1.196e+05
  cps           -0.8761
  margin_bps    -1.227
```

`sharpe` annualizes daily pnl over 252 days and `tstat` says whether the mean daily pnl is
distinguishable from zero. `cps` (cents per share) and `margin_bps` (basis points of notional
traded) are profitability per unit of trading — the two numbers that tell you whether a strategy
survives its own costs. `close_pos` is the gross share position carried into the close.

Underneath, four views of the same run:

```python
result.rows            # (date, ticker): pnl, eod_pos, close_pos, volume, dvolume
result.daily           # one row per date
result.by_ticker()     # totals per ticker, most profitable first
result.drawdowns()     # every peak-to-trough episode, worst first
result.cumulative_pnl().plot()
```

**What the model does and doesn't do.** It fills your whole requested size at the bar price, so
there is no market impact, no queue position, and no partial fills beyond `vol_cap_per_bar`. Costs
are linear in shares and notional. Positions are flat-to-flat within the day unless you skip
`unwind_price`. Treat the output as an upper bound with an explicit cost model attached, not as a
simulation of execution.

## Command line

Installing the package puts a `finarray` command on your path, with two subcommands that cover
getting data in and deriving things from it once it is there.

### `finarray import-csv` — CSVs to date directories

Each CSV needs `time` and `ticker` columns and a `YYYY-MM-DD` date somewhere in its filename:

```bash
finarray import-csv bars/eod quotes-*.csv
```

```
bars/eod/2025-01-13/{ticker.csv,time.csv,bid.nc,ask.nc}
bars/eod/2025-01-14/...
```

`--force` replaces a date directory that already exists. `--add` goes the other way: it keeps an
existing directory's coordinates and writes the CSV's columns into it as extra variables, which is
how you attach a second source to a day you already have.

```bash
finarray import-csv --add bars/eod trades-2025-01-13.csv
finarray import-csv --rename 'raw_%s' bars/eod quotes-2025-01-13.csv   # -> raw_bid, raw_ask
```

A file that fails is reported and skipped; the rest of the batch still runs, and the exit status is
non-zero if anything failed.

### `finarray eval` — derive variables and save them

Expressions run against one date directory, and autoload whatever they name — you don't list inputs:

```bash
finarray eval --dir bars/eod/2025-01-13 'mid=(bid+ask)/2' 'spread=ask-bid'
```

Every variable the expressions create is written to the directory. `--filter` takes a boolean
expression and masks the created variables to where it holds, which is the idiomatic way to drop
bad data at the point of derivation rather than at every use:

```bash
finarray eval --dir bars/eod/2025-01-13 \
    -f '(abs(ask-bid) <= ask*0.01) | (abs(ask-bid) <= 0.05)' \
    'mid=(bid+ask)/2'
```

That writes `mid` as NaN wherever the quote was implausibly wide, and leaves it alone elsewhere.

### The two together

Building a set of bars from scratch is usually one loop:

```bash
finarray import-csv --force bars/eod quotes-*.csv
for d in bars/eod/*/; do
    finarray eval --dir "$d" -f '(abs(ask-bid) <= ask*0.01)' 'mid=(bid+ask)/2'
done
```

After which the Python API above has something to open:

```python
bars = fr.BarsSet("bars/eod")
bars.sel_ticker("AAA").sel_time("15:50:00").cat_var("mid")
```

Set `PY_TRACEBACK=1` to get full tracebacks instead of one-line error messages.

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

**`finarray.backtest`** (extra).
`run_backtest` · `run_date` · `BacktestResult` · `Fees` · `summarize` · `daily_stats` · `drawdowns`

**Command line.**
`finarray import-csv` · `finarray eval`

## Development

```bash
uv sync --all-extras
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
