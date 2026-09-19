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

Installing the package puts a `finarray` command on your path:

```
finarray ls BASEDIR                     what a directory holds
finarray check BASEDIR                  look for gaps and inconsistencies
finarray query BASEDIR -v mid           read data out, to stdout or a file
finarray import-csv BASEDIR FILE...     CSVs -> date directories
finarray import-parquet BASEDIR FILE    daily (date, ticker) values
finarray eval --dir DIR EXPR...         derive variables and save them
finarray link CHILD PARENT              a child directory inheriting a parent
finarray rm BASEDIR VAR...              delete variables
```

Set `PY_TRACEBACK=1` for full tracebacks instead of one-line error messages.

### `finarray ls` — what's in there

```console
$ finarray ls bars/eod
bars/eod
  PARENT: (none)
  363 dates: 2025-01-13 .. 2026-06-30
  at 2026-06-30: 1689 tickers x 661 times (2026-06-30T15:49:00 .. 2026-06-30T16:00:00)
  66 variables:
      Nasize
      Nask
      ...
```

On a child directory it marks what is inherited rather than local:

```console
$ finarray ls bars/eod_bkt
bars/eod_bkt
    PARENT: bars/eod
  363 dates: 2025-01-13 .. 2026-06-30
  66 variables (0 local, 66 inherited):
    ^ Nasize
    ...
```

`--dates-only` and `--vars-only` print one item per line, for shell loops.

### `finarray query` — read data out

Writes CSV to stdout, capped at 50 rows so an exploratory query can't flood your terminal:

```console
$ finarray query bars/eod -v mid,bid --ticker SPY --time 15:54:00
date,mid,bid
2025-01-13,581.0,580.989990234375
2025-01-14,581.5150146484375,581.5
...
```

Selecting a **single** ticker or time drops that column; selecting a **list** keeps it — the same
rule as `sel_ticker`/`sel_time` in Python, with the date always first:

| flags | columns |
| --- | --- |
| *(none)* | `date, time, ticker, …` |
| `--ticker SPY --time 15:54:00` | `date, …` |
| `--time 15:54:00` | `date, ticker, …` |
| `--tickers SPY,AAA` | `date, time, ticker, …` |

```bash
finarray query bars/eod -v mid --time-slice 15:50:00,15:59:59 --ticker SPY
finarray query bars/eod -v mid -D 2025-01-13:2025-01-17        # inclusive date range
finarray query bars/eod -v mid -D 2025-01-13,2025-01-15        # or a list
```

### Filtering the universe

`--where` restricts tickers by a daily (ticker-only) variable. `VAR=MIN:MAX` is an inclusive range
with either end optional; `VAR=VALUE` is an equality test. Repeat it to AND conditions together:

```console
$ finarray query bars/eod -v Ndvol3m,Yadjusted_close --time 15:54:00 -D 2026-06-30 \
      --where 'Ndvol3m=1e8:' --where 'Yadjusted_close=:50'
date,ticker,Ndvol3m,Yadjusted_close
2026-06-30,ASX,128800000.0,42.13
2026-06-30,B,105900000.0,36.93
...
```

The variable names are yours — `--where` knows nothing about what your columns are called, it just
loads the one you name and checks it has `ticker` as its only dimension. `--extra-tickers` keeps
tickers regardless, for a benchmark you always want alongside a filtered universe:

```bash
finarray query bars/eod -v mid --where 'Ndvol3m=1e8:' --extra-tickers SPY --time 15:54:00
```

This is the [`sel_where`](#filtering-the-ticker-universe) filter from the Python API, so the same
rules apply: constraint variables are loaded on demand, and a time-varying variable is rejected.

**The limit is a read budget, not a truncation.** `query` streams: it produces rows date by date and
stops as soon as it has enough, so a capped query against 363 dates of 1689 × 661 bars reads one
date — and within it, one time step:

```console
$ time finarray query bars/eod -v mid
date,time,ticker,mid
2025-01-13,2025-01-13 15:49:00,A,133.3
...
# ... stopped at 50 rows. Use --limit N for more, --limit 0 for all, or --output FILE to export.
0.92s total
```

`--limit N` raises it, `--limit 0` removes it.

### Exporting

`--output` writes to a file instead, with no limit unless you ask for one. The format comes from the
extension (`.csv`, `.tsv`, `.parquet`) or `--format`:

```bash
finarray query bars/eod -v mid,bid --ticker SPY -o spy.csv
finarray query bars/eod -v mid --time 15:54:00 -D 2025-06-01:2025-06-30 -o june.parquet
```

Parquet is written in per-date chunks as the data is collected, so exporting more than fits in
memory is fine.

### `finarray eval` — derive variables

Expressions autoload whatever they name, so nothing has to be listed up front:

```bash
finarray eval --dir bars/eod/2025-01-13 'mid=(bid+ask)/2' 'spread=ask-bid'
```

`--all-dates` runs over a whole base directory instead of one date, and `--dates` narrows that:

```bash
finarray eval --dir bars/eod --all-dates 'mid=(bid+ask)/2'
finarray eval --dir bars/eod -D 2025-01-13:2025-01-17 'mid=(bid+ask)/2'
```

`--filter` masks the created variables to where a boolean expression holds, which is how you drop
bad data once rather than at every use:

```bash
finarray eval --dir bars/eod --all-dates \
    -f '(abs(ask-bid) <= ask*0.01) | (abs(ask-bid) <= 0.05)' \
    'mid=(bid+ask)/2'
```

That writes `mid` as NaN wherever the quote was implausibly wide. `--skip-errors` warns and carries
on past a date that fails instead of stopping.

### Getting data in

`import-csv` builds date directories. Each file needs `time` and `ticker` columns and a
`YYYY-MM-DD` date somewhere in its name:

```bash
finarray import-csv bars/eod quotes-*.csv
finarray import-csv --add bars/eod trades-2025-01-13.csv     # extra variables, existing coords
finarray import-csv --rename 'raw_%s' bars/eod quotes-2025-01-13.csv
```

`--force` replaces a date that already exists. A file that fails is reported and skipped; the rest
of the batch still runs, and the exit status is non-zero if anything failed.

`import-parquet` attaches daily, ticker-only values from a parquet file indexed by `(date, ticker)`
— one value per ticker per day, such as a sector code or an ADV:

```bash
finarray import-parquet bars/eod daily.parquet
finarray import-parquet bars/eod daily.parquet -v adv,sector -D 2025-01-13:2025-01-17
```

Dates in the file that the bars don't have are skipped.

### `finarray link` and `finarray rm`

`link` creates a child directory that inherits a parent's dates and variables — the scratch-space
pattern from [Layering with `PARENT`](#layering-with-parent):

```bash
finarray link bars/eod_bkt bars/eod
```

`rm` deletes variables. It asks before deleting, takes `--dry-run`, and **will not delete out of a
parent**: a variable you only inherit is reported and left alone, so you cannot damage the upstream
data from inside a child directory.

```bash
finarray rm bars/eod stale_signal --dry-run
finarray rm bars/eod stale_signal old_alpha -D 2025-01-13:2025-01-17 -y
```

### `finarray check` — find gaps

```console
$ finarray check bars/eod
363 date(s), 66 variable(s) in bars/eod

2 variable(s) missing on some dates:
  beta_spy                 missing on 2 date(s): 2025-03-20, 2025-04-01
  ei_imbalance3            missing on 1 date(s): 2026-06-30
```

That's how you spot a generation run that half-failed. It exits non-zero if anything is wrong. The
default check is cheap — presence only. `--shapes` additionally opens every variable file and checks
its dimensions against the date's coordinates, which is thorough but slow.

## API summary## API summary

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
`finarray ls` · `finarray check` · `finarray query` · `finarray import-csv` ·
`finarray import-parquet` · `finarray eval` · `finarray link` · `finarray rm`

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
