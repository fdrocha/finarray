# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this repo is

This repo is the public extraction of `finarray`, a small Python library that until now lived
inside a private research monorepo at `~/trading/quant-research/python-libs/finarray`.

**Scope rules for this machine:** only `~/trading/quant-research/` and `~/trading/finarray/`
are in scope. `quant-research/` is **read-only** — it is the live private monorepo and is used
purely as the source to read from. All writes happen here, in `~/trading/finarray/`.

Environment here is managed with **uv** (the source repo has no venv and no `pyproject.toml`;
it just puts `python-libs/` on `PYTHONPATH`).

## What finarray does

`finarray` is a thin, opinionated layer over `xarray` for working with **intraday "bars"**:
per-date panels of financial data indexed by two dimensions, `ticker` and `time`.

The central idea is that a day of data is a *directory on disk*, and each variable ("factor")
in it is a separate netCDF file. You can therefore have hundreds of variables per date and pay
only for the ones you actually touch — the library loads variables lazily, on demand, and can
write new ones back. On top of that it provides a chainable selection API and helpers for
collapsing many dates into a single pandas DataFrame.

In the source monorepo it is imported as `fr` (`import finarray as fr`) and is used for
research on the NYSE closing auction: `~/trading/bars/eod` holds 1-second bars for the last
10 minutes of each trading day for NYSE-listed tickers plus SPY, with ~59 variables per date.

### On-disk format

```
<base_path>/                 <- a BarsSet
├── PARENT -> /other/base    <- optional symlink: inheritance chain (see below)
├── 2025-01-13/              <- a Bars (one date)
│   ├── ticker.csv           <- coordinate labels, single column named "ticker"
│   ├── time.csv             <- coordinate labels, single column named "time" (datetimes)
│   ├── bid.nc               <- one xr.DataArray per file, dims (time, ticker)
│   ├── ask.nc
│   ├── Ndvol3m.nc           <- some vars are (ticker,)-only "daily" vars
│   └── ...
└── 2025-01-14/
```

- `ticker.csv` / `time.csv` define the coordinates for the whole directory; every `.nc` file
  in it is shape-consistent with them. (Despite the historical naming in older notes, the
  `time` coordinate is a *datetime*, not a date.)
- Variables are plain netCDF `DataArray`s, so they are readable with bare `xarray`.
- Aliases are supported as symlinks between `.nc` files (`Bars.set_alias`).
- **Inheritance:** a `PARENT` symlink in a base directory chains it to another base directory.
  A child can add or override variables for a date without copying the parent's data; lookups
  walk `[self] + parent chain` for both coordinate files and variable files. Created with
  `create_child_bars(child, parent)`. This is how backtest scratch dirs (`bars/eod_bkt`) sit on
  top of the real bars (`bars/eod`) without duplicating them.

## Module map (source: `quant-research/python-libs/finarray/`, ~1500 lines total)

| File | Lines | Contents |
|---|---|---|
| `__init__.py` | 11 | Exports `Bars`, `BarsSet`, `bars_getter`, `create_child_bars`, `profile`, `TimeLike`. Also a "wishlist" comment of intended improvements. |
| `bars.py` | 388 | `Bars` — one date. |
| `bars_set.py` | 453 | `BarsSet` — many dates; plus `create_child_bars`, `bars_getter`. |
| `sel_array.py` | 167 | The selection algebra: `BarsSel` ABC, `TimeSel`, `TickerSel`, `TimeSliceSel`, `SelMixin`. |
| `util.py` | 257 | Coordinate/date casting, CSV→bars ingestion (`load_csv`, `to_frdir`), dim/var assertions, warning helpers. |
| `profile.py` | 215 | Signal return profiling (`profile_fixed_single_date`, `profile_fixed`). |

### `Bars` (one date)

Wraps an `xr.Dataset` plus the directory it came from. Designed to feel like the Dataset itself:
`__getattr__` forwards unknown attributes to `self.dataset`, so `bd.mid`, `bd.ticker`, `bd.time`
all work. `bd["mid"]` returns a `DataArray`; `bd[["mid","bid"]]` returns a *new `Bars`*.

- **Lazy loading:** `load_var` / `load_vars`, `get_var` / `get_vars` / `get_value(s)` (which
  autoload on miss), `get_vars_available`, `has_var`.
- **Autoloading `eval`:** `bd.eval("bid*0.5 + ask*0.5")` runs `Dataset.eval`; on a `NameError` /
  `UndefinedVariableError` it parses the missing name out of the exception message
  (`_re_undefined_var`), loads that variable from disk if it exists, and retries in a loop. This
  is the library's signature trick — expressions pull in whatever data they mention.
- **Writing:** `save_var(s)`, `save_created_vars()` (everything live that wasn't loaded from
  disk), `create_var`, `set_alias`. `_can_be_saved` is cleared once a selection has been applied,
  so you cannot accidentally write a sliced dataset back over full-shape files.
- **Time coercion:** `tcast` promotes `"15:54:00"` or a `dt.time` to a full `datetime` using the
  bars' own date, and passes numpy/pandas/xarray objects through untouched. This is why every
  selection API accepts bare time strings.
- **Joining external data:** `add_daily_var` (ticker-only vars), `lookup_indirect` (per-ticker
  *different* times, via DataArray indexing), `merge_df_asof`, `reindex_array`, `_restrict_other`.
- **Universe filtering:** `sel_universe(StockUniverse)` — see "Dependencies" below.

### `BarsSet` (many dates)

A directory of date directories; `bars[date]`, `bars["2025-01-13"]` and `bars[-20]` (negative
index into available dates) all return a `Bars`. `load_date(s)`, `dates_available`,
`dates_loaded`, `has_date`, `reload`, `iter_bars`. `restrict_dates` in the constructor both
limits and eagerly loads; `preload_vars` loads a variable set into every date; `su` applies a
universe filter to every date.

The payoff is the cross-date aggregation:

- `cat_var(var)` / `cat_vars([vars])` — concatenate across dates into a Series/DataFrame.
- `mapcat(func, add_date=|split_datetime=)` — run a `Bars -> DataFrame` function per date and
  concatenate, with `on_errors="raise"|"warn"|"ignore"` and an optional tqdm progress bar.
- `attach_to_df_by_tickers(df, vars)` — enrich a `(date, ticker)`-indexed DataFrame with bars
  variables. Requires a scalar time selection.
- `attach_to_df_by_ticker_time(df, vars, time_column=, method=)` — same, but each row looks up
  its own time (e.g. `method="ffill"` to get the most recent bar at an event's timestamp).
- `restrict_other(df)` — drop `(date, ticker)` rows with no bars data.
- `assign` / `assign_expr` / `assign_func` — define a new variable on every loaded date.
- `bars_getter(base_path)` returns a preconfigured `BarsSet` factory (the monorepo does
  `get_eod_bars = fr.bars_getter("/Users/fabio/trading/bars/eod")`).

### Selection algebra (`sel_array.py`)

`sel_time`, `sel_ticker`, `sel_time_slice` are available on **both** `Bars` and `BarsSet` via
`SelMixin`, and are **immutable** — each returns a copy with one more `BarsSel` appended to
`_active_sels`. Pending selections are replayed onto variables *as they are loaded* and onto
dates *as they are opened*, so slicing before loading avoids reading full arrays.

The subtle part is **scalar tracking**. Selecting a scalar (`sel_ticker("SPY")`,
`sel_time("15:54:00")`) drops that dimension and sets `_ticker_is_scalar` / `_time_is_scalar`;
selecting a one-element *list* keeps it. `BarsSet.cat_var(s)` uses those flags to decide the
resulting index, which is what makes multi-date concatenation come out with a sensible shape:

| selection | `cat_var` index |
|---|---|
| none | `(time, ticker)` |
| `.sel_ticker(["SPY","XOM"])` | `(time, ticker)` |
| `.sel_ticker("SPY")` | `(time,)` |
| `.sel_time(["15:54:00"])` | `(time, ticker)` |
| `.sel_time("15:54:00")` | `(date, ticker)` |
| `.sel_ticker("SPY").sel_time("15:54:00")` | `(date,)` |

Order of chaining does not matter. `notebooks/fr_tests.ipynb` in the source repo asserts exactly
this table and is effectively the library's test suite — it is the best starting point for real
tests here.

### `util.py`

`load_csv` (CSV → `(time, ticker)` Dataset, downcasting to float32/int32 by default) and
`to_frdir` (Dataset → a new date directory) are the ingestion path — the monorepo's `frcsv`
script is a thin CLI over them. Plus `dcast`/`date_cast` (two near-duplicates), `check`,
`check_first_dims`, `check_exact_dim_names`, `check_has_dims`, `check_has_vars`,
`get_empty_bars_dataset`, `level_time_to_date`, `index_loc`, `intersect`, and coloured warning
formatting (`set_warnings_format`, `warn`, `error`).

### `profile.py`

Return/cents-per-share profiles for a signal: given `(date, ticker)`-indexed alphas, take the
*sign* of the alpha per ticker, compute forward returns from a reference time across the bar
grid, optionally beta-hedge against a hedge ticker, average across tickers (optionally weighted),
and stack the dates into an `xr.Dataset` with a `date` dimension. `mode="ret"` (percent) or
`mode="cps"` (cents per share); `t_kind` controls whether the time axis comes out as datetimes,
times, timedeltas, or seconds from a reference. This is the most domain-specific module in the
library and the least general.

## Dependencies

Third-party: `xarray`, `pandas`, `numpy`, `tqdm`, plus a netCDF backend (`netcdf4` or
`h5netcdf`) for the `.nc` IO. Requires Python ≥ 3.11 (`typing.Self`).

Internal to the monorepo — this is the only thing blocking a clean extraction:

- **`stock_universe`** (`python-libs/stock_universe.py`, 22 lines) — a `StockUniverse` dataclass
  (`price_min/max`, `dvol_min/max`, `dvol_type`, `listing_exchange`, `extra_tickers`) and an
  `Exchange` enum. Imported by both `bars.py` and `bars_set.py` for `sel_universe`, which
  hard-codes the monorepo's variable names (`Yadjusted_close`, `Ndvol3m`/`pdvol3m`,
  `listing_exchange`). That naming is not meaningful outside the source repo.

Two imports need deleting rather than porting:

- `from numba.extending import overload_classmethod` in `bars.py` — **unused**, and it makes
  `numba` an accidental hard dependency.
- `from types import MappingProxyType` and several unused `typing` names in `bars_set.py`/`util.py`.

Also note `util.error()` calls `sys.exit(1)`, which is CLI behaviour that does not belong in a
library, and `Bars.sel()` deliberately raises `NotImplementedError` to steer callers to
`sel_time`/`sel_ticker`.

`dds` (`python-libs/dds/`, ~700 lines) is a *sibling* library, not a dependency — finarray does
not import it. It is a parquet-per-date daily dataset store, and it is slated to come across
later. Its own dependency is `butils` (data-directory config, XNYS exchange calendar), and its
`base_datasets.py` is entirely monorepo-specific.

## How it is actually used (from the source notebooks)

```python
import finarray as fr

bars = fr.BarsSet(base_path=".../bars/eod", restrict_dates=dates)
bars = bars.sel_time_slice("15:58:00", "15:59:49")
bars.load_vars(["mid", "close_price", "eret_ols0_cx_irr2_10td"])

# define a derived variable on every date
bars.assign_expr("vol_cap", "0.01 * pvolume3m")
bars.assign_func("ee", lambda bd: entry_exit(bd["eret_..."], 6, 5))

# collapse to pandas
s = bars.sel_ticker("SPY").sel_time("15:54:00").cat_var("mid")   # index: date
q = bars.mapcat(lambda bd: bd.get_var(v).quantile(qs).to_pandas(), add_date=True)

# enrich an event table with the most recent bar at each event's timestamp
events = bars.attach_to_df_by_ticker_time(
    events, ["mbsize", "masize", "mid"], time_column="time_received", method="ffill"
)

# single-date work reads like xarray
bd = bars[-20]
bd.sel_ticker("BA").get_vars(["bid", "ask"]).to_dataframe().plot()
```

Source-repo consumers, for reference when judging what API surface matters:
`futils/data.py` (`bars_getter` factories), `frbkt/core.py` and `backtest.py` (backtests),
`return_models/` (`Bars` in/out), `rblt_eimb.py`, `scripts/fr*` (CLI ingestion/eval),
and ~17 notebooks, most under `notebooks/Sardis/`.

## Conventions

- Type hints where they are not awkward; `Self` for the chainable copy-returning methods.
- Prefer vectorised numpy/pandas/xarray over explicit loops.
- snake_case variable names for bars variables (`close_price`, `ref_price`).
- Keep the public surface as it is unless there is a reason: the source monorepo should be able
  to switch to this package with minimal edits.

## Extraction decisions (2026-09-19)

Settled before work started; revisit only with a reason.

1. **`sel_universe` is generalized, not vendored.** `StockUniverse` and the `Exchange` enum do
   not come across. They are replaced by a variable-agnostic filter on `Bars`/`BarsSet` that
   takes constraints keyed by whatever the caller's variable names are — a `(min, max)` tuple
   (either bound `None` for unbounded) for a range, a scalar for equality — plus an
   `extra_tickers` escape hatch that unions tickers back in. Autoloads each named variable the
   way `sel_universe` did. Caveat to document: `extra_tickers` is a reserved keyword, so a bars
   variable of that name can't be filtered on through the kwargs form.
2. **`profile.py` ships in v1.** Its domain assumptions (`mid`, `beta_spy`, `SPY`) are all
   defaults on parameters, not hardcoded, and it demonstrates what the bars layout is for.
3. **Clean git history.** New commits on top of the existing `Initial commit`; no subtree split
   from the monorepo. The 43 upstream commits touching this path carry private research context
   in their messages.

## Plan status

- [x] **0 — uv skeleton.** src layout, hatchling, `requires-python >=3.11`, deps
      `xarray pandas numpy tqdm netCDF4`, dev group `pytest mypy ruff pandas-stubs`.
      `.python-version` pinned to 3.11 to match the monorepo that will consume this.
- [x] **1 — ported the six modules** into `src/finarray/`, plus the new `where.py`.
      Clean `ruff check`, `ruff format --check` and `mypy`.
- [x] **2 — 100 tests, synthetic data only.** `examples/make_sample_bars.py` writes a random-walk
      bars tree; `tests/conftest.py` builds fixtures from it. CI on 3.11/3.12/3.13.
- [x] **3 — README.** Every output shown in it was run and pasted back from real execution.
- [x] **4 — verified against real bars** (`~/trading/bars/eod`, read-only): 17/17 checks agree
      with the original implementation, plus the inheritance check below.
- [x] **v0.2 — `finarray.backtest`** as an optional extra (see below).
- [ ] **`dds`.** Explicitly deferred by the user; not started, and not planned. See the analysis
      recorded below.

## Changes from the original

Behaviour that deliberately differs from `quant-research/python-libs/finarray`. If the monorepo
ever switches to this package, these are the things to look at.

**Bugs fixed** (both confirmed against the real bars):

1. `BarsSet.dates_available` defaulted to `check_parents=False` while `has_date` and `load_date`
   both consider the parent chain. A `BarsSet` over a `PARENT`-only directory with
   `restrict_dates=` therefore warned on every date and came back **empty** — which is exactly the
   `bars/eod_bkt` pattern in `notebooks/frbkt.ipynb`. The default is now `True`; pass
   `check_parents=False` for local-only listing.
2. `set_alias` produced an unusable alias: the symlinked `.nc` holds an array named after the
   *original* variable, so `load_var("alias")` merged it under the original name and
   `bars["alias"]` raised `KeyError`. `load_var` now renames the loaded array to the requested
   name. (`set_alias` had no callers in the monorepo, so this was never noticed.)

**API changes:**

3. `sel_universe(StockUniverse)` → `sel_where(**constraints)` / `apply_where(Where)`, on both
   `Bars` and `BarsSet`. `BarsSet`'s `su=` constructor argument is now `where=`. Verified to select
   an identical 765-ticker universe on real data.
4. `util.error()` raises `FinArrayError` instead of warning and calling `sys.exit(1)`.
5. `util.date_cast` removed (a duplicate of scalar `dcast`, unused).
6. `~` in `Bars`/`BarsSet` paths is expanded.
7. Dropped the unused `from numba.extending import overload_classmethod` in `bars.py`, so numba is
   no longer an accidental dependency.

## v0.2: the backtest extra

`src/finarray/backtest/` is ported from `quant-research/python-libs/frbkt`, which was 221 lines
with three non-finarray dependencies. All three were resolved rather than carried:

- `numba` → optional. `engine.py` falls back to a no-op `njit` decorator, so the kernels run as
  plain Python (with a one-time `RuntimeWarning`) when numba is absent. CI covers both paths.
- `fees` → `backtest/fees.py`. Only `FeesSimple` was used; the rest of the monorepo's `fees.py`
  reads a private YAML of exchange schedules at import and pulls in butils/futils. Renamed `Fees`,
  and `as_not_sell_only` → `as_two_sided` (the old name described the mechanism, not the intent).
- `backtest.PnlData` → `backtest/result.py`. The original used it only on the last line. Porting it
  would have meant `butils.daily_stats` + `butils.pnl_summary_stats` + 9 plotly call sites out of
  `backtest.py`'s 502 lines, so `BacktestResult` is a fresh, dependency-free replacement.

**Kernels.** The original's two near-identical numba kernels (capped and uncapped) are consolidated
into one `_walk`. Verified against both originals on real bars: `pnl`, `volume` and `dvolume` agree
to float32 precision on 3 dates × both paths.

**Deliberate behaviour differences from frbkt:**

- `unwind_price` defaults to `None`, not `"close_price"`. A public library should not default to a
  variable name that may not exist; the cost is that the default leaves positions open.
- The original's `eod_pos` was the position *before* the closing unwind. That is genuinely useful
  (it is the size you had to liquidate), but so is the residual after it. Both are now reported:
  `close_pos` (pre-unwind) and `eod_pos` (post-unwind, therefore zero whenever the run unwinds).
- Accumulators are float64 rather than float32.
- A date where nothing traded yields an empty frame rather than `None`, so it no longer warns.

**Also fixed in v0.1 code:** `BarsSet.mapcat` raised `ValueError: No objects to concatenate` when
every date failed. It now names the number of dates and points at `on_errors="raise"`.

**Dev-environment note:** `numba` in the `dev` group is pinned `<0.62`. This is purely local —
llvmlite ≥0.45 ships no macOS x86_64 wheel and tries to build against LLVM. The published
`backtest` extra is unpinned (`numba>=0.59`). The extra also drags numpy down (numba pins it
tightly), which is a second reason it is not a hard dependency.

## v0.3: the CLI

`src/finarray/cli.py` ports two of the six `fr*` scripts from `quant-research/scripts/`, as
subcommands of a single `finarray` console script (`[project.scripts]`). One entry point rather
than three, so the package does not squat on generic names like `frcsv` in everyone's PATH.

- `freval` → `finarray eval`. Was already pure finarray, no dependencies at all.
- `frcsv` → `finarray import-csv`. Its only outside dependency was `butils.pp_exception_context`,
  replaced by a local `_report_errors`.

**Verified byte-identical to the originals.** `verify` run in the scratchpad: the same CSV through
`frcsv -f` + `freval -f '<filter>' 'mid=(bid+ask)/2'` and through the new subcommands produces
identical `bid.nc`, `ask.nc`, `mid.nc` (same 20 NaNs from the filter) and identical `ticker.csv` /
`time.csv`. Running the originals needs a stub `butils` on PYTHONPATH, since the real one pulls in
`more_itertools` and `exchange_calendars`.

**Fixes made during the port:**

- `freval` had a dead second `args = parser.parse_args()` after `save_vars`. Gone.
- Both scripts called `util.error()`, which since v0.1 raises instead of calling `sys.exit(1)`.
  `cli.main` now returns proper exit codes, so shell pipelines still detect failure.
- The date is parsed from `os.path.basename(os.path.normpath(dir))`, so a trailing slash on
  `--dir` no longer breaks it, and a non-date directory name gives a clear error.
- `import-csv` reports and skips a bad file rather than abandoning the batch, exiting non-zero
  at the end. `PY_TRACEBACK=1` still gives full tracebacks, as in the originals.

**tqdm is no longer a dependency.** `util.progress_iter` uses `tqdm.auto` when it is installed
(which is what you want in a notebook) and otherwise falls back to a plain stderr counter.
`BarsSet.mapcat(progress=True)` and `profile.profile_fixed(do_progress=True)` go through it. tqdm
stays in the dev group so both branches are exercised; the `ci-minimal` group has neither tqdm nor
numba, and CI runs the suite against it.

**pyarrow is now a dependency**, at the user's instruction. Nothing in the package imports it yet —
it is there for `pd.read_parquet`, which is what a future `finarray daily` (the `frdaily` port)
would need.

**Also fixed in v0.2 code:** `mapcat`'s "produced nothing" error called `len(list(dates_))` on an
iterator that the progress wrapper may already have consumed, so it would have reported 0 dates.

## v0.4: the rest of the CLI

`src/finarray/cli/` is now a package: `_common.py` (date-spec parsing, error reporting),
`info.py` (ls, check), `query.py`, `ingest.py` (import-csv, import-parquet, link),
`derive.py` (eval, rm). Eight subcommands.

**`query` streams, and that is the point.** It emits rows date by date and stops as soon as it has
the requested number, so a capped query against 363 dates of 1689 tickers x 661 times reads one
date in under a second. Within a date it bounds the time dimension to `ceil(limit / n_tickers)`
steps before converting to pandas, and when that leaves a single step it bounds tickers too. Both
are only correct because `Dataset.to_dataframe()` emits rows in (time, ticker) order — so
`test_limited_output_is_a_true_prefix_of_the_full_output` parametrizes over limits and asserts the
bounded result equals the head of the unbounded one. **If that ordering assumption ever breaks, that
test is what catches it.** Two more tests assert the read really is bounded (dates touched, frame
sizes converted) rather than truncated after the fact.

Default limit is 50 rows to stdout; `--output FILE` implies no limit unless `--limit` is explicit.
The truncation notice goes to stdout as `# ...` so CSV consumers can skip it as a comment.
Parquet export streams per date through a `ParquetWriter`.

**`rm` will not delete out of a parent.** `Bars.delete_var` (new) refuses when the variable is only
present upstream, so a child directory cannot damage the data it inherits. It uses `os.path.lexists`
so a broken alias symlink is still removable.

**`eval --all-dates`** closes the TODO on line 8 of the original `freval`
("should be able to run it on basedir and generate it for everything"). `--dir` still means a single
date directory unless `--all-dates`/`--dates` is given, so existing invocations are unaffected.

**Library changes made for the CLI:**

- `FinArrayError` now subclasses `ValueError`, and `util.check` raises it. These always *were*
  value errors; the distinct type lets the CLI separate an expected failure (one line) from a bug
  (traceback). Every existing `pytest.raises(ValueError)` still passes.
- `create_child_bars` raises through `util.check` instead of bare `ValueError`, and expands `~`.
- `Bars.delete_var` / `BarsSet.delete_var` added — there was no way to delete a variable at all
  before, from the CLI or from Python (`__delitem__` only touches the in-memory dataset).
- `cli.main` catches `BrokenPipeError`, so `finarray query ... | head` exits cleanly.
- The CLI translates a `NameError` from a bad expression into a `FinArrayError` listing the
  variables that do exist.

## v0.5: `query --where`

`--where VAR=MIN:MAX` (inclusive range, either end optional) or `VAR=VALUE` (equality), repeatable
and ANDed, plus `--extra-tickers`. Built as a `Where` object directly rather than through
`sel_where`, whose `extra_tickers` keyword would collide with a variable of that name.

Values parse as int, then float, then string, so `listing_exchange=1` compares against an int and
`Ndvol3m=1e7:` against a float. A variable named twice is rejected rather than silently taking the
last one.

Note that `BarsSet.load_date` applies `_active_sels` before `_wheres`, so `--where` is evaluated
after `--ticker`/`--time` regardless of the order `_apply_selections` calls them in. That is
harmless here — the constraint variables are ticker-only, so a time selection cannot affect them,
and a ticker selection just intersects.

Also in v0.5: a query matching nothing prints `# no rows matched` instead of silence, and the CSV
file sink opens lazily on first write, so an empty result no longer leaves a zero-byte file behind.

## Which `fr*` scripts were left behind

`frgendaily`, `frgenimb` and `frgeneimb` are not portable and are not planned. Each is a thin date
loop around a specific private dataset: a hardcoded 16-column list from `dds` plus the `Exchange`
enum dropped in v0.1; NYSE imbalance message fields via `imbalances`; early-imbalance batches via
`rblt_eimb` with `nbatches = 3` baked in. All three also default `--dir` to
`/Users/fabio/trading/bars/eod`. Strip the domain knowledge and nothing is left.

`frdaily` was ported in v0.4 as `finarray import-parquet`, which is what pyarrow is there for.

## Why `dds` was not brought across

Recorded so the question does not get re-litigated. finarray does not import dds — the arrow points
the other way (`dds/base_datasets.py:75` imports finarray inside an ingest function), so the
original premise that it had to come along does not hold. It is also an orthogonal data model:
daily `(date, ticker)` parquet vs intraday `(time, ticker)` netCDF. The decisive problem is the
split: `core.py` (331 lines) is the generic, portable engine, but `base_datasets.py` (208 lines) is
entirely monorepo-specific and unpublishable, and `load_functions.py` — the API actually used day
to day — resolves column names through the registry that `base_datasets` defines. The proven half
cannot be published and the publishable half is a generic parquet-per-date store in a crowded
field. If it ever happens: separate repo, separate package, user-supplied registry.

## Verification

Two read-only scripts in the scratchpad, both hardcoding private paths and so kept out of the repo:

`verify_backtest_kernel.py` exec's frbkt's kernels from source with the unimportable lines stripped,
and diffs them against `_walk` over `~/trading/bars/eod` for both the capped and uncapped paths.

`verify_vs_original.py` imports both implementations side by side (stubbing
numba so the original loads) and diffs them over `~/trading/bars/eod`: structure, the full
`cat_var` shape contract, autoloading `eval`, `mapcat`, both `attach_to_df_*` joins,
`restrict_other`, `sel_universe` vs `sel_where`, and `profile_fixed`. It is read-only and lives in
the scratchpad rather than the repo, since it hardcodes private paths. Re-create it from this
description if the port needs re-checking.
