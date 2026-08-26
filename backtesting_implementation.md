# Signal Backtester — Implementation Notes

The backtester is implemented in `forexgrand_core.backtesting`. The public
entry point is `run_backtest`; `SLTPCalculator`, `SignalExtractor`,
`MarketTableBuilder`, `BacktestEngine`, and `BacktestResult` are also exported
for focused testing and custom workflows. The training-data generator is now
available from `forexgrand_core.generate_train_data` (the old pipeline import
remains as a compatibility path).

For a runnable example and strategy-file contract, see the **Backtest A
Strategy** section in `README.md`.

The original design details follow for reference.

## 0. Scope

Add a new backtesting capability that:
1. Loads a user-supplied strategy `.py` file containing a class that inherits `PreprocessBase` and exposes a signal function returning `{"direction": 0|1|2}` per bar (0=buy, 1=sell, 2=none).
2. Converts those signals into trade candidates (open_price, sl, tp) using one of three configurable SL/TP modes.
3. Replays bar-by-bar ask/bid data forward, opening/closing simulated positions against SL/TP.
4. Returns a `positions` table, equity curves, and summary counts.

Target file: `backtesting.py`. Suggested internal structure (new module, not one giant function):

```
backtesting.py
├── class SLTPCalculator         # stage 3b below — the 3 modes
├── class SignalExtractor        # stages 1-3 below
├── class MarketTableBuilder     # stage 4
├── class BacktestEngine         # stages 5-7 (the bar loop)
└── def run_backtest(...) -> BacktestResult   # orchestrates all of the above
```

`BacktestResult` should be a `TypedDict`/`dataclass`, not a bare dict, so field names are enforced (see §5).

---

## 1. Strategy loading & pre-init

- Dynamically import the given `.py` file (`importlib.util.spec_from_file_location`), locate the subclass of `PreprocessBase`, instantiate it.
- Confirm the object exposes the expected signal method (name TBD — recommend a fixed method name like `generate_signals(batch) -> dict` enforced via `abstractmethod` on `PreprocessBase`, rather than duck-typing).
- Pre-initialization already works per your note — reuse as-is, just wrap in a try/except that raises a clear `StrategyLoadError` on failure (missing class, wrong base, wrong signature).

## 2. Sequential data generation (batched)

- Data is walked in batches sized to fit memory (existing pipeline presumably already does this for training — reuse the same windowing/batch generator rather than writing a new one).
- Each batch must carry its own timestamp index alongside the feature tensor, since signals must be re-attached to a timestamp (in seconds) after inference.
- For every batch: run the strategy's signal function → concatenate `direction` arrays and their aligned timestamps into one long array (do **not** materialize the full dense feature set in memory at once — only keep `timestamp` + `direction` + whatever scalar fields you need per signal, e.g. close/spread/range/atr, discussed next).
- **Batch shape**: the dict returned by the strategy's signal function has each value shaped `(batch_size,)` — i.e. one scalar per bar in the batch it was called on, not a single aggregate value per batch. Concatenation across batches is therefore a simple `np.concatenate` along axis 0, not a stack/reduce.
- **Implementation note — reuse the training pipeline's chunking.** Before writing a new batching/windowing loop here, study how `generate_train_data.py`'s `GenerateTrainData` class shards and streams data (the same 16-way TFRecord sharding used for training). Apply that same chunking approach to both (a) generating the signal sequence in this stage and (b) collecting/appending rows into the positions table in §5–6, rather than building either as one big in-memory array/DataFrame. This keeps memory usage consistent with the training pipeline's existing pattern instead of introducing a second, divergent one.

## 3. Signal extraction & validation

For the concatenated `direction` array:

1. **dtype check** — reject/cast float output. Recommended: if dtype is floating point, verify all values are integral (`np.all(np.mod(arr, 1) == 0)`); if so cast to `int64`; if not, raise `ValueError` (a fractional "direction" indicates a bug upstream, don't silently round).
2. **Unsupported values** — valid set is `{0, 1, 2}`. Anything else (3, 4, -1, …) is invalid and must be dropped from the signal stream, not just 3 specifically. Use `values, counts = np.unique(arr, return_counts=True)` (note: `np.unique_counts` is only in NumPy ≥2.0 — use `np.unique(..., return_counts=True)` for compatibility unless you've pinned NumPy 2.x), then `n_unsupported = counts[~np.isin(values, [0,1,2])].sum()`. Return this count in the result dict so the caller can sanity-check strategy output quality.
3. Filter the signal stream down to only `direction in {0, 1}` (drop `2`/none entirely — no trade — and drop any invalid values, already counted above).
4. For each surviving signal, collect a record: `{time, direction, close_last, spread}` plus, depending on SL/TP mode (see §3b), the raw inputs that mode needs (nothing extra for fixed-points; last-N-bars range for range mode; ATR series for atr mode). `close_last` is assumed to be the **bid** price by default, configurable via `entry_price_type` (see §8).

This produces the **signals table**, indexed `1..n` (n = number of valid buy/sell signals), each row = one trade candidate — but without final `sl`/`tp` prices yet, those are computed next.

## 3b. SL/TP calculation modes

Selected via a single `sl_calculation: dict` argument to `run_backtest()`, keyed by `"mode"`, with the remaining keys depending on which mode is chosen. The engine validates that the dict's keys match the chosen mode (reject unknown keys / fill in defaults for missing ones).

**1. `{"mode": "fixed", "sl_points": 100, "tp_points": 100}`** (default)
- `symbol_points` (the instrument's point size) is **not** part of this dict — it's pulled from the data manager's returned instrument properties, not passed in by the caller.
- Actual price distance = `points / symbol_points`.
- `buy:  sl = open_price - sl_points/symbol_points ;  tp = open_price + tp_points/symbol_points`
- `sell: sl = open_price + sl_points/symbol_points ;  tp = open_price - tp_points/symbol_points`

**2. `{"mode": "range", "range": 60, "sl_ratio": 1.0, "tp_ratio": 1.0}`**
- Compute `bar_range = high[-range:].max() - low[-range:].min()` over the `range` bars immediately preceding (not including) the signal's open bar.
- `buy:  sl = open_price - bar_range * sl_ratio ;  tp = open_price + bar_range * tp_ratio`
- `sell: sl = open_price + bar_range * sl_ratio ;  tp = open_price - bar_range * tp_ratio`

**3. `{"mode": "atr", "sl_multiplier": 3, "tp_multiplier": 3, "atr_period": 14}`**
- `atr_period` defaults to 14 if omitted, but is a first-class key in the dict (not hardcoded).
- Compute ATR on close prices up through the bar preceding the signal's open bar, take `atr[-1]`.
- `buy:  sl = open_price - atr[-1] * sl_multiplier ;  tp = open_price + atr[-1] * tp_multiplier`
- `sell: sl = open_price + atr[-1] * sl_multiplier ;  tp = open_price - atr[-1] * tp_multiplier`

All three modes should share one interface, e.g. `SLTPCalculator.compute(signal_row, market_context) -> (sl, tp)`, dispatching internally on `sl_calculation["mode"]`, so the engine itself doesn't care which mode is active. `symbol_points` should be fetched once per backtest run (from the data manager) and passed into the calculator's context, not looked up per-signal.

## 4. Ask/bid market table

Build a single lookup table over the full backtest window with columns:

```
time (index), high_ask, high_bid, low_ask, low_bid, close
```

- `time` should be the same unit/epoch as the signal timestamps (seconds) — this must be verified, not assumed.
- Confirm the source of `high_ask/low_ask` vs `high_bid/low_bid` — ask = bid + spread per bar, using the dataframe's actual per-bar `spread` column (not a constant); default to `0.0` where spread is unavailable (see §8).

## 5. Positions table schema

```
index (open_time, matches signal time)
direction        : int8      # 0 buy, 1 sell
status           : category  # "pending" | "open" | "closed"
open_price       : float64
sl                : float64
tp                : float64
max_profit       : float64   # running max, init 0.0
min_dd           : float64   # running min, init 0.0
close_time       : Int64     # nullable
close_price      : float64
profit           : float64
close_reason     : category  # "tp" | "sl" | "tiebreak" | "eod" | None
```

All rows start as `"pending"`.

## 6. Main loop (bar-by-bar replay)

Iterate `curr_time` over the market table, starting at the **first signal's open_time** and ending at the **later of** (last position closed) **or** (end of bars) — whichever comes first in practice, since the loop just terminates early once no pending/open positions remain (see step e below).

**Confirmed: a position cannot hit SL/TP on its own entry bar.** Promotion (`pending → open`) happens at the end of the bar in which `open_time == curr_time`; the earliest a position becomes eligible for an SL/TP check is the *next* bar. This resolves the entry-bar-exposure ambiguity from the previous draft — no same-bar exit check is needed.

Per bar:

**a. Update floating P/L for currently-open positions** (vectorized over the `status == "open"` subset, not row-by-row):

```
buy:  max_profit = max(max_profit, min(high_bid, tp) - open_price)
      min_dd      = min(min_dd,      max(low_bid,  sl) - open_price)
      sell: max_profit = max(max_profit, open_price - max(low_ask, tp))
            min_dd      = min(min_dd,      open_price - min(sl, high_ask))
            ```

            **b. Aggregate** `sum(max_profit)` and `sum(min_dd)` across open positions this bar → append to the profit-equity and dd-equity series (one point per bar).

            **c. Check exits**, buy positions (open only). `sl_hit = low_bid <= sl`, `tp_hit = high_bid >= tp`:
            ```
            if sl_hit and tp_hit:   # same-bar tiebreak — decided by the bar's close
                if close >= tp:  status="closed", profit = tp - open_price, close_reason="tiebreak", close_price=tp
                    else:             status="closed", profit = sl - open_price, close_reason="tiebreak", close_price=sl
                    elif sl_hit:  status="closed", profit = sl - open_price, close_reason="sl", close_price=sl
                    elif tp_hit:  status="closed", profit = tp - open_price, close_reason="tp", close_price=tp
                    # (all branches also set close_time=curr_time)
                    ```
                    sell positions (open only), mirrored. `sl_hit = high_ask >= sl`, `tp_hit = low_ask <= tp`:
                    ```
                    if sl_hit and tp_hit:
                        if close <= tp:  status="closed", profit = open_price - tp, close_reason="tiebreak", close_price=tp
                            else:             status="closed", profit = open_price - sl, close_reason="tiebreak", close_price=sl
                            elif sl_hit:  status="closed", profit = open_price - sl, close_reason="sl", close_price=sl
                            elif tp_hit:  status="closed", profit = open_price - tp, close_reason="tp", close_price=tp
                            ```
                            (Use `<=`/`>=`, not strict `<`/`>` — a bar that touches the exact level should still trigger. `sl - open_price` for a buy is correctly negative since `sl < open_price`; the sell formulas were already consistent. `close_reason` values are now `"sl"`, `"tp"`, `"tiebreak"`, or `"eod"` — a single `"tiebreak"` label covers both-touched bars regardless of which side the close resolved toward; which side it was is already recoverable from `close_price` (`== tp` or `== sl`), so the label doesn't need to carry it too.)

                            **d. Promote pending → open**: any row where `index == curr_time` and `status == "pending"` becomes `"open"`.

                            **e. Termination**: if no rows have `status in {"pending", "open"}`, stop.

                            **f. End-of-data handling**: if the loop runs out of bars while positions are still `"open"`/`"pending"`, force-close them at the last available bar's price (bid for buys, ask for sells), recording `profit`, `close_price`, `close_time`, and `close_reason="eod"` just like a normal SL/TP close — every position ends with a status of `"closed"` and a recorded profit, none are left dangling in the output.

                            ## 7. Output

                            ```python
                            {
                              "positions": positions_df,
                                "profit_equity": profit_series,       # per-bar aggregated floating max_profit
                                  "dd_equity": dd_series,                # per-bar aggregated floating min_dd
                                    "positions_total": int,
                                      "buy_count": int,
                                        "sell_count": int,
                                          "unsupported_signal_count": int,       # from stage 3
                                            # "statistics": {}                     # placeholder, later feature
                                            }
                                            ```

                                            ---

                                            ## 8. Loopholes / open questions still outstanding

                                            Resolved: buy SL sign, entry-bar exposure (next-bar only), SL/TP modes (§3b, `sl_calculation` dict, keys validated against the chosen `mode`), end-of-data handling (force-close), same-bar SL+TP tiebreak (decided by bar close, both branches now recorded with a single `close_reason="tiebreak"` rather than a `sl_tiebreak`/`tp_tiebreak` split — see §6c), `symbol_points` source (data manager instrument properties, fetched once per run, not per-signal), ATR period (defaults to 14, exposed as a first-class `atr_period` key in the `sl_calculation` dict), which "close" stage-3 collects / spread double-counting, spread constancy, and timestamp unit (all below).

                                            **Resolved — which "close" is which / spread double-counting.** Stage 3's `close_last` is the **bid** price by default, so `buy open_price = close_last + spread` is correct as written and does not double-count spread (bid + spread = ask, computed once). To avoid re-litigating this per data source, expose it as an explicit parameter rather than a baked-in assumption: add `entry_price_type: "bid" | "ask" | "mid" = "bid"` to `run_backtest()` (or to the `SignalExtractor` config). The extractor uses this to decide how to derive `open_price` from `close_last` and `spread`:
                                            - `"bid"` (default): `buy open_price = close_last + spread`, `sell open_price = close_last`
                                            - `"ask"`: `buy open_price = close_last`, `sell open_price = close_last - spread`
                                            - `"mid"`: `buy open_price = close_last + spread/2`, `sell open_price = close_last - spread/2`

                                            This keeps the default behavior (bid + spread) but makes the assumption explicit and swappable if a different upstream feature source is ever plugged in.

                                            **Resolved — is spread constant per trade or time-varying?** Spread is a genuine per-bar column in the source dataframe, not a constant — pull it from there at signal time (captured once, at entry, per §3 point 4) rather than assuming a fixed value. If the column is missing/unavailable for a given source, default to `0.0` rather than raising, so the pipeline degrades gracefully (entry_price then just equals `close_last`, and `high_ask == high_bid` etc. would need the same `0.0` fallback wherever ask/bid are derived from bid + spread).

                                            **Resolved — timestamp unit/alignment.** Timestamps are `int64` seconds throughout — the signal timestamps carried out of stage 2 and the ask/bid market table's `time` index in §4 both use this representation, so no unit conversion is needed between them.

                                            Still open:

                                            1. **Position sizing / concurrency limits.** Nothing caps how many positions can be open simultaneously, or handles lot sizing/margin. Confirm this is intentional (pure signal-level backtest, unconstrained) vs. needing a max-concurrent-positions rule.

                                            2. **`positions_total`, `buy_count`, `sell_count`** — counts of all rows regardless of status, or only closed? Recommend: all rows ever created (`len(positions_df)`), since with the end-of-data fix in §6f everything ends up `"closed"` anyway, this is now mostly moot.

                                            **Build recommendation, not an open question — performance.** The concern: §6's bar loop is inherently sequential in Python (exits are path-dependent — you can't know if bar N+5 exits a position without having processed bars N through N+4 first, so it can't be vectorized across time the way stages 3/4 can). The risk is specifically doing that sequential walk with pandas row-wise operations (`.iterrows()`, `.loc[i, "col"] = x` inside the loop) — that pattern is slow enough per iteration that it becomes the bottleneck on multi-year, sub-minute bar data (potentially millions of bars). The mitigation isn't to avoid the bar-by-bar structure (that's correct and necessary), it's to keep each bar's *inner* work vectorized: steps 6a–6c should operate on the whole open-positions subset as NumPy arrays in one shot per bar, not loop over individual rows. Recommend validating correctness first with a small, simple (if slower) implementation, then Numba-JIT'ing the loop body once the reference behavior is locked in — same chunked-processing spirit as the §2 note above, applied to the positions table instead of the signal stream.

                                            ---

                                            ## 9. Suggested build order

                                            1. Implement `SLTPCalculator` (§3b) standalone with unit tests for all three modes against hand-computed examples.
                                            2. Implement `SignalExtractor` (stages 1–3) with unit tests using a fake strategy that returns a known array including invalid values (2s, 3s, floats) — verify counting and filtering logic in isolation.
                                            3. Implement `MarketTableBuilder` (stage 4) with a small synthetic OHLC fixture.
                                            4. Implement the bar loop (§6) against a **hand-computed** tiny fixture (e.g., 3 signals, ~20 bars) where you know the expected SL/TP outcomes by hand, including at least one end-of-data force-close case.
                                            5. Wire `run_backtest()` end-to-end, add the summary/count fields.
                                            6. Only after correctness is verified: optimize for speed (Numba/vectorization per the performance note in §8, and chunked processing per the §2 implementation note).