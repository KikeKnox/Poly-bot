# T8b report — REAL training of the numeric loaded model on exported data

**Status: DONE** — the pipeline was actually executed against the real DBs
(read-only), with artifact + manifest + OOS metrics. Main result:
**negative/inconclusive on confirmed labels** — the loaded model does not beat
the OOS baselines; it is kept honestly.

## 1. Honest inventory of real data

| Source | Status | Useful content |
|---|---|---|
| `data/research_events/` (T2 causal tape) | **NONEXISTENT** | `[research_events] enabled=false` in `config/general.toml`; the recorder never recorded. No trade tape and no `origin_ts_ms` for books. |
| `hrf_state_paper.sqlite` (runtime, 52 MB + WAL) | OK | `markets_observed` 41,387 rows (first_seen epoch-**s**), `positions` 878, `transactions` 2,367, `fills` 1,591, `post_close_observations` 116,181, `equity_snapshots` 3,017. **`market_resolutions` / `resolution_rules` DO NOT EXIST** — the T3 pipeline never ran on this DB (0 labels that way). |
| `hrf_market_analytics_paper.sqlite` (runtime, 9.7 GB) | OK | `book_snapshots` 2,110,612 rows, 7,337 markets, 6 assets (btc/sol/eth/xrp/doge/bnb), 2026-09-18→21 UTC (~3 days). Row = atomic snapshot of BOTH books + `spot` + `reference_price`/`clear_at`/`fee`. |
| `hrf_spot_stream_paper.sqlite` (repo root, 380 MB) | OK | `spot_samples` 3,164,495 ticks with real LOCAL `recv_ts_ms`/`avail_ts_ms`; `source_ts_ms` is **approximate** (`spot_stream.rs`: `now−dt` from `Instant` — the `VolSnapshot` carries no remote trade ts). |

### Available labels

- **Confirmed (official, from the venue)**: `winner_label` from
  `positions`/`transactions` — set by `orphan.rs::settle` with the
  `winner` field of Polymarket's CLOB API. Provenance
  `clob_api:orphan_settle`. **639 markets**, zero internal conflicts
  (positions vs transactions agree everywhere). They only exist for
  markets where the bot held a position → **selection bias**
  (near_expiry/arb strategies, brackets already close to the close).
  The real bias is even stricter: the labels come from positions that
  were **alive until the settle** (`redeemed=0 AND bail_sold=0 AND
  outcome_won IS NULL AND size>0`, which the orphan checker resolves) —
  not from "markets with any position".
- **`orphan.rs` fallback hazard (verified)**: if the winning token has
  `winner=Some(true)` but `outcome=None`, the settle writes
  `"Yes"`/`"No"` depending on whether the BOT's position won — not the
  real side. In a `*-updown-*` slug (outcomes `Up`/`Down`) a `Yes`/`No`
  label is the signature of that fallback → potentially INVERTED.
  The exporter cross-checks: all 639 labels are `Up`/`Down`
  (zero with the fallback signature) and the winner implied by
  `positions.outcome_label`+`outcome_won` matches everywhere →
  **`suspicious_fallback_labels = 0`, `outcome_contradictions = 0`**
  (manifest `labels.*`). If any appeared, they would be excluded from the
  confirmed set and listed in `labels.suspicious_cids`.
- **Provisional (NOT certified)**: last determined verdict from
  `post_close_observations` (PCS-1, frozen spot vs strike) — 11,054
  markets. Proxy↔official agreement: **314/322 (97.5%) over the 639
  global labels** (322 markets with both sources, 8 divergences);
  **89/2 over the 176 markets of the confirmed dataset** — the proxy
  is wrong; it is never presented as a certified label.

### Effective coverage

| | confirmed | provisional |
|---|---|---|
| markets with a label | 639 | 11,054 |
| ∩ `book_snapshots` | **176** (27.5%) | **4,797** (43.4%) |
| snapshots in those markets | 55,178 | 1,429,291 |
| median pre-close snapshots/market | 281 | — |

Book coverage (~3 days) is the real bottleneck: the 639 labels cover ~25
days of operation but the analytics recorder only retains the latest days.

## 2. Export pipeline

`research/numeric_model/export_sqlite.py` (new, stdlib) →
`research/data/t8b_confirmed/` and `research/data/t8b_provisional/`:

- Effective path resolution reusing
  `tools/research_tools/research_inventory.effective_sources` (TOML + `.env` +
  `resolve_path` with the `_paper` suffix). Opened with `mode=ro` +
  `PRAGMA query_only=ON`. Never writes/checkpoints the originals.
- One `book_snapshot` event per row with `payload.books =
  {yes_token: …, no_token: …}` (the row IS atomic for both books).
- `price_tick` from `spot_samples` (only `source='binance'`, the oracle
  feed): `origin_ts_ms=None` — the row's `source_ts_ms` is
  APPROXIMATE (see inventory_state) and travels in `payload.source_ts_ms_approx`;
  quality `SYNTHETIC|NO_REMOTE_TS`, same as `event_from_sample`
  in `spot_stream.rs`. `recv`/`avail` are real local timestamps.
- `market_meta` per market (markets_observed + `reference_price` from the
  first snapshot row); `avail = min(first_seen_ts, first
  snapshot)` — a derived local instant, documented in the manifest.
  Cids without a row in `markets_observed` → meta synthesized from the
  first row of `book_snapshots` (`provenance=
  sqlite:book_snapshots:sintetizado`, REDUCED_FIDELITY, counted in
  `metas_synthesized_from_books`): **0 in both datasets** — all labeled
  cids have a real meta.
- `{"sim":"fill"}` from `fills` (economics in `payload.detail`,
  `quality=SYNTHETIC` — paper fills are simulator executions).
- `resolution` per market with `winner`/`known_at_ms` (first REDEEM;
  fallback `clear_at`) + `provenance`.
- `local_seq` reassigned by `(avail_ts_ms, kind_priority)` — the DBs
  do not keep the recorder's seq; `inversions_avail_in_seq=0` after the
  fix (the meta/resolution streams were being reordered by cid — fixed by
  ordering them by avail before the merge).
- Operational guards: explicit `--mode` (default `paper` — an
  `HRF_MODE=live` in the environment does not silently export the live DB) and
  `_validate_out_dir` (rejects `--out` over/next to a source DB).
- `dataset_manifest.json`: sources (path/bytes/mtime), coverage,
  counts per kind, sha256 of the tape, fidelity notes.

Generated tapes:

| | confirmed | provisional |
|---|---|---|
| `events.jsonl` | 601 MB | 2.52 GB |
| events | 2,039,333 | 3,448,531 |
| book_snapshot | 55,178 | 1,429,291 |
| price_tick | 1,982,593 | 1,996,909 |
| market_meta / resolution | 639 / 639 | 11,054 / 11,054 |
| sim fill | 284 | 223 |

## 3. Code changes (all in `research/numeric_model/`)

- `dataset.py`: `build_observations_incremental` (single pass, min-heap
  of pending events by `avail_ts_ms`, drained in seq order; equivalent
  to the batch builder with avail monotonic in seq; `NonMonotonicTape` on
  regression) + `build_observations_auto` (batch fallback) +
  `assume_sorted`/`iter_events_jsonl`/`build_labels` streaming (the 2.5 GB
  tape exhausted RAM when materialized — the first provisional train
  attempt was **OOM-killed at ~22 GB RSS**, documented) + `payload.books`
  support + `_spot_features` fix (it copied the whole tick series per
  observation — a hidden O(n), now only the VOL_WINDOW+1 tail).
- `train.py`: `logistic_train` with a numpy fast path (installed, v2.4.6;
  same math, init 0, deterministic — verified: max diff
  ~5e-17 vs stdlib) + `_logistic_train_stdlib` fallback + `--streaming`.
- `evaluate.py`: `--streaming`.
- `tests/test_causality.py`: +6 tests (exact incremental↔batch
  equivalence, equivalence with labels+split, `books` payload,
  causality under avail inversion, `NonMonotonicTape`, auto fallback)
  +1 numpy↔stdlib parity test (`assertAlmostEqual` per weight,
  skipped if numpy is absent).
- `export_sqlite.py` (new).
- `attach_labels` adds `label_provenance` to each observation
  (settle vs proxy traceability down to the training row).
- The loaded-model manifest's `config_hash` = sha256 of the REAL
  hyperparameter dict (`train_frac/epochs/lr/seed/streaming`), no longer a
  literal; the manifest also stores the dict in plain form.
- Streaming: `NonMonotonicTape` is wrapped in a `RuntimeError` with an
  actionable message in train/evaluate/analyze (no fallback possible
  over a consumed generator); the `build_observations_auto` docstring was
  corrected (it is the entry point for materialized tapes, not for streaming).
- `provisional_labels` with `ORDER BY ts_ms, rowid` (deterministic on
  ties); `analyze_t8b.py` guards against division by zero (empty train →
  base 0.5; empty partitions → `None`).

## 4. Training and OOS evaluation (chronological per market, never
## random; scaler fitted only on train; `split_markets` fixed in
## model.json — evaluate does not re-derive)

### CONFIRMED dataset (`research/data/t8b_confirmed/model/`)

Split: 106 train markets / 70 test (train_frac=0.6 in chronological
order of the market's first `decision_ts_ms`).
Supervised observations: 31,984 train / 23,194 test.
**Methodology note**: the split is chronological by the market's first
decision_ts, so the wall-time windows overlap ~2.36 h (the last train
market opens before the first test one closes — standard practice in
splits over a temporal series; there is no leakage: markets are not
shared and the scaler only sees train).
post_close ≈ 1.6%. label_yes: 49.7% train / 56.8% test.
Feature coverage: `yes_mid` 96%, `spot`/`moneyness` 100%,
`flow_*` absent (no tape), `origin_delay` absent on books.

| OOS metric (test) | logistic loaded model | persistence | market (yes_mid) |
|---|---|---|---|
| accuracy | 0.529 | 0.432 | 0.526 |
| logloss | 0.735 | 0.694 | 0.757 |
| **brier** | **0.2610** | **0.2504** | **0.2746** |

Identical result when restricted to pre-close observations
(`metrics_context.json`: brier 0.264 / 0.250 / 0.275).

**Reading: NEGATIVE.** The numeric encoder (30 causal features +
logistic GD) beats neither persistence nor the market price OOS on
confirmed labels. Train brier 0.216 vs test 0.261 → moderate
overfitting. Caveats: n=176 markets is small; the selection bias (only
markets traded by the bot) limits generalization;
`market_brier` > persistence suggests the set itself is
atypical (extreme mids in brackets close to the close).

### PROVISIONAL dataset (secondary, NOT certified — `proxy:post_close_verdict`)

Split: 2,880 train markets / 1,917 test. Obs: 810,123 / 619,168.

| OOS metric (test) | loaded model | persistence | market |
|---|---|---|---|
| accuracy | 0.724 | — | — |
| logloss | 0.519 | 0.694 | 0.758 |
| **brier** | **0.1743** | **0.2478** | **0.1777** |

The loaded model clearly beats persistence and ties/marginally beats
the market (~2% brier). **Circularity caveat**: the proxy label
is `close_spot > reference` and `moneyness` (decision spot vs reference)
is a legitimate feature but nearly reveals the proxy near the close — the
number measures "predicting the proxy", not the official result (even though the
proxy matches the official one 97.5% of the time). It is reported as
secondary evidence, never as a certified evaluation.

## 5. Not identified / pending

- **T2 recorder off**: no trade tape (`flow_*` = 0/None),
  no `origin_ts_ms` for books (`origin_delay_ms` only from spot ticks),
  no real `book_delta` (only deduplicated snapshots at ~seconds
  cadence). Enabling `[research_events] enabled=true` is the next
  step for full-fidelity data.
- **0 labels via `market_resolutions` (T3)**: the table does not exist in the
  operational DB; the 639 labels come from the orphan checker's settle
  path (valid but biased toward markets with a position).
- **Markout task** (5s/30s horizons): the `markouts` table (1,589
  rows) exists but was not exported — left as the next job.
- **Sequential RL environment** (steps 4–5 of the brief): out of T8b scope —
  there are no real consequences of untaken actions in this data
  (off-policy without propensity), consistent with the brief.
- **`_MarketState.trades` grows without pruning** (deferred, declared): the
  builder accumulates the trade tape per market for `flow_*`; with a
  real high-frequency tape (T2 recorder enabled) per-market memory
  could grow — today it is harmless because there are no trades in the
  tape, but it will require a sliding window once the tape exists.
- Small sample in confirmed: 176 markets → wide CIs; the negative
  result may be lack of signal or lack of data — they cannot be
  told apart with this coverage.

## 6. Verification

- `python3 -m unittest discover -s research/numeric_model/tests` →
  **25 tests OK** (18 previous + 6 incremental + 1 numpy parity).
  RED→GREEN documented: the first 6 new ones failed
  (`AttributeError: build_observations_incremental`) before
  implementing.
- T8b review round (approved-with-minors): BOTH tapes re-exported with the
  corrected `price_tick` schema (`origin=None` +
  `source_ts_ms_approx` + `SYNTHETIC|NO_REMOTE_TS`) and an
  anti-fallback label cross-check (`suspicious_fallback_labels=0` → confirmed
  set intact); both models retrained — OOS metrics
  **identical** to the above (no label moved).
- Export actually executed (read-only) — manifests in
  `research/data/t8b_{confirmed,provisional}/dataset_manifest.json`.
- Artifacts: `model.json`, `metrics_train.json`, `metrics_eval.json`,
  `metrics_context.json`, `manifest.json` per dataset.
- No commits, `src/` and configs untouched.

## Files

- `research/numeric_model/export_sqlite.py` (new)
- `research/numeric_model/report-T8b.md` (this file)
- `research/numeric_model/dataset.py`, `train.py`, `evaluate.py`,
  `tests/test_causality.py`, `README.md` (modified)
- `research/data/t8b_confirmed/{events.jsonl, dataset_manifest.json,
  model/{model.json,metrics_train.json,metrics_eval.json,
  metrics_context.json,manifest.json}}`
- `research/data/t8b_provisional/{events.jsonl, dataset_manifest.json,
  model/{model.json,metrics_train.json,metrics_eval.json,manifest.json}}`

---

## Addendum — closure 2026-09-24 (REAL tape pipeline)

The earlier T8b artifact is **invalidated as OOS evidence** by two
verified provenance defects:

1. **Backdating of `known_at_ms`**: `export_sqlite.py` assigned
   `known_at_ms = clear_at*1000` when there was no REDEEM entry —
   fabricating an availability that could precede the venue's real
   resolution. Fixed: labels without availability stay
   `known_at_ms=None`, are emitted as an upper bound
   (`avail_is_upper_bound`), are excluded from the causal fit via
   `filter_labels_available` and are counted as
   `label_without_availability`.
2. **Train/test overlap of ~141 min** and a window that ends ~2.7 days
   before the new captures — both datasets are marked
   `NO_APTO_LEAKAGE.md` ("not fit, leakage"; not deleted, they keep historical value).

The new evidence comes from `dataset_tape.py` over the recorder's real
tape (`events_post3/`, ~4.8 GB / ~6.7 M events / ~25 min / 342
official Gamma resolutions), with verifiable provenance (sha256 per
source against `post3_capture.json`), labels only from
`resolution`/`market_meta.outcome` with a real `known_at`, a per-market
decision time mesh and the same purged/embargoed split of
`dataset.temporal_split`. Details and commands in `README.md` §
"Closure 2026-09-24".

Real tape format supported in `dataset.py` (regression found while
inspecting `events_post3`): `{"price","size"}` levels, `book_delta`
with null `market_cid` → `payload.market` and `price_changes` dicts by
`asset_id`. Without that support the causal state stayed frozen at the
snapshots — the new tests cover it with synthetic test-only fixtures.

Numeric result of the real pipeline: see the artifacts and the run report
in `research_runs/closure-2026-09-24/` — the ~25 min
window does not reach the OOS sufficiency thresholds
(≥10,000 test obs, ≥30 markets, ≥1 h), so the verdict stays
`insufficient` unless the operator declares otherwise: circuit
technically demonstrated, with no statistical claim.
