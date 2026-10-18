# numeric_model: causal dataset and baseline model

Pipeline that turns recorded `ResearchEvent` tapes (the JSONL written by the bot's research recorder, schema in `src/research/contract.rs`) into numeric observations, trains a small baseline model and evaluates it out of sample. Python standard library only (numpy is used for training if installed).

For the real-data evaluation against the market-implied baseline, see `report-T8b.md` in this folder.

## Files

- `dataset.py`: observation builder. Features come from YES/NO books (best bid/ask, mid, spread, depth, imbalance, microprice), the trade tape (signed flow in a 60 s window), Binance spot (returns, realized volatility, moneyness) and time to close; also quality/age fields and the strategy's own simulated state.
  - **Causality**: each observation only sees events with `local_seq <= asof` and `avail_ts_ms <= decision_ts_ms`. Market outcomes are never features, only targets.
  - `temporal_split`: chronological split by market with global UTC boundaries and an embargo; a market crossing a boundary is purged whole.
  - `FeatureScaler`: z-score with median imputation, fitted on train only.
- `dataset_tape.py`: builds the dataset directly from a tape directory (`<source>/<date>.jsonl`): k-way merge in canonical `(avail_ts_ms, local_seq)` order, per-file sha256 (verified against an optional capture manifest), labels only from official resolutions.
- `export_sqlite.py`: read-only export of the bot's SQLite databases to `events.jsonl` plus a dataset manifest.
- `train.py`: deterministic logistic baseline. Writes `model.json`, `metrics_train.json` and `manifest.json`. `--demo` runs on a synthetic tape.
- `evaluate.py`: out-of-sample accuracy, log-loss and Brier score against two baselines: training-label rate and the market-implied probability (`yes_mid`).
- `shadow_eval_t9.py`: offline replay of the model producing decision records in the same schema as the live shadow observer; it never emits orders.
- `environment.py`, `run_env_episodes.py`: client for the `hrf_research_env` binary (JSONL over stdin/stdout: reset, step, legal actions).
- `analyze_t8b.py`: analysis script for the exported datasets.

## Usage

```bash
# Tests
python3 -m unittest discover -s research/numeric_model/tests

# Demo with a synthetic tape
python3 research/numeric_model/train.py --demo --out /tmp/numeric_demo
python3 research/numeric_model/evaluate.py \
    --events <tape.jsonl> --model /tmp/numeric_demo/model.json

# Real tape pipeline (paths are examples)
python3 research/numeric_model/dataset_tape.py \
    --events-dir <tape_dir> --manifest <capture_manifest.json> \
    --out <run_dir>/dataset --decision-min-interval-ms 1000
python3 research/numeric_model/train.py \
    --events <run_dir>/dataset/events.jsonl --out <run_dir>/model \
    --streaming --decision-min-interval-ms 1000 \
    --dataset-manifest <run_dir>/dataset/dataset_manifest.json
python3 research/numeric_model/evaluate.py \
    --events <run_dir>/dataset/events.jsonl \
    --model <run_dir>/model/model.json --out <run_dir>/model --streaming
```

Write outputs to a research directory, never over an active database. Out-of-sample evidence is reported as `insufficient` unless objective size thresholds are exceeded and `--declare-sufficient` is passed explicitly, so a small capture cannot be mistaken for a performance claim.
