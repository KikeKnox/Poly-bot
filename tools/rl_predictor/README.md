# rl_predictor — LAB package

This package implements the v1 contract, inference, ledger-based reward, the
on-policy REINFORCE trainer, the adapter to the JSONL driver and the migration snapshot.
Stdlib only; `unittest` tests, no installs.

## Layout

- `contract.py` — 22 features / 9 actions in literal order
  (`rl_predictor.features.v1`), bit mask, token identity from
  metadata.
- `scaler.py` — train-only `FeatureScaler`, clamp [-10,10], scale>0.
- `policy.py` — `LinearPolicy`: stable masked softmax, deterministic
  legal argmax (tie -> HOLD), seeded sampling with the recorded
  probability.
- `artifact.py` — policy.json / manifest.json / online manifest
  (parent+hash+update chain, `online_candidate`), fail-closed validation
  and activation verdict. `candidate` never activates.
- `reward.py` — per-episode ledger in micro-USDC: conservative equity
  (bids + depth, remainder -> 0 + flag), reward = flow-adjusted equity
  delta, drawdown over adjusted equity, lambdas >= 0.
- `parity.py` — frozen update math (Rust reference): reward-to-go
  gamma=1, baseline from previous batches, gradient A_t(δ−π), per-step
  mean, global L2 clip.
- `trainer.py` — `ReinforceTrainer`: frozen cohorts with an on-policy
  fingerprint, per-episode derived seeds, exclusion of invalid
  episodes, numeric abort, atomic candidate checkpoints.
- `env_adapter.py` — JSONL driver client + fail-closed v1 gate.
  The `hrf_research_env` driver (`src/bin/hrf_research_env.rs`) does NOT
  produce the `policy_v1` observation: the adapter raises
  `ContractUnsupportedError` with a description of what is missing.
- `cli.py` — see below.
- `snapshot.py` — selective copy of research code -> `migration_stage/`
  with `MANIFEST.json` (src/dest/sha256) and `DEPENDENCIES.md`.
- `tests/` — 89 unittest tests covering all of the above.

## Usage (from `bots/Poly_HRF_V4/tools`)

```bash
# tests
PYTHONPATH=. python3 -m unittest discover -s rl_predictor/tests -t .

# literal contract
PYTHONPATH=. python3 -m rl_predictor.cli contract

# probe the real driver (exit 4: cannot spawn; exit 3: driver contract is not v1)
PYTHONPATH=. python3 -m rl_predictor.cli check-driver \
  --bin <hrf_research_env> --tape <tape.jsonl>

# validate artifact + activation verdict
PYTHONPATH=. python3 -m rl_predictor.cli validate --policy policy.json \
  [--manifest manifest.json] [--online-manifest online.json]

# Python<->Rust parity fixture (deterministic, NOT deployable)
PYTHONPATH=. python3 -m rl_predictor.cli make-fixture --out /tmp/fx --seed 9

# train ONE offline update from recorded transitions
# (schema rl_predictor.transitions.v1, source != fixture)
PYTHONPATH=. python3 -m rl_predictor.cli train \
  --transitions t.jsonl --out out/ \
  --predictor-sha256 <sha> --data-manifest-sha256 <sha> \
  --split-manifest-sha256 <sha> --seed 4

# migration_stage snapshot/verification
PYTHONPATH=. python3 -m rl_predictor.cli snapshot \
  --src <BOT root> --dest rl_predictor/migration_stage
PYTHONPATH=. python3 -m rl_predictor.cli snapshot \
  --src <BOT root> --dest rl_predictor/migration_stage --verify-only
```

## Status / limitations

- The online Rust learner lives in `src/rl/`; `parity.py` and
  `make-fixture` pin down the math it must reproduce.
- The `hrf_research_env` driver does not emit `policy_v1`; see
  `env_adapter.build_handoff()` and the output of `check-driver` for the
  exact gap.
- Real multi-update on-policy training would need a v1-capable driver (or
  recorded `rl_predictor.transitions.v1` transitions from real captures).
  Offline `train` performs ONE update per recorded cohort and emits a
  `candidate` artifact, which is never activated automatically.
- The RL strategy is paper-only and disabled.
