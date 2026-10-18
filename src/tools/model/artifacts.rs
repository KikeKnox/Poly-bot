//! `artifacts` submodule — directory holding the committed copy of the
//! trained numeric model artifact used by the `ModelTool` tool.
//!
//! The artifact itself lives in `src/tools/model/artifacts/btc5m/`:
//! - `manifest.json` (schema_version + sha256 of model.json)
//! - `model.json` (logistic_stdlib_baseline + scaler)
//!
//! These files are NOT source code. They are versioned data from the trained
//! model. They are committed to the repo so the binary can start without an
//! external download. The supervisor validates them by sha256 against
//! `manifest.json` (same contract as `model_observer::ShadowModel`).
//!
//! This `mod.rs` only declares the submodule; the loader lives in
//! `crate::model_observer::ShadowModel::load`.