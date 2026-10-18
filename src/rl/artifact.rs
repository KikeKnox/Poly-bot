//! RL policy artifact (`policy.json` + `manifest.json`) and metadata for
//! online checkpoints.
//!
//! ## Structure
//!
//! - Approved base artifact (offline):
//!   `src/tools/model/artifacts/rl_predictor/<version>/{policy.json,manifest.json}`.
//!   `status == "paper_approved"` + a valid manifest with `approval.scope=="paper"`
//!   is REQUIRED for the strategy to trade. No other state activates a
//!   strategy; an offline `candidate` is for tests only.
//! - Online checkpoints (written by the learner, off the hot path):
//!   `<checkpoint_dir>/checkpoints/v%06d/{policy.json,meta.json}` +
//!   `<checkpoint_dir>/active.json` → pointer to the last valid one.
//!   `meta.json` carries `kind="online_candidate"`/`"rollback"`, `parent_sha256`,
//!   `parent_version`, `base_sha256` (root = approved artifact). The chain must
//!   be verifiable up to the approved artifact (online descendants form a
//!   parent+hash+update chain).
//!
//! A policy's hash is always the `sha256` of the BYTES of its `policy.json`
//! file on disk, not of the in-memory object serialization.
//! `serialize_policy` produces deterministic bytes (fixed field order,
//! floats via ryu/serde_json) so the hash of a written checkpoint is
//! reproducible.

use std::fs;
use std::path::{Path, PathBuf};

use crate::model_observer::sha256_hex;
use crate::rl::contract::{ACTION_NAMES, FEATURE_NAMES, FEATURE_SCHEMA, NUM_ACTIONS, NUM_FEATURES};
use crate::rl::policy::{LinearSoftmaxPolicy, PolicyScaler};

pub const POLICY_SCHEMA_VERSION: u32 = 1;
pub const POLICY_ALGORITHM: &str = "reinforce_linear_v1";
/// `kind`/`schema_version` of the online manifest
/// (`rl_predictor.online_manifest.v1` — mirror of the Python `contract.py`/`artifact.py`).
pub const ONLINE_MANIFEST_KIND: &str = "rl_predictor.online_manifest.v1";
pub const ONLINE_MANIFEST_SCHEMA_VERSION: u32 = 1;
/// Honest hex64 sentinel for online cohorts without an offline data manifest:
/// sha256 of empty content. It is NOT a fabricated data hash; it explicitly
/// means "there is no offline manifest for this cohort" in a format the
/// validators accept (hex64).
pub const EMPTY_MANIFEST_SHA256: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Errors loading/validating the artifact or the checkpoint chain.
#[derive(Debug)]
pub enum PolicyArtifactError {
    Io { path: PathBuf, source: std::io::Error },
    Schema { detail: String },
    /// The real sha256 of the file does not match the declared one — corruption
    /// or a tampered artifact. Rejected fail-closed.
    HashMismatch { expected: String, real: String },
    /// The artifact is structurally valid but not approved for
    /// paper (`status != "paper_approved"` or a manifest without `approval`).
    NotApproved { status: String },
    /// The online chain does not reach the approved base artifact.
    ChainBroken { detail: String },
}

impl std::fmt::Display for PolicyArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
            Self::Schema { detail } => write!(f, "invalid policy artifact: {detail}"),
            Self::HashMismatch { expected, real } => write!(
                f,
                "sha256 of policy.json does not match the declared one (expected {expected}, real {real})"
            ),
            Self::NotApproved { status } => write!(
                f,
                "policy not approved for paper (status='{status}') — the strategy stays disabled"
            ),
            Self::ChainBroken { detail } => write!(f, "broken checkpoint chain: {detail}"),
        }
    }
}

impl std::error::Error for PolicyArtifactError {}

// ── JSON documents ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyTrainingDoc {
    pub run_id: String,
    pub data_manifest_sha256: String,
    pub split_manifest_sha256: String,
    pub seed: u64,
    pub completed_episodes: u64,
    pub updates: u64,
}

/// `policy.json` — strict schema. No other `status` is accepted.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyDoc {
    pub schema_version: u32,
    pub algorithm: String,
    pub feature_schema: String,
    pub feature_names: Vec<String>,
    pub action_names: Vec<String>,
    pub predictor_sha256: String,
    pub scaler: PolicyScaler,
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub training: PolicyTrainingDoc,
    pub status: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyApprovalDoc {
    pub scope: String,
    pub report_sha256: String,
}

/// `manifest.json` of the base artifact — links the offline approval.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyManifestDoc {
    pub schema_version: u32,
    pub policy_sha256: String,
    pub predictor_sha256: String,
    pub feature_schema: String,
    pub approval: Option<PolicyApprovalDoc>,
}

/// `meta.json` of an online checkpoint — a separate online manifest linking
/// the startup approval, parent hash, update config and training counters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnlineMetaDoc {
    pub schema_version: u32,
    /// "online_candidate" | "rollback" — never "paper_approved" (the
    /// offline approval is never forged).
    pub kind: String,
    pub version: u64,
    pub update_index: u64,
    pub policy_sha256: String,
    pub parent_sha256: String,
    pub parent_version: u64,
    /// sha256 of the `policy.json` of the approved base artifact (root).
    pub base_sha256: String,
    /// Path of the base artifact the chain was verified against.
    pub base_artifact_dir: String,
    pub created_ts_ms: i64,
    /// Update config in effect (lr, gates) — for audit/reproduction.
    pub update_config: serde_json::Value,
    /// Learner counters at checkpoint time.
    pub counters: serde_json::Value,
}

/// Atomic pointer to the last valid checkpoint (`active.json`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ActiveCheckpointDoc {
    pub schema_version: u32,
    pub version: u64,
    pub dir: String,
}

// ── Validation ───────────────────────────────────────────────────────────────

fn validate_doc(doc: &PolicyDoc) -> Result<(), PolicyArtifactError> {
    let err = |d: String| PolicyArtifactError::Schema { detail: d };
    if doc.schema_version != POLICY_SCHEMA_VERSION {
        return Err(err(format!(
            "schema_version {} != {POLICY_SCHEMA_VERSION}",
            doc.schema_version
        )));
    }
    if doc.algorithm != POLICY_ALGORITHM {
        return Err(err(format!("algorithm '{}' != '{POLICY_ALGORITHM}'", doc.algorithm)));
    }
    if doc.feature_schema != FEATURE_SCHEMA {
        return Err(err(format!(
            "feature_schema '{}' != '{FEATURE_SCHEMA}'",
            doc.feature_schema
        )));
    }
    if doc.feature_names.len() != NUM_FEATURES
        || doc
            .feature_names
            .iter()
            .zip(FEATURE_NAMES.iter())
            .any(|(a, b)| a != b)
    {
        return Err(err(
            "feature_names does not match the rl_predictor.features.v1 contract (order included)"
                .into(),
        ));
    }
    if doc.action_names.len() != NUM_ACTIONS
        || doc
            .action_names
            .iter()
            .zip(ACTION_NAMES.iter())
            .any(|(a, b)| a != b)
    {
        return Err(err(
            "action_names does not match the contract (order included)".into(),
        ));
    }
    if !matches!(
        doc.status.as_str(),
        "candidate" | "paper_approved" | "online_candidate"
    ) {
        return Err(err(format!("status '{}' not allowed", doc.status)));
    }
    let policy = LinearSoftmaxPolicy {
        scaler: doc.scaler.clone(),
        weights: doc.weights.clone(),
        bias: doc.bias.clone(),
    };
    if !policy.is_valid() {
        return Err(err(
            "wrong dimensions weights[9][22]/bias[9]/scaler[22] or non-finite values".into(),
        ));
    }
    if doc.scaler.scale.iter().any(|s| !(*s > 0.0) || !s.is_finite()) {
        return Err(err("scaler.scale must be >0 and finite for all features".into()));
    }
    Ok(())
}

/// Reads + parses + validates the structure of a `policy.json`. Returns
/// (doc, hash_bytes). Does not require approval — that is the caller's job.
fn read_policy(dir: &Path) -> Result<(PolicyDoc, String), PolicyArtifactError> {
    let path = dir.join("policy.json");
    let raw = fs::read(&path).map_err(|source| PolicyArtifactError::Io {
        path: path.clone(),
        source,
    })?;
    let doc: PolicyDoc =
        serde_json::from_slice(&raw).map_err(|e| PolicyArtifactError::Schema {
            detail: format!("policy.json does not parse: {e}"),
        })?;
    validate_doc(&doc)?;
    Ok((doc, sha256_hex(&raw)))
}

fn read_manifest(dir: &Path) -> Result<PolicyManifestDoc, PolicyArtifactError> {
    let path = dir.join("manifest.json");
    let raw = fs::read(&path).map_err(|source| PolicyArtifactError::Io {
        path: path.clone(),
        source,
    })?;
    let doc: PolicyManifestDoc =
        serde_json::from_slice(&raw).map_err(|e| PolicyArtifactError::Schema {
            detail: format!("manifest.json does not parse: {e}"),
        })?;
    if doc.schema_version != POLICY_SCHEMA_VERSION {
        return Err(PolicyArtifactError::Schema {
            detail: format!(
                "manifest schema_version {} != {POLICY_SCHEMA_VERSION}",
                doc.schema_version
            ),
        });
    }
    Ok(doc)
}

/// Base artifact loaded and VERIFIED as approved for paper.
#[derive(Debug)]
pub struct ApprovedArtifact {
    pub policy: LinearSoftmaxPolicy,
    /// sha256 of the bytes of `policy.json` (root of the online chain).
    pub policy_sha256: String,
    /// sha256 of the predictor the policy was trained against.
    pub predictor_sha256: String,
    pub artifact_dir: PathBuf,
    pub training: PolicyTrainingDoc,
}

/// Loads the base artifact REQUIRING paper approval: a valid manifest with
/// `approval.scope=="paper"`, a matching policy hash, a predictor matching
/// the manifest's, and `status=="paper_approved"`. Any failure → `Err` and
/// the strategy stays disabled (there is no fallback to synthetic weights).
///
/// `expected_predictor_sha256`: if given, the artifact must declare exactly
/// that predictor — prevents activating a policy trained against a different
/// predictor than the one running in the bot.
pub fn load_approved(
    dir: &Path,
    expected_predictor_sha256: Option<&str>,
) -> Result<ApprovedArtifact, PolicyArtifactError> {
    let manifest = read_manifest(dir)?;
    let (doc, real_hash) = read_policy(dir)?;
    if real_hash != manifest.policy_sha256 {
        return Err(PolicyArtifactError::HashMismatch {
            expected: manifest.policy_sha256.clone(),
            real: real_hash,
        });
    }
    if doc.predictor_sha256 != manifest.predictor_sha256 {
        return Err(PolicyArtifactError::Schema {
            detail: format!(
                "policy.json predictor_sha256 ({}) != manifest ({})",
                doc.predictor_sha256, manifest.predictor_sha256
            ),
        });
    }
    if let Some(expected) = expected_predictor_sha256 {
        if doc.predictor_sha256 != expected {
            return Err(PolicyArtifactError::Schema {
                detail: format!(
                    "policy trained for predictor {} but the asset predictor is {expected}",
                    doc.predictor_sha256
                ),
            });
        }
    }
    if manifest.feature_schema != FEATURE_SCHEMA {
        return Err(PolicyArtifactError::Schema {
            detail: format!(
                "manifest feature_schema '{}' != '{FEATURE_SCHEMA}'",
                manifest.feature_schema
            ),
        });
    }
    if doc.status != "paper_approved" {
        return Err(PolicyArtifactError::NotApproved {
            status: doc.status.clone(),
        });
    }
    match &manifest.approval {
        Some(a) if a.scope == "paper" && !a.report_sha256.is_empty() => {}
        _ => {
            return Err(PolicyArtifactError::NotApproved {
                status: "manifest without approval{scope:'paper'}".into(),
            })
        }
    }
    if doc.training.completed_episodes == 0 || doc.training.updates == 0 {
        // "Approved" policy that was never trained — a disguised fixture.
        return Err(PolicyArtifactError::NotApproved {
            status: "training.completed_episodes/updates == 0".into(),
        });
    }
    Ok(ApprovedArtifact {
        policy: LinearSoftmaxPolicy {
            scaler: doc.scaler,
            weights: doc.weights,
            bias: doc.bias,
        },
        policy_sha256: real_hash,
        predictor_sha256: doc.predictor_sha256,
        artifact_dir: dir.to_path_buf(),
        training: doc.training,
    })
}

/// Tests/harness ONLY: loads a `candidate` artifact WITHOUT requiring paper
/// approval. The injection is explicit at the call site — `load_approved`
/// (production) keeps rejecting candidates; this function is NEVER compiled
/// outside `cfg(test)`.
#[cfg(test)]
pub fn load_candidate_for_test(
    dir: &Path,
) -> Result<ApprovedArtifact, PolicyArtifactError> {
    let (doc, real_hash) = read_policy(dir)?;
    if doc.status != "candidate" {
        return Err(PolicyArtifactError::Schema {
            detail: format!(
                "harness: the injected artifact must have status='candidate', \
                 not '{}' (an injected paper_approved would be a forgery)",
                doc.status
            ),
        });
    }
    Ok(ApprovedArtifact {
        policy: LinearSoftmaxPolicy {
            scaler: doc.scaler,
            weights: doc.weights,
            bias: doc.bias,
        },
        policy_sha256: real_hash,
        predictor_sha256: doc.predictor_sha256,
        artifact_dir: dir.to_path_buf(),
        training: doc.training,
    })
}

/// Serializes a policy to the bytes of `policy.json` (deterministic: struct
/// field order, floats via serde_json). For online checkpoints `status` is
/// always "candidate" — online promotion is recorded in `meta.json`
/// (`kind="online_candidate"`), never by forging `paper_approved`.
pub fn serialize_policy(
    policy: &LinearSoftmaxPolicy,
    predictor_sha256: &str,
    training: &PolicyTrainingDoc,
    status: &str,
) -> Result<Vec<u8>, PolicyArtifactError> {
    let doc = PolicyDoc {
        schema_version: POLICY_SCHEMA_VERSION,
        algorithm: POLICY_ALGORITHM.to_string(),
        feature_schema: FEATURE_SCHEMA.to_string(),
        feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
        action_names: ACTION_NAMES.iter().map(|s| s.to_string()).collect(),
        predictor_sha256: predictor_sha256.to_string(),
        scaler: policy.scaler.clone(),
        weights: policy.weights.clone(),
        bias: policy.bias.clone(),
        training: training.clone(),
        status: status.to_string(),
    };
    serde_json::to_vec_pretty(&doc).map_err(|e| PolicyArtifactError::Schema {
        detail: format!("serializing policy.json: {e}"),
    })
}

// ── Online checkpoints ───────────────────────────────────────────────────────

/// `online_manifest.json` — mirror of the Python format
/// (`artifact.py validate_online_manifest_dict`): a chain of
/// `parent_sha256→child_sha256` links with a monotonic `update_index`
/// (= the link's version), `update_config` (lr/grad_clip/seed/
/// cohort_episodes), `counters` and `promoted_as="online_candidate"`.
/// Links are rebuilt by reading the parent's manifest (if the parent
/// version is >0); the root link starts from `base_sha256`.
fn build_online_manifest(
    checkpoint_dir: &Path,
    meta: &OnlineMetaDoc,
) -> Result<serde_json::Value, PolicyArtifactError> {
    let err = |d: String| PolicyArtifactError::Schema { detail: d };
    let mut links: Vec<serde_json::Value> = if meta.parent_version > 0 {
        // Inherits the parent's chain — fail-closed if it is broken.
        let parent_dir = checkpoint_dir
            .join("checkpoints")
            .join(format!("v{:06}", meta.parent_version));
        let raw = fs::read(parent_dir.join("online_manifest.json")).map_err(|e| {
            PolicyArtifactError::ChainBroken {
                detail: format!(
                    "parent online_manifest v{:06} unreadable: {e}",
                    meta.parent_version
                ),
            }
        })?;
        let parent_doc: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| {
                err(format!(
                    "parent online_manifest v{:06} does not parse: {e}",
                    meta.parent_version
                ))
            })?;
        let prev = parent_doc
            .get("links")
            .and_then(|l| l.as_array())
            .cloned()
            .ok_or_else(|| err("parent online_manifest without links".into()))?;
        // Sanity check: the parent's tip must be the declared parent_sha256.
        let tip = prev
            .last()
            .and_then(|l| l.get("child_sha256"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if tip != meta.parent_sha256 {
            return Err(err(format!(
                "online_manifest padre tip {tip} != parent_sha256 {}",
                meta.parent_sha256
            )));
        }
        prev
    } else {
        Vec::new() // root link: parent = approved base artifact
    };
    links.push(serde_json::json!({
        "parent_sha256": meta.parent_sha256,
        "child_sha256": meta.policy_sha256,
        // Link version as a monotonic position in the chain
        // (a rollback does not increment update_index but does increment version).
        "update_index": meta.version,
        "update_config": meta.update_config,
        "counters": {
            "completed_episodes": meta.counters.get("completed_episodes").cloned().unwrap_or(serde_json::json!(0)),
            "updates": meta.counters.get("updates").cloned().unwrap_or(serde_json::json!(0)),
        },
        // Every link in the chain is an online_candidate — NEVER
        // paper_approved (the offline approval is never forged).
        "promoted_as": "online_candidate",
    }));
    Ok(serde_json::json!({
        "schema_version": ONLINE_MANIFEST_SCHEMA_VERSION,
        "kind": ONLINE_MANIFEST_KIND,
        "root_policy_sha256": meta.base_sha256,
        "links": links,
    }))
}

/// Writes an atomic checkpoint: `checkpoints/v%06d/{policy.json,meta.json,
/// online_manifest.json}` via tmp+rename, and updates `active.json` (same
/// pattern). `online_manifest.json` is the chain manifest (parent+hash+update);
/// `meta.json` is the internal Rust manifest.
/// Returns the `policy_sha256` of the written bytes.
pub fn write_checkpoint(
    checkpoint_dir: &Path,
    _policy: &LinearSoftmaxPolicy,
    meta: &OnlineMetaDoc,
    policy_bytes: &[u8],
) -> Result<PathBuf, PolicyArtifactError> {
    let dir_name = format!("v{:06}", meta.version);
    let dir = checkpoint_dir.join("checkpoints").join(&dir_name);
    let tmp_dir = checkpoint_dir.join("checkpoints").join(format!(".{dir_name}.tmp"));
    let io = |e: std::io::Error, p: PathBuf| PolicyArtifactError::Io {
        path: p,
        source: e,
    };
    // Build the online manifest BEFORE creating the tmp dir: if the
    // parent's chain is broken, abort without writing anything.
    let manifest = build_online_manifest(checkpoint_dir, meta)?;
    fs::create_dir_all(&tmp_dir).map_err(|e| io(e, tmp_dir.clone()))?;
    fs::write(tmp_dir.join("policy.json"), policy_bytes)
        .map_err(|e| io(e, tmp_dir.join("policy.json")))?;
    let meta_bytes = serde_json::to_vec_pretty(meta).map_err(|e| PolicyArtifactError::Schema {
        detail: format!("serializing meta.json: {e}"),
    })?;
    fs::write(tmp_dir.join("meta.json"), &meta_bytes)
        .map_err(|e| io(e, tmp_dir.join("meta.json")))?;
    let manifest_bytes =
        serde_json::to_vec_pretty(&manifest).map_err(|e| PolicyArtifactError::Schema {
            detail: format!("serializing online_manifest.json: {e}"),
        })?;
    fs::write(tmp_dir.join("online_manifest.json"), &manifest_bytes)
        .map_err(|e| io(e, tmp_dir.join("online_manifest.json")))?;
    if dir.exists() {
        // A published version is never rewritten — a version collision is a
        // learner counter bug; fail loudly instead of overwriting.
        return Err(PolicyArtifactError::Schema {
            detail: format!("checkpoint {dir_name} already exists — duplicate version"),
        });
    }
    fs::rename(&tmp_dir, &dir).map_err(|e| io(e, dir.clone()))?;
    // Atomic active.json: tmp + rename (same FS, atomic operation).
    let active = ActiveCheckpointDoc {
        schema_version: POLICY_SCHEMA_VERSION,
        version: meta.version,
        dir: format!("checkpoints/{dir_name}"),
    };
    let active_bytes = serde_json::to_vec_pretty(&active).map_err(|e| {
        PolicyArtifactError::Schema {
            detail: format!("serializing active.json: {e}"),
        }
    })?;
    let active_tmp = checkpoint_dir.join(".active.json.tmp");
    fs::write(&active_tmp, &active_bytes).map_err(|e| io(e, active_tmp.clone()))?;
    fs::rename(&active_tmp, checkpoint_dir.join("active.json"))
        .map_err(|e| io(e, checkpoint_dir.join("active.json")))?;
    Ok(dir)
}

/// Reads `meta.json` of a checkpoint and verifies its hash + internal consistency.
fn read_meta(dir: &Path) -> Result<(OnlineMetaDoc, String), PolicyArtifactError> {
    let (doc, policy_sha) = read_policy(dir)?;
    let meta_path = dir.join("meta.json");
    let raw = fs::read(&meta_path).map_err(|source| PolicyArtifactError::Io {
        path: meta_path.clone(),
        source,
    })?;
    let meta: OnlineMetaDoc =
        serde_json::from_slice(&raw).map_err(|e| PolicyArtifactError::Schema {
            detail: format!("meta.json does not parse: {e}"),
        })?;
    let err = |d: String| PolicyArtifactError::Schema { detail: d };
    if meta.schema_version != POLICY_SCHEMA_VERSION {
        return Err(err(format!("meta schema_version {}", meta.schema_version)));
    }
    if meta.policy_sha256 != policy_sha {
        return Err(PolicyArtifactError::HashMismatch {
            expected: meta.policy_sha256.clone(),
            real: policy_sha,
        });
    }
    if !matches!(meta.kind.as_str(), "online_candidate" | "rollback") {
        return Err(err(format!("meta kind '{}' not allowed", meta.kind)));
    }
    // An online checkpoint must never claim an offline approval.
    if doc.status == "paper_approved" {
        return Err(err(
            "online checkpoint with status 'paper_approved' — approval forgery".into(),
        ));
    }
    Ok((meta, policy_sha))
}

/// Walks the checkpoint chain backwards from `active.json` (or the highest
/// version) up to the approved base. Returns the last valid link:
/// `(meta, policy, policy_sha)`. `Err` if the chain is broken or does not
/// reach `base_sha256` — the caller decides the fallback (reload the base).
pub fn load_checkpoint_chain(
    checkpoint_dir: &Path,
    base_sha256: &str,
) -> Result<(OnlineMetaDoc, LinearSoftmaxPolicy, String), PolicyArtifactError> {
    let err = |d: String| PolicyArtifactError::ChainBroken { detail: d };
    // Entry point: active.json if it exists; otherwise the highest version v*.
    let start_dir = {
        let active_path = checkpoint_dir.join("active.json");
        if active_path.exists() {
            let raw = fs::read(&active_path).map_err(|source| PolicyArtifactError::Io {
                path: active_path.clone(),
                source,
            })?;
            let active: ActiveCheckpointDoc =
                serde_json::from_slice(&raw).map_err(|e| PolicyArtifactError::Schema {
                    detail: format!("active.json does not parse: {e}"),
                })?;
            checkpoint_dir.join(&active.dir)
        } else {
            greatest_version_dir(checkpoint_dir).ok_or_else(|| err("no checkpoints".into()))?
        }
    };
    let (meta, policy_sha) = read_meta(&start_dir)?;
    // Verify that the declared root matches the approved artifact.
    if meta.base_sha256 != base_sha256 {
        return Err(err(format!(
            "checkpoint base_sha256 {} != approved artifact {}",
            meta.base_sha256, base_sha256
        )));
    }
    // Walk the chain up to the approved base artifact (parent_version=0).
    // Each link must exist and its real `policy_sha256` must be exactly the
    // `parent_sha256` that the child declared.
    let mut cursor = meta.clone();
    while cursor.parent_version > 0 {
        let parent_dir = checkpoint_dir
            .join("checkpoints")
            .join(format!("v{:06}", cursor.parent_version));
        let (parent_meta, parent_sha) = read_meta(&parent_dir).map_err(|e| {
            err(format!("parent v{:06} unreadable: {e}", cursor.parent_version))
        })?;
        if parent_sha != cursor.parent_sha256 {
            return Err(err(format!(
                "v{:06} declares a parent_sha256 different from the parent's real hash",
                cursor.version
            )));
        }
        cursor = parent_meta;
    }
    if cursor.parent_sha256 != base_sha256 {
        return Err(err(
            "the root link does not point to the approved base artifact".into(),
        ));
    }
    let (doc, _) = read_policy(&start_dir)?;
    let policy = LinearSoftmaxPolicy {
        scaler: doc.scaler,
        weights: doc.weights,
        bias: doc.bias,
    };
    Ok((meta, policy, policy_sha))
}

fn greatest_version_dir(checkpoint_dir: &Path) -> Option<PathBuf> {
    let base = checkpoint_dir.join("checkpoints");
    let mut best: Option<(u64, PathBuf)> = None;
    for e in fs::read_dir(&base).ok()? .flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if let Some(v) = name
            .strip_prefix('v')
            .and_then(|s| s.parse::<u64>().ok())
        {
            if best.as_ref().map(|(bv, _)| v > *bv).unwrap_or(true) {
                best = Some((v, e.path()));
            }
        }
    }
    best.map(|(_, p)| p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_doc(status: &str) -> PolicyDoc {
        PolicyDoc {
            schema_version: POLICY_SCHEMA_VERSION,
            algorithm: POLICY_ALGORITHM.to_string(),
            feature_schema: FEATURE_SCHEMA.to_string(),
            feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
            action_names: ACTION_NAMES.iter().map(|s| s.to_string()).collect(),
            predictor_sha256: "aa".repeat(32),
            scaler: PolicyScaler {
                mean: vec![0.0; NUM_FEATURES],
                scale: vec![1.0; NUM_FEATURES],
            },
            weights: vec![vec![0.0; NUM_FEATURES]; NUM_ACTIONS],
            bias: vec![0.0; NUM_ACTIONS],
            training: PolicyTrainingDoc {
                run_id: "test".into(),
                data_manifest_sha256: "bb".repeat(32),
                split_manifest_sha256: "cc".repeat(32),
                seed: 7,
                completed_episodes: 10,
                updates: 3,
            },
            status: status.to_string(),
        }
    }

    fn write_artifact(dir: &Path, status: &str) -> String {
        fs::create_dir_all(dir).unwrap();
        let bytes = serde_json::to_vec_pretty(&valid_doc(status)).unwrap();
        fs::write(dir.join("policy.json"), &bytes).unwrap();
        let sha = sha256_hex(&bytes);
        let manifest = PolicyManifestDoc {
            schema_version: POLICY_SCHEMA_VERSION,
            policy_sha256: sha.clone(),
            predictor_sha256: "aa".repeat(32),
            feature_schema: FEATURE_SCHEMA.to_string(),
            approval: Some(PolicyApprovalDoc {
                scope: "paper".into(),
                report_sha256: "dd".repeat(32),
            }),
        };
        fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        sha
    }

    #[test]
    fn approved_artifact_load() {
        let tmp = tempfile::tempdir().unwrap();
        write_artifact(tmp.path(), "paper_approved");
        let art = load_approved(tmp.path(), Some(&"aa".repeat(32))).unwrap();
        assert_eq!(art.training.run_id, "test");
    }

    #[test]
    fn candidate_not_active() {
        let tmp = tempfile::tempdir().unwrap();
        write_artifact(tmp.path(), "candidate");
        let e = load_approved(tmp.path(), None).unwrap_err();
        assert!(matches!(e, PolicyArtifactError::NotApproved { .. }));
    }

    #[test]
    fn corrupt_hash_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        write_artifact(tmp.path(), "paper_approved");
        let mut bytes = fs::read(tmp.path().join("policy.json")).unwrap();
        bytes.truncate(bytes.len() - 20);
        bytes.extend_from_slice(b"\"corrupt\":1}");
        fs::write(tmp.path().join("policy.json"), &bytes).unwrap();
        let e = load_approved(tmp.path(), None).unwrap_err();
        // May be HashMismatch or Schema depending on the parser — both are rejections.
        assert!(matches!(
            e,
            PolicyArtifactError::HashMismatch { .. } | PolicyArtifactError::Schema { .. }
        ));
    }

    #[test]
    fn different_predictor_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        write_artifact(tmp.path(), "paper_approved");
        let e = load_approved(tmp.path(), Some(&"ff".repeat(32))).unwrap_err();
        assert!(matches!(e, PolicyArtifactError::Schema { .. }));
    }

    #[test]
    fn action_names_unordered_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        let mut doc = valid_doc("paper_approved");
        doc.action_names.swap(1, 2);
        fs::create_dir_all(tmp.path()).unwrap();
        let bytes = serde_json::to_vec_pretty(&doc).unwrap();
        fs::write(tmp.path().join("policy.json"), &bytes).unwrap();
        let manifest = PolicyManifestDoc {
            schema_version: POLICY_SCHEMA_VERSION,
            policy_sha256: sha256_hex(&bytes),
            predictor_sha256: "aa".repeat(32),
            feature_schema: FEATURE_SCHEMA.to_string(),
            approval: Some(PolicyApprovalDoc {
                scope: "paper".into(),
                report_sha256: "dd".repeat(32),
            }),
        };
        fs::write(
            tmp.path().join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        assert!(load_approved(tmp.path(), None).is_err());
    }

    fn empty_policy() -> LinearSoftmaxPolicy {
        LinearSoftmaxPolicy {
            scaler: PolicyScaler {
                mean: vec![0.0; NUM_FEATURES],
                scale: vec![1.0; NUM_FEATURES],
            },
            weights: vec![vec![0.0; NUM_FEATURES]; NUM_ACTIONS],
            bias: vec![0.0; NUM_ACTIONS],
        }
    }

    fn meta_v(version: u64, parent_version: u64, parent_sha: &str, sha: &str, base: &str) -> OnlineMetaDoc {
        OnlineMetaDoc {
            schema_version: POLICY_SCHEMA_VERSION,
            kind: "online_candidate".into(),
            version,
            update_index: version,
            policy_sha256: sha.into(),
            parent_sha256: parent_sha.into(),
            parent_version,
            base_sha256: base.into(),
            base_artifact_dir: "/base".into(),
            created_ts_ms: 0,
            update_config: serde_json::json!({
                "lr": 0.05, "grad_clip": 1.0, "seed": 0, "cohort_episodes": 4
            }),
            counters: serde_json::json!({"completed_episodes": 4, "updates": 1}),
        }
    }

    /// The v1→v2 chain writes `online_manifest.json` (chain format: parent→child
    /// links with `promoted_as="online_candidate"`), the hashes are the real
    /// sha256 of the bytes on disk, and `load_checkpoint_chain` verifies each
    /// link up to the base.
    #[test]
    fn honest_manifest_checkpoint_chain() {
        let tmp = tempfile::tempdir().unwrap();
        let ckpt = tmp.path().join("ckpt");
        let base_sha = "bb".repeat(32);
        let training = valid_doc("online_candidate").training;

        let b1 = serialize_policy(&empty_policy(), &"aa".repeat(32), &training, "online_candidate").unwrap();
        let m1 = meta_v(1, 0, &base_sha, &sha256_hex(&b1), &base_sha);
        write_checkpoint(&ckpt, &empty_policy(), &m1, &b1).unwrap();

        let mut p2 = empty_policy();
        p2.bias[0] = 0.1;
        let b2 = serialize_policy(&p2, &"aa".repeat(32), &training, "online_candidate").unwrap();
        let m2 = meta_v(2, 1, &sha256_hex(&b1), &sha256_hex(&b2), &base_sha);
        write_checkpoint(&ckpt, &p2, &m2, &b2).unwrap();

        let man: serde_json::Value = serde_json::from_slice(
            &fs::read(ckpt.join("checkpoints/v000002/online_manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(man["kind"], ONLINE_MANIFEST_KIND);
        assert_eq!(man["schema_version"], ONLINE_MANIFEST_SCHEMA_VERSION);
        assert_eq!(man["root_policy_sha256"], base_sha);
        let links = man["links"].as_array().unwrap();
        assert_eq!(links.len(), 2);
        assert_eq!(links[0]["child_sha256"], sha256_hex(&b1));
        assert_eq!(links[1]["parent_sha256"], sha256_hex(&b1));
        assert_eq!(links[1]["child_sha256"], sha256_hex(&b2));
        for l in links {
            assert_eq!(l["promoted_as"], "online_candidate");
        }

        let (meta, _p, sha) = load_checkpoint_chain(&ckpt, &base_sha).unwrap();
        assert_eq!(meta.version, 2);
        assert_eq!(sha, sha256_hex(&b2), "real hash of the bytes on disk");
    }

    /// A checkpoint whose `policy.json` claims `paper_approved` is an approval
    /// forgery — `read_meta` always rejects it.
    #[test]
    fn checkpoint_paper_approved_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("checkpoints/v000001");
        fs::create_dir_all(&dir).unwrap();
        let bytes = serde_json::to_vec_pretty(&valid_doc("paper_approved")).unwrap();
        fs::write(dir.join("policy.json"), &bytes).unwrap();
        let meta = meta_v(1, 0, &"bb".repeat(32), &sha256_hex(&bytes), &"bb".repeat(32));
        fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta).unwrap()).unwrap();
        let e = read_meta(&dir).unwrap_err();
        assert!(
            matches!(e, PolicyArtifactError::Schema { .. }),
            "paper_approved in a checkpoint must be rejected: {e:?}"
        );
    }

    /// Declared hash different from the real bytes → rejection
    /// (`HashMismatch`) — fabricated provenance is not accepted.
    #[test]
    fn false_meta_hash_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("checkpoints/v000001");
        fs::create_dir_all(&dir).unwrap();
        let bytes = serde_json::to_vec_pretty(&valid_doc("online_candidate")).unwrap();
        fs::write(dir.join("policy.json"), &bytes).unwrap();
        let mut meta = meta_v(1, 0, &"bb".repeat(32), &"ee".repeat(32), &"bb".repeat(32));
        meta.policy_sha256 = "ee".repeat(32);
        fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta).unwrap()).unwrap();
        assert!(matches!(
            read_meta(&dir).unwrap_err(),
            PolicyArtifactError::HashMismatch { .. }
        ));
    }

    /// The promoted `paper_bootstrap_v1` artifact (weights/scaler/bias of the
    /// tip f6801832… of the real run) MUST load through the production path —
    /// a regression test binding the promotion to `load_approved` with no bypass.
    #[test]
    fn paper_bootstrap_v1_loads_real_approved() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/tools/model/artifacts/rl_predictor/paper_bootstrap_v1");
        let art = load_approved(
            &dir,
            Some("ef4e4b0860ef9601fb21462539f4d43d2c8af378060eb2dcb63d153502e4c396"),
        )
        .expect("paper_bootstrap_v1 must pass the production load_approved");
        assert_eq!(art.policy_sha256.len(), 64);
        assert_eq!(art.training.completed_episodes, 1);
        assert_eq!(art.training.updates, 1);
        // Weights from the bootstrap of the real update — not a zero fixture.
        assert!(art.policy.weights.iter().flatten().any(|w| *w != 0.0));
        // Real scaler fitted over the trajectory — not the identity.
        assert!(art.policy.scaler.scale.iter().any(|s| *s != 1.0));
    }

    /// The same artifact with a tampered weight must be rejected
    /// (`HashMismatch`) — the hash binds policy.json to the approved manifest.
    #[test]
    fn paper_bootstrap_v1_tamper_rejects() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/tools/model/artifacts/rl_predictor/paper_bootstrap_v1");
        let tmp = tempfile::tempdir().unwrap();
        for f in ["policy.json", "manifest.json"] {
            fs::copy(src.join(f), tmp.path().join(f)).unwrap();
        }
        let raw = fs::read_to_string(tmp.path().join("policy.json")).unwrap();
        let tampered = raw.replacen("-0.000024298902755136318", "0.000024298902755136318", 1);
        assert_ne!(raw, tampered, "the tampered byte must exist in the artifact");
        fs::write(tmp.path().join("policy.json"), tampered).unwrap();
        let e = load_approved(tmp.path(), None).unwrap_err();
        assert!(
            matches!(e, PolicyArtifactError::HashMismatch { .. }),
            "tampering must be a HashMismatch: {e:?}"
        );
    }

    /// Cross-language parity: writes a real v1→v2 chain into
    /// `target/rl_parity_ckpt/` so that
    /// `rl_predictor.integration.validate_rust_checkpoint` (Python)
    /// verifies it with `validate_online_manifest_dict` +
    /// `check_activation`. The directory is generated by the test itself
    /// (target/ is a build artifact).
    #[test]
    fn checkpoint_exportable_to_python() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/rl_parity_ckpt");
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        let base_sha = "bb".repeat(32);
        let training = valid_doc("online_candidate").training;
        let p = empty_policy();
        let b1 = serialize_policy(&p, &"aa".repeat(32), &training, "online_candidate").unwrap();
        write_checkpoint(&dir, &p, &meta_v(1, 0, &base_sha, &sha256_hex(&b1), &base_sha), &b1)
            .unwrap();
        let mut p2 = empty_policy();
        p2.bias[0] = 0.01;
        let b2 = serialize_policy(&p2, &"aa".repeat(32), &training, "online_candidate").unwrap();
        write_checkpoint(&dir, &p2, &meta_v(2, 1, &sha256_hex(&b1), &sha256_hex(&b2), &base_sha), &b2)
            .unwrap();
        // Self-check: the Rust loader also verifies the chain.
        let (meta, ..) = load_checkpoint_chain(&dir, &base_sha).unwrap();
        assert_eq!(meta.version, 2);
        eprintln!("ckpt dir: {}", dir.display());
    }
}
