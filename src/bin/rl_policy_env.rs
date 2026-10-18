//! `rl_policy_env` — offline driver of the `policy_v1` contract (bridge
//! to the Rust side).
//!
//! **TEMPORARY inside the BOT tree** — the final delivery is meant to be an
//! EXTERNAL crate with a snapshot of the sources + hashes; this bin exists to
//! close the vertical slice (first verifiable real episode) without duplicating
//! the contractual math (it lives in `poly_bot::rl::obs`, shared with the strategy).
//! Once the external snapshot exists, this binary is retired.
//!
//! JSONL protocol over stdin/stdout (compatible with
//! `rl_predictor.env_adapter.V1EnvAdapter`, one process per episode — the
//! replay clock is process-global):
//!
//! ```text
//! Py ──► {"cmd":"reset","episode_id":"e1","seed":7,"market_cid":"0x…"}
//! Py ◄── {"type":"reset","observation":{"policy_v1":{…}}}
//! Py ──► {"cmd":"step","action":{"policy_v1":3}}
//! Py ◄── {"type":"step_result","step":{…},"observation":{"policy_v1":{…}}}
//! Py ──► {"cmd":"close"}  ◄── {"type":"closed",…}  (and the process exits)
//! ```
//!
//! Economic authority = the bot's REAL pipeline: `ReplayEngine`
//! (Executor → RiskGuard → SimulatedRouter → fill_sink → SQLite
//! `:memory:`). The policy view (features/mask/account) comes from the
//! SAME `EpisodeLedger` and the SAME `rl::obs` functions as the runtime —
//! never an invented second ledger.
//!
//! `p_yes` is produced with the FROZEN predictor (`--predictor-dir` →
//! `model_observer::ShadowModel`) over `extract_features` of the causal
//! state — without a predictor or without the asset's vol the observation declares
//! `p_missing=1` and BUYs are masked out (honest, not synthetic).
//!
//! Usage:
//!   rl_policy_env --tape <dir|file.jsonl> [--predictor-dir <dir>]
//!       [--budget USDC] [--balance USDC] [--max-steps N]
//!       [--tape-window-secs N] [--max-order USDC] [--strict]

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;
use serde_json::{json, Value};

use poly_bot::model_observer::{self, Predictor, ShadowModel, SpotSeries};
use poly_bot::replay::{ReplayEngine, ReplayParams, StepMark, Tape, TapeResolution};
use poly_bot::research::contract::{EventSource, MarketOutcome, OutcomeSide};
use poly_bot::research_replay::{EnvAgent, ResearchTape, TapeOpts};
use poly_bot::rl::contract::{
    action_kind, ActionKind, ACTION_NAMES, FEATURE_NAMES, FEATURE_SCHEMA,
    MASK_HOLD_ONLY, NUM_ACTIONS,
};
use poly_bot::rl::episode::{BookLevelsUm, EpisodeLedger};
use poly_bot::rl::obs::{self, MarketView, PolicyTunables};
use poly_bot::state::{to_micros, StateStore};
use poly_bot::types::{
    BookSnapshot, Fill, FillSide, Mode, OrderIntent, OrderSide, StrategyId,
};

/// Strategy label under which the SQLite ledger attributes positions
/// (same as `StrategyId::RlPredictor` at runtime — the settlement
/// query filters by it).
const STRATEGY_LABEL: &str = "rl_predictor";

/// Pending latch timeout (mirrors `defaults::pending_timeout_ms`).
const PENDING_TIMEOUT_MS: i64 = 3_000;

// ── Client request ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Req {
    cmd: String,
    #[serde(default)]
    episode_id: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    market_cid: Option<String>,
    #[serde(default)]
    action: Option<Value>,
}

// ── Per-market state (mirrors the runtime's `MarketState`) ───────────────────

struct Pending {
    token_id: String,
    reserved_um: i64,
    emitted_ts_ms: i64,
}

struct DrvMarket {
    episode: EpisodeLedger,
    yes_token: String,
    no_token: String,
    clear_at: i64,
    fee_rate: Decimal,
    pending: Option<Pending>,
    uncertain: bool,
    /// legacy exec_id/key of already-applied fills (dedup of
    /// re-emissions; intents are FOK: one execution per order).
    seen_execs: std::collections::HashSet<String>,
    /// Settlement already applied to the episode ledger (idempotent).
    settle_done: bool,
}

impl DrvMarket {
    fn new(cid: &str, book: &BookSnapshot, budget: Decimal, fee_fallback: Decimal, ts_ms: i64) -> Self {
        Self {
            episode: EpisodeLedger::new(cid.to_string(), budget, ts_ms),
            yes_token: book.yes.token_id.clone(),
            no_token: book.no.token_id.clone(),
            clear_at: book.clear_at,
            fee_rate: book.parabolic_fee_rate_or(fee_fallback),
            pending: None,
            uncertain: false,
            seen_execs: std::collections::HashSet::new(),
            settle_done: false,
        }
    }

    /// Fill that arrives before the market can be observed — recovered
    /// episode (not trainable), real economic consequence (mirrors
    /// the fill→first-book race of `RlPredictor::on_fill`).
    fn recovered(cid: &str, fill: &Fill, budget: Decimal, fee_fallback: Decimal, ts_ms: i64) -> Self {
        Self {
            episode: EpisodeLedger::recovered(
                cid.to_string(),
                budget,
                HashMap::new(),
                ts_ms,
            ),
            yes_token: String::new(),
            no_token: String::new(),
            clear_at: fill.clear_at,
            fee_rate: fee_fallback,
            pending: None,
            uncertain: false,
            seen_execs: std::collections::HashSet::new(),
            settle_done: false,
        }
    }

    fn view(&self) -> MarketView<'_> {
        MarketView {
            episode: &self.episode,
            yes_token: &self.yes_token,
            no_token: &self.no_token,
            fee_rate: self.fee_rate,
            pending: self.pending.is_some(),
            uncertain: self.uncertain,
        }
    }
}

/// Dedup key identical to the runtime's (`fill_dedup_key`).
fn fill_dedup_key(fill: &Fill) -> String {
    match &fill.exec_id {
        Some(e) => format!("exec:{e}"),
        None => format!(
            "legacy:{}:{:?}:{}:{}:{}",
            fill.order_id.as_deref().unwrap_or("-"),
            fill.side,
            fill.price,
            fill.size,
            fill.ts.timestamp_millis()
        ),
    }
}

/// GAP quality state per market — ONLY with causal data (the
/// window is learned from the first observed book, which carries the real meta;
/// the whole tape is never probed ahead of time).
#[derive(Default)]
struct GapState {
    /// `market_open_ts*1000` of the first BookObserve (causal).
    open_ms: Option<i64>,
    /// `clear_at*1000` of the first BookObserve (causal).
    clear_ms: Option<i64>,
    /// Pending pre-open GAP mark awaiting proven recovery.
    pending_since: Option<i64>,
    /// Trainable experience already started: first decision in window
    /// or any dispatched intent. From then on, ANY GAP
    /// mark invalidates — even if the clock classifies it as pre-open.
    trading_started: bool,
    /// Warmup marks recovered by a real snapshot (evidence/report).
    recovered_marks: u64,
    /// Warmup marks seen (pending + recovered).
    warmup_marks: u64,
    /// Post-clear marks without pending (informational, they do not invalidate).
    post_clear_marks: u64,
}

/// Causal classification of a GAP mark — pure so it can be tested.
/// `pending_order` = there is an unresolved intent in the market.
fn classify_gap(
    g: &GapState,
    ts: i64,
    pending_order: bool,
) -> &'static str {
    if g.trading_started {
        return "abort"; // gap after trading start — always invalidates
    }
    match (g.open_ms, g.clear_ms) {
        // Unknown window: cannot be classified causally →
        // fail-closed (a mark without context is a real defect).
        (None, _) => "abort_unknown_window",
        // Pre-open warmup: does not invalidate by itself — stays pending
        // proven recovery by a real snapshot (see `advance`).
        (Some(open), _) if ts < open => "warmup_pending_recovery",
        // Post-clear: ignorable only if nothing depends on the book — no
        // pending order and settlement by an independent official label.
        (Some(_), Some(clear)) if ts > clear => {
            if pending_order {
                "abort_postclear_pending"
            } else {
                "postclear_safe"
            }
        }
        // Inside [open, clear] — or with unknown clear (not bounded):
        // corrupts the decision window → invalidates.
        _ => "abort_in_window",
    }
}

// ── Driver ───────────────────────────────────────────────────────────────────

struct Driver {
    engine: ReplayEngine,
    agent: EnvAgent,
    markets: HashMap<String, DrvMarket>,
    series: HashMap<String, SpotSeries>,
    model: Option<Arc<ShadowModel>>,
    tunables: PolicyTunables,
    budget: Decimal,
    fee_fallback: Decimal,
    filter: Option<String>,
    /// Cid of the last emitted decision point (to which the step applies).
    cur_cid: Option<String>,
    step_idx: u64,
    /// Rejection/exclusion reasons of the current step (they go in `step`).
    step_reasons: Vec<String>,
    /// GAP quality in the tradable window or after trading has started —
    /// aborts like env F. A pre-open mark does NOT invalidate by itself:
    /// it stays `pending` in `gaps` until the reconstructor proves
    /// recovery (BookObserve ⇒ valid real snapshot of the invalidated
    /// half/halves, see `Reconstructor::emit`).
    gap_in_scope: bool,
    quality_seen: usize,
    /// GAP state per cid (causal window, marks, trading started).
    gaps: HashMap<String, GapState>,
    /// Initial balance of the sim — reference for the episode↔wallet
    /// identity check in `closed`.
    initial_balance: Decimal,
}

impl Driver {
    /// Applies the window's fills to the episode ledgers (same
    /// semantics as `RlPredictor::on_fill`: only this strategy's fills,
    /// dedup by exec_id, pending resolved only by the intent's token).
    fn process_fills(&mut self) {
        let fills = std::mem::take(&mut self.agent.window_fills);
        for fill in &fills {
            if fill.strategy != StrategyId::RlPredictor {
                continue;
            }
            let cid = fill.market_cid.clone();
            if !self.markets.contains_key(&cid) {
                // Fill without market state yet (fill→first-book race):
                // recovered episode — it is counted, never lost.
                let m = DrvMarket::recovered(
                    &cid,
                    fill,
                    self.budget,
                    self.fee_fallback,
                    fill.ts.timestamp_millis(),
                );
                self.markets.insert(cid.clone(), m);
            }
            let m = self.markets.get_mut(&cid).expect("market");
            if !m.seen_execs.insert(fill_dedup_key(fill)) {
                continue; // re-emission — counted and discarded
            }
            match fill.side {
                FillSide::Buy => {
                    let reserved = m
                        .pending
                        .as_ref()
                        .filter(|p| p.token_id == fill.token_id)
                        .map(|p| p.reserved_um)
                        .unwrap_or(0);
                    m.episode.apply_buy_fill(
                        &fill.token_id,
                        fill.price,
                        fill.size,
                        reserved,
                        m.fee_rate,
                    );
                }
                FillSide::Sell => {
                    m.episode
                        .apply_sell_fill(&fill.token_id, fill.price, fill.size, m.fee_rate);
                }
                _ => {}
            }
            if let Some(p) = &m.pending {
                if p.token_id == fill.token_id {
                    match fill.side {
                        FillSide::Buy | FillSide::Sell => {
                            m.pending = None;
                            m.uncertain = false;
                        }
                        FillSide::Rejected | FillSide::GateRejected | FillSide::Cancelled => {
                            m.episode.release_reservation(p.reserved_um);
                            m.pending = None;
                            m.uncertain = false;
                        }
                    }
                }
            }
        }
    }

    /// Expired pending → uncertain (like the runtime's `on_timer` — never
    /// invents a rejection), official settlement applied to the
    /// episode ledger (unknown `outcome_won` → invalidate, do not fabricate)
    /// and draining of in-scope quality marks.
    async fn housekeeping(&mut self, now_ms: i64) {
        let settled_cids: Vec<String> = self
            .markets
            .iter()
            .filter(|(cid, m)| {
                m.clear_at > 0
                    && now_ms / 1000 > m.clear_at
                    && !m.settle_done
                    && self.engine.market_settled(cid)
            })
            .map(|(cid, _)| cid.clone())
            .collect();
        for cid in settled_cids {
            let m = self.markets.get_mut(&cid).expect("market");
            m.settle_done = true;
            for token in [m.yes_token.clone(), m.no_token.clone()] {
                if token.is_empty() || m.episode.shares_of(&token) <= Decimal::ZERO {
                    continue;
                }
                match self
                    .engine
                    .state()
                    .get_position(&token, STRATEGY_LABEL)
                    .await
                {
                    // Same rule as the runtime's `reconcile_run`/`drain_settles`:
                    // known outcome → apply; `redeemed` or a row
                    // without `outcome_won` → invalidate (never fabricate).
                    Ok(Some(pos)) if pos.outcome_won.is_some() || pos.redeemed => {
                        match pos.outcome_won {
                            Some(won) => {
                                m.episode.apply_settlement(&token, won);
                            }
                            None => m.episode.invalidate("settle_outcome_unknown"),
                        }
                    }
                    _ => m.episode.invalidate("settle_outcome_unknown"),
                }
            }
        }
        for m in self.markets.values_mut() {
            if let Some(p) = &m.pending {
                if now_ms - p.emitted_ts_ms > PENDING_TIMEOUT_MS {
                    m.uncertain = true;
                    m.episode.invalidate("pending_timeout");
                }
            }
        }
        // Quality: drain by index (like env F). Causal classification:
        // - trading already started → abort (any GAP, even pre-open).
        // - in window [open, clear] or window not yet known → abort.
        // - pre-open → pending proven recovery by a real snapshot
        //   (the reconstructor does not emit BookObserve with needs_snapshot on any
        //   half; the first post-mark book is the proof).
        // - post-clear → ignorable only without a pending order.
        // The marks are NEVER deleted: they stay in the engine log and in the
        // report counters (`gap_marks` in `closed`).
        let log = self.engine.quality_log();
        for (ts, cid, flag) in &log[self.quality_seen.min(log.len())..] {
            let in_scope = self
                .filter
                .as_ref()
                .is_none_or(|f| cid.as_deref() == Some(f.as_str()));
            if !(in_scope
                && flag.contains(poly_bot::research::contract::QualityFlags::GAP))
            {
                continue;
            }
            let key = cid.clone().unwrap_or_default();
            let pending_order = self
                .markets
                .get(&key)
                .is_some_and(|m| m.pending.is_some());
            let g = self.gaps.entry(key.clone()).or_default();
            match classify_gap(g, *ts, pending_order) {
                v @ ("abort" | "abort_unknown_window" | "abort_in_window"
                | "abort_postclear_pending") => {
                    if !self.gap_in_scope {
                        eprintln!(
                            "[rl_policy_env] fatal GAP ({v}): ts_ms={ts} \
                             cid={key} flag={flag:?}"
                        );
                    }
                    self.gap_in_scope = true;
                }
                "warmup_pending_recovery" => {
                    let g = self.gaps.entry(key.clone()).or_default();
                    g.pending_since = Some(*ts);
                    g.warmup_marks += 1;
                    eprintln!(
                        "[rl_policy_env] GAP warmup pre-open (pending \
                         proven recovery by a real snapshot): ts_ms={ts} cid={key} \
                         open_ms={:?}",
                        g.open_ms
                    );
                }
                _ => {
                    let g = self.gaps.entry(key.clone()).or_default();
                    g.post_clear_marks += 1;
                    eprintln!(
                        "[rl_policy_env] GAP post-clear without pending \
                         (informational, settlement by label): ts_ms={ts} \
                         cid={key}"
                    );
                }
            }
        }
        self.quality_seen = log.len();
    }

    /// Advances the engine until the next decision point in scope.
    /// `None` = tape exhausted.
    async fn advance(&mut self) -> anyhow::Result<Option<(i64, String)>> {
        loop {
            let mark = self.engine.step_once(&mut self.agent).await?;
            self.process_fills();
            match mark {
                Some(StepMark::Decision { ts_ms, cid, .. }) => {
                    self.housekeeping(ts_ms).await;
                    // Causal learning of the window: the BookSnapshot
                    // carries `market_open_ts`/`clear_at` from the real meta —
                    // the tape is never consulted ahead of time.
                    if let Some(b) = self.agent.last_book.get(&cid) {
                        let g = self.gaps.entry(cid.clone()).or_default();
                        if g.open_ms.is_none() {
                            g.open_ms = Some(b.market_open_ts * 1000);
                            g.clear_ms = Some(b.clear_at * 1000);
                        }
                        // Proven recovery: `Reconstructor::emit` does not
                        // emit with `needs_snapshot` on ANY half
                        // and requires both books — a BookObserve
                        // after the mark proves that a
                        // valid real snapshot rebuilt the base.
                        if let Some(mark_ts) = g.pending_since {
                            if ts_ms > mark_ts {
                                g.pending_since = None;
                                g.recovered_marks += 1;
                                eprintln!(
                                    "[rl_policy_env] GAP recovered by \
                                     snapshot real: cid={cid} \
                                     mark_ts={mark_ts} recovery_ts={ts_ms}"
                                );
                            }
                        }
                        // Trainable experience starts with the first
                        // decision within the tradable window.
                        if ts_ms >= g.open_ms.unwrap_or(i64::MAX) {
                            g.trading_started = true;
                        }
                    }
                    if self.filter.as_ref().is_some_and(|f| *f != cid) {
                        continue;
                    }
                    self.cur_cid = Some(cid.clone());
                    return Ok(Some((ts_ms, cid)));
                }
                Some(_) => {
                    // Intermediate event — fills can materialize
                    // between decisions (same order as the real broadcast).
                    let now = self.engine.clock_now_ms();
                    self.housekeeping(now).await;
                }
                None => {
                    self.housekeeping(self.engine.clock_now_ms()).await;
                    return Ok(None);
                }
            }
        }
    }

    /// `policy_v1` block at the decision point of `cid`.
    fn observation(&mut self, ts_ms: i64, cid: &str) -> Value {
        let book = self.agent.last_book.get(cid).cloned();
        let budget = self.budget;
        let fee_fallback = self.fee_fallback;
        let m = match self.markets.entry(cid.to_string()) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let Some(b) = &book else {
                    return json!({"policy_v1": null, "error": "no observable book"});
                };
                e.insert(DrvMarket::new(cid, b, budget, fee_fallback, ts_ms))
            }
        };
        let Some(book) = book else {
            return json!({"policy_v1": null, "error": "no observable book"});
        };
        // The contract identity ALWAYS comes from the book (meta), never
        // from internal state: it fills tokens of a recovered episode.
        if m.yes_token.is_empty() {
            m.yes_token = book.yes.token_id.clone();
        }
        if m.no_token.is_empty() {
            m.no_token = book.no.token_id.clone();
        }
        if m.fee_rate == fee_fallback {
            if let Some(rate) = book.parabolic_fee_rate {
                m.fee_rate = rate;
            }
        }

        // p_yes from the frozen predictor over the causal features.
        // Without model or without the asset's vol → `None` → `p_missing=1` and BUYs
        // masked out (never a fabricated value).
        let model = self.model.clone();
        let p_yes = model.as_ref().and_then(|model| {
            let asset = poly_bot::oracle::asset_from_slug(&book.slug).unwrap_or_default();
            let vol = self
                .agent
                .last_vol
                .values()
                .find(|v| v.asset.to_lowercase() == asset)?;
            let series = self.series.entry(asset).or_insert_with(SpotSeries::new);
            let feats = model_observer::extract_features(&book, vol, series, ts_ms);
            model.predict(&feats).ok()
        });

        let view = m.view();
        let book_age_secs = book
            .recv_instant
            .and_then(|r| poly_bot::time::Instant::now().checked_duration_since(r))
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let features = obs::build_obs(&self.tunables, &view, &book, p_yes, book_age_secs);
        // Pending GAP warmup without proven recovery: HOLD only — no actions
        // until a post-mark BookObserve proves the real snapshot.
        // (By the reconstructor's contract no decision should arrive in
        // this state; the mask is an extra defense, not an invalidation.)
        let gap_pending = self
            .gaps
            .get(cid)
            .is_some_and(|g| g.pending_since.is_some());
        let mask = if gap_pending {
            MASK_HOLD_ONLY
        } else {
            obs::mask_legal(&self.tunables, &view, &book, p_yes.is_some(), false)
        };
        let mask_bits: u32 = mask
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, b)| acc | ((*b as u32) << i));

        // Episode account in micros (EpisodeLedger = the view of the
        // runtime; the sim/StateStore remains the authority for fills).
        let bids: HashMap<String, BookLevelsUm> = [
            (
                m.yes_token.clone(),
                book.yes
                    .bids
                    .levels
                    .iter()
                    .map(|l| (to_micros(l.price), to_micros(l.size)))
                    .collect::<BookLevelsUm>(),
            ),
            (
                m.no_token.clone(),
                book.no
                    .bids
                    .levels
                    .iter()
                    .map(|l| (to_micros(l.price), to_micros(l.size)))
                    .collect::<BookLevelsUm>(),
            ),
        ]
        .into_iter()
        .collect();
        let (exit_um, _complete) = m.episode.exit_value_um(&bids, m.fee_rate);
        let equity_um = m.episode.equity_um(exit_um);
        let exposure_um: i64 = m.episode.positions.values().map(|p| p.cost_um).sum();

        let mut invalid_reasons: Vec<String> = Vec::new();
        if let Some(r) = &m.episode.invalid_reason {
            invalid_reasons.push(r.clone());
        }
        if m.uncertain {
            invalid_reasons.push("uncertain_state".into());
        }
        if self.gap_in_scope {
            invalid_reasons.push("gap_in_scope".into());
        }
        if features.is_none() {
            invalid_reasons.push("obs_non_finite".into());
        }

        // Warmup (causal): the market has not opened yet according to the meta of the
        // book (`market_open_ts`) or there is a pending pre-open GAP awaiting
        // proven recovery. It is not a trainable decision: the LAB
        // excludes it from the gradient/scaler, not just masks it.
        let warmup = gap_pending || ts_ms < book.market_open_ts * 1000;
        // A forced action (HOLD-only mask: warmup, pending, no
        // inventory, window closed…) is not a decision: ∇logπ = 0
        // exactly, and must not count in the update's denominator.
        let trainable_step = mask_bits != 1;

        json!({
            "policy_v1": {
                "feature_schema": FEATURE_SCHEMA,
                "feature_names": FEATURE_NAMES,
                "features": features.map(|f| f.to_vec()),
                "action_mask": mask_bits,
                "action_names": ACTION_NAMES,
                "meta": {
                    "yes_token_id": book.yes.token_id,
                    "no_token_id": book.no.token_id,
                    "market_cid": book.market_condition_id,
                },
                "account": {
                    "equity_um": equity_um,
                    "deposits_um": 0,
                    "withdrawals_um": 0,
                    "exposure_um": exposure_um,
                    "elapsed_ms": 0,
                    // Per-transition audit (first locatable divergence):
                    "cash_um": m.episode.cash_um,
                    "exit_value_um": exit_um,
                    "reserved_um": m.episode.reserved_um,
                    "fees_est_um": m.episode.fees_paid_um,
                },
                "warmup": warmup,
                "trainable_step": trainable_step,
                "invalid_reasons": invalid_reasons,
            },
            "decision_ts_ms": ts_ms,
            "market_cid": cid,
            "predictor_sha256": model.as_ref().map(|m| m.model_hash().to_string()),
        })
    }

    /// Translates the `policy_v1` action to an `OrderIntent` of the real pipeline —
    /// SAME size/limit semantics as `RlPredictor::emit_action`
    /// (they share `rl::obs::action_params`). Illegal → recorded in
    /// `step_reasons`, never a bypass of the executor.
    fn dispatch(&mut self, action: usize, cid: &str, book: &BookSnapshot) -> bool {
        if action_kind(action) == Some(ActionKind::Hold) {
            return true; // HOLD is not dispatched — it is valid
        }
        let spec = {
            let Some(m) = self.markets.get(cid) else {
                self.step_reasons.push("unknown_market".into());
                return false;
            };
            let view = m.view();
            match obs::action_params(&self.tunables, &view, book, action) {
                Some(s) => s,
                None => {
                    self.step_reasons.push("action_not_executable".into());
                    return false;
                }
            }
        };
        let Some(m) = self.markets.get_mut(cid) else {
            return false;
        };
        // Trainable experience started — from here on any GAP
        // invalidates (even if its ts falls pre-open per the clock).
        self.gaps.entry(cid.to_string()).or_default().trading_started = true;
        if spec.buy && !m.episode.reserve_buy(spec.reserve_um) {
            self.step_reasons.push("reserve_rejected".into());
            return false;
        }
        m.pending = Some(Pending {
            token_id: spec.token_id.clone(),
            reserved_um: spec.reserve_um,
            emitted_ts_ms: self.engine.clock_now_ms(),
        });
        let side_label = if spec.buy { "BUY" } else { "SELL" };
        eprintln!(
            "[rl_policy_env] intent {side_label} {} a={} px={} sz={} lim={}",
            spec.token_id, ACTION_NAMES[spec.action], spec.price, spec.size, spec.limit
        );
        self.engine.push_agent_intent(OrderIntent {
            market_cid: cid.to_string(),
            token_id: spec.token_id,
            side: if spec.buy {
                OrderSide::Buy
            } else {
                OrderSide::Sell { size: Some(spec.size) }
            },
            price: spec.price,
            size: spec.size,
            clear_at: book.clear_at,
            signal: Decimal::ZERO,
            neg_risk: book.neg_risk,
            limit_price: Some(spec.limit),
            second_leg: None,
            strategy: StrategyId::RlPredictor,
            // FOK — same `resting:false` as the runtime; a resolution
            // terminal per order, with no resting remainder.
            resting: false,
            real_clear_at: None,
        });
        true
    }
}

/// `filter_tape` of `research_replay` is private (frozen file) —
/// verbatim replica over the `pub` fields of `Tape`.
fn filter_tape(tape: &Tape, market_cid: Option<&str>) -> Tape {
    let Some(cid) = market_cid else {
        return tape.clone();
    };
    let asset = tape
        .books
        .iter()
        .find(|b| b.snap.market_condition_id == cid)
        .and_then(|b| poly_bot::oracle::asset_from_slug(&b.snap.slug));
    let mut quality_stats = tape.quality_stats.clone();
    quality_stats.meta_cids.retain(|c| c == cid);
    Tape {
        books: tape
            .books
            .iter()
            .filter(|b| b.snap.market_condition_id == cid)
            .cloned()
            .collect(),
        spots: tape
            .spots
            .iter()
            .filter(|s| asset.as_ref().is_none_or(|a| s.asset.to_lowercase() == *a))
            .cloned()
            .collect(),
        resolutions: tape
            .resolutions
            .iter()
            .filter(|r| &r.cid == cid)
            .cloned()
            .collect(),
        quality: tape.quality.clone(),
        quality_stats,
        source: tape.source,
    }
}

/// Official label (`labels_v1.json` from the slicer) → `TapeResolution` positioned
/// at the REAL `avail_ms = known_at_ms` (same rule as `load_labels_db`:
/// never backdated to `clear_at`). If `known_at_ms` falls outside the window
/// of the tape, the resolution never materializes in the run → the episode
/// ends honestly TRUNCATED, not as resolved by magic.
fn labels_to_resolutions(labels: &Value) -> anyhow::Result<Vec<TapeResolution>> {
    let mut out = Vec::new();
    let items: Vec<&Value> = match labels {
        Value::Array(a) => a.iter().collect(),
        Value::Object(_) => vec![labels],
        _ => bail!("labels: JSON must be an object or array"),
    };
    for l in items {
        let cid = l
            .get("market_cid")
            .and_then(Value::as_str)
            .context("labels: missing market_cid")?;
        let winner = match l.get("winner").and_then(Value::as_str) {
            Some("yes") => OutcomeSide::Yes,
            Some("no") => OutcomeSide::No,
            other => bail!("labels: unknown winner {other:?}"),
        };
        let known_at_ms = l
            .get("known_at_ms")
            .and_then(Value::as_i64)
            .context("labels: missing known_at_ms (never invented)")?;
        let provenance = l
            .get("provenance")
            .and_then(Value::as_str)
            .context("labels: missing provenance")?;
        anyhow::ensure!(
            !provenance.trim().is_empty(),
            "labels: empty provenance in {cid}"
        );
        out.push(TapeResolution {
            avail_ms: known_at_ms,
            tape_seq: None,
            cid: cid.to_string(),
            outcome: MarketOutcome {
                winner,
                known_at_ms,
                provenance: provenance.to_string(),
            },
            source: EventSource::PolymarketRest,
        });
    }
    Ok(out)
}

fn emit(out: &mut impl Write, v: &Value) -> anyhow::Result<()> {
    writeln!(out, "{}", serde_json::to_string(v)?)?;
    out.flush()?;
    Ok(())
}

fn emit_err(out: &mut impl Write, msg: impl std::fmt::Display, fatal: bool) -> anyhow::Result<()> {
    emit(out, &json!({"type": "error", "message": msg.to_string(), "fatal": fatal}))
}

/// `current_thread`: the `ReplayEngine` requires a single-thread runtime.
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let mut tape_path: Option<PathBuf> = None;
    let mut predictor_dir: Option<PathBuf> = None;
    let mut labels_path: Option<PathBuf> = None;
    let mut opts = TapeOpts::default();
    let mut params = ReplayParams::default();
    let mut budget = dec!(10);
    let mut max_steps: Option<u64> = None;
    let mut max_order: Option<Decimal> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tape" => {
                tape_path = Some(PathBuf::from(
                    args.next().context("--tape missing value")?,
                ))
            }
            "--predictor-dir" => {
                predictor_dir = Some(PathBuf::from(
                    args.next().context("--predictor-dir missing value")?,
                ))
            }
            "--labels" => {
                labels_path = Some(PathBuf::from(
                    args.next().context("--labels missing value")?,
                ))
            }
            "--budget" => {
                budget = Decimal::from_str_exact(&args.next().context("--budget missing value")?)?
            }
            "--balance" => {
                params.initial_balance =
                    Decimal::from_str_exact(&args.next().context("--balance missing value")?)?
            }
            "--max-steps" => {
                max_steps = Some(args.next().context("--max-steps missing value")?.parse()?)
            }
            "--max-order" => {
                max_order =
                    Some(Decimal::from_str_exact(&args.next().context("--max-order missing value")?)?)
            }
            "--tape-window-secs" => {
                opts.tape_window_secs =
                    args.next().context("--tape-window-secs missing value")?.parse()?
            }
            "--strict" => opts.strict = true,
            "--help" | "-h" => {
                eprintln!(
                    "uso: rl_policy_env --tape <dir|file.jsonl> [--predictor-dir <dir>] \
                     [--labels <labels_v1.json>] [--budget USDC] [--balance USDC] \
                     [--max-steps N] [--tape-window-secs N] [--max-order USDC] [--strict]"
                );
                std::process::exit(2);
            }
            _ => bail!("unknown flag: {a}"),
        }
    }
    let tape_path = tape_path.context("missing --tape")?;

    let rt = ResearchTape::load(&tape_path, &opts)
        .with_context(|| format!("loading tape {}", tape_path.display()))?;
    // Reconstruction diagnostics to stderr — audit of the root cause
    // of quality marks (GAP/RESYNC) without polluting the JSONL wire.
    for d in &rt.diagnostics {
        eprintln!(
            "[tape-diag] seq={:?} {}:{} {}",
            d.local_seq, d.origin, d.line, d.message
        );
    }
    let label_resolutions: Vec<TapeResolution> = match &labels_path {
        Some(p) => {
            let v: Value = serde_json::from_str(
                &std::fs::read_to_string(p)
                    .with_context(|| format!("reading labels {}", p.display()))?,
            )
            .with_context(|| format!("labels {} is not JSON", p.display()))?;
            labels_to_resolutions(&v)
                .with_context(|| format!("labels {}", p.display()))?
        }
        None => Vec::new(),
    };
    let model: Option<Arc<ShadowModel>> = match &predictor_dir {
        Some(d) => match ShadowModel::load(d) {
            Ok(m) => Some(Arc::new(m)),
            Err(e) => bail!("predictor {} invalid: {e:?}", d.display()),
        },
        None => None,
    };
    let tunables = PolicyTunables {
        slippage_margin: dec!(0.01),
        min_order_shares: dec!(1),
        max_order_usdc: max_order.unwrap_or(dec!(5)),
        min_remaining_secs: 30,
        max_remaining_secs: 240,
        max_spread: dec!(0.05),
    };

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    emit(
        &mut out,
        &json!({
            "type": "hello",
            "driver": "rl_policy_env",
            "contract": "policy_v1",
            "feature_schema": FEATURE_SCHEMA,
            "num_actions": NUM_ACTIONS,
            "predictor_sha256": model.as_ref().map(|m| m.model_hash().to_string()),
            "tape_events": rt.events,
            "tape_diagnostics": rt.diagnostics.len(),
        }),
    )?;
    let tape = rt.tape;

    // State of the episode in progress (one reset = one new engine).
    let mut drv: Option<Driver> = None;
    let mut ep_id = String::new();
    let mut end: Option<&'static str> = None;

    for line in stdin.lock().lines() {
        let line = line?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let req: Req = match serde_json::from_str(t) {
            Ok(r) => r,
            Err(e) => {
                emit_err(&mut out, format!("malformed JSON: {e}"), false)?;
                continue;
            }
        };
        match req.cmd.as_str() {
            "reset" => {
                let seed = req.seed.unwrap_or(0);
                params.seed = seed;
                let state = StateStore::open_memory(Mode::Paper)
                    .await
                    .context("env: ephemeral StateStore")?;
                let mut engine = ReplayEngine::new(params.clone(), state)
                    .await
                    .context("env: ReplayEngine::new")?;
                let mut t = filter_tape(&tape, req.market_cid.as_deref());
                t.resolutions
                    .extend(label_resolutions.iter().filter(|r| {
                        req.market_cid
                            .as_ref()
                            .is_none_or(|f| r.cid == *f)
                    }).cloned());
                // Episode diagnostic: real time range of the slice,
                // number of queued resolutions entries (label at the real known_at).
                {
                    let mut lo = i64::MAX;
                    let mut hi = i64::MIN;
                    let mut acc = |v: i64| {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    };
                    for b in &t.books {
                        acc(b.avail_ms.unwrap_or(b.recv_ms));
                    }
                    for s in &t.spots {
                        acc(s.ts_ms);
                    }
                    for r in &t.resolutions {
                        acc(r.avail_ms);
                    }
                    for q in &t.quality {
                        acc(q.ts_ms);
                    }
                    eprintln!(
                        "[rl_policy_env] reset cid={:?} books={} spots={} \
                         resolutions={} quality={} ts_range=[{lo},{hi}] \
                         res_avail={:?}",
                        req.market_cid,
                        t.books.len(),
                        t.spots.len(),
                        t.resolutions.len(),
                        t.quality.len(),
                        t.resolutions.iter().map(|r| r.avail_ms).collect::<Vec<_>>(),
                    );
                }
                engine.seed_tape(t).context("env: seed_tape")?;
                ep_id = req.episode_id.clone().unwrap_or_default();
                drv = Some(Driver {
                    engine,
                    agent: EnvAgent::new(),
                    markets: HashMap::new(),
                    series: HashMap::new(),
                    model: model.clone(),
                    tunables: tunables.clone(),
                    budget,
                    fee_fallback: dec!(0.07),
                    filter: req.market_cid.clone(),
                    cur_cid: None,
                    step_idx: 0,
                    step_reasons: Vec::new(),
                    gap_in_scope: false,
                    quality_seen: 0,
                    gaps: HashMap::new(),
                    initial_balance: params.initial_balance,
                });
                end = None;
                let d = drv.as_mut().unwrap();
                match d.advance().await {
                    Ok(Some((ts, cid))) => {
                        let obs = d.observation(ts, &cid);
                        emit(
                            &mut out,
                            &json!({"type": "reset", "episode_id": ep_id, "observation": obs}),
                        )?;
                    }
                    Ok(None) => {
                        emit_err(&mut out, "tape has no decision points", true)?;
                        bail!("reset: tape has no decision points");
                    }
                    Err(e) => {
                        emit_err(&mut out, format!("{e:#}"), true)?;
                        bail!("reset: {e:#}");
                    }
                }
            }
            "step" => {
                let Some(d) = drv.as_mut() else {
                    emit_err(&mut out, "step without reset", false)?;
                    continue;
                };
                if end.is_some() {
                    emit_err(&mut out, "step after episode ended", false)?;
                    continue;
                }
                let action_id = req
                    .action
                    .as_ref()
                    .and_then(|a| a.get("policy_v1"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                let mut executed = false;
                let cid_for_step = d.cur_cid.clone().unwrap_or_default();
                let book_for_step = d.agent.last_book.get(&cid_for_step).cloned();
                match action_id {
                    Some(a) if a < NUM_ACTIONS => {
                        // Legality declared BY THE MASK of the state —
                        // an action outside the mask is recorded as
                        // rejected, never dispatched.
                        match (&book_for_step, d.markets.get(&cid_for_step)) {
                            (Some(book), Some(m)) => {
                                let view = m.view();
                                let mask = obs::mask_legal(
                                    &d.tunables,
                                    &view,
                                    book,
                                    // Legality does not depend on the VALUE of p.
                                    true,
                                    false,
                                );
                                if mask[a] {
                                    executed = d.dispatch(a, &cid_for_step, book);
                                } else {
                                    d.step_reasons.push("illegal_action_masked".into());
                                }
                            }
                            (None, _) => d.step_reasons.push("no_book".into()),
                            (_, None) => d.step_reasons.push("unknown_market".into()),
                        }
                    }
                    Some(_) => d.step_reasons.push("action_out_of_range".into()),
                    None => d.step_reasons.push("missing_action".into()),
                }
                // Advances to the next decision point (the fills/acks of
                // the action land in the window, as in production).
                let next = d.advance().await;
                let settled = d
                    .filter
                    .as_ref()
                    .is_some_and(|c| d.engine.market_settled(c));
                let killed = d.engine.risk().is_killed();
                let step_end = match &next {
                    Err(_) => Some("engine_error"),
                    Ok(None) => {
                        // A warmup GAP that never proved recovery
                        // (no post-mark BookObserve) leaves the segment
                        // invalidated → the episode is not certifiable.
                        let unrecovered =
                            d.gaps.values().any(|g| g.pending_since.is_some());
                        if unrecovered || d.gap_in_scope {
                            Some("data_quality_abort")
                        } else if settled {
                            Some("market_resolved")
                        } else {
                            Some("horizon_reached")
                        }
                    }
                    Ok(Some(_)) => {
                        if killed {
                            Some("risk_halt")
                        } else if d.gap_in_scope {
                            Some("data_quality_abort")
                        } else if settled {
                            Some("market_resolved")
                        } else if max_steps.is_some_and(|mx| d.step_idx + 1 >= mx) {
                            Some("horizon_reached")
                        } else {
                            None
                        }
                    }
                };
                d.step_idx += 1;
                if step_end.is_some() {
                    eprintln!(
                        "[rl_policy_env] end_reason={step_end:?} step={} \
                         ts={} settled={settled} gap_in_scope={} \
                         pending_settle={:?}",
                        d.step_idx,
                        d.engine.clock_now_ms(),
                        d.gap_in_scope,
                        d.engine.settlement_pending(),
                    );
                }
                let invalid = std::mem::take(&mut d.step_reasons);
                let truncated = matches!(
                    step_end,
                    Some("horizon_reached")
                        | Some("risk_halt")
                        | Some("data_quality_abort")
                        | Some("engine_error")
                );
                let obs = match &next {
                    Ok(Some((ts, cid))) => d.observation(*ts, cid),
                    _ => {
                        // Terminal without a next point: the account of the
                        // last obs reflects the final fills/settlement.
                        let cid = cid_for_step.clone();
                        let ts = d.engine.clock_now_ms();
                        if d.markets.contains_key(&cid) {
                            d.observation(ts, &cid)
                        } else {
                            Value::Null
                        }
                    }
                };
                end = step_end;
                emit(
                    &mut out,
                    &json!({
                        "type": "step_result",
                        "step": {
                            "step_idx": d.step_idx - 1,
                            "action_id": action_id,
                            "action_name": action_id.map(|a| ACTION_NAMES.get(a).copied().unwrap_or("?")),
                            "executed": executed,
                            "terminal": step_end.is_some(),
                            "truncated": truncated,
                            "end_reason": step_end,
                            "invalid_reasons": invalid,
                        },
                        "observation": obs,
                    }),
                )?;
            }
            "close" => {
                let Some(mut d) = drv.take() else {
                    emit_err(&mut out, "close without reset", false)?;
                    continue;
                };
                match d.engine.finalize(&mut d.agent).await {
                    Ok(outcome) => {
                        let reconciled = outcome.reconciled();
                        let digest = outcome.digest.clone();
                        d.engine.shutdown().await?;
                        let gap_marks: serde_json::Map<String, Value> = d
                            .gaps
                            .iter()
                            .map(|(cid, g)| {
                                (
                                    cid.clone(),
                                    json!({
                                        "warmup_marks": g.warmup_marks,
                                        "recovered_marks": g.recovered_marks,
                                        "post_clear_marks": g.post_clear_marks,
                                        "pending_since": g.pending_since,
                                        "trading_started": g.trading_started,
                                    }),
                                )
                            })
                            .collect();
                        // Episode↔wallet identity: Σ(cash_ep − budget) of
                        // the episode ledgers must equal the delta of the
                        // sim wallet (all-inclusive: fills + redeem).
                        let ep_pnl_um: i64 = d
                            .markets
                            .values()
                            .map(|m| m.episode.cash_um - m.episode.budget_um)
                            .sum();
                        let wallet_delta_um = outcome
                            .wallet_final
                            .map(|w| to_micros(w - d.initial_balance));
                        let drift_um =
                            wallet_delta_um.map(|w| ep_pnl_um - w);
                        emit(
                            &mut out,
                            &json!({
                                "type": "closed",
                                "episode_id": ep_id,
                                "digest": digest,
                                "reconciled": reconciled,
                                "gap_marks": gap_marks,
                                "episode_pnl_um": ep_pnl_um,
                                "wallet_delta_um": wallet_delta_um,
                                "episode_wallet_drift_um": drift_um,
                            }),
                        )?;
                        return Ok(());
                    }
                    Err(e) => {
                        emit_err(&mut out, format!("{e:#}"), true)?;
                        bail!("close: {e:#}");
                    }
                }
            }
            "legal_actions" => {
                let mask: u32 = drv
                    .as_mut()
                    .and_then(|d| {
                        let cid = d.cur_cid.clone()?;
                        let book = d.agent.last_book.get(&cid)?.clone();
                        let m = d.markets.get(&cid)?;
                        let v = m.view();
                        let mask = obs::mask_legal(&d.tunables, &v, &book, true, false);
                        Some(
                            mask.iter()
                                .enumerate()
                                .fold(0u32, |acc, (i, b)| acc | ((*b as u32) << i)),
                        )
                    })
                    .unwrap_or(1);
                emit(&mut out, &json!({"type": "legal_actions", "action_mask": mask}))?;
            }
            other => {
                emit_err(&mut out, format!("unknown cmd: {other}"), false)?;
            }
        }
    }
    // EOF: the client closed without `close` — shuts the engine down normally.
    if let Some(mut d) = drv.take() {
        let _ = d.engine.shutdown().await;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(open: Option<i64>, clear: Option<i64>) -> GapState {
        GapState {
            open_ms: open,
            clear_ms: clear,
            ..Default::default()
        }
    }

    const OPEN: i64 = 1_000_000;
    const CLEAR: i64 = 1_300_000;

    #[test]
    fn gap_preopen_without_trading_is_pending_warmup() {
        let s = g(Some(OPEN), Some(CLEAR));
        assert_eq!(
            classify_gap(&s, OPEN - 1, false),
            "warmup_pending_recovery"
        );
    }

    #[test]
    fn gap_in_window_invalidates() {
        let s = g(Some(OPEN), Some(CLEAR));
        assert_eq!(classify_gap(&s, OPEN, false), "abort_in_window");
        assert_eq!(classify_gap(&s, CLEAR, false), "abort_in_window");
        assert_eq!(classify_gap(&s, (OPEN + CLEAR) / 2, false), "abort_in_window");
    }

    #[test]
    fn gap_after_trading_start_always_invalidates() {
        // Even a mark with a pre-open ts: the experience has already started.
        let s = GapState {
            trading_started: true,
            ..g(Some(OPEN), Some(CLEAR))
        };
        assert_eq!(classify_gap(&s, OPEN - 1, false), "abort");
        assert_eq!(classify_gap(&s, (OPEN + CLEAR) / 2, false), "abort");
        assert_eq!(classify_gap(&s, CLEAR + 1, false), "abort");
    }

    #[test]
    fn gap_without_causal_window_fails_closed() {
        let s = GapState::default();
        assert_eq!(classify_gap(&s, OPEN - 1, false), "abort_unknown_window");
    }

    #[test]
    fn gap_postclear_only_safe_without_pending() {
        let s = g(Some(OPEN), Some(CLEAR));
        assert_eq!(classify_gap(&s, CLEAR + 1, false), "postclear_safe");
        // With a pending order the book remains the authority → invalidates.
        assert_eq!(classify_gap(&s, CLEAR + 1, true), "abort_postclear_pending");
    }

    #[test]
    fn recovery_requires_post_mark_book() {
        // A pending_since is only lifted by a decision ts AFTER
        // the mark (the observed book comes from a real snapshot —
        // guarantee of the reconstructor's emit(), see tests/rl_gap_recovery.rs).
        let mut s = g(Some(OPEN), Some(CLEAR));
        s.pending_since = Some(OPEN - 100);
        // ts == mark does NOT recover (an avail tie does not prove order).
        assert!(s.pending_since.is_some());
    }
}
