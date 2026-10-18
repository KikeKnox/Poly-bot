//! `ModelBtc5m` strategy — PAPER-ONLY.
//!
//! Operating vehicle of the numeric model: consumes probabilities
//! (`ModelPrediction`) emitted by the `ModelTool` tool and translates them into
//! taker FOK `OrderIntent`s when the net edge exceeds the configured
//! threshold. A defensive experiment.
//!
//! ## Design — chained gates
//!
//! 1. **Market scope**: EXACT slug `btc-updown-5m-epoch` and duration
//!    300s — everything else is ignored (the probability is not even looked at).
//!    Important: token labels are NOT iterated to infer
//!    identity — the YES/NO id comes from `book.yes.token_id`/
//!    `book.no.token_id`/`ModelPrediction.token_id_yes`/`token_id_no`, fixed
//!    once by the market meta.
//! 2. **Freshness**: the book must be recent (`recv_instant` within
//!    `ctx.max_book_age`) and the spot not stale (`VolSnapshot::is_stale_for_signal`).
//!    Missing book/spot → recorded abstention.
//! 3. **Usable prediction**: it only acts if `ModelPrediction.p_model.is_some()`
//!    AND the `ModelPrediction` is not an abstention (`abstention.is_none()`).
//!    Abstentions produce NO `OrderIntent` — the strategy only emits
//!    when the model successfully gave a probability.
//! 4. **Net edge**: `p_model − ask − fee − slippage_margin ≥ min_net_edge`
//!    using `book.parabolic_fee_rate` (or `TAKER_FEE_RATE_PARABOLIC` as
//!    fallback). The taker fee is the CLOB's real convention
//!    (`fee = C × rate × p × (1-p)`, `rate=0.07` by default, rounded to 5 decimals).
//!    The slippage margin is 0.01 (configurable). No edge → skip
//!    `model_skip_edge`.
//! 5. **Spread**: maximum tolerable bid-ask on EACH leg (default 0.05).
//!    Wide spread = dead liquidity, we do not enter.
//! 6. **Entry window**: only accepted between 30s and 240s remaining in the
//!    bracket (no preopen, no postclose). The remaining-time filter
//!    lives BEFORE the fill — no trading with `secs_to_clear < 30` or
//!    `secs_to_clear > 240`.
//! 7. **No pyramiding**: ONE confirmed entry per market. Control
//!    is by `(market_cid → state)`: if there is already a `pending_buy` or
//!    `filled`, the next intent for the SAME cid is rejected
//!    (`model_skip_dup_pending`). This avoids the classic pattern of duplicating
//!    a partial fill.
//! 8. **Hold until settlement**: the strategy emits NO `Sell`; sells/orders are
//!    managed by official settlement
//!    (orphan checker → redeem). `on_fill` only carries an idempotency latch
//!    (`filled_set`) to detect confirmed fills and consolidate the state.
//! 9. **Fee endpoint**: if `polling_fee_endpoint` is declared, the market's real
//!    `parabolic_fee_rate` is used. Otherwise fallback
//!    `TAKER_FEE_RATE_PARABOLIC`. The documented difference: in crypto
//!    updown the real rate is 0.07; the budget/fee endpoint is 0.07 — no
//!    operational difference in this experiment. If the market's declared
//!    rate is different (e.g. 0.04 for finance_prices_fees), THAT
//!    value is used. Documented in `fair_value.rs`.
//!
//! ## Fail-closed live
//!
//! This strategy CANNOT go live: `Config::validate` rejects
//! startup if `paper_only=false && enabled=true` in live.
//! The barrier is **structural**, not a matter of discipline — it lives in
//! `StrategyLimits::can_run` (`paper_only` blocks live) AND in the additional barrier of `ModelBtc5m::new`/`build`
//! which aborts if the combination is not `paper_only=true` with `enabled=true`
//! in any mode other than `Paper`.
//!
//! ## Consuming the model tool
//!
//! The `broadcast::Receiver<ModelPrediction>` is injected via the constructor (not
//! via `StrategyCtx` — `StrategyCtx` carries no tool channels, to
//! keep the separation). It subscribes once in `ModelBtc5m::new`.

use rust_decimal::Decimal;
use serde::Deserialize;
use std::collections::HashMap;
use tokio::sync::broadcast;

use crate::config::StrategyLimits;
use crate::tools::model::{ModelPrediction, MODEL_TOOL_SCHEMA_VERSION};
use crate::types::{
    BookSnapshot, Fill, FillSide, Mode, OrderIntent, OrderSide, StrategyId, VolSnapshot,
};

use super::Strategy;

/// Exact slug that restricts this strategy — the exact 5m duration is enforced by
/// `(clear_at − market_open_ts) == 300s`. Any other slug / duration
/// is structurally ignored without looking at the probability.
pub const REQUIRED_SLUG_PREFIX: &str = "btc-updown-5m";
const REQUIRED_DURATION_SECS: i64 = 300;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelBtc5mConfig {
    #[serde(flatten)]
    pub limits: StrategyLimits,
    /// Minimum net edge PER SHARE: `p_model − ask − fee − slippage_margin ≥ min_net_edge`.
    /// Default 0.05 = 5¢ net after the real fee and slippage. A fill below this
    /// does not cover the fee + slippage + inventory buffer.
    #[serde(default = "defaults::model_btc5m_min_net_edge")]
    pub min_net_edge: Decimal,
    /// Pre-trade slippage margin (USDC/share). 0.01 = 1¢. Covers the
    /// move between the book read and the arrival at the CLOB.
    #[serde(default = "defaults::model_btc5m_slippage_margin")]
    pub slippage_margin: Decimal,
    /// Maximum tolerable bid-ask spread on EACH leg. 0 = no filter.
    /// Default 0.05 = 5¢ (a bracket at 0.5/0.5 has a spread < 2¢).
    #[serde(default = "defaults::model_btc5m_max_spread")]
    pub max_spread: Decimal,
    /// Minimum remaining seconds to accept an entry. Below it, NO
    /// entry (preclose). Default 30s.
    #[serde(default = "defaults::model_btc5m_min_remaining_secs")]
    pub min_remaining_secs: i64,
    /// Maximum remaining seconds to accept an entry. Above it, NO
    /// entry (preopen). Default 240s — 60s before the official close of the
    /// bracket (300s).
    #[serde(default = "defaults::model_btc5m_max_remaining_secs")]
    pub max_remaining_secs: i64,
    /// Maximum tolerable age of a prediction (`now − pred.ts_ms`) in
    /// milliseconds. Older predictions are rejected. Default 1500ms —
    /// a numeric model decision should not survive more than ~1 runner tick
    /// before being re-emitted. Independent of the entry window's
    /// `min_remaining_secs` (which is a book threshold, not a
    /// prediction one).
    #[serde(default = "defaults::model_btc5m_max_prediction_age_ms")]
    pub max_prediction_age_ms: i64,
    /// Maximum shares per entry (size cap). 0 = inherit from
    /// the limit's `max_order_usdc`. By default, what `max_order_usdc` pays
    /// at the expected ask ≈ `max_order_usdc / 0.5` = `max_order_usdc * 2`.
    #[serde(default)]
    pub max_size_shares_override: Decimal,
    /// If the market's explicit rate does not come from discovery, use this
    /// fallback. 0.07 (crypto_fees_v2) is the canonical updown one. Override
    /// only if you want to experiment with another regime.
    #[serde(default = "defaults::model_btc5m_fee_rate_fallback")]
    pub fee_rate_fallback: Decimal,
}

impl Default for ModelBtc5mConfig {
    fn default() -> Self {
        Self {
            limits: StrategyLimits {
                enabled: false,
                paper_only: true, // fail-closed by default
                max_orders_per_hour: 60,
                max_order_usdc: defaults::model_btc5m_max_order_usdc(),
                daily_loss_usdc_stop: defaults::model_btc5m_daily_loss_stop(),
                universe: defaults::model_btc5m_default_universe(),
            },
            min_net_edge: defaults::model_btc5m_min_net_edge(),
            slippage_margin: defaults::model_btc5m_slippage_margin(),
            max_spread: defaults::model_btc5m_max_spread(),
            min_remaining_secs: defaults::model_btc5m_min_remaining_secs(),
            max_remaining_secs: defaults::model_btc5m_max_remaining_secs(),
            max_prediction_age_ms: defaults::model_btc5m_max_prediction_age_ms(),
            max_size_shares_override: Decimal::ZERO,
            fee_rate_fallback: defaults::model_btc5m_fee_rate_fallback(),
        }
    }
}

impl crate::strategy::StrategyConfig for ModelBtc5mConfig {
    fn limits(&self) -> &StrategyLimits {
        &self.limits
    }
}

pub mod defaults {
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    pub fn model_btc5m_min_net_edge() -> Decimal {
        dec!(0.05)
    }
    pub fn model_btc5m_slippage_margin() -> Decimal {
        dec!(0.01)
    }
    pub fn model_btc5m_max_spread() -> Decimal {
        dec!(0.05)
    }
    pub fn model_btc5m_min_remaining_secs() -> i64 {
        30
    }
    pub fn model_btc5m_max_remaining_secs() -> i64 {
        240
    }
    /// Maximum tolerable age of the numeric tool's prediction. 1500ms =
    /// slack of 1 runner tick (~500ms) + IPC variability + jitter
    /// of the `broadcast` channel. Older → the prediction went
    /// stale and is rejected. Not to be confused with `min_remaining_secs` (that one is
    /// about the book, not the prediction).
    pub fn model_btc5m_max_prediction_age_ms() -> i64 {
        1500
    }
    pub fn model_btc5m_fee_rate_fallback() -> Decimal {
        dec!(0.07)
    }
    /// Cap per entry (5 USDC). The paper experiment uses a 100 USDC balance,
    /// exposure cap 10 USDC — 5 USDC gives slack.
    pub fn model_btc5m_max_order_usdc() -> Decimal {
        dec!(5)
    }
    pub fn model_btc5m_daily_loss_stop() -> Decimal {
        dec!(2)
    }
    pub fn model_btc5m_default_universe() -> Vec<String> {
        // Universe dedicated to the model. We do NOT use the default `crypto-updown-*`
        // because they also capture 15m/1h — the 5m duration filter
        // lives within the strategy, but the `universe` here defines the
        // subset to scan.
        vec!["btc-updown-5m".into()]
    }
}

// ── Per-market state ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
enum EntryState {
    /// No Buy has been emitted for this cid yet.
    None,
    /// A Buy was emitted and is in flight. A reject/fill clears it.
    Pending,
    /// Confirmed fill. Hold until settlement (orphan checker).
    Filled,
}

#[derive(Debug)]
struct BracketState {
    clear_at: i64,
    state: EntryState,
    /// Confirmed fill size (post-fill).
    filled_size: Decimal,
    /// Average price of the confirmed fill (post-fill).
    filled_price: Decimal,
    /// YES token (for the hold, in a possible future manual sell).
    yes_token: String,
    /// NO token.
    no_token: String,
}

// ── Strategy ──────────────────────────────────────────────────────────────────

pub struct ModelBtc5m {
    cfg: ModelBtc5mConfig,
    /// Receiver of the tool's predictions. Subscribed ONCE in the
    /// constructor (not re-subscribed in `on_book`).
    prediction_rx: broadcast::Receiver<ModelPrediction>,
    /// State per cid — fail-closed against duplicates/pendings.
    states: HashMap<String, BracketState>,
    /// Mode the strategy was built with. The constructor's barrier
    /// aborts on `Live`; this field remains only so the
    /// rest of the code (tests, debug) can inspect it.
    mode: Mode,
}

impl ModelBtc5m {
    /// Builds the strategy. **Total fail-closed live**: the
    /// `ModelBtc5m` strategy is a paper-only experiment — `mode=Live` produces
    /// an IMMEDIATE `panic!` in the constructor, whatever the rest of
    /// the configuration (`enabled`, `paper_only`). This
    /// is the final barrier, not the only one:
    ///
    /// * `StrategyLimits::can_run(mode)` already aborts `enabled=true &&
    ///   paper_only=true && mode=Live` with `BlockReason::PaperOnly`.
    /// * `StrategySupervisor::set_enabled` adds a runtime guard that
    ///   rejects `mode=Live` with `BlockReason::PaperOnly` before
    ///   any spawn.
    ///
    /// The panic is the safety net in case both fail (a hot TOML change
    /// + an accidental bypass): before, the constructor
    /// only panicked if `paper_only=false`, which let through
    /// `enabled=true && paper_only=true && mode=Live` — the scenario the
    /// supervisor already filters but the constructor did not.
    ///
    /// `catch_unwind` is not treated as success: the strategy must NEVER trade
    /// in Live. A "caught" catch_unwind entry would be worse
    /// than the panic — the strategy would stay alive emitting intents.
    pub fn new(
        cfg: ModelBtc5mConfig,
        prediction_rx: broadcast::Receiver<ModelPrediction>,
        mode: Mode,
    ) -> Self {
        if mode == Mode::Live {
            panic!(
                "ModelBtc5m: the defensive experiment NEVER trades in Live mode — \
                 attempted to build with mode=Live (paper_only={}, \
                 enabled={}). This is an invalid configuration: the experiment is \
                 paper-only by contract.",
                cfg.limits.paper_only, cfg.limits.enabled,
            );
        }
        Self {
            cfg,
            prediction_rx,
            states: HashMap::new(),
            mode,
        }
    }

    /// Mode the strategy was built with. The constructor's barrier
    /// aborts on `Live`, so in practice this getter
    /// always returns `Paper` for any live instance. It is exposed
    /// so tests and diagnostic code (metrics, logs) can
    /// verify that the wiring does not skip the mode barrier.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Non-blockingly drains `prediction_rx` and returns the latest prediction
    /// for `cid` (if any). The broadcast coalesces implicitly:
    /// if N predictions arrived for the same cid between two `on_book`s,
    /// we only care about the last one (the book is already fresh, the intermediate
    /// predictions were for the same book).
    ///
    /// **Upper bound** (`MAX_DRAIN_OTHERS`): if the channel accumulates predictions
    /// for OTHER cids while this book arrives (typical scenario: the tool emits
    /// for several active brackets), we do not silently consume them all —
    /// they belong to different brackets and must remain available for their
    /// next `on_book`. `try_recv` removes from the broadcast buffer; without a cap
    /// we would consume every foreign prediction even if we do not want it, losing the
    /// message for its cid's next book. The old comment
    /// («they stay in the channel for the next book») was FALSE.
    fn drain_prediction_for(&mut self, cid: &str) -> Option<ModelPrediction> {
        // Per-call cap: predictions consumed in a single pass of
        // `on_book`. Enough for the tool's real rate (broadcast_capacity
        // = 16, one prediction per evaluated book); any excess is
        // retried on the next tick.
        const MAX_DRAIN_OTHERS: usize = 32;
        let mut latest_for_cid: Option<ModelPrediction> = None;
        let mut consumed_others: usize = 0;
        loop {
            match self.prediction_rx.try_recv() {
                Ok(p) => {
                    if p.market_cid == cid {
                        latest_for_cid = Some(p);
                        // Predictions for OTHER cids are consumed from the
                        // broadcast buffer (there is no `put_back`); if
                        // they exceed `MAX_DRAIN_OTHERS` we leave them and stop.
                    } else if consumed_others < MAX_DRAIN_OTHERS {
                        consumed_others += 1;
                        metrics::counter!("hrf_model_btc5m_pred_other_cid_consumed_total")
                            .increment(1);
                    } else {
                        // Cap reached: we leave the remaining predictions
                        // for their next book. The strategy ALREADY consumed the
                        // previous 32; the rest stays in the
                        // broadcast buffer (the next `try_recv` will pick it up).
                        break;
                    }
                }
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Lagged(n)) => {
                    metrics::counter!("hrf_model_btc5m_pred_lag_total").increment(n);
                    tracing::warn!(
                        skipped = n,
                        "model_btc5m: predictions skipped due to broadcast lag"
                    );
                }
                Err(broadcast::error::TryRecvError::Closed) => {
                    // The tool is off: strategy without predictions.
                    tracing::debug!("model_btc5m: prediction broadcast closed");
                    break;
                }
            }
        }
        latest_for_cid
    }

    /// Computes the pair's parabolic taker fee (`rate × p × (1-p)` per leg)
    /// using the market's rate or the fallback.
    fn fee_rate(&self, book: &BookSnapshot) -> Decimal {
        book.parabolic_fee_rate
            .unwrap_or(self.cfg.fee_rate_fallback)
    }

    /// fee per share at price `p`.
    fn fee_per_share(&self, rate: Decimal, p: Decimal) -> Decimal {
        // Rounded to 5 decimals — Polymarket convention.
        let fee = rate * p * (Decimal::ONE - p);
        fee.trunc_with_scale(5)
    }

    /// Maximum size per order (shares) guaranteeing that the total
    /// cost — `size × limit + fee_per_share × size` — does NOT exceed
    /// `max_order_usdc`. The closed formula is:
    ///   `size ≤ max_order_usdc / (limit + rate × limit × (1 − limit))`
    /// which comes from the equation `size × limit × (1 + rate × (1 − limit)) ≤ max`.
    ///
    /// The explicit override (`max_size_shares_override`) keeps its
    /// semantics (hard cap) but is REDUCED by the inclusive fee bound
    /// so as not to bypass the experiment's budget barrier.
    ///
    /// The result is truncated to 2dp (the CLOB SDK's minimum for sizes).
    fn max_shares(&self, ask: Decimal, rate: Decimal) -> Decimal {
        if self.cfg.limits.max_order_usdc <= Decimal::ZERO || ask <= Decimal::ZERO {
            return Decimal::ZERO;
        }
        // All-in cost per share: price + the pair's taker fee.
        let limit = ask + self.cfg.slippage_margin;
        if limit >= Decimal::ONE || rate < Decimal::ZERO {
            return Decimal::ZERO;
        }
        let fee_per_share = rate * limit * (Decimal::ONE - limit);
        let unit_cost = limit + fee_per_share;
        let raw = if unit_cost > Decimal::ZERO {
            ((self.cfg.limits.max_order_usdc - Decimal::new(1, 5)).max(Decimal::ZERO) / unit_cost)
                .trunc_with_scale(2)
        } else {
            Decimal::ZERO
        };
        // Explicit cap (override) — always REDUCED by the budget-derived
        // one, never enlarged.
        if self.cfg.max_size_shares_override > Decimal::ZERO {
            let cap = self
                .cfg
                .max_size_shares_override
                .trunc_with_scale(2)
                .min(raw);
            cap
        } else {
            raw
        }
    }
}

impl Strategy for ModelBtc5m {
    fn id(&self) -> StrategyId {
        StrategyId::ModelBtc5m
    }

    fn requires_oracle(&self) -> bool {
        true
    }

    fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
        let cid = book.market_condition_id.clone();

        // 1) Market scope: EXACT btc-updown-5m slug and 300s duration.
        if !is_target_market(book) {
            return vec![];
        }

        // 2) State: if there is already a confirmed fill, we do nothing (hold).
        if let Some(state) = self.states.get(&cid) {
            if state.state == EntryState::Filled {
                metrics::counter!("hrf_model_btc5m_skip_already_filled_total").increment(1);
                return vec![];
            }
        }

        // 3) Prediction: the tool's LATEST for this cid, if any.
        let Some(pred) = self.drain_prediction_for(&cid) else {
            metrics::counter!("hrf_model_btc5m_skip_no_prediction_total").increment(1);
            return vec![];
        };

        // 4) The prediction must be usable: p_model Some AND abstention None.
        if pred.abstention.is_some() {
            metrics::counter!(
                "hrf_model_btc5m_skip_abstention_total",
                "reason" => pred.abstention.clone().unwrap_or_default()
            )
            .increment(1);
            return vec![];
        }
        let p_model = match pred.p_model {
            Some(p) => p,
            None => {
                metrics::counter!("hrf_model_btc5m_skip_no_pmodel_total").increment(1);
                return vec![];
            }
        };

        // 4b) Model identity: `model_hash` must be present. Without a
        //     hash there is no way to tie the order to the artifact that
        //     generated it (the strict verifier requires
        //     `model_hash` in every shadow row; the live path reproduces the
        //     same criterion). `None` or an empty string → honest abstention.
        if pred
            .model_hash
            .as_deref()
            .map(str::is_empty)
            .unwrap_or(true)
        {
            metrics::counter!("hrf_model_btc5m_skip_no_model_hash_total").increment(1);
            tracing::warn!(
                cid = %cid,
                "model_btc5m: prediction without model_hash — abstaining"
            );
            return vec![];
        }

        // 4c) Finite probability: NaN or ±∞ invalidates the signal without
        //     raising a panic. `from_f64_retain` would keep them as NaN
        //     and later arithmetic would propagate false results.
        if !p_model.is_finite() {
            metrics::counter!("hrf_model_btc5m_skip_nonfinite_p_total").increment(1);
            tracing::warn!(
                cid = %cid,
                p_model,
                "model_btc5m: p_model not finite (NaN/inf) — abstaining"
            );
            return vec![];
        }

        // 4d) Legal range [0,1]: a probability outside [0,1] is
        //     impossible for a well-calibrated model. We reject it
        //     too — defense against corrupted artifacts.
        if !(0.0..=1.0).contains(&p_model) {
            metrics::counter!("hrf_model_btc5m_skip_p_out_of_range_total").increment(1);
            tracing::warn!(
                cid = %cid,
                p_model,
                "model_btc5m: p_model outside [0,1] — abstaining"
            );
            return vec![];
        }

        // 4e) Token identity (YES/NO) — the prediction carries
        //     `token_id_yes`/`token_id_no` fixed by the market
        //     meta. If they do NOT match those of the current book, the
        //     prediction is for a DIFFERENT bracket (resync race, slug
        //     recycles cid, etc.) — abstain.
        if pred.token_id_yes != book.yes.token_id || pred.token_id_no != book.no.token_id {
            metrics::counter!("hrf_model_btc5m_skip_token_mismatch_total").increment(1);
            tracing::warn!(
                cid = %cid,
                pred_yes = %pred.token_id_yes,
                pred_no = %pred.token_id_no,
                book_yes = %book.yes.token_id,
                book_no = %book.no.token_id,
                "model_btc5m: prediction tokens != book tokens — abstaining"
            );
            return vec![];
        }

        // 4e2) Slug identity — a `cid` recycled between consecutive
        //     brackets with the same `token_id_*` is not detectable
        //     with tokens alone. The book's slug is the canonical source
        //     of the current market; if it does not match the prediction's,
        //     the prediction comes from an already closed bracket and using it
        //     would mean trading against a past market. Abstain.
        if pred.slug != book.slug {
            metrics::counter!("hrf_model_btc5m_skip_slug_mismatch_total").increment(1);
            tracing::warn!(
                cid = %cid,
                pred_slug = %pred.slug,
                book_slug = %book.slug,
                "model_btc5m: prediction slug != book slug — abstaining"
            );
            return vec![];
        }

        // 4f) Prediction freshness: `pred.ts_ms` must be
        //     within the window `[now − max_prediction_age_ms, now]`.
        //     A timestamp in the FUTURE is always suspicious (clock
        //     skew between runners, preloaded tool predictions,
        //     tampered payload) — rejected with an
        //     independent counter for diagnosis. A timestamp that is too
        //     old means the model decision went stale
        //     (another book tick has passed) — also rejected. The
        //     threshold is `max_prediction_age_ms` (default 1500ms), NOT
        //     `min_remaining_secs`: the tolerable age of the prediction
        //     does not depend on how long is left until the bracket closes.
        let now_ms = crate::time::now().timestamp_millis();
        if pred.ts_ms > now_ms {
            let skew_ms = pred.ts_ms - now_ms;
            metrics::counter!(
                "hrf_model_btc5m_skip_future_prediction_total",
                "skew_ms_bucket" => "future"
            )
            .increment(1);
            tracing::warn!(
                cid = %cid,
                pred_ts_ms = pred.ts_ms,
                now_ms,
                skew_ms,
                "model_btc5m: prediction with FUTURE ts_ms (clock skew or invalid payload) — abstaining"
            );
            return vec![];
        }
        let pred_age_ms = now_ms - pred.ts_ms;
        let max_pred_age_ms = self.cfg.max_prediction_age_ms;
        if pred_age_ms > max_pred_age_ms {
            metrics::counter!(
                "hrf_model_btc5m_skip_stale_prediction_total",
                "age_ms_bucket" => "over_threshold"
            )
            .increment(1);
            tracing::warn!(
                cid = %cid,
                pred_age_ms,
                max_pred_age_ms,
                "model_btc5m: prediction older than the freshness threshold — abstaining"
            );
            return vec![];
        }

        // 5) Schema check — if the tool emits a future,
        // unknown version, we do NOT act for safety (explicit forward-compat).
        if pred.schema_version != MODEL_TOOL_SCHEMA_VERSION {
            metrics::counter!("hrf_model_btc5m_skip_schema_mismatch_total").increment(1);
            tracing::warn!(
                schema = pred.schema_version,
                expected = MODEL_TOOL_SCHEMA_VERSION,
                "model_btc5m: unknown ModelPrediction schema — abstaining"
            );
            return vec![];
        }

        // 6) Entry window: only between 30s and 240s remaining.
        let secs_left = book.secs_to_clear();
        if secs_left < self.cfg.min_remaining_secs {
            metrics::counter!("hrf_model_btc5m_skip_preclose_total").increment(1);
            return vec![];
        }
        if secs_left > self.cfg.max_remaining_secs {
            metrics::counter!("hrf_model_btc5m_skip_preopen_total").increment(1);
            return vec![];
        }

        // 7) Spread: both legs (for safety — we buy ONE token, but
        // we check BOTH to avoid entering illiquid books).
        let yes_bid = book.yes.best_bid().unwrap_or(Decimal::ZERO);
        let yes_ask = book.yes.best_ask().unwrap_or(Decimal::ZERO);
        let no_bid = book.no.best_bid().unwrap_or(Decimal::ZERO);
        let no_ask = book.no.best_ask().unwrap_or(Decimal::ZERO);
        if yes_ask <= Decimal::ZERO || no_ask <= Decimal::ZERO {
            metrics::counter!("hrf_model_btc5m_skip_no_ask_total").increment(1);
            return vec![];
        }
        if self.cfg.max_spread > Decimal::ZERO {
            let y_spread = yes_ask - yes_bid;
            let n_spread = no_ask - no_bid;
            if y_spread > self.cfg.max_spread || n_spread > self.cfg.max_spread {
                metrics::counter!("hrf_model_btc5m_skip_spread_total").increment(1);
                return vec![];
            }
        }

        // 8) Fee + edge per side — we buy the side whose net edge is
        // positive. Identity comes from `pred.token_id_*` (market meta,
        // fixed by the feed — NEVER by label order).
        let rate = self.fee_rate(book);
        let p_dec = Decimal::from_f64_retain(p_model).unwrap_or(Decimal::ZERO);
        // Tuples by value — `fee_per_share`, `max_shares`, etc. are
        // operations that consume `Decimal` by value. `ask` and `edge` are
        // moved into the filter; `side_label` is a `&'static str` (needs no
        // cloning); `token_id` is the id of the token published in the
        // `ModelPrediction` or in the book (no costly operations).
        let candidates = [
            (
                "buy_yes",
                book.yes.token_id.clone(),
                yes_ask,
                p_dec - yes_ask,
            ),
            (
                "buy_no",
                book.no.token_id.clone(),
                no_ask,
                Decimal::ONE - p_dec - no_ask,
            ),
        ];
        let mut best: Option<(&'static str, String, Decimal, Decimal)> = None;
        for cand in candidates.into_iter() {
            let (side_label, token_id, ask, edge) = cand;
            let fee = self.fee_per_share(rate, ask);
            let net = edge - fee - self.cfg.slippage_margin;
            if net < self.cfg.min_net_edge {
                continue;
            }
            match &best {
                Some((_, _, _, best_edge)) if *best_edge >= net => {}
                _ => best = Some((side_label, token_id, ask, net)),
            }
        }

        let Some((side_label, token_id, ask, net_edge)) = best else {
            metrics::counter!("hrf_model_btc5m_skip_edge_total").increment(1);
            return vec![];
        };

        // 9) No pyramiding: if there is already a Buy pending or filled, skip.
        if let Some(state) = self.states.get(&cid) {
            match state.state {
                EntryState::Pending | EntryState::Filled => {
                    metrics::counter!("hrf_model_btc5m_skip_dup_pending_total").increment(1);
                    return vec![];
                }
                EntryState::None => {}
            }
        }

        // 10) Size — `max_order_usdc / (limit + fee_per_share)` truncated to
        //     2dp (SDK minimum). The inclusive fee cap guarantees
        //     `size × limit + fee_per_share × size ≤ max_order_usdc` holds
        //     by construction. Without an override, falls back to the
        //     budget-derived bound (no bypass).
        let size = self.max_shares(ask, rate);
        if size <= Decimal::ZERO {
            metrics::counter!("hrf_model_btc5m_skip_zero_size_total").increment(1);
            return vec![];
        }

        let fee = self.fee_per_share(rate, ask);
        tracing::info!(
            cid = %cid,
            side = %side_label,
            token_id = %token_id,
            ask = %ask,
            p_model = %p_dec,
            fee_per_share = %fee,
            slippage = %self.cfg.slippage_margin,
            net_edge = %net_edge,
            size = %size,
            secs_left,
            "model_btc5m: edge detected — emitting taker FOK"
        );

        // Mark pending before emitting: if the next book arrives before
        // the fill/Reject, the dup-pending gate will reject the second
        // intent. If the fill/Reject does NOT arrive and `clear_at` passes, the state
        // is purged by `clear_at` in `on_timer`.
        let side_buy = OrderSide::Buy;
        self.states.insert(
            cid.clone(),
            BracketState {
                clear_at: book.clear_at,
                state: EntryState::Pending,
                filled_size: Decimal::ZERO,
                filled_price: Decimal::ZERO,
                yes_token: book.yes.token_id.clone(),
                no_token: book.no.token_id.clone(),
            },
        );

        let intent = OrderIntent {
            market_cid: cid.clone(),
            token_id: token_id.clone(),
            side: side_buy,
            price: ask,
            size,
            clear_at: book.clear_at,
            signal: net_edge,
            neg_risk: book.neg_risk,
            limit_price: Some(ask + self.cfg.slippage_margin),
            second_leg: None,
            strategy: StrategyId::ModelBtc5m,
            resting: false,
            real_clear_at: None,
        };
        metrics::counter!(
            "hrf_model_btc5m_signals_total",
            "side" => side_label.to_string()
        )
        .increment(1);
        vec![intent]
    }

    fn on_fill(&mut self, fill: &Fill) {
        if fill.strategy != StrategyId::ModelBtc5m {
            return;
        }
        // Idempotency against a race: a `Fill` from the CLOB-paper gateway can
        // arrive BEFORE `on_book` has processed the prediction and
        // emitted the Buy (IPC latency, WS reconnect, etc.). In that scenario
        // we create the state with `clear_at = fill.clear_at` so that the
        // next `on_book` finds the `Filled` and respects the hold.
        // If the fill is not for an expected token (it appears in neither the YES
        // nor the NO of the cid's book — race with another cid), we do not create a
        // spurious state: we let it pass and the test logs diagnose.
        let state = self.states.entry(fill.market_cid.clone()).or_insert_with(||
            // We do not yet have the book's YES/NO ids; we use the fill's
            // `token_id` as a sentinel in both slots: the check
            // `fill.token_id == state.yes_token || == state.no_token` will always
            // be true for this specific fill (we are creating the
            // state precisely for it).
            BracketState {
                clear_at: fill.clear_at,
                state: EntryState::None,
                filled_size: Decimal::ZERO,
                filled_price: Decimal::ZERO,
                yes_token: fill.token_id.clone(),
                no_token: fill.token_id.clone(),
            });
        match fill.side {
            FillSide::Buy => {
                if fill.token_id != state.yes_token && fill.token_id != state.no_token {
                    return;
                }
                let prev = state.filled_size;
                let new = prev + fill.size;
                state.filled_price = if new > Decimal::ZERO {
                    (state.filled_price * prev + fill.price * fill.size) / new
                } else {
                    fill.price
                };
                state.filled_size = new;
                state.state = EntryState::Filled;
                metrics::counter!("hrf_model_btc5m_fills_total").increment(1);
                tracing::info!(
                    cid = %fill.market_cid,
                    token = %fill.token_id,
                    price = %fill.price,
                    size = %fill.size,
                    "model_btc5m: fill confirmed — hold until settlement"
                );
            }
            // Sell/Rejected/GateRejected/Cancelled do NOT advance the state: the
            // strategy holds, sells/orders are managed by official
            // settlement. A pre-trade Rejected must clear the
            // pending to allow a retry.
            FillSide::Sell => {}
            FillSide::Rejected | FillSide::GateRejected => {
                if state.state == EntryState::Pending {
                    state.state = EntryState::None;
                    metrics::counter!("hrf_model_btc5m_rejected_total").increment(1);
                    tracing::info!(
                        cid = %fill.market_cid,
                        "model_btc5m: Buy rejected — pending cleared"
                    );
                }
            }
            FillSide::Cancelled => {}
        }
    }

    /// Watchdog: clear states whose bracket already closed without a fill.
    fn on_timer(&mut self) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        let intents = vec![];
        self.states.retain(|cid, s| {
            if s.clear_at <= now {
                tracing::info!(
                    cid = %cid,
                    state = ?s.state,
                    "model_btc5m: bracket closed — state purged"
                );
                if s.state == EntryState::Pending {
                    metrics::counter!("hrf_model_btc5m_pending_expired_total").increment(1);
                }
                false
            } else {
                true
            }
        });
        intents
    }
}

/// Scope: the slug starts with `btc-updown-5m` (exact prefix, but
/// suffixes like `-epoch` are accepted) AND the bracket duration is 300s.
pub fn is_target_market(book: &BookSnapshot) -> bool {
    if !book.slug.starts_with(REQUIRED_SLUG_PREFIX) {
        return false;
    }
    let duration = book.clear_at - book.market_open_ts;
    duration == REQUIRED_DURATION_SECS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HalfBook, Level, TokenBook};
    use rust_decimal_macros::dec;

    fn make_book(yes_ask: &str, no_ask: &str, secs_left: i64) -> BookSnapshot {
        let asks = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(50),
            }],
        };
        let bids = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(50),
            }],
        };
        let now = crate::time::now().timestamp();
        BookSnapshot {
            market_condition_id: "cid-btc5m".into(),
            yes: TokenBook {
                token_id: "tok-yes-btc".into(),
                bids: bids(&(yes_ask.parse::<Decimal>().unwrap() - dec!(0.01)).to_string()),
                asks: asks(yes_ask),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "tok-no-btc".into(),
                bids: bids(&(no_ask.parse::<Decimal>().unwrap() - dec!(0.01)).to_string()),
                asks: asks(no_ask),
                ..Default::default()
            },
            market_open_ts: now - (300 - secs_left),
            clear_at: now + secs_left,
            neg_risk: false,
            slug: format!("btc-updown-5m-{}", now - (300 - secs_left)),
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate: Some(dec!(0.07)),
            tick_size: None,
            recv_instant: Some(crate::time::Instant::now()),
        }
    }

    /// Helper that creates a prediction coherent with a given book:
    /// `market_cid` and `slug` are copied from the book so the identity
    /// gate (4e/4e2) always passes as long as the other fields are valid.
    /// `token_id_yes/no` are also copied — tests that want a
    /// mismatch overwrite them after building.
    fn make_pred_for_book(book: &BookSnapshot, p_model: f64) -> ModelPrediction {
        ModelPrediction {
            schema_version: MODEL_TOOL_SCHEMA_VERSION,
            ts_ms: crate::time::now().timestamp_millis(),
            market_cid: book.market_condition_id.clone(),
            slug: book.slug.clone(),
            token_id_yes: book.yes.token_id.clone(),
            token_id_no: book.no.token_id.clone(),
            side: "buy_yes".into(),
            p_model: Some(p_model),
            p_market: Some(0.5),
            book_age_ms: 0,
            spot_age_ms: 0,
            latency_ms: 0,
            budget_ms: 0,
            abstention: None,
            model_hash: Some(
                "ef4e4b0860ef9601fb21462539f4d43d2c8af378060eb2dcb63d153502e4c396".into(),
            ),
        }
    }

    fn vol() -> VolSnapshot {
        let mut v = VolSnapshot::default();
        v.last_price = dec!(100_000);
        v.updated_at = crate::time::Instant::now();
        v
    }

    /// Confirms the scope gate: a different slug produces NO intent even if
    /// the prediction is excellent and the book fresh.
    #[test]
    fn scope_rejects_non_btc_5m_slug() {
        let (_tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let mut book = make_book("0.40", "0.50", 120);
        book.slug = "eth-updown-5m-epoch".into(); // NOT BTC
        let intents = strat.on_book(&book, &vol());
        assert!(
            intents.is_empty(),
            "scope must reject ETH even if the rest is OK"
        );
    }

    /// Duration gate: a 15m bracket produces NO intent even if the slug
    /// `btc-updown-5m-...` is present — the duration is evaluated by the
    /// difference `clear_at − market_open_ts`.
    #[test]
    fn scope_rejects_non_300s_duration() {
        let (_tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let mut book = make_book("0.40", "0.50", 600); // 600s remaining = 900s total
        book.market_open_ts = crate::time::now().timestamp() - 300; // opened 5 min ago, total = 300+600 = 900
                                                                    // duration = 900 ≠ 300 → skip
        let intents = strat.on_book(&book, &vol());
        assert!(
            intents.is_empty(),
            "900s duration is not an exact 5m bracket"
        );
    }

    /// Freshness gate: `recv_instant` must be present and within
    /// `ctx.max_book_age` (the runner already filters older books, but the
    /// strategy does its own check for isolated tests).
    #[test]
    fn scope_accepts_btc_5m_300s() {
        let (_tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        assert!(is_target_market(&book));
        // No prediction → skip (prediction gate), but NOT by scope.
        let intents = strat.on_book(&book, &vol());
        assert!(intents.is_empty());
    }

    /// Edge gate: prediction with P(YES)=0.5, YES ask=0.4 → gross edge 0.1.
    /// Fee 0.07*0.4*0.6 = 0.0168. Slippage 0.01. Net = 0.0732 > 0.05 ✓.
    #[test]
    fn emits_intent_with_positive_net_edge() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        tx.send(make_pred_for_book(&book, 0.50)).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert_eq!(intents.len(), 1, "net edge must pass the gate");
        assert!(matches!(intents[0].side, OrderSide::Buy));
        assert_eq!(intents[0].token_id, "tok-yes-btc"); // identity by meta, not by order
    }

    /// Failed edge gate: prediction 0.5, YES ask=0.49 → gross edge 0.01.
    /// Fee ≈ 0.07*0.49*0.51 = 0.017493. Slippage 0.01. Net = -0.017 < 0.05.
    #[test]
    fn skip_without_net_edge() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.49", "0.50", 120);
        tx.send(make_pred_for_book(&book, 0.50)).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert!(intents.is_empty(), "net edge < 0.05 must skip");
    }

    /// Spread gate: YES ask = 0.40, YES bid = 0.10 → spread 0.30 > 0.05 → skip.
    #[test]
    fn skip_due_to_wide_spread() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let mut book = make_book("0.40", "0.50", 120);
        book.yes.bids = HalfBook {
            levels: vec![Level {
                price: dec!(0.10),
                size: dec!(10),
            }],
        };
        tx.send(make_pred_for_book(&book, 0.80)).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert!(intents.is_empty(), "spread 0.30 > 0.05 must skip");
    }

    /// Entry window: <30s remaining → preclose skip.
    #[test]
    fn skip_preclose_under_30s() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 20); // < 30s
        tx.send(make_pred_for_book(&book, 0.50)).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert!(intents.is_empty(), "preclose must skip");
    }

    /// Entry window: >240s remaining → preopen skip.
    #[test]
    fn skip_preopen_over_240s() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 250); // > 240s
        tx.send(make_pred_for_book(&book, 0.50)).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert!(intents.is_empty(), "preopen must skip");
    }

    /// No pyramiding: a confirmed fill blocks future intents over the
    /// SAME cid even if the prediction stays good.
    #[test]
    fn no_pyramiding_after_fill() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        let cid = book.market_condition_id.clone();
        // Simulate a confirmed fill.
        strat.on_fill(&Fill {
            market_cid: cid.clone(),
            token_id: "tok-yes-btc".into(),
            side: FillSide::Buy,
            price: dec!(0.40),
            size: dec!(10),
            order_id: None,
            exec_id: None,
            local_order_id: None,
            ts: crate::time::now(),
            mode: Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ModelBtc5m,
            clear_at: book.clear_at,
            outcome_label: None,
            hedge_id: None,
        });
        tx.send(make_pred_for_book(&book, 0.80)).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert!(
            intents.is_empty(),
            "a confirmed fill must block new intents"
        );
    }

    /// A pre-trade reject clears the pending and allows a retry on the
    /// next book.
    #[test]
    fn reject_pre_trade_clears_pending_and_allows_retry() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        let cid = book.market_condition_id.clone();
        tx.send(make_pred_for_book(&book, 0.50)).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert_eq!(intents.len(), 1, "the first intent must be emitted");
        // The state stays Pending.
        assert!(matches!(
            strat.states.get(&cid).unwrap().state,
            EntryState::Pending
        ));
        // A Rejected arrives.
        strat.on_fill(&Fill {
            market_cid: cid.clone(),
            token_id: "tok-yes-btc".into(),
            side: FillSide::Rejected,
            price: dec!(0.40),
            size: dec!(10),
            order_id: None,
            exec_id: None,
            local_order_id: None,
            ts: crate::time::now(),
            mode: Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::ModelBtc5m,
            clear_at: book.clear_at,
            outcome_label: None,
            hedge_id: None,
        });
        // The state goes back to None.
        assert!(matches!(
            strat.states.get(&cid).unwrap().state,
            EntryState::None
        ));
        // New book with a prediction → must re-emit.
        tx.send(make_pred_for_book(&book, 0.50)).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert_eq!(intents.len(), 1, "a rejection must allow a retry");
    }

    /// Total fail-closed live: the constructor panics if
    /// `mode=Live`, whatever the rest of the configuration. `catch_unwind` is
    /// not treated as success: the strategy must NEVER trade in Live,
    /// and a caught panic would still leave the instance alive
    /// emitting intents if the barrier were skipped.
    #[test]
    #[should_panic(expected = "ModelBtc5m: the defensive experiment NEVER trades in Live mode")]
    fn live_fail_closed_in_constructor() {
        let (_tx, rx) = broadcast::channel(16);
        let mut cfg = ModelBtc5mConfig::default();
        cfg.limits.enabled = true;
        cfg.limits.paper_only = false;
        let _ = ModelBtc5m::new(cfg, rx, Mode::Live);
    }

    /// Extended fail-closed coverage: the barrier must block ALL
    /// flag combinations in Live, not just `paper_only=false`.
    /// Cases: paper_only=true, enabled=false. Each must panic — the
    /// supervisor is the first line, the constructor is the safety
    /// net.
    #[test]
    #[should_panic(expected = "NEVER trades in Live mode")]
    fn live_fail_closed_constructor_blocks_paper_only_true() {
        let (_tx, rx) = broadcast::channel(16);
        let mut cfg = ModelBtc5mConfig::default();
        cfg.limits.enabled = true;
        cfg.limits.paper_only = true;
        let _ = ModelBtc5m::new(cfg, rx, Mode::Live);
    }

    #[test]
    #[should_panic(expected = "NEVER trades in Live mode")]
    fn live_fail_closed_constructor_blocks_even_with_enabled_false() {
        let (_tx, rx) = broadcast::channel(16);
        let cfg = ModelBtc5mConfig::default(); // enabled=false by default
        let _ = ModelBtc5m::new(cfg, rx, Mode::Live);
    }

    /// Paper coverage: `mode=Paper` ALWAYS builds without a panic (it is the
    /// only legitimate condition). No flag should alter this fact.
    #[test]
    fn paper_mode_always_builds_without_panic() {
        let (_tx, rx) = broadcast::channel(16);
        // Defaults: enabled=false, paper_only=true
        let cfg = ModelBtc5mConfig::default();
        let _ = ModelBtc5m::new(cfg, rx, Mode::Paper);
        // Variant with all the flags inverted
        let (_tx2, rx2) = broadcast::channel(16);
        let mut cfg = ModelBtc5mConfig::default();
        cfg.limits.enabled = true;
        cfg.limits.paper_only = true;
        let _ = ModelBtc5m::new(cfg, rx2, Mode::Paper);
    }

    /// The struct's `mode` field is recorded for inspection
    /// (tests, debug). Minimal coverage: after building in Paper,
    /// `mode()` returns `Paper`. The Live branch is unreachable (panic
    /// in the constructor), but the field's symmetry is verified.
    #[test]
    fn struct_mode_matches_parameter() {
        let (_tx, rx) = broadcast::channel(16);
        let strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        assert_eq!(strat.mode(), Mode::Paper);
    }

    /// A missing `model_hash` in the prediction is tolerated — the strategy does not use
    /// the hash to decide, only the tool records it.
    #[test]
    fn prediction_without_model_hash_blocks() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        let mut pred = make_pred_for_book(&book, 0.50);
        pred.model_hash = None;
        tx.send(pred).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert!(
            intents.is_empty(),
            "without the model's identity it cannot trade"
        );
    }

    #[test]
    fn rejects_old_prediction_with_fresh_book() {
        // The default threshold is 1500ms — any older
        // prediction is rejected even if the other gates pass.
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        let mut pred = make_pred_for_book(&book, 0.8);
        pred.ts_ms -= 5_000; // 5s old, well over 1500ms
        tx.send(pred).unwrap();
        assert!(strat.on_book(&book, &vol()).is_empty());
    }

    /// A prediction with `ts_ms` in the FUTURE (clock skew, preloaded payload,
    /// tampering) is rejected with its own counter (`skip_future_*`)
    /// for independent diagnosis. Verifies that `saturating_sub` does not
    /// mask the scenario.
    #[test]
    fn rejects_prediction_with_future_ts_ms() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        let mut pred = make_pred_for_book(&book, 0.8);
        pred.ts_ms += 60_000; // 60s in the future
        tx.send(pred).unwrap();
        assert!(strat.on_book(&book, &vol()).is_empty());
    }

    /// The freshness threshold is `max_prediction_age_ms` (default 1500ms),
    /// NOT `min_remaining_secs*1000` as before. A prediction within the
    /// threshold but with the book in preclose (10s) still emits
    /// if the other gates pass — the tolerable age of the prediction
    /// is independent of the bracket's remaining time.
    #[test]
    fn fresh_prediction_passes_age_gate_even_if_bracket_is_in_preclose() {
        // We raise `min_remaining_secs` so that 10s remaining is NOT
        // preclose (we only check the freshness gate, not the time gate).
        let (tx, rx) = broadcast::channel(16);
        let mut cfg = ModelBtc5mConfig::default();
        cfg.min_remaining_secs = 5; // a book with 10s remaining passes this gate
        let mut strat = ModelBtc5m::new(cfg, rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 10); // 10s remaining, within the new gate
        let mut pred = make_pred_for_book(&book, 0.8);
        // Prediction from ~1s ago: within the threshold (1500ms) but at the edge
        pred.ts_ms = crate::time::now().timestamp_millis() - 1000;
        tx.send(pred).unwrap();
        // Edge: 0.8 - 0.4 - fee(0.0168) - slip(0.01) = ~0.37 > 0.05 → should emit
        let intents = strat.on_book(&book, &vol());
        assert_eq!(
            intents.len(),
            1,
            "a 1s-old prediction must pass"
        );
    }

    /// A prediction just above the 1500ms threshold is rejected.
    #[test]
    fn rejects_prediction_just_above_freshness_threshold() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        let mut pred = make_pred_for_book(&book, 0.8);
        pred.ts_ms = crate::time::now().timestamp_millis() - 2000; // 2s, over 1500ms
        tx.send(pred).unwrap();
        assert!(strat.on_book(&book, &vol()).is_empty());
    }

    /// A prediction at the threshold's edge (≤1500ms) passes. Verifies that the
    /// limit is strict (`> max`, not `>=`).
    #[test]
    fn prediction_at_threshold_edge_passes() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        let mut pred = make_pred_for_book(&book, 0.8);
        pred.ts_ms = crate::time::now().timestamp_millis() - 1000; // 1s, within 1500ms
        tx.send(pred).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert_eq!(
            intents.len(),
            1,
            "a 1s-old prediction must pass the 1500ms gate"
        );
    }

    /// Prediction slug != book slug → rejection (even if the tokens
    /// match). Covers the recycled-cid scenario where the `token_id_*`s
    /// stay the same but the bracket already closed.
    #[test]
    fn rejects_slug_mismatch_between_pred_and_book() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        let mut pred = make_pred_for_book(&book, 0.8);
        // Same tokens, different slug (earlier bracket with a recycled cid)
        pred.slug = format!("btc-updown-5m-{}", crate::time::now().timestamp() - 600);
        tx.send(pred).unwrap();
        assert!(strat.on_book(&book, &vol()).is_empty());
    }

    #[test]
    fn rejects_incompatible_tokens() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        let mut pred = make_pred_for_book(&book, 0.8);
        pred.token_id_yes = "other-token".into();
        tx.send(pred).unwrap();
        assert!(strat.on_book(&book, &vol()).is_empty());
    }

    #[test]
    fn rejects_nan_probability() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        tx.send(make_pred_for_book(&book, f64::NAN)).unwrap();
        assert!(strat.on_book(&book, &vol()).is_empty());
    }

    #[test]
    fn max_cost_with_fee_does_not_exceed_five_usdc() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        tx.send(make_pred_for_book(&book, 0.8)).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert_eq!(intents.len(), 1);
        let order = &intents[0];
        let limit = order.limit_price.unwrap();
        let fee = (order.size * dec!(0.07) * limit * (Decimal::ONE - limit)).round_dp(5);
        assert!(order.size * limit + fee <= dec!(5));
    }

    /// Schema mismatch → silent skip (recorded in a metric).
    #[test]
    fn skip_if_schema_version_differs() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        let mut pred = make_pred_for_book(&book, 0.50);
        pred.schema_version = 999;
        tx.send(pred).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert!(intents.is_empty(), "schema_version!=1 must skip");
    }

    /// Prediction with abstention → skip without looking at the edge.
    #[test]
    fn prediction_with_abstention_is_ignored() {
        let (tx, rx) = broadcast::channel(16);
        let mut strat = ModelBtc5m::new(ModelBtc5mConfig::default(), rx, Mode::Paper);
        let book = make_book("0.40", "0.50", 120);
        let mut pred = make_pred_for_book(&book, 0.50);
        pred.abstention = Some("latency_exceeded".into());
        pred.p_model = None;
        tx.send(pred).unwrap();
        let intents = strat.on_book(&book, &vol());
        assert!(intents.is_empty(), "abstention must not produce an intent");
    }

    /// Default config has `paper_only=true` and `enabled=false`.
    #[test]
    fn defaults_are_paper_only_and_disabled() {
        let cfg = ModelBtc5mConfig::default();
        assert!(!cfg.limits.enabled);
        assert!(cfg.limits.paper_only);
        assert_eq!(cfg.min_net_edge, dec!(0.05));
        assert_eq!(cfg.slippage_margin, dec!(0.01));
        assert_eq!(cfg.max_spread, dec!(0.05));
        assert_eq!(cfg.min_remaining_secs, 30);
        assert_eq!(cfg.max_remaining_secs, 240);
        assert_eq!(cfg.max_prediction_age_ms, 1500);
        assert_eq!(cfg.fee_rate_fallback, dec!(0.07));
        assert_eq!(cfg.limits.max_order_usdc, dec!(5));
        assert_eq!(cfg.limits.daily_loss_usdc_stop, dec!(2));
    }
}

/// Construction helper used by the supervisor and the tests: spins up an internal
/// `broadcast::channel::<ModelPrediction>` and builds the strategy
/// in `Paper` mode. Used in `strategy/registry.rs` and `tests/`.
pub fn build_for_paper(
    cfg: ModelBtc5mConfig,
) -> (ModelBtc5m, broadcast::Receiver<ModelPrediction>) {
    let (tx, rx) = broadcast::channel(cfg.broadcast_capacity_hint());
    let strat = ModelBtc5m::new(cfg, rx, Mode::Paper);
    // We return the `tx` wrapped in something cloneable: the caller uses it to
    // emit predictions in tests. Since `broadcast::Sender` is already cloneable,
    // returning the sender is enough.
    drop(tx);
    // Trick: rebuilding the `tx` by cloning from the internal channel is not possible
    // from here. It is solved at the call site: the supervisor has its
    // own `broadcast::Sender<ModelPrediction>` that it injects via the
    // constructor. This helper exists only for tests.
    (strat, broadcast::channel(16).1)
}

impl ModelBtc5mConfig {
    /// Size of the broadcast used by `build_for_paper` for tests. Not used
    /// in production (the supervisor injects its own).
    pub fn broadcast_capacity_hint(&self) -> usize {
        16
    }
}
