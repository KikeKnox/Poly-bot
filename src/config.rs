//! Configuration loading.
//!
//! Configuration lives in a `config/` directory with one file per domain.
//! Figment merges them in order and the `HRF_*` environment variable override is
//! applied on top (double underscore for nesting: `HRF_RISK__MAX_TOTAL_EXPOSURE_USDC`).
//!
//! Design: `Config` uses nested structs so each section is independent
//! and the TOML files stay small and typed. The strategies directory
//! (`config/strategies/*.toml`) is discovered automatically.

use figment::{
    providers::{Env, Format, Toml},
    Figment,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::types::{Mode, StrategyId};

// ── Main struct ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub mode: Mode,
    pub db_path: String,
    #[serde(default = "defaults::clob_api_url")]
    pub clob_api_url: String,
    #[serde(default = "defaults::gamma_api_url")]
    pub gamma_api_url: String,
    pub oracle: OracleConfig,
    pub market_feed: MarketFeedConfig,
    pub risk: RiskConfig,
    pub api: ApiConfig,
    #[serde(default)]
    pub telegram: TelegramConfig,
    /// Discovery universes declared in
    /// `[universes.<name_str>]`. Empty in old TOMLs (without `[universes]`) —
    /// `synthesize_default_universes` fills it with the two crypto up/down universes
    /// derived from `market_feed` so the bot still starts.
    #[serde(default)]
    pub universes: HashMap<String, UniverseConfig>,
    /// Fair-value model parameters shared across several strategies
    /// (not owned by any single one, hence kept apart from
    /// `config/strategies/`).
    #[serde(default)]
    pub fair_value: FairValueConfig,

    #[serde(default)]
    pub ladder_arb: LadderArbConfig,
    #[serde(default)]
    pub vol_surface: VolSurfaceConfig,
    #[serde(default)]
    pub post_close: PostCloseConfig,
    /// Market analytics recorder (read-only,
    /// `src/tools/recorders/analytics.rs`):
    /// records the full book (YES/NO bids+asks) of every market
    /// asset every 1s in a database SEPARATE from `StateStore`. It never emits `OrderIntent`,
    /// does not go through `RiskGuard`/executor — it is not a toggleable strategy.
    #[serde(default)]
    pub analytics: AnalyticsConfig,
    /// Spot/oracle series for research
    /// (`src/tools/recorders/spot_stream.rs`): records one
    /// records one sample per spot CHANGE (binance) for each asset in a
    /// separate DB. Read-only — never emits `OrderIntent`, does not go through
    /// `RiskGuard`/executor. Same observer pattern as `analytics`.
    #[serde(default)]
    pub spot_stream: SpotStreamConfig,
    /// On-chain redemption of winning positions in live when
    /// `risk.auto_redeem = false` (EOA account, no magic/pro). Read-only on
    /// config; the redeem is executed by `src/settlement/redeem.rs` from `src/settlement/orphan.rs`.
    #[serde(default)]
    pub redeem: RedeemConfig,
    /// Config for the 12 strategies registered in `strategy::registry`
    /// (all items except `Follower`, which does not implement `Strategy` — see its
    /// doc comment below). `#[serde(flatten)]` keeps each TOML table
    /// (`[arb]`, `[near_certain]`, ...) at the top level, as before.
    #[serde(flatten)]
    pub strategies: crate::strategy::StrategyConfigs,
    /// Wallet follower — NOT a `Strategy`
    /// (standalone task in `strategy/follower.rs`, see the doc comment there), but
    /// it reuses `StrategyLimits` to hook into `RiskGuard` (kill switch,
    /// rate limit, daily loss stop) like the others.
    #[serde(default)]
    pub follower: crate::strategy::follower::FollowerConfig,
    /// Simulation profile for RESEARCH (`[research_sim]`). NEW section,
    /// separate from the operational `risk.paper_*` profile — only consumed by
    /// the research path (`SimulatedRouter::apply_research_profile`,
    /// replay/tests); the operational bot never reads it. The default reproduces
    /// current behavior even if the section exists in the TOML.
    #[serde(default)]
    pub research_sim: ResearchSimConfig,
    /// Causal recorder of research events
    /// (src/research/events.rs) — `[research_events]` section. Pure observer,
    /// like `analytics`/`spot_stream`: it never emits `OrderIntent` nor
    /// touches `RiskGuard`/executor. Writes JSONL to its OWN directory
    /// (research data outside the active DB).
    /// `enabled=false` by default: enabling it is opt-in because it multiplies
    /// disk writes on the hot path.
    #[serde(default)]
    pub research_events: ResearchEventsConfig,
    /// Official resolutions fetcher (`src/tools/recorders/resolution.rs`) —
    /// `[resolution_fetcher]` section. Sweeps `markets_observed` already closed
    /// without a confirmed label and queries Gamma by slug; the result enters through
    /// `crate::research::labels::apply_fetch` (only `Resolved` writes). Read-only
    /// HTTP + label writes — never trades.
    #[serde(default)]
    pub resolution_fetcher: ResolutionFetchConfig,
    /// *Shadow* observer of the numeric loaded model (`model_observer.rs`) —
    /// `[model_shadow]` section. Pure observer of the same kind as
    /// `analytics`/`spot_stream`/`research_events`: consumes the broadcast of
    /// books and the vol watch, records predictions and VIRTUAL decisions
    /// in its own JSONL, and can NEVER emit `OrderIntent` nor
    /// touch `RiskGuard`/executor/ledger (its `ShadowCtx` carries none of those
    /// channels — it is structural, not discipline). `enabled=false` by
    /// default: no gate enables live automatically; the shadow is
    /// evidence only.
    #[serde(default)]
    pub model_shadow: ModelShadowConfig,
    /// Numeric loaded-model tool (`src/tools/model/`). `[model_tool]` section.
    /// Producer of P(YES) probabilities; the `ModelBtc5m` strategy is the
    /// ONLY one authorized to turn them into `OrderIntent`. `enabled=false` by
    /// default; no gate to live.
    #[serde(default)]
    pub model_tool: crate::tools::model::ModelToolConfig,
}

impl Config {
    /// Loads the configuration from `config/` and overrides it with `HRF_*` env vars.
    ///
    /// Env vars use `__` as the nesting separator:
    /// `HRF_RISK__MAX_TOTAL_EXPOSURE_USDC=50` overrides `risk.max_total_exposure_usdc`.
    pub fn load() -> anyhow::Result<Self> {
        let config_dir = Path::new("config");
        let strat_dir = config_dir.join("strategies");

        let mut fig = Figment::new()
            .merge(Toml::file(config_dir.join("general.toml")))
            .merge(Toml::file(config_dir.join("telegram.toml")));

        // Discover and merge all the strategies/*.toml files
        if strat_dir.is_dir() {
            let mut paths: Vec<_> = std::fs::read_dir(&strat_dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("toml"))
                .collect();
            paths.sort(); // deterministic order
            for path in paths {
                fig = fig.merge(Toml::file(path));
            }
        }

        // The `mode` declared by `config/general.toml`, read BEFORE
        // `HRF_MODE` can override it (see `check_mode_conflict`).
        let toml_mode = classify_declared_mode(&fig);

        // Override with HRF_* env vars (__ = nesting separator)
        fig = fig.merge(
            Env::prefixed("HRF_")
                .lowercase(true)
                .map(|k| k.as_str().replace("__", ".").into()),
        );

        let mut cfg: Config = fig.extract()?;
        check_mode_conflict(toml_mode, cfg.mode)?;
        cfg.synthesize_default_universes();
        cfg.validate()?;
        Ok(cfg)
    }

    /// BACKWARD COMPAT: if the TOML has no `[universes]`,
    /// synthesizes the two crypto up/down universes from the current
    /// `market_feed` config — same `slug_patterns`/`end_date_max_hours` as were used
    /// before this change, so the bot discovers exactly the same
    /// set of markets with a stale TOML. Both end up in each strategy's default
    /// `universe` (`defaults::default_universe`), so which
    /// of the two a given market is attributed to in the discovery dedup
    /// does not change the dispatch (see `strategy::run`).
    fn synthesize_default_universes(&mut self) {
        if !self.universes.is_empty() {
            return;
        }
        let base = UniverseConfig {
            discovery: "slug".into(),
            slug_patterns: self.market_feed.bracket_slug_patterns.clone(),
            tag_slugs: Vec::new(),
            end_date_min_hours: 0,
            end_date_max_hours: self.market_feed.end_date_max_hours,
            duration_filter: Vec::new(),
            has_oracle: true,
            scan: false,
            max_markets: defaults::universe_max_markets(),
        };
        self.universes
            .insert("crypto-updown-short".into(), base.clone());
        self.universes.insert("crypto-updown-long".into(), base);
    }

    /// Names of all "asset" universes: referenced by an
    /// `enabled=true` strategy, or marked `scan=true` (they are discovered and
    /// logged even if no strategy trades there).
    pub fn active_universe_names(&self) -> std::collections::HashSet<String> {
        let mut set = std::collections::HashSet::new();
        for &id in StrategyId::ALL {
            let lim = self.strategy_limits(id);
            if lim.enabled {
                set.extend(lim.universe.iter().cloned());
            }
        }
        for (name, uni) in &self.universes {
            if uni.scan {
                set.insert(name.clone());
            }
        }
        set
    }

    /// Universes declared by at least one `enabled=true` strategy —
    /// their tokens go to the WebSocket (real time). The remaining
    /// asset universes (scan-only) are served by batched REST poll (`poll.rs`): the
    /// venue limits the WS by TOTAL subscribed tokens. The strategy
    /// supervisor keeps the set hot on toggle (`ws_universes`
    /// in `MarketRegistry`).
    pub fn ws_universe_names(&self) -> std::collections::HashSet<String> {
        let mut set = std::collections::HashSet::new();
        for &id in StrategyId::ALL {
            let lim = self.strategy_limits(id);
            if lim.enabled {
                set.extend(lim.universe.iter().cloned());
            }
        }
        set
    }

    /// `StrategyLimits` of a strategy by its `StrategyId` — delegates to
    /// `strategy::registry::strategy_limits` (single id -> config mapping
    /// point, see `strategy/registry.rs`).
    pub fn strategy_limits(&self, id: StrategyId) -> &StrategyLimits {
        crate::strategy::registry::strategy_limits(id, self)
    }

    fn validate(&self) -> anyhow::Result<()> {
        // Live requires CLOB credentials in the environment
        if self.mode == Mode::Live {
            let has_key = std::env::var("POLY_PRIVATE_KEY")
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            anyhow::ensure!(has_key, "live mode requires POLY_PRIVATE_KEY in the environment");
            // Without auto_redeem (EOA account) the only way to collect
            // winnings is on-chain `redeemPositions` — without an RPC the
            // positions would pile up unredeemable. Aborting is fail-closed.
            anyhow::ensure!(
                self.risk.auto_redeem || !self.redeem.rpc_url.is_empty(),
                "live with risk.auto_redeem=false requires [redeem].rpc_url \
                 (HRF_REDEEM__RPC_URL) — without an RPC the winners cannot be redeemed"
            );
        }

        // API on non-loopback requires a token
        if !self.api.bind.starts_with("127.") && !self.api.bind.starts_with("::1") {
            anyhow::ensure!(
                !self.api.token.is_empty(),
                "api.bind is not loopback — set api.token (HRF_API__TOKEN)"
            );
        }

        // Ladder arb: remember the calibration gate
        if self.ladder_arb.limits.enabled {
            tracing::warn!(
                "ladder_arb enabled — prior gate: >= 1 week of scan confirming volume"
            );
        }

        self.validate_universes()?;
        self.validate_strategy_ranges()?;
        self.validate_live_loss_stop()?;
        self.validate_theta_capture_blocked_live()?;

        Ok(())
    }

    /// `daily_loss_usdc_stop = "0"` disables the loss stop
    /// (`risk.rs`: `if stop > Decimal::ZERO`). In paper that is legitimate — there is no
    /// real capital and a calibration run must not auto-pause on the noise
    /// of simulated PnL. In live it is not: a strategy allowed to move real
    /// money without a stop can lose without automatic brake, so startup
    /// fails CLOSED.
    ///
    /// The criterion is `!paper_only`, NOT `can_run`: a
    /// filter by `can_run` exempts `enabled = false` strategies, and
    /// those can be turned on hot via API/Telegram on a process already
    /// in live (the task is spawned on demand). Depending on `enabled`,
    /// the invariant only lasted until the first toggle.
    fn validate_live_loss_stop(&self) -> anyhow::Result<()> {
        if self.mode != Mode::Live {
            return Ok(());
        }
        for id in StrategyId::ALL {
            let lim = self.strategy_limits(*id);
            if lim.enabled && !lim.paper_only && lim.daily_loss_usdc_stop <= Decimal::ZERO {
                anyhow::bail!(
                    "config: {id} is enabled for live (paper_only = false) but has \
                     daily_loss_usdc_stop = {} — a 0 disables the loss stop and in live that \
                     is trading without a brake. Give it a cap > 0 in its TOML under config/strategies/ or \
                     set paper_only = true",
                    lim.daily_loss_usdc_stop
                );
            }
        }
        Ok(())
    }

    /// ThetaCapture resting sells
    /// (`sell_loser_side`, maker asks on tokens that
    /// are NOT held) are naked shorts: the GTD claim disappears when
    /// cancelled and `state.rs` does not model open short positions. While
    /// the ledger does not distinguish an "inventory sell" from a "short", TC cannot go
    /// live — startup fails CLOSED if it is not `paper_only`.
    /// Same criterion as `validate_live_loss_stop`: `paper_only` is checked,
    /// not `enabled`, because the hot toggle comes later.
    fn validate_theta_capture_blocked_live(&self) -> anyhow::Result<()> {
        if self.mode != Mode::Live {
            return Ok(());
        }
        let lim = self.strategy_limits(StrategyId::ThetaCapture);
        anyhow::ensure!(
            lim.paper_only,
            "config: theta_capture is not paper_only but its resting sells are not \
             modeled (naked shorts: claim that disappears with the GTD, no \
             short position in state.rs). Leave it with paper_only = true in \
             config/strategies/theta_capture.toml until the ledger models shorts."
        );
        Ok(())
    }

    /// Config part of the strategy range validation (the defensive runtime `clamp`
    /// in `market_making.rs` covers the other side): previously no min/max pair
    /// in the strategies config was validated at startup, so
    /// a TOML with `min > max` did not fail fast — it depended on what the
    /// consuming code assumed at runtime (in the best case, the strategy never
    /// quotes/enters because of a condition that is never met; in the worst, a
    /// later panic brings down the whole process — ANY task
    /// that returns today shuts down the whole bot). It fails startup with a
    /// clear message instead of letting that happen in production.
    ///
    /// The min/max pairs are only required for `enabled=true` strategies
    /// (same criterion as `validate_universes`: a disabled one may carry
    /// stale config without blocking startup). Non-negative thresholds
    /// are always required — a negative value in a price/size/spread is
    /// almost certainly an operator typo, not a deliberate configuration
    /// for a switched-off strategy.
    fn validate_strategy_ranges(&self) -> anyhow::Result<()> {
        if self.strategies.market_making.limits.enabled {
            anyhow::ensure!(
                self.strategies.market_making.min_spread_bps
                    <= self.strategies.market_making.max_spread_bps,
                "config: market_making.min_spread_bps ({}) > max_spread_bps ({})",
                self.strategies.market_making.min_spread_bps,
                self.strategies.market_making.max_spread_bps
            );
        }
        if self.strategies.near_certain.limits.enabled {
            anyhow::ensure!(
                self.strategies.near_certain.min_ask <= self.strategies.near_certain.max_ask,
                "config: near_certain.min_ask ({}) > max_ask ({})",
                self.strategies.near_certain.min_ask,
                self.strategies.near_certain.max_ask
            );
        }
        if self.strategies.near_expiry.limits.enabled {
            anyhow::ensure!(
                self.strategies.near_expiry.min_ask <= self.strategies.near_expiry.max_ask,
                "config: near_expiry.min_ask ({}) > max_ask ({})",
                self.strategies.near_expiry.min_ask,
                self.strategies.near_expiry.max_ask
            );
        }
        if self.strategies.directional.limits.enabled {
            anyhow::ensure!(
                self.strategies.directional.min_entry_price
                    <= self.strategies.directional.max_entry_price,
                "config: directional.min_entry_price ({}) > max_entry_price ({})",
                self.strategies.directional.min_entry_price,
                self.strategies.directional.max_entry_price
            );
        }
        if self.strategies.drift_box.limits.enabled {
            anyhow::ensure!(
                self.strategies.drift_box.min_lead_secs
                    <= self.strategies.drift_box.max_window_secs,
                "config: drift_box.min_lead_secs ({}) > max_window_secs ({})",
                self.strategies.drift_box.min_lead_secs,
                self.strategies.drift_box.max_window_secs
            );
        }

        // Thresholds that do not allow negatives. `u32` (bps, counters) already excludes
        // negatives by type — only the `Decimal`s are listed here.
        macro_rules! ensure_non_negative {
            ($val:expr, $label:literal) => {
                anyhow::ensure!(
                    $val >= Decimal::ZERO,
                    concat!("config: ", $label, " cannot be negative ({})"),
                    $val
                );
            };
        }
        ensure_non_negative!(
            self.risk.max_total_exposure_usdc,
            "risk.max_total_exposure_usdc"
        );
        ensure_non_negative!(self.risk.max_bracket_usdc, "risk.max_bracket_usdc");
        ensure_non_negative!(self.strategies.near_certain.min_ask, "near_certain.min_ask");
        ensure_non_negative!(self.strategies.near_certain.max_ask, "near_certain.max_ask");
        ensure_non_negative!(self.strategies.near_expiry.min_ask, "near_expiry.min_ask");
        ensure_non_negative!(self.strategies.near_expiry.max_ask, "near_expiry.max_ask");
        ensure_non_negative!(
            self.strategies.directional.min_entry_price,
            "directional.min_entry_price"
        );
        ensure_non_negative!(
            self.strategies.directional.max_entry_price,
            "directional.max_entry_price"
        );

        // Fields common to StrategyId::ALL via StrategyLimits — negatives
        // here are always a typo, validated whether or not the strategy is
        // enabled=true.
        for &id in StrategyId::ALL {
            let lim = self.strategy_limits(id);
            anyhow::ensure!(
                lim.max_order_usdc >= Decimal::ZERO,
                "config: {id}.max_order_usdc ({}) cannot be negative",
                lim.max_order_usdc
            );
            anyhow::ensure!(
                lim.daily_loss_usdc_stop >= Decimal::ZERO,
                "config: {id}.daily_loss_usdc_stop ({}) cannot be negative",
                lim.daily_loss_usdc_stop
            );
        }

        Ok(())
    }

    /// Loud failure: two rules, only for
    /// `enabled=true` strategies (a disabled one may carry stale config
    /// without stopping the bot from starting).
    /// 1. Every referenced universe must exist in `[universes.*]`.
    /// 2. An oracle-driven strategy (`strategy_requires_oracle=true`) cannot
    ///    declare a universe with `has_oracle=false`.
    fn validate_universes(&self) -> anyhow::Result<()> {
        for &id in StrategyId::ALL {
            let lim = self.strategy_limits(id);
            if !lim.enabled {
                continue;
            }
            for uname in &lim.universe {
                let uni = self.universes.get(uname).ok_or_else(|| {
                    anyhow::anyhow!(
                        "config: strategy {id} references universe '{uname}', \
                         which does not exist in [universes.*]"
                    )
                })?;
                anyhow::ensure!(
                    !(crate::strategy::registry::strategy_requires_oracle(id) && !uni.has_oracle),
                    "config: strategy {id} requires an oracle (requires_oracle=true) \
                     but universe '{uname}' has has_oracle=false"
                );
            }
        }
        Ok(())
    }
}

// ── Sections ──────────────────────────────────────────────────────────────────

/// Fair-value model parameters that apply to several
/// strategies. Lives in `general.toml`, not in `config/strategies/`.
#[derive(Debug, Clone, Deserialize)]
pub struct FairValueConfig {
    /// Additive haircut on `fv_side` in the BUY gates
    /// that compare the fair value with a price (NC, drift_box,
    /// open_momentum, directional). Calibration gap measured at `FV >= 0.90`:
    /// -3.82pp, cluster 95% CI [-4.54, -3.07] over 12,042 brackets — 0.04 is
    /// the conservative rounding of the CI upper bound.
    ///
    /// The haircut ALWAYS moves the decision toward not trading: it is SUBTRACTED from the FV
    /// in buy gates, ADDED in sell gates (theta_capture, sign
    /// inverted — over-selling would be the error).
    ///
    /// Do NOT apply in `parity_maker.rs` (`fv_skew`) nor `reward_farm.rs`: there the
    /// FV is not compared with a price, it skews a symmetric maker quote —
    /// a symmetric haircut cancels out and only adds noise. Do not "complete" this
    /// in a future PR without rereading the calibration notes.
    #[serde(default = "defaults::fv_confidence_haircut")]
    pub confidence_haircut: Decimal,
}

impl Default for FairValueConfig {
    fn default() -> Self {
        Self {
            confidence_haircut: defaults::fv_confidence_haircut(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OracleConfig {
    #[serde(default = "defaults::oracle_ws_url")]
    pub ws_url: String,
    #[serde(default = "defaults::oracle_stale_secs")]
    pub stale_secs: u64,
    #[serde(default = "defaults::oracle_ping_interval_secs")]
    pub ping_interval_secs: u64,
    #[serde(default = "defaults::oracle_max_jump_ratio")]
    pub max_jump_ratio: Decimal,
    #[serde(default = "defaults::min_session_secs")]
    pub min_session_secs_for_backoff_reset: u64,

    // ── Binance futures liquidations (observation) ─────────────────────────
    /// Enables the forced-liquidation feed (`forceOrder`) of Binance
    /// USDS-M futures. Requires no credentials (public endpoint). false
    /// by default: it is the most expensive pipe (new WS) and its signal is not yet
    /// validated — a missing signal never blocks the strategies.
    #[serde(default)]
    pub futures_liq_enabled: bool,
    /// Binance Futures USDS-M WS endpoint for the combined `forceOrder` stream.
    #[serde(default = "defaults::futures_ws_url")]
    pub futures_ws_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MarketFeedConfig {
    pub bracket_slug_patterns: Vec<String>,
    #[serde(default = "defaults::end_date_max_hours")]
    pub end_date_max_hours: u64,
    #[serde(default = "defaults::market_poll_interval_secs")]
    pub market_poll_interval_secs: u64,
    #[serde(default = "defaults::resync_max_concurrent")]
    pub resync_max_concurrent: usize,
    #[serde(default = "defaults::evict_grace_secs")]
    pub evict_grace_secs: i64,
    /// TTL (seconds) of the long-lived `clear_at_cache` of
    /// `MarketRegistry` — much larger than `evict_grace_secs`, each entry is
    /// a minimal tuple. See the doc of `MarketRegistry::clear_at_cache`.
    #[serde(default = "defaults::clear_at_cache_ttl_secs")]
    pub clear_at_cache_ttl_secs: i64,
    #[serde(default = "defaults::broadcast_throttle_ms")]
    pub broadcast_throttle_ms: u64,
    /// Maximum age of a book (ms) for the strategies to process it.
    /// During WS gaps the last book stays frozen — do not trade on stale prices.
    #[serde(default = "defaults::max_book_age_ms")]
    pub max_book_age_ms: u64,
    /// Enables the REST sweeper of `market_feed::resync` (batched POST /books
    /// over the tokens still in `needs_snapshot` after the grace
    /// window). The historical name is `resync_on_reconnect` — it no longer enqueues
    /// a per-token resync on every reconnect (that was the 429/404 storm):
    /// now it only covers the stragglers the WS does not repopulate. `false` = no
    /// REST repair (stale books are blocked via `max_book_age`).
    #[serde(default = "defaults::resync_on_reconnect")]
    pub resync_on_reconnect: bool,
    /// Period of the sweeper pass over `needs_snapshot` (ms).
    #[serde(default = "defaults::resync_sweep_ms")]
    pub resync_sweep_ms: u64,
    /// Window given to the WS to repopulate a pending token before
    /// requesting it via REST (ms) — avoids duplicating the resubscription snapshot.
    #[serde(default = "defaults::resync_grace_ms")]
    pub resync_grace_ms: u64,
    /// Maximum tokens per `POST /books` (the endpoint accepts up to 500).
    #[serde(default = "defaults::resync_batch_size")]
    pub resync_batch_size: usize,
    /// Minimum interval between sweeper HTTP requests (ms) — global rate limit.
    #[serde(default = "defaults::resync_min_interval_ms")]
    pub resync_min_interval_ms: u64,
    /// Base cooldown after a failed attempt (ms); grows x2 per failure per token.
    #[serde(default = "defaults::resync_cooldown_base_ms")]
    pub resync_cooldown_base_ms: u64,
    /// Per-token cooldown ceiling (ms) — dead tokens stay at this cadence.
    #[serde(default = "defaults::resync_cooldown_max_ms")]
    pub resync_cooldown_max_ms: u64,
    /// Cadence of the textual `PING` keepalive (seconds). The market channel
    /// docs require <=10s — at 30s the server closed the session every ~10-60s.
    #[serde(default = "defaults::ws_ping_interval_secs")]
    pub ws_ping_interval_secs: u64,
    #[serde(default = "defaults::ws_stale_threshold_secs")]
    pub ws_stale_threshold_secs: u64,
    /// If `true`, when a new bracket is discovered the CLOB's
    /// `GET /fee-rate?token_id=...` is queried to set the real taker fee of the
    /// market (`MarketEntry.fee_rate_bps`). `false` = use the constants
    /// (`TAKER_FEE_RATE`, each strategy's `taker_fee_rate`).
    #[serde(default = "defaults::fetch_taker_fee")]
    pub fetch_taker_fee: bool,
    /// Window (seconds) of the trade tape (`last_trade_price`) used
    /// to accumulate signed flow per token (`registry::token_flow`). `0` =
    /// disabled: the subscription flag is not set and nothing is accumulated
    /// (missing signal, never blocks).
    #[serde(default = "defaults::tape_window_secs")]
    pub tape_window_secs: u64,
    /// Cadence (seconds) of the batched REST poll of `market_feed::poll` for
    /// the tokens of scan-only universes (no enabled strategy declares
    /// them). The venue limits the WS by total subscribed tokens, so
    /// these are taken out of the stream — 10s gives enough resolution for the
    /// observation tools (LadderArb/VolSurface only use the mid).
    /// `0` = disabled (scan universes receive no books).
    #[serde(default = "defaults::scan_poll_interval_secs")]
    pub scan_poll_interval_secs: u64,
    /// Maximum tokens per `POST /books` of the scan poller (same endpoint as the
    /// sweeper — accepts up to ~500).
    #[serde(default = "defaults::scan_poll_batch_size")]
    pub scan_poll_batch_size: usize,
    /// Predictive bracket prefetch: how often, in seconds,
    /// Gamma's `GET /markets?slug=` is queried for the next expected slug of each
    /// active (asset, duration) — brackets exist in Gamma before
    /// opening, so they are discovered and subscribed ahead of time instead of
    /// waiting for the general poll (`market_poll_interval_secs`).
    /// `0` = disabled (general poll only).
    #[serde(default = "defaults::prefetch_interval_secs")]
    pub prefetch_interval_secs: u64,
}

/// A universe = a set of discovery filters. `discovery` selects the query branch in `market_feed::discover`:
/// - `"slug"`: like today's discovery — filters by `slug_patterns`.
/// - `"tag"`: `?tag_id=<n>&related_tags=true` — `tag_slugs` is resolved to
///   `tag_id` at runtime via `/events?tag_slug=...` (see `discover::resolve_tag_id`).
/// - `"rewards"`: no `end_date` ceiling (if `end_date_max_hours=0`), filters
///   client-side by `parse_rewards(gm).2.is_some()` (rate > 0).
///   client-side by `parse_rewards(gm).2.is_some()` (rate > 0).
#[derive(Debug, Clone, Deserialize)]
pub struct UniverseConfig {
    pub discovery: String,
    #[serde(default)]
    pub slug_patterns: Vec<String>,
    #[serde(default)]
    pub tag_slugs: Vec<String>,
    /// Markets that close before `now + this` are excluded. 0 = no floor.
    #[serde(default)]
    pub end_date_min_hours: u64,
    /// Markets that close after `now + this` are excluded. 0 = no ceiling
    /// (only real scenario: `rewards-longdated`, which is long-dated by definition).
    #[serde(default)]
    pub end_date_max_hours: u64,
    /// Nominal bracket durations to include (e.g. `["5m","15m","1h"]`).
    /// Declared by the spec but NOT enforced yet in discovery
    /// (out of scope for the scan phase): the field is accepted and
    /// preserved for when the real filter is implemented.
    #[serde(default)]
    pub duration_filter: Vec<String>,
    /// Whether the markets of this universe have an asset with an oracle feed
    /// (Binance). Validated against `Strategy::requires_oracle` at startup.
    #[serde(default)]
    pub has_oracle: bool,
    /// `true` = universe that is discovered and logged (target `uni_scan`) even if
    /// no strategy declares it in its `universe` — pure scan phase.
    #[serde(default)]
    pub scan: bool,
    /// Cap of markets discovered for this universe per poll cycle.
    #[serde(default = "defaults::universe_max_markets")]
    pub max_markets: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RiskConfig {
    pub max_total_exposure_usdc: Decimal,
    /// Cap of net directional exposure per bracket (`|YES - NO|` in USDC).
    /// Covered pairs net to ~0 and hedge legs are never blocked;
    /// it only stops the same side stacking across several strategies.
    /// 0 = disabled (default): no per-bracket limit, only the global one.
    #[serde(default)]
    pub max_bracket_usdc: Decimal,
    pub daily_loss_usdc_stop: Decimal,
    #[serde(default = "defaults::paper_disable_kill_switch")]
    pub paper_disable_kill_switch: bool,
    #[serde(default = "defaults::paper_slippage_bps")]
    pub paper_slippage_bps: u32,
    /// Fill probability for FOK in paper (0.0-1.0). Models rejection due to insufficient liquidity.
    #[serde(default = "defaults::paper_fok_fill_rate")]
    pub paper_fok_fill_rate: f64,
    /// Per-strategy override of `paper_fok_fill_rate` (key = strategy
    /// name, e.g. `arb_15m5m`). With `1.0` that strategy's FOK is
    /// decided by the book alone (depth within the limit after latency), with no random draw.
    #[serde(default)]
    pub paper_fok_fill_rate_per_strategy: std::collections::HashMap<String, f64>,
    /// Correlation of the liquidity rejection between the legs of a YES+NO pair.
    /// In live post_orders evaluates both at the same instant, hence the high default (0.9).
    #[serde(default = "defaults::paper_pair_fill_correlation")]
    pub paper_pair_fill_correlation: f64,
    /// Simulated network latency in ms before processing each order in paper.
    #[serde(default = "defaults::paper_latency_ms")]
    pub paper_latency_ms: u64,
    /// Base rate of the simulated taker fee in paper: `fee = rate × p × (1−p)` per
    /// share (parabolic) — the same model Polymarket charges
    /// in live for crypto up/down. Applies to FOK/Market buys and to sells
    /// (FAK); passive GTDs pay no taker fee (see `post_only` in
    /// `sim_router.rs`). Fallback when the book carries no `parabolic_fee_rate`
    /// of its own.
    #[serde(default = "defaults::paper_taker_fee_rate")]
    pub paper_taker_fee_rate: Decimal,
    /// Sim chaos: probability (0.0-1.0) that an ACCEPTED submit returns
    /// `unknown` in paper — exercises the Unknown path end-to-end (journal,
    /// mirror, reconciler) without `ScriptedRouter`. 0.0 = deterministic.
    #[serde(default = "defaults::paper_unknown_rate")]
    pub paper_unknown_rate: f64,
    #[serde(default = "defaults::paper_initial_deposit_usdc")]
    pub paper_initial_deposit_usdc: Decimal,
    /// If true, Polymarket redeems automatically (magic/pro account).
    /// The bot only marks the position in the DB without sending the on-chain transaction.
    /// If false (regular account), the bot must send the redemption manually — implementation pending.
    #[serde(default = "defaults::auto_redeem")]
    pub auto_redeem: bool,
    /// Per-bracket cooldown (Freqtrade's CooldownPeriod): after a losing
    /// exit on a `market_cid`, blocks NEW entries (Buy) on that cid
    /// for this many seconds. Entries only — sells/cancels/unwinds are never
    /// blocked. 0 = disabled (default).
    #[serde(default)]
    pub entry_cooldown_secs: u64,
    /// Per-strategy loss streak (Freqtrade's StoplossGuard): N consecutive
    /// losing exits (pnl<0 in `record_strategy_pnl`) pause the
    /// entries of that strategy for `loss_strike_cooldown_secs`. A winning exit
    /// resets the streak. 0 = disabled (default).
    #[serde(default)]
    pub loss_strike: u32,
    /// Duration (seconds) of the entry pause when `loss_strike` triggers.
    /// 0 = disabled (default): even if `loss_strike` is configured, a
    /// 0s pause never blocks.
    #[serde(default)]
    pub loss_strike_cooldown_secs: u64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        use rust_decimal_macros::dec;
        Self {
            max_total_exposure_usdc: dec!(1000),
            max_bracket_usdc: Decimal::ZERO,
            daily_loss_usdc_stop: dec!(100),
            paper_disable_kill_switch: true,
            paper_slippage_bps: 5,
            paper_fok_fill_rate: 0.75,
            paper_fok_fill_rate_per_strategy: Default::default(),
            paper_pair_fill_correlation: 0.9,
            paper_latency_ms: 150,
            paper_taker_fee_rate: defaults::paper_taker_fee_rate(),
            paper_unknown_rate: 0.0,
            paper_initial_deposit_usdc: dec!(1000),
            auto_redeem: true,
            entry_cooldown_secs: 0,
            loss_strike: 0,
            loss_strike_cooldown_secs: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiConfig {
    #[serde(default = "defaults::api_bind")]
    pub bind: String,
    #[serde(default = "defaults::api_port")]
    pub port: u16,
    #[serde(default)]
    pub token: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TelegramConfig {
    #[serde(default = "defaults::fill_alerts")]
    pub fill_alerts: bool,
    #[serde(default = "defaults::periodic_summary_hours")]
    pub periodic_summary_hours: u32,
}

/// Passive post-close observer (`src/settlement/post_close.rs`). It is not a toggleable
/// strategy (it does not trade, only persists observations), so it lives in `general.toml`
/// and not in `config/strategies/`.
#[derive(Debug, Clone, Deserialize)]
pub struct PostCloseConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "defaults::pcs_sample_throttle_ms")]
    pub sample_throttle_ms: u64,
}

impl Default for PostCloseConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_throttle_ms: defaults::pcs_sample_throttle_ms(),
        }
    }
}

/// Market analytics recorder (`src/tools/recorders/analytics.rs`): read-only,
/// never touches `RiskGuard`/executor/strategies. Its parameters live in
/// `general.toml`; only the `enabled` key lives in
/// `config/strategies/analytics.toml` — persisted by the hot toggle
/// (same destination as the rest of the tools).
#[derive(Debug, Clone, Deserialize)]
pub struct AnalyticsConfig {
    /// `true` by default (unlike `PostCloseConfig`): the user
    /// asked for it enabled from the first startup.
    #[serde(default = "defaults::analytics_enabled")]
    pub enabled: bool,
    /// Path of the dedicated SQLite DB — ALWAYS separate from `db_path`
    /// (`hrf_state*.sqlite`, StateStore). Same folder as the bot's other DBs.
    #[serde(default = "defaults::analytics_db_path")]
    pub db_path: String,
    /// Retention of `book_snapshots`, in hours. `0` = NO RETENTION (never
    /// deleted, unbounded growth — see the doc comment of
    /// `tools::recorders::analytics::run_purger`). The rest of the arithmetic that
    /// justifies the default lives in the `[analytics]` comment in
    /// `config/general.toml`.
    #[serde(default = "defaults::analytics_retention_hours")]
    pub retention_hours: u64,
    /// Every how many seconds a purge cycle runs — its OWN cadence,
    /// independent of the 1s tick that records snapshots (they never share a
    /// loop, see `tools::recorders::analytics::run` vs `run_purger`).
    #[serde(default = "defaults::analytics_purge_interval_secs")]
    pub purge_interval_secs: u64,
    /// Rows per batch in each purge `DELETE` — bounds the transaction so as
    /// not to block the WAL with a single massive delete.
    #[serde(default = "defaults::analytics_purge_batch_size")]
    pub purge_batch_size: u32,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enabled: defaults::analytics_enabled(),
            db_path: defaults::analytics_db_path(),
            retention_hours: defaults::analytics_retention_hours(),
            purge_interval_secs: defaults::analytics_purge_interval_secs(),
            purge_batch_size: defaults::analytics_purge_batch_size(),
        }
    }
}

/// Spot/oracle series for research (`src/tools/recorders/spot_stream.rs`):
/// read-only,
/// never touches `RiskGuard`/executor/strategies. Same split as
/// `AnalyticsConfig`: parameters in `general.toml`, `enabled` in
/// `config/strategies/spot_stream.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct SpotStreamConfig {
    /// `true` by default, like `analytics` (the tick series is meant to be
    /// recording from the first startup).
    #[serde(default = "defaults::spot_stream_enabled")]
    pub enabled: bool,
    /// Path of the research DB — ALWAYS separate from `db_path`
    /// (StateStore) and from `analytics.db_path`. `main` applies the
    /// mode suffix (`_paper`/`_live`) with `StateStore::resolve_path`, as
    /// `tools::recorders::analytics` does, so paper and live do not share ticks.
    #[serde(default = "defaults::spot_stream_db_path")]
    pub db_path: String,
    /// Capacity of the bounded producer→writer channel. Samples that do not
    /// fit are DROPPED (`try_send`), they never block the producer — losing
    /// research samples does not affect trading.
    #[serde(default = "defaults::spot_stream_channel_capacity")]
    pub channel_capacity: usize,
}

impl Default for SpotStreamConfig {
    fn default() -> Self {
        Self {
            enabled: defaults::spot_stream_enabled(),
            db_path: defaults::spot_stream_db_path(),
            channel_capacity: defaults::spot_stream_channel_capacity(),
        }
    }
}

/// Causal recorder of research events (src/research/events.rs).
/// Read-only on the flow (non-blocking hooks with `try_send` — an event
/// that does not fit is dropped and counted, never blocks trading). Same
/// observer pattern as `AnalyticsConfig`/`SpotStreamConfig`.
#[derive(Debug, Clone, Deserialize)]
pub struct ResearchEventsConfig {
    /// `false` by default: bulk recording of raw events is opt-in
    /// (disk volume on the hot path). Unlike
    /// `analytics`/`spot_stream`, which record small aggregated series.
    #[serde(default)]
    pub enabled: bool,
    /// Base directory of the JSONL files (`<dir>/<source>/<yyyymmdd>.jsonl`).
    /// Must be a research path SEPARATE from the operational DBs;
    /// research data stays outside the active DB.
    #[serde(default = "defaults::research_events_dir")]
    pub dir: String,
    /// Capacity of the bounded producer→writer channel (see
    /// `ResearchRecorderConfig::channel_capacity`).
    #[serde(default = "defaults::research_events_channel_capacity")]
    pub channel_capacity: usize,
    /// FIFO window of dedup keys remembered by the writer (see
    /// `ResearchRecorderConfig::dedup_capacity`).
    #[serde(default = "defaults::research_events_dedup_capacity")]
    pub dedup_capacity: usize,
    /// `true` forces `custom_feature_enabled` in the WS subscription even if
    /// no strategy consumes the tape — needed to record
    /// `last_trade_price`. NOTE: the flag also enables
    /// `best_bid_ask`/`new_market`/`market_resolved` and multiplies
    /// per-connection traffic (it contributed to the 1013 disconnects). Opt-in.
    #[serde(default)]
    pub record_tape: bool,
}

impl Default for ResearchEventsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dir: defaults::research_events_dir(),
            channel_capacity: defaults::research_events_channel_capacity(),
            dedup_capacity: defaults::research_events_dedup_capacity(),
            record_tape: false,
        }
    }
}

/// Official resolutions fetcher (`src/tools/recorders/resolution.rs`). Periodic task
/// that sweeps `markets_observed` with a past `clear_at` and no `confirmed`
/// label, queries Gamma by slug and feeds
/// `crate::research::labels::apply_fetch`. It never invents a winner: an
/// inaccessible/ambiguous fetch leaves the market pending.
#[derive(Debug, Clone, Deserialize)]
pub struct ResolutionFetchConfig {
    /// `true` by default (same criterion as `tools::recorders::spot_stream`: labels are
    /// input to the research pipeline and the fetch is read-only).
    #[serde(default = "defaults::resolution_fetcher_enabled")]
    pub enabled: bool,
    /// Interval between full sweeps (seconds).
    #[serde(default = "defaults::resolution_fetcher_poll_secs")]
    pub poll_secs: u64,
    /// Minimum spacing between requests to Gamma within a sweep (ms) —
    /// polite rate limit, like the resync sweeper.
    #[serde(default = "defaults::resolution_fetcher_min_request_ms")]
    pub min_request_ms: u64,
    /// Maximum markets queried per sweep (the most recently closed
    /// first; the rest wait for the next cycle).
    #[serde(default = "defaults::resolution_fetcher_batch_size")]
    pub batch_size: usize,
    /// Sweep horizon in days: markets with an older `clear_at`
    /// drop out of the job set. Without it, dead markets (disputed,
    /// purged from Gamma, corrupt `clear_at`) would keep occupying the head of the
    /// batch forever and fresh labels would stop arriving.
    #[serde(default = "defaults::resolution_fetcher_horizon_days")]
    pub pending_horizon_days: u64,
}

impl Default for ResolutionFetchConfig {
    fn default() -> Self {
        Self {
            enabled: defaults::resolution_fetcher_enabled(),
            poll_secs: defaults::resolution_fetcher_poll_secs(),
            min_request_ms: defaults::resolution_fetcher_min_request_ms(),
            batch_size: defaults::resolution_fetcher_batch_size(),
            pending_horizon_days: defaults::resolution_fetcher_horizon_days(),
        }
    }
}

/// *Shadow* observer of the numeric loaded model (`model_observer.rs`).
/// Read-only on the feeds + its own JSONL writes — the artifact is
/// validated by manifest (schema + sha256 of `model.json`) at startup and the
/// weights stay frozen per run. `enabled=false` by default.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelShadowConfig {
    /// `false` by default: the shadow is a research opt-in.
    #[serde(default)]
    pub enabled: bool,
    /// Directory of the versioned artifact (`manifest.json` + `model.json`).
    /// An invalid artifact → the observer does not start (error
    /// logged, never aborts the bot).
    #[serde(default = "defaults::model_shadow_artifact_dir")]
    pub artifact_dir: String,
    /// Directory of the decisions/abstentions/stats JSONL record —
    /// research path separate from the active DB.
    #[serde(default = "defaults::model_shadow_decisions_dir")]
    pub decisions_dir: String,
    /// Signal horizon in ms: how long a prediction is considered
    /// actionable. The latency budget is DERIVED from it
    /// (`latency_budget_ms == 0` → horizon/10) — never from the length
    /// of the bracket.
    #[serde(default = "defaults::model_shadow_horizon_ms")]
    pub signal_horizon_ms: u64,
    /// Recv→decision latency budget in ms. `0` = derived from the
    /// signal horizon (`signal_horizon_ms / 10`, minimum 1 ms).
    /// Exceeding it produces a recorded `latency_exceeded` abstention.
    #[serde(default)]
    pub latency_budget_ms: u64,
    /// Scope by condition_id. Empty = all observed markets.
    #[serde(default)]
    pub markets: Vec<String>,
    /// Scope by asset (slug prefix, lowercase: "btc", "eth"…).
    /// Empty = all.
    #[serde(default)]
    pub assets: Vec<String>,
    /// Minimum edge over the ask for the prediction to trigger a virtual
    /// buy (`p − ask > edge_min`). Threshold of the decision rule —
    /// recorded on every line for the replay.
    #[serde(default = "defaults::model_shadow_edge_min")]
    pub edge_min: Decimal,
    /// Accounting size of each virtual decision (shares) — only for the
    /// offline virtual record/PnL; the shadow does not reserve cash.
    #[serde(default = "defaults::model_shadow_shares_virtual")]
    pub shares_virtual: Decimal,
    /// Capacity of the observer→writer channel (`try_send` — a record
    /// that does not fit is dropped and counted, never blocks).
    #[serde(default = "defaults::model_shadow_channel_capacity")]
    pub channel_capacity: usize,
    /// Minimum cadence between evaluations of the same market (ms). `0` =
    /// evaluate every book; abstention below the interval is not
    /// recorded (pure skip, metric only).
    #[serde(default = "defaults::model_shadow_min_interval_ms")]
    pub min_decision_interval_ms: u64,
    /// Cadence of the `stats` lines (p50/p95/p99 latencies and counters)
    /// in the record. Effective minimum 5s.
    #[serde(default = "defaults::model_shadow_stats_interval_secs")]
    pub stats_interval_secs: u64,
}

impl ModelShadowConfig {
    /// Effective latency budget (ms): explicit if declared, otherwise
    /// derived from the signal horizon (`horizon/10`, minimum 1 ms).
    pub fn effective_budget_ms(&self) -> u64 {
        if self.latency_budget_ms > 0 {
            self.latency_budget_ms
        } else {
            (self.signal_horizon_ms / 10).max(1)
        }
    }
}

impl Default for ModelShadowConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            artifact_dir: defaults::model_shadow_artifact_dir(),
            decisions_dir: defaults::model_shadow_decisions_dir(),
            signal_horizon_ms: defaults::model_shadow_horizon_ms(),
            latency_budget_ms: 0,
            markets: vec![],
            assets: vec![],
            edge_min: defaults::model_shadow_edge_min(),
            shares_virtual: defaults::model_shadow_shares_virtual(),
            channel_capacity: defaults::model_shadow_channel_capacity(),
            min_decision_interval_ms: defaults::model_shadow_min_interval_ms(),
            stats_interval_secs: defaults::model_shadow_stats_interval_secs(),
        }
    }
}

/// On-chain redemption in live when `auto_redeem=false`
/// (EOA account, no magic/pro). Inert in paper — `src/settlement/redeem.rs` runs it
/// from `src/settlement/orphan.rs` only in `Mode::Live`.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RedeemConfig {
    /// HTTPS endpoint of a Polygon RPC. Mandatory in live with
    /// `auto_redeem=false` — without it the winners cannot be redeemed and
    /// startup aborts (fail-closed: better not to trade than to accumulate positions
    /// with no way to collect). Empty is valid in paper and with auto_redeem=true.
    #[serde(default)]
    pub rpc_url: String,
    /// Market collateral — USDC.e on Polygon mainnet by default; only
    /// touched for testnet.
    #[serde(default)]
    pub collateral: String,
}

/// What `config/general.toml` says about `mode`, seen BEFORE the merge
/// of `HRF_MODE`. "Absent" and "unreadable" are not the same: a `.ok()` conflated
/// them and a corrupt `mode` (typo, wrong type) let
/// `HRF_MODE=live` win with no log or abort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeclaredMode {
    /// No TOML declares `mode` — the `Config` default decides.
    Absent,
    /// `mode` is present, but does not parse as `Mode`.
    Unreadable,
    Declared(Mode),
}

/// Classifies the `mode` of the TOML figment (without the env merge).
fn classify_declared_mode(fig: &Figment) -> DeclaredMode {
    match fig.find_value("mode") {
        Err(_) => DeclaredMode::Absent,
        Ok(value) => match value.deserialize::<Mode>() {
            Ok(mode) => DeclaredMode::Declared(mode),
            Err(_) => DeclaredMode::Unreadable,
        },
    }
}

/// Compares the `mode` declared in `config/general.toml` with the EFFECTIVE
/// mode (the one left after the `HRF_MODE` override, which Figment merges
/// after the TOMLs and therefore wins). Previously the env overrode the TOML leaving no
/// trace: the operator could revert to paper by editing the TOML and stay in live.
///
/// The asymmetry is DELIBERATE. Ending up in **live** without the TOML saying so is the
/// dangerous scenario (real money where the config file says there is none)
/// → startup is aborted. The other way round — effective paper with the TOML in live —
/// the ERROR log is enough: falling to paper by accident cannot lose money.
fn check_mode_conflict(declared: DeclaredMode, effective_mode: Mode) -> anyhow::Result<()> {
    let toml_mode = match declared {
        // Nobody declares `mode`: there are no two sources to compare.
        DeclaredMode::Absent => return Ok(()),
        // `mode` present but unreadable. The TOML can no longer contradict
        // `HRF_MODE`, so if the effective mode is live the same asymmetric
        // rule applies: abort. Silencing it was treating it as "absent".
        DeclaredMode::Unreadable => {
            tracing::error!(
                %effective_mode,
                "config/general.toml has a `mode` that is present but UNREADABLE (valid values: \
                 \"paper\" / \"live\") — the effective mode is then decided by HRF_MODE or the default: \
                 '{effective_mode}'"
            );
            anyhow::ensure!(
                effective_mode != Mode::Live,
                "the effective mode is live and the `mode` in config/general.toml is unreadable, so it \
                 cannot confirm it. Startup aborted for safety: fix `mode` in \
                 config/general.toml (\"paper\" or \"live\")"
            );
            return Ok(());
        }
        DeclaredMode::Declared(m) => m,
    };
    if toml_mode == effective_mode {
        return Ok(());
    }
    tracing::error!(
        %toml_mode, %effective_mode,
        "mode conflict: config/general.toml says '{toml_mode}' but the effective mode is \
         '{effective_mode}' — the environment variable HRF_MODE wins, which Figment merges after \
         the TOMLs"
    );
    anyhow::ensure!(
        effective_mode != Mode::Live,
        "mode conflict: HRF_MODE=live overrides mode = \"{toml_mode}\" of config/general.toml. \
         Startup aborted for safety. If you really want live, set it in \
         config/general.toml too; if you wanted paper, remove HRF_MODE from the environment (or from .env) or set it to paper"
    );
    Ok(())
}

// ── Strategies ───────────────────────────────────────────────────────────────

/// Fields common to all strategies.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StrategyLimits {
    #[serde(default)]
    pub enabled: bool,
    /// Experiment that must NEVER touch real money, whatever the global
    /// mode and `enabled` say (see `can_run`).
    #[serde(default)]
    pub paper_only: bool,
    #[serde(default = "defaults::max_orders_per_hour")]
    pub max_orders_per_hour: u32,
    #[serde(default = "defaults::max_order_usdc")]
    pub max_order_usdc: Decimal,
    #[serde(default = "defaults::daily_loss_usdc_stop")]
    pub daily_loss_usdc_stop: Decimal,
    /// Universes where this strategy trades.
    /// Default = current behavior (the two crypto up/down universes) — the
    /// new universes (`crypto-multistrike`, `rewards-longdated`) are never
    /// declared by default, so no existing strategy trades them
    /// even if discovery finds them through `scan=true` (`strategy::run` filters
    /// on this field before `on_book`).
    #[serde(default = "defaults::default_universe")]
    pub universe: Vec<String>,
}

// Manual default so it matches the serde defaults (not 0/0/0).
// The test `strategy_limits_default_matches_serde` fails if they drift apart.
impl Default for StrategyLimits {
    fn default() -> Self {
        Self {
            enabled: false,
            paper_only: false,
            max_orders_per_hour: defaults::max_orders_per_hour(),
            max_order_usdc: defaults::max_order_usdc(),
            daily_loss_usdc_stop: defaults::daily_loss_usdc_stop(),
            universe: defaults::default_universe(),
        }
    }
}

/// Why a strategy cannot run. Returned by
/// `StrategyLimits::can_run`; its `Display` is the message the
/// operator sees in the log, the API and Telegram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockReason {
    /// `enabled = false` in its TOML.
    Disabled,
    /// `paper_only = true`: experiment that never trades with real money.
    PaperOnly,
    /// The supervisor did not receive `model_prediction_tx` (the
    /// injection from `main.rs` was not done) — the strategy cannot
    /// start without the model tool's channel.
    ModelPredictionTxMissing,
    /// Live mode with `daily_loss_usdc_stop = 0`: no daily loss brake.
    NoLossStop,
}

impl std::fmt::Display for BlockReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockReason::Disabled => write!(f, "disabled (enabled = false)"),
            BlockReason::PaperOnly => write!(
                f, "marked paper_only = true: it is an experiment that never trades with real money"
            ),
            BlockReason::ModelPredictionTxMissing => write!(
                f,
                "no prediction channel from model_tool — the supervisor did not inject \
                 `model_prediction_tx` on construction; the runtime wiring is incomplete"
            ),
            BlockReason::NoLossStop => write!(
                f,
                "daily_loss_usdc_stop = 0 disables the daily loss stop; give it a cap > 0 \
                 in its TOML before running it in live"
            ),
        }
    }
}

impl StrategyLimits {
    /// SINGLE source of truth on whether a strategy can run.
    /// It is consulted by startup (`strategy::registry::spawn_all` and
    /// `register_limits`), the hot toggle
    /// (`strategy::StrategySupervisor::set_enabled`, and hence the API and
    /// Telegram) and the manual spawns in `main.rs` (Follower, post_close_sweep). Do not
    /// duplicate the logic in the callers.
    ///
    /// Deliberate case order: first what does not depend on the mode
    /// (`enabled`), then the absolute prohibition (`paper_only`). In live a
    /// strategy runs iff `enabled && !paper_only`.
    pub fn can_run(&self, mode: Mode) -> Result<(), BlockReason> {
        if !self.enabled {
            return Err(BlockReason::Disabled);
        }
        if mode == Mode::Live && self.paper_only {
            return Err(BlockReason::PaperOnly);
        }
        if mode == Mode::Live && self.daily_loss_usdc_stop <= Decimal::ZERO {
            return Err(BlockReason::NoLossStop);
        }
        Ok(())
    }
}

/// Configuration of the LadderArb tool (`crate::tools`) — scan-only, does
/// not go through RiskGuard. The field is still called `ladder_arb` in `Config` and the
/// TOML shape did not change: only the code that consumes it moved.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct LadderArbConfig {
    #[serde(flatten)]
    pub limits: StrategyLimits,
    #[serde(default = "defaults::arb_fee_threshold")]
    pub fee_threshold: Decimal,
    #[serde(default = "defaults::ladder_min_edge")]
    pub min_edge_after_fee: Decimal,
    #[serde(default = "defaults::arb_max_size_shares")]
    pub max_size_shares: Decimal,
}
/// Configuration of the VolSurface tool (VS, `crate::tools`) — scan-only,
/// no RiskGuard. The field is still called `vol_surface` in `Config`
/// and the TOML shape did not change: only the code that consumes it moved.
///
/// Groups brackets by (asset, strike) and compares the implied volatility across
/// different durations. Emits a log + metric when σ_short / σ_long > threshold.
/// It emits no orders until the scan confirms an edge and trading is enabled
/// (at that point it would stop being a `Tool` and become a `Strategy` again).
#[derive(Debug, Clone, Deserialize)]
pub struct VolSurfaceConfig {
    #[serde(flatten)]
    pub limits: StrategyLimits,
    /// Minimum σ_short/σ_long ratio to log a divergence.
    #[serde(default = "defaults::vs_vol_ratio_threshold")]
    pub vol_ratio_threshold: f64,
    /// Minimum remaining time of the short bracket to include it in the analysis.
    #[serde(default = "defaults::vs_min_remaining_short")]
    pub min_remaining_short: u64,
    /// Maximum remaining time of the short bracket. Brackets newer than this are ignored.
    #[serde(default = "defaults::vs_max_remaining_short")]
    pub max_remaining_short: u64,
}

impl Default for VolSurfaceConfig {
    fn default() -> Self {
        Self {
            limits: StrategyLimits::default(),
            vol_ratio_threshold: defaults::vs_vol_ratio_threshold(),
            min_remaining_short: defaults::vs_min_remaining_short(),
            max_remaining_short: defaults::vs_max_remaining_short(),
        }
    }
}

// ── Simulation profile for RESEARCH ────────────────────────────────────────────
//
// NEW section, separate from the operational `risk.paper_*` profile: the bot in
// production never reads it (`SimulatedRouter` only applies it when the
// caller invokes `apply_research_profile` — replay/tests). No default
// here changes the meaning of an existing key.

/// Latency leg of the simulated pipeline. Each segment has its own
/// base — in live they are not interchangeable (a cancel takes different time than a
/// submit, and the fill notification is another path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyLeg {
    /// Book/feed arrival at the consumer.
    Feed,
    /// Strategy decision time.
    Decision,
    /// Submit agent→venue (post_orders).
    Submit,
    /// Venue ACK (POST response).
    Ack,
    /// Cancel agent→venue.
    Cancel,
    /// Fill notification (user WS / trade channel).
    Notify,
}

/// Per-stage latency distribution for research.
///
/// **These are parameterizable estimates, NOT measurements**: no field has been
/// calibrated against real latencies observed on the CLOB — a profile that
/// uses them must say so in its report. Deliberately simple model:
/// `base + u·jitter` (uniform ±jitter) scaled by load
/// (`× (1 + load · load_mult)`, where `load` = events in the same batch).
/// The replay engine applies them to the virtual queue — NEVER as sleeps:
/// `feed` is added to the `obs_delay` of each `BookObserve`, `decision` delays the
/// intent's `Dispatch`, `submit`/`cancel` move the job's arrival at the
/// venue and `ack` defers delivery of the response to the executor. Queries
/// (`order_status`, `open_gtd`…) travel through the `submit` stage.
/// **`notify` is NOT modeled**: in the real pipeline the fill reaches the
/// strategy via fill_sink→broadcast, a path the replay reproduces
/// synchronously on every tick — a notification latency would need its
/// own deferred event and remains as a declared pending job.
#[derive(Debug, Clone, Deserialize)]
pub struct LatencyProfile {
    #[serde(default)]
    pub feed_ms: f64,
    #[serde(default)]
    pub decision_ms: f64,
    #[serde(default)]
    pub submit_ms: f64,
    #[serde(default)]
    pub ack_ms: f64,
    #[serde(default)]
    pub cancel_ms: f64,
    #[serde(default)]
    pub notify_ms: f64,
    /// Uniform ± jitter applied to each stage (ms). 0 = fixed latencies.
    #[serde(default)]
    pub jitter_ms: f64,
    /// Load-conditioned: `effective_lat = base × (1 + load × load_mult)`.
    /// 0 = load has no effect (flat).
    #[serde(default)]
    pub load_mult: f64,
}

impl Default for LatencyProfile {
    fn default() -> Self {
        Self {
            feed_ms: 0.0,
            decision_ms: 0.0,
            submit_ms: 0.0,
            ack_ms: 0.0,
            cancel_ms: 0.0,
            notify_ms: 0.0,
            jitter_ms: 0.0,
            load_mult: 0.0,
        }
    }
}

impl LatencyProfile {
    /// Effective ms of a stage under `load` (events in the same batch) and
    /// a uniform draw `u ∈ [0,1)` — PURE, deterministic function: the same
    /// (stage, load, u) always gives the same ms, so the replay is
    /// reproducible by seed.
    pub fn sample(&self, leg: LatencyLeg, load: f64, u: f64) -> i64 {
        let base = match leg {
            LatencyLeg::Feed => self.feed_ms,
            LatencyLeg::Decision => self.decision_ms,
            LatencyLeg::Submit => self.submit_ms,
            LatencyLeg::Ack => self.ack_ms,
            LatencyLeg::Cancel => self.cancel_ms,
            LatencyLeg::Notify => self.notify_ms,
        };
        let jitter = self.jitter_ms * (2.0 * u.clamp(0.0, 1.0) - 1.0);
        let ms = (base + jitter) * (1.0 + load.max(0.0) * self.load_mult);
        ms.max(0.0).round() as i64
    }
}

/// Simulator research profile — `[research_sim]` section.
///
/// Explicit hypotheses of the queue model (no trade tape, L2 data):
///
/// - **`cancel_attr` (κ)**: fraction of a level shrink WITHOUT crossing evidence
///   attributed to cancels/consumption AHEAD of our position.
///   κ=1 (nominal, status quo): every shrink drains `queue_ahead`. κ=0
///   (conservative): every shrink is assumed BEHIND — the queue only advances with
///   real crossing flow. A shrink larger than the queue ahead is
///   ambiguous consumption under any hypothesis (trades that reached us or cancels
///   behind — indistinguishable without a tape) and marks the execution as
///   `Stressed`.
/// - **`share`**: cap of the capturable flow per level/tick ∈ (0,1] —
///   the displayed liquidity is not all ours. 1 = no limit.
/// - **Stale/adverse selection**: a quote shown unchanged for >=
///   `quote_stale_ms` and further than `consensus_margin` from the pair's cross
///   consensus (`1 − mid` of the sibling token) is treated as a stale quote — fills
///   against it are `SimConfidence::Stressed` (low confidence, not
///   free liquidity). 0 in either = disabled.
/// - **`gap_ms`/`gap_abstain`**: a clock jump between consecutive ticks
///   of a token > `gap_ms` invalidates the episode — with `gap_abstain` the post-gap tick
///   neither fills nor advances the queue (it only resyncs the L2 delta base
///   and degrades the order's evidence); without it the fills are marked
///   `Stressed`. 0 = disabled.
/// - **`track_inventory`**: per-token inventory — sells cannot
///   exceed the accumulated purchases (inventory is never fabricated) and resting
///   `LimitSell`s reserve it just as buys reserve cash.
/// - **`latencies`**: per-stage estimates for the replay engine.
#[derive(Debug, Clone, Deserialize)]
pub struct ResearchSimConfig {
    /// Profile label ("nominal" | "conservative" | free-form) — informational;
    /// what matters are the field values.
    #[serde(default = "defaults::research_sim_profile")]
    pub profile: String,
    /// κ for attributing shrink to cancels ahead ∈ [0,1] (see doc above).
    #[serde(default = "defaults::research_sim_cancel_attr")]
    pub cancel_attr: Decimal,
    /// Participation limit per level/tick ∈ (0,1].
    #[serde(default = "defaults::research_sim_participation")]
    pub participation: Decimal,
    /// Quote persistence for the stale mark (ms, sim clock).
    #[serde(default)]
    pub quote_stale_ms: i64,
    /// Out-of-consensus margin (in price) for the stale mark.
    #[serde(default)]
    pub consensus_margin: Decimal,
    /// Tape gap that invalidates the episode (ms). 0 = disabled.
    #[serde(default)]
    pub gap_ms: i64,
    /// true = abstain on the post-gap tick; false = fill marked.
    #[serde(default)]
    pub gap_abstain: bool,
    /// true = sells limited to the simulated inventory + share reservation
    /// for resting `LimitSell`s.
    #[serde(default)]
    pub track_inventory: bool,
    /// Per-stage latencies (estimates — see `LatencyProfile`).
    #[serde(default)]
    pub latencies: LatencyProfile,
}

impl ResearchSimConfig {
    /// NOMINAL profile: reproduces current operational behavior (κ=1,
    /// no participation limit, no stale/gap marks, no inventory tracking) —
    /// same result as applying no profile.
    pub fn nominal() -> Self {
        Self::default()
    }

    /// CONSERVATIVE profile: worst identifiable scenario for each hypothesis —
    /// shrinks do not advance the queue (κ=0), participation limited to 25%
    /// of flow/level, quotes stale >5min and >0.02 from consensus degraded,
    /// gaps >60s abstain, inventory required. Latencies are
    /// generous estimates (not measurements) — see `LatencyProfile`.
    pub fn conservative() -> Self {
        Self {
            profile: "conservative".into(),
            cancel_attr: Decimal::ZERO,
            participation: defaults::research_sim_participation_conservative(),
            quote_stale_ms: 300_000,
            consensus_margin: rust_decimal_macros::dec!(0.02),
            gap_ms: 60_000,
            gap_abstain: true,
            track_inventory: true,
            latencies: LatencyProfile {
                feed_ms: 100.0,
                decision_ms: 50.0,
                submit_ms: 250.0,
                ack_ms: 100.0,
                cancel_ms: 250.0,
                notify_ms: 150.0,
                jitter_ms: 50.0,
                load_mult: 0.5,
            },
        }
    }
}

impl Default for ResearchSimConfig {
    fn default() -> Self {
        Self {
            profile: defaults::research_sim_profile(),
            cancel_attr: defaults::research_sim_cancel_attr(),
            participation: defaults::research_sim_participation(),
            quote_stale_ms: 0,
            consensus_margin: Decimal::ZERO,
            gap_ms: 0,
            gap_abstain: false,
            track_inventory: false,
            latencies: LatencyProfile::default(),
        }
    }
}

// ── Defaults ──────────────────────────────────────────────────────────────────

pub(crate) mod defaults {
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    pub fn clob_api_url() -> String {
        "https://clob.polymarket.com".into()
    }
    pub fn gamma_api_url() -> String {
        "https://gamma-api.polymarket.com".into()
    }

    // See the full justification in the docstring of
    // `FairValueConfig::confidence_haircut`.
    pub fn fv_confidence_haircut() -> Decimal {
        dec!(0.04)
    }

    pub fn oracle_ws_url() -> String {
        "wss://stream.binance.com:9443".into()
    }
    pub fn oracle_stale_secs() -> u64 {
        60
    }
    pub fn oracle_ping_interval_secs() -> u64 {
        20
    }
    pub fn oracle_max_jump_ratio() -> Decimal {
        dec!(0.05)
    }
    pub fn min_session_secs() -> u64 {
        60
    }
    // Public USDS-M endpoint (no auth) of the forceOrder combined stream.
    pub fn futures_ws_url() -> String {
        "wss://fstream.binance.com".into()
    }

    pub fn end_date_max_hours() -> u64 {
        2
    }
    pub fn market_poll_interval_secs() -> u64 {
        30
    }
    pub fn resync_max_concurrent() -> usize {
        10
    }
    // >= max_sweep_secs + gtd_ttl_secs of post_close_sweep (200 today) + margin —
    // see the comment in config/general.toml.
    pub fn evict_grace_secs() -> i64 {
        300
    }
    // 24h — cheap to retain (minimal tuple), enough margin
    // over any legitimate late fill of the current session.
    pub fn clear_at_cache_ttl_secs() -> i64 {
        86_400
    }
    pub fn broadcast_throttle_ms() -> u64 {
        50
    }
    // 3000→500 — with decision windows of 45-90s and taker FOKs, a 3s-old book
    // signs orders at dead prices. 500ms covers the normal backlog of the
    // broadcast without accepting WS gaps.
    pub const fn max_book_age_ms() -> u64 {
        500
    }

    pub fn api_bind() -> String {
        "127.0.0.1".into()
    }
    pub fn api_port() -> u16 {
        8766
    }

    pub fn resync_on_reconnect() -> bool {
        true
    }
    pub fn resync_sweep_ms() -> u64 {
        2_000
    }
    pub fn resync_grace_ms() -> u64 {
        1_500
    }
    pub fn resync_batch_size() -> usize {
        400
    }
    pub fn resync_min_interval_ms() -> u64 {
        250
    }
    pub fn resync_cooldown_base_ms() -> u64 {
        5_000
    }
    pub fn resync_cooldown_max_ms() -> u64 {
        300_000
    }
    pub fn ws_ping_interval_secs() -> u64 {
        // Official market channel docs: "Send PING every 10 seconds to
        // keep the connection alive" — with 30s the server closed the session
        // every ~10-60s (~840 reconnects/6h). user_ws already uses 10s.
        10
    }
    pub fn ws_stale_threshold_secs() -> u64 {
        90
    }
    // 60s tape window — enough to see recent directional pressure
    // without diluting it in the noise of 5-min brackets.
    pub fn tape_window_secs() -> u64 {
        60
    }
    // Scan-only universes: batched REST poll every 10s — enough for the
    // observation tools (they only use the mid) without the cost of ~1,200 extra
    // tokens on the WS (venue throttle ∝ total subscribed tokens).
    pub fn scan_poll_interval_secs() -> u64 {
        10
    }
    pub fn scan_poll_batch_size() -> usize {
        200
    }
    // Predictive prefetch: the next bracket of each up/down series is
    // queried by direct slug every 2s — it comes out of the general 30s poll.
    pub fn prefetch_interval_secs() -> u64 {
        2
    }
    // Real taker fee per market via the CLOB's /fee-rate (public endpoint,
    // verified without auth). true by default — more precise than
    // the constants; disable only if the endpoint causes problems in production.
    pub fn fetch_taker_fee() -> bool {
        true
    }

    pub fn paper_disable_kill_switch() -> bool {
        true
    }
    pub fn paper_slippage_bps() -> u32 {
        5
    }
    pub fn paper_fok_fill_rate() -> f64 {
        0.75
    }
    pub fn paper_pair_fill_correlation() -> f64 {
        0.9
    }
    pub fn paper_latency_ms() -> u64 {
        150
    }
    // 0.07, base rate of the PARABOLIC formula (`rate × p × (1−p)`),
    // the contract's real one for crypto up/down — verified from three sources,
    // including the Gamma API `feeSchedule` (6/6 assets). The earlier value
    // (0.10) was the `MAX_FEE_RATE_BIPS` returned by the CLOB's `/fee-rate`
    // combined with the triangular formula `rate × min(p,1−p)` — a ceiling
    // authorized by the contract, not the fee actually charged; it overcharged
    // by up to 186% at p=0.50 and biased all historical paper PnL downward.
    pub fn paper_taker_fee_rate() -> Decimal {
        dec!(0.07)
    }
    pub fn paper_unknown_rate() -> f64 {
        0.0
    }
    pub fn paper_initial_deposit_usdc() -> rust_decimal::Decimal {
        rust_decimal_macros::dec!(1000)
    }

    // `[research_sim]` (research profile, separate from `risk.paper_*`).
    // NOMINAL defaults = parity with current operational behavior:
    // applying the nominal profile changes no sim result.
    pub fn research_sim_profile() -> String {
        "nominal".into()
    }
    pub fn research_sim_cancel_attr() -> Decimal {
        // κ=1: every shrink without crossing drains the queue ahead (status quo).
        dec!(1)
    }
    pub fn research_sim_participation() -> Decimal {
        dec!(1)
    }
    pub fn research_sim_participation_conservative() -> Decimal {
        // Conservative estimate (not a measurement): we capture at most a
        // quarter of the displayed flow/level.
        dec!(0.25)
    }
    pub fn auto_redeem() -> bool {
        true
    }

    pub fn fill_alerts() -> bool {
        true
    }
    pub fn periodic_summary_hours() -> u32 {
        4
    }

    pub fn max_orders_per_hour() -> u32 {
        20
    }
    pub fn max_order_usdc() -> Decimal {
        dec!(50)
    }
    pub fn daily_loss_usdc_stop() -> Decimal {
        dec!(100)
    }

    // NC
    pub fn nc_z_threshold() -> Decimal {
        dec!(2.5)
    }
    pub fn nc_window_secs() -> u64 {
        90
    }
    pub fn nc_min_ask() -> Decimal {
        dec!(0.75)
    }
    pub fn nc_max_ask() -> Decimal {
        dec!(0.93)
    }
    pub fn nc_min_book_imbalance() -> Decimal {
        dec!(1.0)
    }
    pub fn nc_bail_threshold_pct() -> Decimal {
        dec!(0.04)
    }
    pub fn nc_min_bail_bid() -> Decimal {
        dec!(0.10)
    } // do not sell in a dry market
    pub fn nc_bail_retry_secs() -> u64 {
        10
    }

    // NE
    pub fn ne_min_dist_pct() -> Decimal {
        dec!(0.003)
    }
    pub fn ne_window_secs() -> u64 {
        45
    }
    pub fn ne_min_lead_secs() -> u64 {
        5
    }
    pub fn ne_min_ask() -> Decimal {
        dec!(0.60)
    }
    pub fn ne_max_ask() -> Decimal {
        dec!(0.97)
    }
    pub fn ne_order_usdc() -> Decimal {
        dec!(25)
    }
    pub fn ne_bail_threshold_pct() -> Decimal {
        dec!(0.20)
    }
    pub fn ne_min_sigma() -> Decimal {
        dec!(0.0001)
    }
    pub fn ne_min_bail_bid() -> Decimal {
        dec!(0.10)
    } // do not sell in a dry market
    pub fn ne_bail_retry_secs() -> u64 {
        10
    }

    // ARB
    pub fn arb_fee_threshold() -> Decimal {
        dec!(0.02)
    }
    pub fn arb_min_edge() -> Decimal {
        dec!(0.005)
    }
    pub fn arb_max_size_shares() -> Decimal {
        dec!(10)
    }
    pub fn arb_sell_retry_secs() -> u64 {
        10
    }
    pub fn arb_orphan_confirm_secs() -> u64 {
        2
    }
    pub fn arb_orphan_stop_loss() -> Decimal {
        dec!(0.05)
    }
    // 1000 bps = 10%, verified on `/fee-rate` (previously 0.07, the docs' coefficient).
    pub fn arb_taker_fee_rate() -> Decimal {
        dec!(0.10)
    }
    pub fn arb_min_remaining_secs() -> i64 {
        60
    }
    pub fn arb_depth_safety_factor() -> Decimal {
        dec!(2)
    }
    pub fn arb_max_plausible_edge() -> Decimal {
        dec!(0.10)
    }

    // DIR
    pub fn dir_min_edge() -> Decimal {
        dec!(0.03)
    }
    pub fn dir_max_entry() -> Decimal {
        dec!(0.97)
    }
    pub fn dir_min_entry() -> Decimal {
        dec!(0.95)
    }
    pub fn dir_decision_window() -> u64 {
        60
    }
    pub fn dir_max_spread() -> Decimal {
        dec!(0.10)
    } // 10 cts max bid-ask

    // MM
    pub fn mm_spread_bps() -> u32 {
        30
    }
    pub fn mm_min_spread_bps() -> u32 {
        10
    }
    pub fn mm_max_spread_bps() -> u32 {
        100
    }
    pub fn mm_vol_factor() -> Decimal {
        dec!(1.5)
    }
    pub fn mm_order_size_shares() -> Decimal {
        dec!(5)
    }
    // g_rw1_maker
    pub fn grw1_ticket_shares() -> Decimal {
        dec!(50)
    }
    pub fn grw1_requote_threshold_bps() -> u32 {
        20
    }
    pub fn grw1_order_ttl_secs() -> u64 {
        5
    }
    pub fn mm_max_inventory_shares() -> Decimal {
        dec!(50)
    }
    pub fn mm_requote_threshold_bps() -> u32 {
        15
    }
    pub fn mm_order_ttl_secs() -> u64 {
        5
    }
    pub fn mm_min_unwind_bid() -> Decimal {
        dec!(0.05)
    }

    // PM
    pub fn pm_margin() -> Decimal {
        dec!(0.05)
    }
    pub fn pm_single_leg_unwind_secs() -> u64 {
        10
    }
    pub fn pm_min_unwind_bid() -> Decimal {
        dec!(0.05)
    }
    pub fn pm_min_leg_price() -> Decimal {
        dec!(0.05)
    }
    pub fn pm_order_size_shares() -> Decimal {
        dec!(3)
    }
    pub fn pm_requote_threshold_bps() -> u32 {
        20
    }
    pub fn pm_pair_ttl_secs() -> u64 {
        10
    }
    // Skew/no-quote/smart unwind by FV.
    pub fn pm_fv_skew_threshold() -> Decimal {
        dec!(0.08)
    }
    pub fn pm_fv_skew_bps() -> u32 {
        100
    }
    pub fn pm_fv_no_quote_threshold() -> Decimal {
        dec!(0.22)
    }
    pub fn pm_unwind_hold_fv() -> Decimal {
        dec!(0.60)
    }
    pub fn pm_unwind_dump_fv() -> Decimal {
        dec!(0.40)
    }
    pub fn pm_unwind_repost_min_secs() -> u64 {
        60
    }
    pub fn pm_unwind_maker_edge() -> Decimal {
        dec!(0.01)
    }
    pub fn pm_oracle_drift_secs() -> u64 {
        10
    }
    pub fn pm_oracle_drift_bps() -> u32 {
        5
    }
    pub fn pm_min_expected_move_bps() -> u32 {
        20
    }
    pub fn pm_min_spread() -> Decimal {
        dec!(0.02)
    }
    pub fn pm_max_queue_ahead_shares() -> Decimal {
        dec!(500)
    }
    pub fn pm_queue_imbalance_ratio() -> Decimal {
        dec!(5)
    }
    pub fn pm_xbracket_window_secs() -> u64 {
        5
    }
    pub fn pm_xbracket_move_bps() -> u32 {
        100
    }
    pub fn pm_xbracket_common_ratio() -> Decimal {
        dec!(0.5)
    }
    pub fn pm_chase_after_secs() -> u64 {
        3
    }
    pub fn pm_max_dumps_per_bracket() -> u32 {
        1
    }
    pub fn pm_bracket_history_len() -> usize {
        20
    }
    pub fn pm_min_duration_secs() -> u64 {
        900
    }

    // LadderArb
    pub fn ladder_min_edge() -> Decimal {
        dec!(0.008)
    }

    // DriftBox
    pub fn db_min_confidence() -> Decimal {
        dec!(0.08)
    } // minimum |fv-0.5| to pick a side
    pub fn db_max_entry_price() -> Decimal {
        dec!(0.58)
    } // A's maximum ask (leaves room for cash)
    pub fn db_min_box_profit() -> Decimal {
        dec!(0.03)
    } // minimum profit when completing the box
    pub fn db_bail_threshold_pct() -> Decimal {
        dec!(0.02)
    } // relative pullback for bail-sell (same as ARB oracle_bad)
    pub fn db_min_lead_secs() -> u64 {
        30
    } // min. remaining time to enter
    pub fn db_max_window_secs() -> u64 {
        240
    } // max. remaining time to enter
    pub fn db_bail_secs() -> u64 {
        30
    } // time-based bail if the position is at a loss

    // OpenMomentum
    pub fn om_max_open_secs() -> u64 {
        30
    } // opening window in seconds
    pub fn om_min_confidence() -> Decimal {
        dec!(0.08)
    } // minimum |fv-0.5| (same as DriftBox)
    pub fn om_max_entry_price() -> Decimal {
        dec!(0.60)
    } // if the market already corrected, skip
    pub fn om_min_remaining() -> u64 {
        120
    } // do not enter with < 2 min to close
    pub fn om_bail_sigma() -> Decimal {
        dec!(2.5)
    } // σ × √t against for bail
    pub fn om_bail_checks() -> u32 {
        3
    } // consecutive ticks to trigger bail

    // ThetaCapture
    pub fn tc_min_yes_price() -> Decimal {
        dec!(0.93)
    } // YES >= 93 cts → near-winning side
    pub fn tc_bid_discount() -> Decimal {
        dec!(0.015)
    } // bid = ask - 1.5 cts (better than NC taker)
    pub fn tc_max_window_secs() -> u64 {
        90
    } // last 90s (shorter than NC 90s, same window)
    pub fn tc_min_loser_ask() -> Decimal {
        dec!(0.04)
    } // minimum NO sell at 4 cts

    // VolSurface
    pub fn vs_vol_ratio_threshold() -> f64 {
        2.0
    } // σ_short/σ_long ratio to log a divergence
    pub fn vs_min_remaining_short() -> u64 {
        60
    } // short bracket: at least 60s so it has direction
    pub fn vs_max_remaining_short() -> u64 {
        600
    } // short bracket: at most 10 min (avoid almost-new ones)

    // PostCloseConfig
    pub fn pcs_sample_throttle_ms() -> u64 {
        1000
    } // <=1 sample per market every 1s

    // AnalyticsConfig (src/tools/recorders/analytics.rs)
    pub fn analytics_enabled() -> bool {
        true
    }
    pub fn analytics_db_path() -> String {
        "hrf_market_analytics.sqlite".to_string()
    }
    // 72h (3 days): with the measured growth of 4-8 GB/day (depth already
    // capped at 15 levels + hash dedup) it bounds the size in steady
    // state to ~12-24 GiB. See the `[analytics]` comment in general.toml.
    pub fn analytics_retention_hours() -> u64 {
        72
    }
    pub fn analytics_purge_interval_secs() -> u64 {
        600
    } // 10 min, own to the purge cycle
    pub fn analytics_purge_batch_size() -> u32 {
        5_000
    } // rows per DELETE batch

    // SpotStreamConfig (src/tools/recorders/spot_stream.rs)
    pub fn spot_stream_enabled() -> bool {
        true
    }
    pub fn spot_stream_db_path() -> String {
        "hrf_spot_stream.sqlite".to_string()
    }
    // 4,096 samples ≈ plenty of headroom: 6 assets × ~1 tick/s = ~6
    // samples/s; the channel absorbs ~10 min of a
    // stalled writer before it starts dropping.
    pub fn spot_stream_channel_capacity() -> usize {
        4_096
    }

    // ResearchEventsConfig (src/research/events.rs)
    pub fn research_events_dir() -> String {
        "data/research_events".to_string()
    }
    // 8,192 envelopes: same default as `ResearchRecorderConfig` — absorbs
    // dispatcher bursts without dropping; the excess is counted as `GAP`.
    pub fn research_events_channel_capacity() -> usize {
        8_192
    }
    pub fn research_events_dedup_capacity() -> usize {
        262_144
    }

    // ResolutionFetchConfig (src/tools/recorders/resolution.rs)
    pub fn resolution_fetcher_enabled() -> bool {
        true
    }
    // 2 min: an up/down bracket resolves in the post-close window; frequent
    // sweeps cost little (pending ones are few and resolved ones leave
    // the set as soon as they are confirmed).
    pub fn resolution_fetcher_poll_secs() -> u64 {
        120
    }
    pub fn resolution_fetcher_min_request_ms() -> u64 {
        400
    }
    pub fn resolution_fetcher_horizon_days() -> u64 {
        30
    }
    pub fn resolution_fetcher_batch_size() -> usize {
        50
    }

    // ModelShadowConfig (model_observer.rs)
    pub fn model_shadow_artifact_dir() -> String {
        "research/data/t8b_confirmed/model".to_string()
    }
    pub fn model_shadow_decisions_dir() -> String {
        "data/model_shadow".to_string()
    }
    /// 30s: an outcome signal in up/down brackets degrades in seconds
    /// (the price converges to the result); horizon >> expected latency.
    pub fn model_shadow_horizon_ms() -> u64 {
        30_000
    }
    /// 0.02 net edge over the ask: below the parabolic fee and the
    /// average spread there is nothing to evaluate as a decision.
    pub fn model_shadow_edge_min() -> Decimal {
        dec!(0.02)
    }
    pub fn model_shadow_shares_virtual() -> Decimal {
        dec!(1)
    }
    /// 4,096 lines ≈ minutes of headroom at a cadence bounded by
    /// `min_decision_interval_ms` — same criterion as spot_stream.
    pub fn model_shadow_channel_capacity() -> usize {
        4_096
    }
    /// 1s per market: same cadence as the analytics recorder — enough
    /// for the comparison with replay without multiplying disk usage.
    pub fn model_shadow_min_interval_ms() -> u64 {
        1_000
    }
    pub fn model_shadow_stats_interval_secs() -> u64 {
        60
    }

    // PostCloseSweep
    pub fn pcs2_min_rel_move() -> Decimal {
        dec!(0.001)
    } // deterministic verdict threshold
    pub fn pcs2_maker_bid_price() -> Decimal {
        dec!(0.99)
    } // maker GTD bid on the winning side
    pub fn pcs2_taker_max_ask() -> Decimal {
        dec!(0.98)
    } // maximum ask for the taker sweep
    pub fn pcs2_gtd_ttl_secs() -> u64 {
        90
    } // GTD TTL before cancelling
    pub fn pcs2_max_sweep_secs() -> u64 {
        110
    } // post-close window to trade
    pub fn pcs2_max_spot_age_secs() -> u64 {
        8
    } // maximum age of the frozen spot
    pub fn pcs2_exposure_carveout_usdc() -> Decimal {
        dec!(30)
    } // reserve over the global cap, post_close_sweep only

    // RewardFarm (scan-only)
    pub fn rf_scan_log_interval_secs() -> u64 {
        60
    } // 1 log per market per minute — enough to mine offline

    // ── Real posting ─────────────────────────────────────────────────────────
    pub fn rf_durations() -> Vec<String> {
        vec!["daily".into(), "weekly".into()]
    }
    pub fn rf_max_markets() -> u32 {
        5
    }
    pub fn rf_mid_min() -> Decimal {
        dec!(0.10)
    }
    pub fn rf_mid_max() -> Decimal {
        dec!(0.90)
    }
    pub fn rf_edge_buffer_cents() -> Decimal {
        dec!(0.5)
    }
    pub fn rf_requote_threshold_cents() -> Decimal {
        dec!(1.0)
    }
    pub fn rf_queue_behind() -> bool {
        true
    }
    pub fn rf_single_level() -> bool {
        true
    }
    pub fn rf_size_mode() -> String {
        "min_size".into()
    }
    pub fn rf_max_total_usdc() -> Decimal {
        dec!(300)
    }
    pub fn rf_require_no_pm_mm() -> bool {
        true
    }
    pub fn rf_cvd_toxic() -> Decimal {
        dec!(0.35)
    }
    pub fn rf_vol_ratio_pause() -> Decimal {
        dec!(2.0)
    }
    pub fn rf_max_fills_per_bracket() -> u32 {
        1
    }
    pub fn rf_unwind_hold_fv() -> Decimal {
        dec!(0.85)
    }
    pub fn rf_unwind_dump_fv() -> Decimal {
        dec!(0.55)
    }
    pub fn rf_repost_min_secs() -> u64 {
        300
    }
    pub fn rf_min_secs_to_clear() -> i64 {
        3600
    }

    // Wallet follower
    pub fn follower_target_wallet() -> String {
        "0x00000000000000000000000000000000000000a1".into()
    } // target wallet to follow
    pub fn follower_poll_interval_secs() -> u64 {
        5
    }
    pub fn follower_size_usdc() -> Decimal {
        dec!(10)
    }
    pub fn follower_max_slippage_pct() -> Decimal {
        dec!(0.02)
    }
    pub fn follower_mirror_exits() -> bool {
        true
    }

    // Scan universes
    // Previously this value stopped Gamma PAGINATION (with
    // max_markets=50 < PAGE=100, the cut was always met after the first
    // page). Now `discover::fetch_gamma_page` paginates until the date window
    // is exhausted and this number only limits how many NEW brackets are registered
    // per universe in a single poll cycle (post-filter cap) — the
    // default is raised so as not to artificially defer the discovery of a universe
    // with many pending markets (e.g. after a cold startup).
    pub fn universe_max_markets() -> u32 {
        500
    }
    pub fn default_universe() -> Vec<String> {
        vec!["crypto-updown-short".into(), "crypto-updown-long".into()]
    }
}

// ── Wallet identity ───────────────────────────────────────────────────────────

/// Wallet credentials extracted from the environment to bootstrap the CLOB.
pub struct WalletIdentity {
    pub private_key: String,
    pub signature_type: u8,
    pub funder_address: Option<String>,
}

impl Config {
    /// Reads the wallet credentials from the environment.
    ///
    /// - `POLY_PRIVATE_KEY`: hex private key (required in live mode).
    /// - `POLY_SIGNATURE_TYPE`: 0=EOA (default), 1=Proxy, 2=GnosisSafe, 3=Poly1271.
    /// - `POLY_PROXY_ADDRESS`: funder address (only for Proxy/GnosisSafe).
    pub fn resolve_wallet_identity(&self) -> anyhow::Result<WalletIdentity> {
        let private_key = std::env::var("POLY_PRIVATE_KEY")
            .map_err(|_| anyhow::anyhow!("POLY_PRIVATE_KEY is not set"))?;
        anyhow::ensure!(
            !private_key.trim().is_empty(),
            "POLY_PRIVATE_KEY is empty"
        );

        let signature_type: u8 = std::env::var("POLY_SIGNATURE_TYPE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        let funder_address = std::env::var("POLY_PROXY_ADDRESS")
            .ok()
            .filter(|v| !v.is_empty());

        Ok(WalletIdentity {
            private_key,
            signature_type,
            funder_address,
        })
    }
}

// ── Strategy toggle persistence ─────────────────────────────────────────────

/// Enables or disables a strategy in its TOML configuration file,
/// preserving comments and the rest of the content (`toml_edit`).
///
/// Write-to-tmp + atomic rename so there are no truncated files if the process
/// dies mid-write.
pub fn persist_strategy_toggle(strategy: &str, enabled: bool) -> anyhow::Result<()> {
    use toml_edit::{DocumentMut, Item, Value};

    let path = format!("config/strategies/{strategy}.toml");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| format!("[{strategy}]\nenabled = false\n"));

    let mut doc = content
        .parse::<DocumentMut>()
        .map_err(|e| anyhow::anyhow!("error parsing {path}: {e}"))?;

    doc[strategy]["enabled"] = Item::Value(Value::from(enabled));

    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, doc.to_string())?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Persists a strategy's operational limits in its TOML (write-tmp + atomic rename).
/// Only writes the `Some` fields; the rest is left intact.
pub fn persist_strategy_limits(
    strategy: &str,
    max_order_usdc: Option<&str>,
    max_orders_per_hour: Option<u32>,
    daily_loss_usdc_stop: Option<&str>,
) -> anyhow::Result<()> {
    use toml_edit::{DocumentMut, Item, Value};

    let path = format!("config/strategies/{strategy}.toml");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| format!("[{strategy}]\nenabled = false\n"));

    let mut doc = content
        .parse::<DocumentMut>()
        .map_err(|e| anyhow::anyhow!("error parsing {path}: {e}"))?;

    if let Some(v) = max_order_usdc {
        doc[strategy]["max_order_usdc"] = Item::Value(Value::from(v));
    }
    if let Some(v) = max_orders_per_hour {
        doc[strategy]["max_orders_per_hour"] = Item::Value(Value::from(v as i64));
    }
    if let Some(v) = daily_loss_usdc_stop {
        doc[strategy]["daily_loss_usdc_stop"] = Item::Value(Value::from(v));
    }

    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, doc.to_string())?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Validates and parses the global exposure limit from text (HTTP/Telegram).
/// Rules: must parse to `Decimal`, be > 0 and <= 1_000_000.
pub fn parse_exposure_limit(s: &str) -> Result<Decimal, String> {
    let v: Decimal = s
        .trim()
        .parse()
        .map_err(|_| format!("invalid value: '{s}'"))?;
    if v <= Decimal::ZERO {
        return Err("the limit must be greater than 0".into());
    }
    if v > Decimal::from(1_000_000) {
        return Err("the limit cannot exceed 1,000,000".into());
    }
    Ok(v)
}

/// Persists `risk.max_total_exposure_usdc` in `config/general.toml` (write-tmp +
/// atomic rename, preserving comments and the rest of the content via `toml_edit`).
pub fn persist_max_total_exposure(value: &str) -> anyhow::Result<()> {
    use toml_edit::{DocumentMut, Item, Value};

    let path = "config/general.toml";
    let content = std::fs::read_to_string(path)?;

    let mut doc = content
        .parse::<DocumentMut>()
        .map_err(|e| anyhow::anyhow!("error parsing {path}: {e}"))?;

    doc["risk"]["max_total_exposure_usdc"] = Item::Value(Value::from(value));

    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, doc.to_string())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// `true` if `HRF_RISK__MAX_TOTAL_EXPOSURE_USDC` is present in the process
/// environment — a bot restart would mask the value persisted in the TOML.
/// Does this tool's `enabled` come from an environment variable
/// `HRF_<TOOL>__ENABLED` instead of its `config/strategies/<tool>.toml`? (the key Figment
/// consumes is `HRF_<TOOL>__ENABLED`; both spellings are checked).
fn tool_enabled_from_env(id: crate::tools::ToolId) -> bool {
    let name = id.to_string();
    std::env::var(format!("HRF_{}__ENABLED", name.to_uppercase())).is_ok()
        || std::env::var(format!("hrf_{name}__enabled")).is_ok()
}

/// Each tool's `enabled` lives in `config/strategies/<tool>.toml`
/// — that is where the hot toggle (API/Telegram) persists it. An
/// `HRF_<TOOL>__ENABLED` environment variable would silently override it (Figment gives
/// the env priority over all the TOMLs) and startup would ignore the
/// persisted state without a trace. It is not forbidden — the env is the
/// legitimate override of the whole config — a warning goes in the startup log. Call
/// from `main` AFTER `tracing_subscriber::init` (a `warn!` emitted
/// before registering the subscriber is lost).
pub fn warn_tool_enabled_env_overrides() {
    let with_env: Vec<String> = crate::tools::ToolId::ALL
        .iter()
        .filter(|id| tool_enabled_from_env(**id))
        .map(|id| id.to_string())
        .collect();
    if !with_env.is_empty() {
        tracing::warn!(
            "config: {} has `enabled` overridden by ENVIRONMENT VARIABLE \
             (HRF_<TOOL>__ENABLED) — the value in config/strategies/<tool>.toml \
             is ignored while the variable exists",
            with_env.join(", ")
        );
    }
}

pub fn exposure_env_override_present() -> bool {
    std::env::var("HRF_RISK__MAX_TOTAL_EXPOSURE_USDC").is_ok()
}

/// Minimal valid config for tests: a single universe `[universes.foo]` with
/// configurable `has_oracle` and no strategy enabled (so the rest
/// of `validate()` does not interfere). `pub(crate)` (not just private to `mod
/// tests`) so other test modules — e.g.
/// `strategy::supervisor::tests` — can reuse it instead of duplicating a
/// full `Config` by hand.
#[cfg(test)]
pub(crate) fn base_config_with_universe(has_oracle: bool) -> Config {
    use rust_decimal_macros::dec;
    let mut universes = HashMap::new();
    universes.insert(
        "foo".to_string(),
        UniverseConfig {
            discovery: "slug".into(),
            slug_patterns: vec!["btc-updown".into()],
            tag_slugs: Vec::new(),
            end_date_min_hours: 0,
            end_date_max_hours: 24,
            duration_filter: Vec::new(),
            has_oracle,
            scan: false,
            max_markets: defaults::universe_max_markets(),
        },
    );
    Config {
        mode: crate::types::Mode::Paper,
        db_path: ":memory:".into(),
        clob_api_url: defaults::clob_api_url(),
        gamma_api_url: defaults::gamma_api_url(),
        oracle: OracleConfig {
            ws_url: defaults::oracle_ws_url(),
            stale_secs: defaults::oracle_stale_secs(),
            ping_interval_secs: defaults::oracle_ping_interval_secs(),
            max_jump_ratio: defaults::oracle_max_jump_ratio(),
            min_session_secs_for_backoff_reset: defaults::min_session_secs(),
            futures_liq_enabled: false,
            futures_ws_url: defaults::futures_ws_url(),
        },
        market_feed: MarketFeedConfig {
            bracket_slug_patterns: vec!["btc-updown".into()],
            end_date_max_hours: defaults::end_date_max_hours(),
            market_poll_interval_secs: defaults::market_poll_interval_secs(),
            resync_max_concurrent: defaults::resync_max_concurrent(),
            evict_grace_secs: defaults::evict_grace_secs(),
            clear_at_cache_ttl_secs: defaults::clear_at_cache_ttl_secs(),
            broadcast_throttle_ms: defaults::broadcast_throttle_ms(),
            max_book_age_ms: defaults::max_book_age_ms(),
            resync_on_reconnect: defaults::resync_on_reconnect(),
            resync_sweep_ms: defaults::resync_sweep_ms(),
            resync_grace_ms: defaults::resync_grace_ms(),
            resync_batch_size: defaults::resync_batch_size(),
            resync_min_interval_ms: defaults::resync_min_interval_ms(),
            resync_cooldown_base_ms: defaults::resync_cooldown_base_ms(),
            resync_cooldown_max_ms: defaults::resync_cooldown_max_ms(),
            ws_ping_interval_secs: defaults::ws_ping_interval_secs(),
            ws_stale_threshold_secs: defaults::ws_stale_threshold_secs(),
            fetch_taker_fee: false,
            tape_window_secs: 0,
            scan_poll_interval_secs: defaults::scan_poll_interval_secs(),
            scan_poll_batch_size: defaults::scan_poll_batch_size(),
            prefetch_interval_secs: defaults::prefetch_interval_secs(),
        },
        risk: RiskConfig {
            max_total_exposure_usdc: dec!(20),
            ..RiskConfig::default()
        },
        api: ApiConfig {
            bind: defaults::api_bind(),
            port: defaults::api_port(),
            token: String::new(),
        },
        telegram: TelegramConfig::default(),
        universes,
        fair_value: FairValueConfig::default(),
        ladder_arb: LadderArbConfig::default(),
        vol_surface: VolSurfaceConfig::default(),
        post_close: PostCloseConfig::default(),
        analytics: AnalyticsConfig::default(),
        spot_stream: SpotStreamConfig::default(),
        redeem: RedeemConfig::default(),
        research_sim: ResearchSimConfig::default(),
        research_events: ResearchEventsConfig::default(),
        resolution_fetcher: ResolutionFetchConfig::default(),
        model_shadow: ModelShadowConfig::default(),
        model_tool: crate::tools::model::ModelToolConfig::default(),
        strategies: crate::strategy::StrategyConfigs {
            near_certain: crate::strategy::near_certain::NearCertainConfig::default(),
            near_expiry: crate::strategy::near_expiry::NearExpiryConfig::default(),
            arb: crate::strategy::arb::ArbConfig::default(),
            arb_15m5m: crate::strategy::arb_15m5m::Arb15m5mConfig::default(),
            directional: crate::strategy::directional::DirectionalConfig::default(),
            market_making: crate::strategy::market_making::MarketMakingConfig::default(),
            parity_maker: crate::strategy::parity_maker::ParityMakerConfig::default(),
            drift_box: crate::strategy::drift_box::DriftBoxConfig::default(),
            open_momentum: crate::strategy::open_momentum::OpenMomentumConfig::default(),
            theta_capture: crate::strategy::theta_capture::ThetaCaptureConfig::default(),
            post_close_sweep: crate::strategy::post_close_sweep::PostCloseSweepConfig::default(),
            reward_farm: crate::strategy::reward_farm::RewardFarmConfig::default(),
            g_rw1_maker: crate::strategy::g_rw1_maker::GRw1MakerConfig::default(),
            model_btc5m: crate::strategy::model_btc5m::ModelBtc5mConfig::default(),
            rl_predictor: crate::strategy::rl_predictor::RlPredictorConfig::default(),
        },
        follower: crate::strategy::follower::FollowerConfig::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Used by the validate_strategy_ranges tests further
    // down (base_config_with_universe already brings its own local `use` for its
    // own body — this one is separate, at module level, for the rest of the tests).
    use rust_decimal_macros::dec;

    #[test]
    fn parse_exposure_limit_valid() {
        assert_eq!(parse_exposure_limit("20").unwrap(), Decimal::from(20));
    }

    #[test]
    fn parse_exposure_limit_rejects_zero() {
        assert!(parse_exposure_limit("0").is_err());
    }

    #[test]
    fn parse_exposure_limit_rejects_negative() {
        assert!(parse_exposure_limit("-5").is_err());
    }

    #[test]
    fn parse_exposure_limit_rejects_non_numeric() {
        assert!(parse_exposure_limit("abc").is_err());
    }

    #[test]
    fn parse_exposure_limit_rejects_excessive() {
        assert!(parse_exposure_limit("2000000").is_err());
    }

    #[test]
    fn strategy_limits_default_matches_serde() {
        let from_serde: StrategyLimits = serde_json::from_str("{}").unwrap();
        let d = StrategyLimits::default();
        assert_eq!(
            d.max_orders_per_hour, from_serde.max_orders_per_hour,
            "max_orders_per_hour out of sync between Default and the serde impl"
        );
        assert_eq!(
            d.max_order_usdc, from_serde.max_order_usdc,
            "max_order_usdc out of sync between Default and the serde impl"
        );
        assert_eq!(
            d.daily_loss_usdc_stop, from_serde.daily_loss_usdc_stop,
            "daily_loss_usdc_stop out of sync between Default and the serde impl"
        );
        assert_eq!(
            d.universe, from_serde.universe,
            "universe out of sync between Default and the serde impl"
        );
    }

    // ── Scan universes ─────────────────────────────────────────────────────

    #[test]
    fn validate_universes_rejects_nonexistent_universe() {
        let mut cfg = base_config_with_universe(true);
        cfg.strategies.arb.limits.enabled = true;
        cfg.strategies.arb.limits.universe = vec!["no-exists".into()];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("no-exists"), "error: {err}");
    }

    #[test]
    fn validate_universes_rejects_oracle_driven_without_oracle() {
        // NearCertain is oracle-driven (requires_oracle=true by default) — it
        // cannot declare a universe with has_oracle=false.
        let mut cfg = base_config_with_universe(false);
        cfg.strategies.near_certain.limits.enabled = true;
        cfg.strategies.near_certain.limits.universe = vec!["foo".into()];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("near_certain"), "error: {err}");
    }

    #[test]
    fn validate_universes_accepts_agnostic_without_oracle() {
        // ARB is agnostic (strategy_requires_oracle=false) — it can declare
        // a universe without an oracle without validation failing.
        let mut cfg = base_config_with_universe(false);
        cfg.strategies.arb.limits.enabled = true;
        cfg.strategies.arb.limits.universe = vec!["foo".into()];
        cfg.validate().unwrap();
    }

    #[test]
    fn validate_universes_disabled_strategy_is_not_validated() {
        // enabled=false: even if it references a nonexistent universe, it does not block startup.
        let mut cfg = base_config_with_universe(true);
        cfg.strategies.arb.limits.enabled = false;
        cfg.strategies.arb.limits.universe = vec!["no-exists".into()];
        cfg.validate().unwrap();
    }

    // ── validate_strategy_ranges ────────────────────────────────────────────

    #[test]
    fn validate_rejects_min_spread_bps_greater_than_max() {
        let mut cfg = base_config_with_universe(true);
        cfg.strategies.market_making.limits.enabled = true;
        cfg.strategies.market_making.limits.universe = vec!["foo".into()];
        cfg.strategies.market_making.min_spread_bps = 300;
        cfg.strategies.market_making.max_spread_bps = 50;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("min_spread_bps"), "error: {err}");
    }

    #[test]
    fn validate_rejects_min_ask_greater_than_max_ask_near_certain() {
        let mut cfg = base_config_with_universe(true);
        cfg.strategies.near_certain.limits.enabled = true;
        cfg.strategies.near_certain.limits.universe = vec!["foo".into()];
        cfg.strategies.near_certain.min_ask = dec!(0.90);
        cfg.strategies.near_certain.max_ask = dec!(0.80);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("near_certain.min_ask"),
            "error: {err}"
        );
    }

    #[test]
    fn validate_ignores_min_max_pairs_of_disabled_strategy() {
        // enabled=false: even if min > max, it does not block startup (stale
        // config of a switched-off strategy — same criterion as universes).
        let mut cfg = base_config_with_universe(true);
        cfg.strategies.market_making.limits.enabled = false;
        cfg.strategies.market_making.min_spread_bps = 999;
        cfg.strategies.market_making.max_spread_bps = 1;
        cfg.validate().unwrap();
    }

    #[test]
    fn validate_rejects_negative_exposure() {
        let mut cfg = base_config_with_universe(true);
        cfg.risk.max_total_exposure_usdc = dec!(-5);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("max_total_exposure_usdc"),
            "error: {err}"
        );
    }

    #[test]
    fn validate_rejects_negative_daily_loss_stop_even_if_disabled() {
        // Common StrategyLimits are always validated, whether or not enabled=true.
        let mut cfg = base_config_with_universe(true);
        cfg.strategies.arb.limits.enabled = false;
        cfg.strategies.arb.limits.daily_loss_usdc_stop = dec!(-1);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("daily_loss_usdc_stop"),
            "error: {err}"
        );
    }

    #[test]
    fn validate_accepts_default_config() {
        // The base config (all strategies off, default values)
        // must still start without errors after adding validate_strategy_ranges.
        let cfg = base_config_with_universe(true);
        cfg.validate().unwrap();
    }

    // live with auto_redeem=false requires [redeem].rpc_url — without an RPC
    // the EOA winners would be unredeemable. A single test for the three cases:
    // touching POLY_PRIVATE_KEY is global and the tests run in parallel.
    #[test]
    fn validate_live_auto_redeem_requires_rpc_only_without_auto_redeem() {
        let prev = std::env::var("POLY_PRIVATE_KEY").ok();
        std::env::set_var("POLY_PRIVATE_KEY", "0xtest");
        let mut cfg = base_config_with_universe(true);
        cfg.mode = Mode::Live;
        // A valid live config keeps theta_capture paper_only (unmodeled shorts).
        cfg.strategies.theta_capture.limits.paper_only = true;

        cfg.risk.auto_redeem = false;
        cfg.redeem.rpc_url = String::new();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("rpc_url"), "error: {err}");

        cfg.redeem.rpc_url = "https://polygon.example/rpc".to_string();
        cfg.validate().unwrap();

        cfg.risk.auto_redeem = true;
        cfg.redeem.rpc_url = String::new();
        cfg.validate().unwrap();

        match prev {
            Some(v) => std::env::set_var("POLY_PRIVATE_KEY", v),
            None => std::env::remove_var("POLY_PRIVATE_KEY"),
        }
    }

    #[test]
    fn strategy_requires_oracle_table_2_5() {
        use crate::strategy::registry::strategy_requires_oracle;
        assert!(!strategy_requires_oracle(StrategyId::RewardFarm));
        assert!(!strategy_requires_oracle(StrategyId::Arb));
        assert!(!strategy_requires_oracle(StrategyId::ParityMaker));
        assert!(strategy_requires_oracle(StrategyId::NearCertain));
        assert!(strategy_requires_oracle(StrategyId::NearExpiry));
        assert!(strategy_requires_oracle(StrategyId::Directional));
        assert!(strategy_requires_oracle(StrategyId::DriftBox));
        assert!(strategy_requires_oracle(StrategyId::OpenMomentum));
        assert!(strategy_requires_oracle(StrategyId::ThetaCapture));
        assert!(!strategy_requires_oracle(StrategyId::Follower));
        assert!(!strategy_requires_oracle(StrategyId::GRw1Maker));
    }

    /// BACKWARD COMPAT: a TOML without `[universes]` synthesizes the two crypto up/down
    /// from `market_feed` — the bot must still start with old configs.
    #[test]
    fn synthesize_default_universes_fills_from_market_feed() {
        let mut cfg = base_config_with_universe(true);
        cfg.universes.clear();
        cfg.market_feed.bracket_slug_patterns = vec!["btc-updown".into(), "eth-updown".into()];
        cfg.market_feed.end_date_max_hours = 24;
        cfg.synthesize_default_universes();
        assert_eq!(cfg.universes.len(), 2);
        for name in ["crypto-updown-short", "crypto-updown-long"] {
            let uni = cfg
                .universes
                .get(name)
                .unwrap_or_else(|| panic!("missing universe {name}"));
            assert_eq!(uni.discovery, "slug");
            assert_eq!(
                uni.slug_patterns,
                vec!["btc-updown".to_string(), "eth-updown".to_string()]
            );
            assert_eq!(uni.end_date_max_hours, 24);
            assert!(uni.has_oracle);
            assert!(!uni.scan);
        }
    }

    /// If `[universes]` ALREADY has content (new TOML), the synthesis does not overwrite it.
    #[test]
    fn synthesize_default_universes_does_not_overwrite_existing_config() {
        let mut cfg = base_config_with_universe(true);
        cfg.synthesize_default_universes();
        assert_eq!(
            cfg.universes.len(),
            1,
            "must not synthesize if [universes] already has content"
        );
        assert!(cfg.universes.contains_key("foo"));
    }

    #[test]
    fn active_universe_names_includes_scan_true_even_if_no_strategy_declares_it() {
        let mut cfg = base_config_with_universe(true);
        cfg.universes.get_mut("foo").unwrap().scan = true;
        // No enabled=true strategy declares "foo" nor any other universe.
        let set = cfg.active_universe_names();
        assert!(
            set.contains("foo"),
            "a scan=true universe must be active even if nobody declares it"
        );
    }

    /// Regression of the real `[universes.*]` block of `config/general.toml`:
    /// it must parse and pass `validate()` exactly
    /// as it is in the repo — the 4 universes of the spec, with
    /// crypto-multistrike/rewards-longdated at `scan=true`.
    #[test]
    fn repo_config_general_toml_loads_and_validates() {
        let cfg = Config::load().expect("the repo's config/ must load without errors");
        assert_eq!(cfg.universes.len(), 5);
        for name in [
            "crypto-updown-short",
            "crypto-updown-long",
            "crypto-multistrike",
            "rewards-longdated",
            "g-rw1",
        ] {
            assert!(
                cfg.universes.contains_key(name),
                "universe {name} is missing in config/general.toml"
            );
        }
        assert!(cfg.universes["crypto-multistrike"].scan);
        assert!(cfg.universes["rewards-longdated"].scan);
        assert!(!cfg.universes["crypto-updown-short"].scan);
        assert!(!cfg.universes["crypto-updown-long"].scan);
        assert!(!cfg.universes["rewards-longdated"].has_oracle);

        // If `#[serde(flatten)]` ever stopped routing `[arb]` to
        // `StrategyConfigs`, `cfg.strategies.arb` would silently fall back to
        // `ArbConfig::default()` and the assertions above (which do not touch
        // `strategies`) would stay green. These verify real values from
        // the TOMLs (not defaults): `max_order_usdc` of arb.toml (10, the default
        // is 50) and of arb_15m5m.toml (25).
        // ALL strategies are `enabled = false` on purpose,
        // so `enabled` is no longer usable to test the flatten.
        assert!(
            !cfg.strategies.arb.limits.enabled,
            "arb.toml has enabled=false since 2026-10-03"
        );
        assert_eq!(
            cfg.strategies.arb.limits.max_order_usdc,
            dec!(10),
            "real value from arb.toml, not the default"
        );
        let arb15 = &cfg.strategies.arb_15m5m;
        assert!(!arb15.limits.enabled, "arb_15m5m.toml: enabled=false");
        assert_eq!(arb15.limits.max_order_usdc, dec!(25), "real value from arb_15m5m.toml");

        // The repo TOML must keep `paper_only = true` on
        // theta_capture — without it, `validate_theta_capture_blocked_live` is the
        // only barrier between TC-live and its unmodeled naked shorts.
        assert!(
            cfg.strategies.theta_capture.limits.paper_only,
            "theta_capture.toml must have paper_only=true (resting sells = unmodeled shorts)"
        );
    }

    #[test]
    fn active_universe_names_includes_those_declared_by_enabled_strategies() {
        let mut cfg = base_config_with_universe(true);
        cfg.strategies.arb.limits.enabled = true;
        cfg.strategies.arb.limits.universe = vec!["foo".into()];
        let set = cfg.active_universe_names();
        assert!(set.contains("foo"));
    }
}

#[cfg(test)]
mod flatten_smoke_test {
    use super::*;
    use figment::providers::{Format, Toml};

    #[derive(Debug, Clone, Deserialize, Default)]
    struct FakeAConfig {
        #[serde(default)]
        pub x: i64,
    }

    fn fake_b_default_y() -> String {
        "default".into()
    }

    #[derive(Debug, Clone, Deserialize)]
    struct FakeBConfig {
        #[serde(default = "fake_b_default_y")]
        pub y: String,
    }

    impl Default for FakeBConfig {
        fn default() -> Self {
            Self {
                y: fake_b_default_y(),
            }
        }
    }

    #[derive(Debug, Clone, Deserialize, Default)]
    struct FakeStrategyConfigs {
        #[serde(default)]
        pub a: FakeAConfig,
        #[serde(default)]
        pub b: FakeBConfig,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct FakeCore {
        pub core_field: String,
        #[serde(flatten)]
        pub strategies: FakeStrategyConfigs,
    }

    /// Technical risk: `#[serde(flatten)]` over a
    /// struct composed of sub-structs must (a) keep the TOML layout of
    /// top-level tables (`[a]`, `[b]`, like `[arb]`/
    /// `[near_certain]` today) and (b) keep applying the field defaults
    /// (`#[serde(default = "fn")]`) when an entire table is absent.
    /// If this test FAILS: do not continue — the fallback
    /// is to generate the named fields in `Config`
    /// without flatten.
    #[test]
    fn flatten_preserves_top_level_tables_and_defaults() {
        let toml = r#"
            core_field = "hello"

            [a]
            x = 42
        "#;
        // "b" is entirely absent from the document — it must fall back to
        // FakeBConfig::default(), which in turn uses fake_b_default_y().
        let cfg: FakeCore = Figment::new()
            .merge(Toml::string(toml))
            .extract()
            .expect("flatten + absent table must deserialize with defaults");

        assert_eq!(cfg.core_field, "hello");
        assert_eq!(cfg.strategies.a.x, 42);
        assert_eq!(cfg.strategies.b.y, "default");
    }

    // ── Per-strategy run barrier (enabled / paper_only) ──────────────────────

    /// Test limits with only the two barrier levers set.
    fn limits(enabled: bool, paper_only: bool) -> StrategyLimits {
        StrategyLimits {
            enabled,
            paper_only,
            ..StrategyLimits::default()
        }
    }

    /// A TOML that does not declare `paper_only` leaves it at false.
    #[test]
    fn paper_only_absent_in_toml_deserializes_to_false() {
        #[derive(Debug, Deserialize)]
        struct Wrapper {
            #[serde(flatten)]
            limits: StrategyLimits,
        }
        let cfg: Wrapper = Figment::new()
            .merge(Toml::string("enabled = true\n"))
            .extract()
            .expect("a TOML without paper_only must still deserialize");
        assert!(cfg.limits.enabled);
        assert!(
            !cfg.limits.paper_only,
            "absent paper_only must fall back to false"
        );
    }

    #[test]
    fn enabled_runs_in_both_modes() {
        assert!(limits(true, false).can_run(Mode::Paper).is_ok());
        assert!(limits(true, false).can_run(Mode::Live).is_ok());
    }

    #[test]
    fn paper_only_blocks_in_live() {
        assert_eq!(
            limits(true, true).can_run(Mode::Live),
            Err(BlockReason::PaperOnly)
        );
    }

    #[test]
    fn paper_only_does_not_obstruct_in_paper() {
        assert!(limits(true, true).can_run(Mode::Paper).is_ok());
    }

    #[test]
    fn disabled_blocks_in_both_modes() {
        assert_eq!(
            limits(false, false).can_run(Mode::Paper),
            Err(BlockReason::Disabled)
        );
        assert_eq!(
            limits(false, false).can_run(Mode::Live),
            Err(BlockReason::Disabled)
        );
    }

    #[test]
    fn mode_conflict_ending_in_live_aborts_startup() {
        let err = check_mode_conflict(DeclaredMode::Declared(Mode::Paper), Mode::Live)
            .expect_err("effective live with TOML on paper must abort");
        assert!(
            err.to_string().contains("general.toml"),
            "the message must explain how to resolve it: {err}"
        );
    }

    #[test]
    fn mode_conflict_ending_in_paper_does_not_abort() {
        // Falling to paper by accident is safe: the ERROR log is enough.
        assert!(check_mode_conflict(DeclaredMode::Declared(Mode::Live), Mode::Paper).is_ok());
    }

    #[test]
    fn without_mode_conflict_there_is_no_error() {
        assert!(check_mode_conflict(DeclaredMode::Declared(Mode::Live), Mode::Live).is_ok());
        assert!(check_mode_conflict(DeclaredMode::Declared(Mode::Paper), Mode::Paper).is_ok());
        assert!(check_mode_conflict(DeclaredMode::Absent, Mode::Live).is_ok());
    }

    /// A `mode` that is present but unreadable (typo: `mode = "papr"`) is NOT the
    /// same as not declaring it. With `HRF_MODE=live` the effective mode is live and the
    /// TOML cannot contradict it, so it aborts.
    #[test]
    fn unreadable_mode_in_toml_with_effective_live_aborts() {
        let err = check_mode_conflict(DeclaredMode::Unreadable, Mode::Live)
            .expect_err("a corrupt mode cannot authorize live silently");
        assert!(
            err.to_string().contains("general.toml"),
            "the message must explain how to resolve it: {err}"
        );
    }

    #[test]
    fn unreadable_mode_in_toml_with_effective_paper_does_not_abort() {
        // Same asymmetric criterion: ending in paper is safe, the log is enough.
        assert!(check_mode_conflict(DeclaredMode::Unreadable, Mode::Paper).is_ok());
    }

    /// The `load()` classifier: distinguishes the three states without reading disk.
    #[test]
    fn classify_declared_mode_distinguishes_absent_from_unreadable() {
        assert_eq!(
            classify_declared_mode(&Figment::new()),
            DeclaredMode::Absent
        );
        assert_eq!(
            classify_declared_mode(&Figment::new().merge(Toml::string("mode = \"live\"\n"))),
            DeclaredMode::Declared(Mode::Live)
        );
        assert_eq!(
            classify_declared_mode(&Figment::new().merge(Toml::string("mode = \"papr\"\n"))),
            DeclaredMode::Unreadable
        );
        assert_eq!(
            classify_declared_mode(&Figment::new().merge(Toml::string("mode = 7\n"))),
            DeclaredMode::Unreadable
        );
    }

    /// `validate_live_loss_stop` is tested and not the whole `validate()` because
    /// the latter requires `POLY_PRIVATE_KEY` in the environment when the mode is
    /// live, and manipulating environment variables from a test affects the process
    /// as a whole (tests run in parallel).
    #[test]
    fn loss_stop_at_zero_in_live_with_live_capable_strategy_fails() {
        let mut cfg = base_config_with_universe(true);
        cfg.mode = Mode::Live;
        cfg.strategies.arb.limits.enabled = true;
        cfg.strategies.arb.limits.daily_loss_usdc_stop = Decimal::ZERO;
        let err = cfg
            .validate_live_loss_stop()
            .expect_err("stop at 0 in live must abort");
        assert!(
            err.to_string().contains("daily_loss_usdc_stop"),
            "the message must name the field: {err}"
        );
    }

    /// A disabled strategy with the stop at 0 does not block live startup, but
    /// it cannot be turned on later: `can_run` refuses it in live
    /// (`BlockReason::NoLossStop`), so an API/Telegram toggle cannot start it
    /// trading with real money and no daily stop.
    #[test]
    fn stop_at_zero_in_live_with_disabled_strategy_blocks_the_toggle_not_startup() {
        let mut cfg = base_config_with_universe(true);
        cfg.mode = Mode::Live;
        cfg.strategies.arb.limits.enabled = false;
        cfg.strategies.arb.limits.daily_loss_usdc_stop = Decimal::ZERO;
        cfg.validate_live_loss_stop()
            .expect("a disabled strategy must not abort live startup");
        let mut toggled = cfg.strategies.arb.limits.clone();
        toggled.enabled = true;
        assert_eq!(toggled.can_run(Mode::Live), Err(BlockReason::NoLossStop));
    }

    /// `paper_only` can exempt: that strategy will never trade in live,
    /// regardless of its other flags.
    #[test]
    fn stop_at_zero_in_live_with_paper_only_does_not_block() {
        let mut cfg = base_config_with_universe(true);
        cfg.mode = Mode::Live;
        cfg.strategies.arb.limits.enabled = true;
        cfg.strategies.arb.limits.paper_only = true;
        cfg.strategies.arb.limits.daily_loss_usdc_stop = Decimal::ZERO;
        assert!(cfg.validate_live_loss_stop().is_ok());
    }

    #[test]
    fn loss_stop_at_zero_in_paper_is_legitimate() {
        let mut cfg = base_config_with_universe(true);
        cfg.strategies.arb.limits.enabled = true;
        cfg.strategies.arb.limits.daily_loss_usdc_stop = Decimal::ZERO;
        assert!(cfg.validate_live_loss_stop().is_ok());
    }

    /// TC without `paper_only` in live aborts
    /// startup — its resting sells are unmodeled naked shorts in
    /// `state.rs` (claim that disappears with the GTD). Fail-closed even if the
    /// strategy is `enabled = false`: the hot toggle comes later.
    #[test]
    fn theta_capture_without_paper_only_fails_startup_in_live() {
        let mut cfg = base_config_with_universe(true);
        cfg.mode = Mode::Live;
        cfg.strategies.theta_capture.limits.enabled = false;
        cfg.strategies.theta_capture.limits.paper_only = false;
        let err = cfg
            .validate_theta_capture_blocked_live()
            .expect_err("TC without paper_only in live must abort startup");
        assert!(
            err.to_string().contains("theta_capture"),
            "the message must name the strategy: {err}"
        );
    }

    /// The `paper_only` flag is the documented escape hatch: with it, TC never
    /// runs in live and startup is not blocked.
    #[test]
    fn theta_capture_with_paper_only_does_not_block_live() {
        let mut cfg = base_config_with_universe(true);
        cfg.mode = Mode::Live;
        cfg.strategies.theta_capture.limits.paper_only = true;
        assert!(cfg.validate_theta_capture_blocked_live().is_ok());
    }

    /// In paper the restriction does not apply: TC runs with resting sells with no
    /// risk of real money.
    #[test]
    fn theta_capture_in_paper_does_not_block() {
        let mut cfg = base_config_with_universe(true);
        cfg.mode = Mode::Paper;
        cfg.strategies.theta_capture.limits.paper_only = false;
        assert!(cfg.validate_theta_capture_blocked_live().is_ok());
    }
}

#[cfg(test)]
mod live_loss_stop_tests {
    use super::*;

    #[test]
    fn can_run_rejects_live_without_loss_stop() {
        let lim = StrategyLimits {
            enabled: true,
            daily_loss_usdc_stop: Decimal::ZERO,
            ..Default::default()
        };
        assert_eq!(lim.can_run(Mode::Live), Err(BlockReason::NoLossStop));
        assert_eq!(lim.can_run(Mode::Paper), Ok(()));
    }

    #[test]
    fn disabled_strategy_without_loss_stop_does_not_block_live_startup() {
        let mut cfg = base_config_with_universe(true);
        cfg.mode = Mode::Live;
        cfg.strategies.reward_farm.limits.enabled = false;
        cfg.strategies.reward_farm.limits.paper_only = false;
        cfg.strategies.reward_farm.limits.daily_loss_usdc_stop = Decimal::ZERO;
        assert!(cfg.validate_live_loss_stop().is_ok());
        cfg.strategies.reward_farm.limits.enabled = true;
        assert!(cfg.validate_live_loss_stop().is_err());
    }
}
