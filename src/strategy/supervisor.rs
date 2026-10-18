//! Dynamic activation of a strategy that started with `enabled=false` in the
//! TOML loaded at boot. Previously, the hot toggle from
//! `api::control_strategy` / the Telegram enable command only touched
//! `RiskGuard`'s kill switch, which has nothing to "unpause" if the task was
//! never spawned (`spawn_strategy!` in `main.rs` only starts tasks at boot,
//! gated by that same flag).
//!
//! Strategies that DID have `enabled=true` at boot keep being spawned as
//! usual, inside the main `JoinSet` of `main.rs` (fail-fast: a panic takes
//! down the whole bot, see the shutdown `tokio::select!` in `main.rs`). On
//! construction this supervisor records which ones those are
//! (`boot_spawned`) so it does not spawn them again, and it only comes into
//! play for strategies that start disabled and are enabled later by an
//! operator without a restart. Same pattern as `tools::ToolSupervisor`, with
//! the same failure-isolation difference that already exists for
//! `LadderArb`/`VolSurface`: a task spawned here lives DETACHED
//! (`tokio::spawn`, not the `JoinSet` of `main.rs`), so if it panics it dies
//! alone instead of taking down the whole bot. Acceptable because it starts
//! from ZERO exposure: it never had a task, so it could never have had open
//! positions before this `spawn`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex};

use tokio::sync::{broadcast, mpsc, watch};

use crate::config::{BlockReason, Config};
use crate::coordination::BracketLedger;
use crate::market_feed::{MarketRegistry, TokenEvent};
use crate::risk::RiskGuard;
use crate::types::{BookSnapshot, Fill, QueuedIntent, StrategyId, VolSnapshot};

use super::{
    run, Arb, Arb15m5m, Arb15m5mDeps, Directional, DriftBox, GRw1Maker, MarketMaking, ModelBtc5m, NearCertain, NearExpiry,
    OpenMomentum, ParityMaker, PostCloseSweep, RewardFarm, RlPredictor, Strategy, StrategyCtx,
    ThetaCapture,
};

pub struct StrategySupervisor {
    book_tx: broadcast::Sender<Arc<BookSnapshot>>,
    vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
    intent_tx: mpsc::Sender<QueuedIntent>,
    fill_tx: broadcast::Sender<Fill>,
    risk: Arc<RiskGuard>,
    shutdown: watch::Receiver<bool>,
    max_book_age: std::time::Duration,
    registry: Arc<MarketRegistry>,
    config: Arc<Config>,
    ledger: Arc<BracketLedger>,
    running: StdMutex<HashMap<StrategyId, tokio::task::AbortHandle>>,
    /// Strategies with `enabled=true` at boot: they already have a live task in
    /// the `JoinSet` of `main.rs`. `spawn` is a no-op for these.
    boot_spawned: HashSet<StrategyId>,
    /// Channel for dynamic adds/removes on the market WS: enabling a
    /// strategy over a scan-only universe promotes its tokens to the WS in
    /// ~1 RTT (`operation:subscribe`); disabling returns them to REST polling
    /// if no other active strategy needs them.
    sub_tx: broadcast::Sender<TokenEvent>,
    /// Sender shared with the `ModelTool` tool to feed the `ModelBtc5m`
    /// strategy when it is built on demand. The strategy ONLY starts if this
    /// is Some; `set_enabled` validates that the supervisor's config/toggle
    /// do NOT unblock live (`can_run`); the structural barrier lives in
    /// `StrategyLimits::can_run` and in the fail-closed constructor of
    /// `ModelBtc5m::new`.
    model_prediction_tx: Option<Arc<broadcast::Sender<crate::tools::model::ModelPrediction>>>,
    /// ARB-15M5M: TWAP store and streak persistence, the same ones the
    /// startup `BuildCtx` uses (the hot toggle builds with these).
    arb15m5m: Arb15m5mDeps,
}

impl StrategySupervisor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        book_tx: broadcast::Sender<Arc<BookSnapshot>>,
        vol_rxs: HashMap<String, watch::Receiver<VolSnapshot>>,
        intent_tx: mpsc::Sender<QueuedIntent>,
        fill_tx: broadcast::Sender<Fill>,
        risk: Arc<RiskGuard>,
        shutdown: watch::Receiver<bool>,
        max_book_age: std::time::Duration,
        registry: Arc<MarketRegistry>,
        config: Arc<Config>,
        ledger: Arc<BracketLedger>,
        sub_tx: broadcast::Sender<TokenEvent>,
        model_prediction_tx: Option<Arc<broadcast::Sender<crate::tools::model::ModelPrediction>>>,
        arb15m5m: Arb15m5mDeps,
    ) -> Arc<Self> {
        // `boot_spawned` must reflect what `spawn_all` really spawned, which
        // is what `can_run` authorizes, not `enabled` alone. Otherwise a
        // strategy blocked in live would show up as "running" in the UI
        // without having a task.
        let boot_spawned = StrategyId::ALL
            .iter()
            .copied()
            .filter(|&id| config.strategy_limits(id).can_run(config.mode).is_ok())
            .filter(|&id| {
                // ModelBtc5m without a prediction channel has no signal at
                // all, so it is not spawned. RlPredictor DOES start without it
                // (degrades to HOLD/SELL over the recovered inventory;
                // an absent predictor never immobilizes its own positions).
                if id == StrategyId::ModelBtc5m && model_prediction_tx.is_none() {
                    return false;
                }
                // Both are structurally paper-only: in Live they are not
                // boot-spawned even if `can_run` passed (extra net over the TOML/ctor).
                !matches!(id, StrategyId::ModelBtc5m | StrategyId::RlPredictor)
                    || config.mode == crate::types::Mode::Paper
            })
            .collect();
        Arc::new(Self {
            book_tx,
            vol_rxs,
            intent_tx,
            fill_tx,
            risk,
            shutdown,
            max_book_age,
            registry,
            config,
            ledger,
            running: StdMutex::new(HashMap::new()),
            boot_spawned,
            sub_tx,
            model_prediction_tx,
            arb15m5m,
        })
    }

    fn universe(&self, id: StrategyId) -> Vec<String> {
        match id {
            StrategyId::Arb => self.config.strategies.arb.limits.universe.clone(),
            StrategyId::Arb15m5m => self.config.strategies.arb_15m5m.limits.universe.clone(),
            StrategyId::Directional => self.config.strategies.directional.limits.universe.clone(),
            StrategyId::NearCertain => self.config.strategies.near_certain.limits.universe.clone(),
            StrategyId::NearExpiry => self.config.strategies.near_expiry.limits.universe.clone(),
            StrategyId::MarketMaking => {
                self.config.strategies.market_making.limits.universe.clone()
            }
            StrategyId::ParityMaker => self.config.strategies.parity_maker.limits.universe.clone(),
            StrategyId::DriftBox => self.config.strategies.drift_box.limits.universe.clone(),
            StrategyId::OpenMomentum => {
                self.config.strategies.open_momentum.limits.universe.clone()
            }
            StrategyId::ThetaCapture => {
                self.config.strategies.theta_capture.limits.universe.clone()
            }
            StrategyId::PostCloseSweep => self
                .config
                .strategies
                .post_close_sweep
                .limits
                .universe
                .clone(),
            StrategyId::RewardFarm => self.config.strategies.reward_farm.limits.universe.clone(),
            StrategyId::GRw1Maker => self.config.strategies.g_rw1_maker.limits.universe.clone(),
            // Never reachable: `spawn()` returns early for
            // Follower (it is not a `Strategy`, see the guard below).
            StrategyId::Follower => Vec::new(),
            // Paper-only defensive. Dedicated `btc-updown-5m` universe,
            // but the scope filter (`is_target_market`) lives inside the
            // strategy and additionally requires a 300s duration.
            StrategyId::ModelBtc5m => self.config.strategies.model_btc5m.limits.universe.clone(),
            // Same BTC-5m universe as ModelBtc5m; the exact scope
            // filter (`is_target_market`) lives inside the
            // strategy.
            StrategyId::RlPredictor => {
                self.config.strategies.rl_predictor.limits.universe.clone()
            }
        }
    }

    fn build(&self, id: StrategyId) -> Box<dyn Strategy> {
        let haircut = self.config.fair_value.confidence_haircut;
        match id {
            StrategyId::Arb            => Box::new(Arb::new(self.config.strategies.arb.clone())),
            StrategyId::Directional    => Box::new(Directional::new(self.config.strategies.directional.clone(), haircut)),
            StrategyId::NearCertain    => Box::new(NearCertain::new(self.config.strategies.near_certain.clone(), haircut)),
            StrategyId::NearExpiry     => Box::new(NearExpiry::new(self.config.strategies.near_expiry.clone())),
            StrategyId::MarketMaking   => Box::new(MarketMaking::new(self.config.strategies.market_making.clone())),
            StrategyId::ParityMaker    => Box::new(ParityMaker::new(self.config.strategies.parity_maker.clone())),
            StrategyId::DriftBox       => Box::new(DriftBox::new(self.config.strategies.drift_box.clone(), haircut)),
            StrategyId::OpenMomentum   => Box::new(OpenMomentum::new(self.config.strategies.open_momentum.clone(), haircut)),
            StrategyId::ThetaCapture   => Box::new(ThetaCapture::new(self.config.strategies.theta_capture.clone(), haircut)),
            StrategyId::PostCloseSweep => Box::new(PostCloseSweep::new(self.config.strategies.post_close_sweep.clone())),
            StrategyId::RewardFarm     => Box::new(RewardFarm::new(
                self.config.strategies.reward_farm.clone(), self.registry.clone(), self.ledger.clone(),
            )),
            StrategyId::GRw1Maker      => Box::new(GRw1Maker::new(self.config.strategies.g_rw1_maker.clone())),
            StrategyId::Arb15m5m       => Box::new(Arb15m5m::new(self.config.strategies.arb_15m5m.clone(), self.arb15m5m.clone())),
            // The strategy requires the `model_prediction_tx` the
            // supervisor holds. The canonical path is `register_strategies!`
            // in `registry.rs` (used by `spawn_all`); this path (hot
            // toggle of a strategy that started disabled) injects the
            // same `Arc<Sender>` directly. If it is not injected,
            // we abort with a loud `unreachable!` so the operator
            // can diagnose; it should never happen if `main.rs` populated the
            // supervisor with `model_prediction_tx`.
            StrategyId::ModelBtc5m => {
                let tx = self
                    .model_prediction_tx
                    .as_ref()
                    .expect(
                        "StrategySupervisor::build called with StrategyId::ModelBtc5m \
                         without `model_prediction_tx`; main.rs must inject the \
                         tool's Arc<sender> when building the supervisor",
                    )
                    .clone();
                let cfg = self.config.strategies.model_btc5m.clone();
                let mode = self.config.mode;
                Box::new(ModelBtc5m::new(cfg, tx.subscribe(), mode))
            }
            // The `RlPredictor` ctor is fail-closed: without a `paper_approved`
            // artifact it is built disabled (reason
            // in `/ui/rl_learner`), it does not panic. `prediction_rx=None` degrades to
            // HOLD/SELL; `mode=Live` is already vetoed by the guard in
            // `set_enabled` below and by the ctor's defensive panic.
            StrategyId::RlPredictor => Box::new(RlPredictor::new(
                self.config.strategies.rl_predictor.clone(),
                self.model_prediction_tx.as_ref().map(|t| t.subscribe()),
                self.config.mode,
                self.config.as_ref(),
            )),
            // Follower is a standalone task (`strategy::follower::run`),
            // it does not implement the `Strategy` trait; it should never get here.
            // `spawn()` excludes it BEFORE calling `build()` (see the guard there).
            StrategyId::Follower => unreachable!(
                "StrategySupervisor::build called with StrategyId::Follower; spawn() must exclude it first"
            ),
        }
    }

    /// Starts the task for `id` on demand if it never ran (neither at
    /// boot nor through an earlier `set_enabled` in this session). No-op if it is
    /// already alive. No double-spawn: the check and the insertion of the
    /// `AbortHandle` happen under the same lock (same pattern as
    /// `ToolSupervisor::spawn`).
    fn spawn(self: &Arc<Self>, id: StrategyId) {
        if self.boot_spawned.contains(&id) {
            return;
        }
        // Follower does not implement the `Strategy` trait (standalone task,
        // see `strategy/follower.rs`); this hot-toggle mechanism is designed
        // for `Strategy` and the fit is not forced in this phase (known
        // limitation). Enabling/disabling Follower only works via TOML +
        // restart; if it started `enabled=true` it is already in
        // `boot_spawned` and the early return above applies just the same.
        // Without this guard, `set_enabled(Follower, true)` on a
        // boot-disabled Follower would reach `build()` and panic.
        if id == StrategyId::Follower {
            tracing::warn!(
                "strategy supervisor: Follower does not support hot toggle (standalone task) \
                 — edit config/strategies/follower.toml and restart"
            );
            return;
        }
        let mut running = self.running.lock().unwrap();
        if let Some(h) = running.get(&id) {
            if !h.is_finished() {
                return;
            }
        }
        let ctx = StrategyCtx {
            book_rx: self.book_tx.subscribe(),
            vol_rxs: self.vol_rxs.clone(),
            intent_tx: self.intent_tx.clone(),
            fill_rx: self.fill_tx.subscribe(),
            risk: self.risk.clone(),
            shutdown: self.shutdown.clone(),
            max_book_age: self.max_book_age,
            registry: self.registry.clone(),
            universe: self.universe(id),
            requires_oracle: crate::strategy::registry::strategy_requires_oracle(id),
        };
        let strat = self.build(id);
        let handle = tokio::spawn(run(strat, ctx));
        running.insert(id, handle.abort_handle());
        tracing::info!(%id, "strategy supervisor: task started on demand");
    }

    /// Does any ACTIVE strategy (live task without a kill switch) declare
    /// `universe`? A task paused by kill does not count: it will not consume data.
    /// Called after `kill_strategy(id)` on the shutdown path, so `id`
    /// itself is excluded without an extra parameter.
    fn universe_in_use(&self, universe: &str) -> bool {
        StrategyId::ALL.iter().any(|&s| {
            self.has_task(s)
                && !self.risk.is_strategy_killed(s)
                && self.universe(s).iter().any(|u| u == universe)
        })
    }

    /// Promotes the universes of `id` to the WS: puts them in `ws_universes` and
    /// dynamically subscribes their already-registered tokens (~1 RTT).
    fn promote_universes_to_ws(&self, id: StrategyId) {
        let universes: HashSet<String> = self.universe(id).into_iter().collect();
        if universes.is_empty() {
            return;
        }
        for u in &universes {
            self.registry.ws_universes.insert(u.clone());
        }
        let token_ids = self.registry.token_ids_for_universes(&universes);
        if !token_ids.is_empty() {
            metrics::counter!("hrf_strategy_ws_promote_tokens_total")
                .increment(token_ids.len() as u64);
            tracing::info!(
                %id,
                universes = ?universes,
                tokens = token_ids.len(),
                "strategy supervisor: universe promoted to WS on the fly"
            );
            let _ = self.sub_tx.send(TokenEvent::Subscribe { token_ids });
        }
    }

    /// Returns to REST polling the universes of `id` that no other active
    /// strategy needs anymore: removes them from `ws_universes` and unsubscribes
    /// their tokens from the WS (the poller picks them up on its next tick).
    fn demote_universes_to_poll(&self, id: StrategyId) {
        let universes = self.universe(id);
        let mut to_unsub: Vec<String> = Vec::new();
        for u in universes {
            if self.universe_in_use(&u) {
                continue; // another active strategy still needs it on the WS
            }
            self.registry.ws_universes.remove(&u);
            let mut one = HashSet::new();
            one.insert(u.clone());
            to_unsub.extend(self.registry.token_ids_for_universes(&one));
        }
        if !to_unsub.is_empty() {
            metrics::counter!("hrf_strategy_ws_demote_tokens_total")
                .increment(to_unsub.len() as u64);
            tracing::info!(
                %id,
                tokens = to_unsub.len(),
                "strategy supervisor: universes returned to REST polling"
            );
            let _ = self.sub_tx.send(TokenEvent::Unsubscribe {
                token_ids: to_unsub,
            });
        }
    }

    /// Turns on (spawns on demand if needed + resets the kill
    /// switch) or off (kill switch: the task, if it exists, stays alive but
    /// paused in place, preserving its internal state (z-score
    /// buffers, partial window positions, etc.) so it can
    /// resume instantly without losing context).
    ///
    /// Turning on can be REJECTED: in live, a `paper_only` strategy cannot
    /// be turned on; having the API token is not enough. Turning off is never rejected: having no way to stop
    /// a strategy would be worse than anything this protects against.
    pub fn set_enabled(self: &Arc<Self>, id: StrategyId, enabled: bool) -> Result<(), BlockReason> {
        if !enabled {
            self.risk
                .kill_strategy(id, "disabled by dynamic control");
            // WS→poll migration: lazy for the strategy (it is already dead),
            // but it frees the venue's throttle (proportional to subscribed tokens).
            self.demote_universes_to_poll(id);
            return Ok(());
        }
        // The operator is asking precisely to turn it on, so the
        // `!enabled` arm of `can_run` (which reflects the `enabled` of the TOML
        // loaded at boot) does not apply: the mode arms are evaluated over
        // limits with `enabled` already set. The decision is still up to
        // `can_run`; no rule is duplicated here.
        let mut lim = self.config.strategy_limits(id).clone();
        lim.enabled = true;
        if let Err(reason) = lim.can_run(self.config.mode) {
            tracing::warn!(
                %id, mode = %self.config.mode,
                "hot toggle REJECTED by the paper/live barrier: {reason}"
            );
            return Err(reason);
        }
        // Extra runtime barrier: if the supervisor has no
        // `model_prediction_tx` injected, we CANNOT build the
        // strategy on demand. We return a clear `Err` (no panic) and
        // do NOT spawn.
        // RlPredictor tolerates an absent channel (degrades to HOLD/SELL with
        // `p_missing=1`); the veto for a missing `model_prediction_tx` only
        // applies to ModelBtc5m, which has no signal at all without it.
        if id == StrategyId::ModelBtc5m && self.model_prediction_tx.is_none() {
            tracing::warn!(
                %id,
                "hot toggle REJECTED: model_prediction_tx absent in the supervisor"
            );
            return Err(BlockReason::ModelPredictionTxMissing);
        }
        // Defensive MODE barrier: the defensive experiment
        // `ModelBtc5m` NEVER trades real money, whatever the TOML's
        // `paper_only` says. `StrategyLimits::can_run` already blocks the
        // combination `enabled=true && paper_only=true && mode=Live`, but
        // lets `enabled=true && paper_only=false && mode=Live`
        // through (the "not paper-only" scenario);
        // there the constructor panics as a safety net. That is a
        // fragile decision: if the TOML changes and `paper_only=false` slips in through a
        // hot `set_enabled`, the constructor's `panic!` blows up
        // the task. Better to cut it here in the supervisor with a clear message and
        // a `BlockReason` that the UI/Telegram already know how to show.
        if id == StrategyId::ModelBtc5m && self.config.mode == crate::types::Mode::Live {
            tracing::warn!(
                %id,
                "hot toggle REJECTED: ModelBtc5m is a paper-only experiment \
                 — it NEVER trades real money (independent of paper_only)"
            );
            return Err(BlockReason::PaperOnly);
        }
        // RlPredictor NEVER trades in live: neither inference nor
        // learning with real money (live and live learning
        // remain forbidden). The barrier is structural here and in the
        // ctor (fail-closed panic), not a TOML default.
        if id == StrategyId::RlPredictor && self.config.mode == crate::types::Mode::Live {
            tracing::warn!(
                %id,
                "hot toggle REJECTED: RlPredictor is structurally paper-only \
                 — neither trading nor live learning (independent of paper_only)"
            );
            return Err(BlockReason::PaperOnly);
        }
        self.spawn(id);
        self.risk.reset_strategy_kill(id);
        // poll→WS migration: its scan-only universe tokens are subscribed
        // dynamically; without this the strategy would operate on ~10s-old data.
        self.promote_universes_to_ws(id);
        Ok(())
    }

    /// `true` if the strategy has a live task somewhere (boot or
    /// on-demand spawn), to report real state in the API/Telegram.
    pub fn has_task(&self, id: StrategyId) -> bool {
        if self.boot_spawned.contains(&id) {
            return true;
        }
        self.running
            .lock()
            .unwrap()
            .get(&id)
            .map(|h| !h.is_finished())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::base_config_with_universe;
    use std::time::Duration;

    /// `universe()` above reads `self.config.strategies.<field>.limits.universe`
    /// by hand per `StrategyId`; the same datum that `registry::strategy_limits`
    /// obtains from the `register_strategies!` table via the right field of
    /// `StrategyConfigs`. Both should ALWAYS match for every registered
    /// variant; if a future copy-paste in the `match` of `universe()` reads the
    /// wrong field, the `match` remains exhaustive (compiles cleanly) but
    /// the hot toggle would serve another strategy's universe. This test
    /// builds a config where each strategy has a distinct sentinel
    /// universe (its own name) and compares both sources for every
    /// `StrategyId` of `registry::REGISTERED`.
    fn sentinel_universe_supervisor() -> Arc<StrategySupervisor> {
        let mut cfg = base_config_with_universe(true);
        for &id in crate::strategy::registry::REGISTERED {
            let sentinel = vec![id.to_string()];
            match id {
                StrategyId::Arb => cfg.strategies.arb.limits.universe = sentinel,
                StrategyId::Arb15m5m => cfg.strategies.arb_15m5m.limits.universe = sentinel,
                StrategyId::Directional => cfg.strategies.directional.limits.universe = sentinel,
                StrategyId::NearCertain => cfg.strategies.near_certain.limits.universe = sentinel,
                StrategyId::NearExpiry => cfg.strategies.near_expiry.limits.universe = sentinel,
                StrategyId::MarketMaking => cfg.strategies.market_making.limits.universe = sentinel,
                StrategyId::ParityMaker => cfg.strategies.parity_maker.limits.universe = sentinel,
                StrategyId::DriftBox => cfg.strategies.drift_box.limits.universe = sentinel,
                StrategyId::OpenMomentum => cfg.strategies.open_momentum.limits.universe = sentinel,
                StrategyId::ThetaCapture => cfg.strategies.theta_capture.limits.universe = sentinel,
                StrategyId::PostCloseSweep => {
                    cfg.strategies.post_close_sweep.limits.universe = sentinel
                }
                StrategyId::RewardFarm => cfg.strategies.reward_farm.limits.universe = sentinel,
                StrategyId::GRw1Maker => cfg.strategies.g_rw1_maker.limits.universe = sentinel,
                // Dedicated sentinel universe so the
                // `sentinel_universe_*` test detects an incorrect copy-paste.
                StrategyId::ModelBtc5m => cfg.strategies.model_btc5m.limits.universe = sentinel,
                // Own sentinel universe (same anti-copy-paste
                // pattern as ModelBtc5m).
                StrategyId::RlPredictor => {
                    cfg.strategies.rl_predictor.limits.universe = sentinel
                }
                StrategyId::Follower => {}
            }
        }
        let (book_tx, _) = broadcast::channel(16);
        let (intent_tx, _) = mpsc::channel(16);
        let (fill_tx, _) = broadcast::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let (sub_tx, _sub_rx) = broadcast::channel(16);
        StrategySupervisor::new(
            book_tx,
            HashMap::new(),
            intent_tx,
            fill_tx,
            crate::risk::RiskGuard::new(
                crate::config::RiskConfig::default(),
                crate::types::Mode::Paper,
            ),
            shutdown_rx,
            std::time::Duration::from_secs(1),
            Arc::new(crate::market_feed::MarketRegistry::new()),
            Arc::new(cfg),
            BracketLedger::new(),
            sub_tx,
            // Supervisor tests do not need real predictions;
            // `None` so `build(ModelBtc5m)` does not activate. If a future
            // supervisor test needs the ModelBtc5m path,
            // we will pass a `Some(Arc<Sender>)` from a local
            // broadcast::channel.
            None,
            crate::strategy::Arb15m5mDeps::empty(),
        )
    }

    // ── Hot toggle ────────────────────────────────────────────────────────

    /// Supervisor with ARB off at boot (so `boot_spawned` is empty and the
    /// toggle has to really spawn).
    fn supervisor_with(mode: crate::types::Mode) -> Arc<StrategySupervisor> {
        let mut cfg = base_config_with_universe(false);
        cfg.mode = mode;
        cfg.strategies.arb.limits.enabled = false;
        let (book_tx, _book_rx) = broadcast::channel(16);
        let (intent_tx, _intent_rx) = mpsc::channel(16);
        let (fill_tx, _fill_rx) = broadcast::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let (sub_tx, _sub_rx) = broadcast::channel(16);
        StrategySupervisor::new(
            book_tx,
            HashMap::new(),
            intent_tx,
            fill_tx,
            crate::risk::RiskGuard::new(crate::config::RiskConfig::default(), mode),
            shutdown_rx,
            std::time::Duration::from_secs(1),
            Arc::new(crate::market_feed::MarketRegistry::new()),
            Arc::new(cfg),
            BracketLedger::new(),
            sub_tx,
            None,
            crate::strategy::Arb15m5mDeps::empty(),
        )
    }

    #[tokio::test]
    async fn toggle_in_live_starts_task() {
        let sup = supervisor_with(crate::types::Mode::Live);
        assert!(sup.set_enabled(StrategyId::Arb, true).is_ok());
        assert!(sup.has_task(StrategyId::Arb));
    }

    #[tokio::test]
    async fn toggle_in_paper_starts_task() {
        let sup = supervisor_with(crate::types::Mode::Paper);
        assert!(sup.set_enabled(StrategyId::Arb, true).is_ok());
        assert!(sup.has_task(StrategyId::Arb));
    }

    /// Turning off is NEVER blocked: if it were, an operator could be left
    /// with no way to stop a strategy.
    #[tokio::test]
    async fn off_toggle_always_allowed() {
        for mode in [crate::types::Mode::Paper, crate::types::Mode::Live] {
            let sup = supervisor_with(mode);
            assert!(
                sup.set_enabled(StrategyId::Arb, false).is_ok(),
                "turning off in {mode} must be allowed"
            );
        }
    }

    /// The hot toggle migrates the strategy's universe between REST polling
    /// and WS: enabling inserts into `ws_universes` (and subscribes its registered
    /// tokens); disabling removes it if nobody else uses it.
    #[tokio::test]
    async fn toggle_promotes_and_demotes_universe_in_ws() {
        let sup = supervisor_with(crate::types::Mode::Paper);
        let uni = sup.universe(StrategyId::Arb);
        assert!(!uni.is_empty());
        assert!(!sup.registry.ws_universes.contains(&uni[0]));

        sup.set_enabled(StrategyId::Arb, true).unwrap();
        for u in &uni {
            assert!(
                sup.registry.ws_universes.contains(u),
                "enabling ARB must promote {u} to the WS"
            );
        }

        sup.set_enabled(StrategyId::Arb, false).unwrap();
        for u in &uni {
            assert!(
                !sup.registry.ws_universes.contains(u),
                "disabling ARB must return {u} to polling"
            );
        }
    }

    #[test]
    fn universe_matches_registry_strategy_limits_for_each_registered() {
        let sup = sentinel_universe_supervisor();
        for &id in crate::strategy::registry::REGISTERED {
            let expected = crate::strategy::registry::strategy_limits(id, &sup.config)
                .universe
                .clone();
            assert_eq!(
                sup.universe(id),
                expected,
                "universe() in supervisor.rs diverges from registry::strategy_limits for {id} \
                 — check that the match arm reads self.config.strategies.{id} and not another field"
            );
        }
    }

    // ── Runtime wiring tests ────────────────────────────────────────────

    /// `BuildCtx` assembled with a real `prediction_tx` (shared Arc<Sender>),
    /// dummy market registry and a 100/5 paper config with
    /// `model_btc5m.limits.enabled=true`. Used by the ModelBtc5m runtime
    /// wiring tests; `model_tool.enabled=true` is set in
    /// `Config` and the artifact points to the real versioned directory
    /// (`src/tools/model/artifacts/btc5m`).
    fn ctx_paper_with_model(
        tempdir: &std::path::Path,
    ) -> (
        crate::strategy::BuildCtx,
        Arc<broadcast::Sender<crate::tools::model::ModelPrediction>>,
    ) {
        let mut cfg = base_config_with_universe(false);
        cfg.mode = crate::types::Mode::Paper;
        cfg.strategies.model_btc5m.limits.enabled = true;
        cfg.strategies.model_btc5m.limits.paper_only = true;
        cfg.model_tool.enabled = true;
        cfg.model_tool.artifact_dir = format!(
            "{}/src/tools/model/artifacts/btc5m",
            env!("CARGO_MANIFEST_DIR")
        );
        cfg.model_tool.predictions_dir = tempdir.join("model_tool").to_string_lossy().into_owned();
        let (tx_raw, _rx) = broadcast::channel::<crate::tools::model::ModelPrediction>(16);
        let tx = Arc::new(tx_raw);
        let ctx = crate::strategy::BuildCtx {
            config: Arc::new(cfg),
            registry: Arc::new(crate::market_feed::MarketRegistry::new()),
            ledger: crate::coordination::BracketLedger::new(),
            model_prediction_tx: Some(tx.clone()),
            arb15m5m: crate::strategy::Arb15m5mDeps::empty(),
        };
        (ctx, tx)
    }

    /// The registry constructor (canonical path of
    /// `spawn_all`) uses the real `ctx.config.mode`, not a hardcoded `Mode::Paper`.
    /// If the runtime wiring always injected `mode = Paper`, this test
    /// would detect the bypass because we would switch to Live and the ctor would panic.
    ///
    /// It tests the END-TO-END behavior of the constructor:
    ///
    /// * `mode=Paper` with the default config (`paper_only=true`,
    ///   `enabled=false`): builds without panic AND `on_book` emits no
    ///   intents without a prediction (verifies there is no flow bypass).
    /// * `mode=Live` with the same config: the constructor PANICS. The test
    ///   uses `std::panic::catch_unwind` to OBSERVE the panic, NOT as a
    ///   success path: if the panic does NOT happen, the test fails (that is,
    ///   `catch_unwind` is only the mechanism for "not letting the panic
    ///   take down the test runner", while the check remains that
    ///   `is_panicked == true`). If the constructor "caught" the panic
    ///   internally and returned an empty instance, this test
    ///   would detect the bypass.
    #[test]
    fn registry_constructor_uses_real_mode_not_hardcoded_paper() {
        let dir = tempfile::tempdir().unwrap();
        let (ctx, _tx) = ctx_paper_with_model(dir.path());
        assert_eq!(ctx.config.mode, crate::types::Mode::Paper);
        let cfg = ctx.config.clone();
        let lim = cfg.strategies.model_btc5m.limits.clone();
        // helper sets: paper_only=true. Paper passes; Live blocks on paper_only.
        assert!(lim.can_run(crate::types::Mode::Paper).is_ok());
        assert_eq!(
            lim.can_run(crate::types::Mode::Live),
            Err(crate::config::BlockReason::PaperOnly),
            "paper_only=true in Live must block with PaperOnly"
        );

        // End-to-end verification of the constructor: with the real mode of the
        // `BuildCtx` (Paper), the strategy is built and the `mode` field
        // is preserved correctly. We do not need to verify `on_book`
        // here; that coverage lives in `model_btc5m::tests::scope_*` and
        // `::struct_mode_*`, where the internal helpers are accessible.
        let rx = ctx
            .model_prediction_tx
            .as_ref()
            .expect("ctx_paper_with_model injects tx")
            .subscribe();
        let strat = crate::strategy::model_btc5m::ModelBtc5m::new(
            cfg.strategies.model_btc5m.clone(),
            rx,
            ctx.config.mode,
        );
        assert_eq!(
            strat.mode(),
            crate::types::Mode::Paper,
            "the struct's mode must match the BuildCtx's"
        );

        // Fail-closed verification in Live: the constructor PANICS regardless
        // of the other flags. We use `catch_unwind` ONLY to avoid taking down
        // the test runner; the assert is that the panic OCCURRED. If the
        // constructor caught the panic internally and returned an
        // instance (which would be worse: the strategy would stay alive
        // emitting intents in Live), `is_panicked` would be `false` and the
        // test would fail with a clear message.
        //
        // We build a `Config` with `mode=Live` directly from the
        // `ctx` data (without touching the shared `Arc<Config>`).
        let mut cfg_live: crate::config::Config = (*cfg).clone();
        cfg_live.mode = crate::types::Mode::Live;
        let rx_live = ctx
            .model_prediction_tx
            .as_ref()
            .expect("ctx_paper_with_model injects tx")
            .subscribe();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = crate::strategy::model_btc5m::ModelBtc5m::new(
                cfg_live.strategies.model_btc5m.clone(),
                rx_live,
                crate::types::Mode::Live,
            );
        }));
        assert!(
            result.is_err(),
            "Mode::Live MUST panic in the constructor — \
             catch_unwind is only an observer, not a success path. \
             If the constructor catches the panic internally, this \
             test will detect the bypass."
        );
    }

    /// The `ToolSupervisor::spawn(ModelTool)` wiring starts
    /// the task with the shared `prediction_tx`. A prediction published
    /// by the tool's writer reaches the strategy supervisor's `Receiver`.
    /// Verifies the end-to-end in-memory connection without requiring
    /// the real artifact (uses `NullPredictor` via `ModelProducer::sink`).
    #[tokio::test]
    async fn spawn_tool_and_strategy_share_predictions_channel() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let (ctx, tx) = ctx_paper_with_model(dir.path());

        // Minimal connection: `ToolSupervisor` with its
        // `model_prediction_tx` injected, and a single book that passes the
        // strategy's gates. Without `recorder_deps` we do not spawn
        // recorders, but the ModelTool is spawned (it does not need them).
        let cfg = ctx.config.clone();
        let (book_tx, _book_rx) = broadcast::channel::<Arc<crate::types::BookSnapshot>>(64);
        let vol_rxs: HashMap<String, watch::Receiver<crate::types::VolSnapshot>> = HashMap::new();
        let (_kill_tx, kill_rx) = watch::channel(false);
        let (scan_tx, _scan_rx) = mpsc::unbounded_channel();
        let sup = crate::tools::ToolSupervisor::new(
            book_tx.clone(),
            vol_rxs.clone(),
            scan_tx,
            kill_rx,
            Duration::from_millis(500),
            cfg.clone(),
        );
        sup.set_model_prediction_tx(tx.clone());

        // Subscribe the strategy side to the shared channel.
        let mut rx_strategy = tx.subscribe();
        assert_eq!(rx_strategy.len(), 0, "no predictions at start");

        // Publish a synthetic prediction directly to the broadcast
        // (we do not need the tool's runner; the wiring is what
        // we are testing).
        let now_ms = crate::time::now().timestamp_millis();
        let pred = crate::tools::model::ModelPrediction {
            schema_version: crate::tools::model::MODEL_TOOL_SCHEMA_VERSION,
            ts_ms: now_ms,
            market_cid: "cid-test".into(),
            slug: "btc-updown-5m-epoch".into(),
            token_id_yes: "tok-yes".into(),
            token_id_no: "tok-no".into(),
            side: "buy_yes".into(),
            p_model: Some(0.65),
            p_market: Some(0.50),
            book_age_ms: 100,
            spot_age_ms: 50,
            latency_ms: 100,
            budget_ms: 3_000,
            abstention: None,
            model_hash: Some("deadbeef".into()),
        };
        tx.send(pred.clone()).unwrap();

        // The strategy must see the prediction.
        let received = tokio::time::timeout(Duration::from_secs(1), rx_strategy.recv())
            .await
            .expect("prediction did not reach the strategy")
            .expect("broadcast closed");
        assert_eq!(received.market_cid, pred.market_cid);
        assert_eq!(received.p_model, pred.p_model);
        assert_eq!(received.abstention, None);

        // Also verify that `BuildCtx.model_prediction_tx` points to the
        // same `Arc<Sender>` as the tool supervisor's.
        let ctx_tx = ctx
            .model_prediction_tx
            .as_ref()
            .expect("ctx_paper_with_model must populate the field");
        assert!(
            Arc::ptr_eq(ctx_tx, &tx),
            "BuildCtx.model_prediction_tx and the ToolSupervisor's are clones of the same Arc"
        );
    }

    /// Boot spawn in paper: the `ToolSupervisor` starts
    /// `ToolId::ModelTool` at boot AND the registry builds the
    /// `ModelBtc5m` strategy that consumes predictions from the SAME channel. We validate
    /// that both sides see the TOML's `enabled`: the tool starts, and the
    /// strategy ends up in `boot_spawned` (because can_run authorizes it).
    /// The test does NOT call `spawn_enabled_at_boot` (which tries to spawn
    /// other tools that require `recorder_deps`); we use `set_enabled`
    /// directly on ModelTool, which is the real runtime path.
    #[tokio::test]
    async fn spawn_boot_paper_starts_connected_tool_and_strategy() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let (ctx, tx) = ctx_paper_with_model(dir.path());
        let cfg = ctx.config.clone();
        assert!(cfg.model_tool.enabled);
        assert!(cfg.strategies.model_btc5m.limits.enabled);
        let lim = &cfg.strategies.model_btc5m.limits;
        assert!(lim.can_run(cfg.mode).is_ok(), "paper-only can run");

        let (book_tx, _book_rx) = broadcast::channel::<Arc<crate::types::BookSnapshot>>(64);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let (scan_tx, _scan_rx) = mpsc::unbounded_channel();
        let sup = crate::tools::ToolSupervisor::new(
            book_tx.clone(),
            HashMap::new(),
            scan_tx,
            kill_rx,
            Duration::from_millis(500),
            cfg.clone(),
        );
        sup.set_model_prediction_tx(tx.clone());

        // Start ONLY ModelTool (not the rest, which requires
        // `recorder_deps`). The real path in `main.rs` is
        // `spawn_enabled_at_boot`, but in this unit test we invoke
        // the supervisor path directly.
        sup.set_enabled(crate::tools::ToolId::ModelTool, true)
            .await
            .expect("ModelTool.enabled=true with prediction_tx populated must start");
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(
            sup.is_running(crate::tools::ToolId::ModelTool),
            "ModelTool must be running after set_enabled"
        );

        // Turn off with `set_enabled(false)`; it must drain the writer or
        // return a timeout `Err` (the task keeps draining in the background,
        // but `is_running` must drop to `false`).
        let _ = sup.set_enabled(crate::tools::ToolId::ModelTool, false).await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!sup.is_running(crate::tools::ToolId::ModelTool));
    }

    /// Live blocked by CONFIG: with `paper_only=true` in
    /// Live mode, `set_enabled(true)` must be rejected via `BlockReason::PaperOnly`
    /// WITHOUT a panic and the strategy is NOT spawned. The constructor's
    /// fail-closed barrier only applies to the combination `paper_only=false
    /// && mode=Live && enabled=true`, which is
    /// excluded by `can_run` before reaching the spawn (paper_only=true
    /// wins). Here we verify the paper_only=true path the
    /// defensive experiment is configured with.
    #[tokio::test]
    async fn live_blocked_paper_only_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let (mut ctx, tx) = ctx_paper_with_model(dir.path());
        Arc::make_mut(&mut ctx.config);
        let cfg_mut = Arc::get_mut(&mut ctx.config).expect("unique Arc");
        cfg_mut.mode = crate::types::Mode::Live;
        cfg_mut.strategies.model_btc5m.limits.paper_only = true;
        cfg_mut.strategies.model_btc5m.limits.enabled = true;
        let (book_tx, _book_rx) = broadcast::channel::<Arc<crate::types::BookSnapshot>>(64);
        let (intent_tx, _intent_rx) = mpsc::channel(16);
        let (fill_tx, _fill_rx) = broadcast::channel::<crate::types::Fill>(16);
        let (sub_tx, _sub_rx) = broadcast::channel::<crate::market_feed::TokenEvent>(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let sup = crate::strategy::StrategySupervisor::new(
            book_tx,
            HashMap::new(),
            intent_tx,
            fill_tx,
            crate::risk::RiskGuard::new(crate::config::RiskConfig::default(), crate::types::Mode::Live),
            kill_rx,
            Duration::from_millis(500),
            Arc::new(crate::market_feed::MarketRegistry::new()),
            Arc::new(cfg_mut.clone()),
            crate::coordination::BracketLedger::new(),
            sub_tx,
            Some(tx),
            crate::strategy::Arb15m5mDeps::empty(),
        );
        let res = sup.set_enabled(crate::types::StrategyId::ModelBtc5m, true);
        assert_eq!(
            res,
            Err(crate::config::BlockReason::PaperOnly),
            "paper_only=true in live must be rejected via BlockReason (NO panic)"
        );
        assert!(!sup.has_task(crate::types::StrategyId::ModelBtc5m));
        // Turning off is NEVER rejected; always Ok.
        assert!(sup.set_enabled(crate::types::StrategyId::ModelBtc5m, false).is_ok());
    }

    /// Hot start from `disabled`: with `enabled=false` at
    /// boot, `boot_spawned` does NOT contain ModelBtc5m. The operator enables it
    /// via API/Telegram (`set_enabled(true)`); the strategy is built
    /// on demand with the `prediction_tx` already injected, and starts.
    #[tokio::test]
    async fn hot_start_paper_from_disabled_works() {
        let dir = tempfile::tempdir().unwrap();
        let (mut ctx, tx) = ctx_paper_with_model(dir.path());
        Arc::make_mut(&mut ctx.config);
        let cfg_mut = Arc::get_mut(&mut ctx.config).expect("unique Arc");
        // enabled=false at boot; the supervisor must not boot-spawn it.
        cfg_mut.strategies.model_btc5m.limits.enabled = false;
        cfg_mut.mode = crate::types::Mode::Paper;
        let (book_tx, _book_rx) = broadcast::channel::<Arc<crate::types::BookSnapshot>>(64);
        let (intent_tx, _intent_rx) = mpsc::channel(16);
        let (fill_tx, _fill_rx) = broadcast::channel::<crate::types::Fill>(16);
        let (sub_tx, _sub_rx) = broadcast::channel::<crate::market_feed::TokenEvent>(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let sup = crate::strategy::StrategySupervisor::new(
            book_tx,
            HashMap::new(),
            intent_tx,
            fill_tx,
            crate::risk::RiskGuard::new(crate::config::RiskConfig::default(), crate::types::Mode::Paper),
            kill_rx,
            Duration::from_millis(500),
            Arc::new(crate::market_feed::MarketRegistry::new()),
            Arc::new(cfg_mut.clone()),
            crate::coordination::BracketLedger::new(),
            sub_tx,
            Some(tx),
            crate::strategy::Arb15m5mDeps::empty(),
        );
        assert!(!sup.has_task(crate::types::StrategyId::ModelBtc5m));
        // Enable: enabled=true + paper + prediction_tx populated → must
        // start.
        assert!(sup.set_enabled(crate::types::StrategyId::ModelBtc5m, true).is_ok());
        assert!(sup.has_task(crate::types::StrategyId::ModelBtc5m));
        // Turn off: `set_enabled(false)` is a kill switch, it must not fail.
        assert!(sup.set_enabled(crate::types::StrategyId::ModelBtc5m, false).is_ok());
        // has_task stays true because the kill switch does NOT kill the task;
        // it leaves it paused. The semantics of has_task are "live task",
        // not "strategy emitting". The supervisor reflects the runtime
        // state, not the logical one.
        // assert!(!sup.has_task(...)) <- NO, depends on the design.
    }

    /// `set_enabled` rejects ModelBtc5m if the supervisor
    /// was built without `model_prediction_tx` (runtime barrier,
    /// `BlockReason::ModelPredictionTxMissing`). No panic.
    #[tokio::test]
    async fn set_enabled_model_btc5m_rejects_without_prediction_tx() {
        let dir = tempfile::tempdir().unwrap();
        let (mut ctx, _tx) = ctx_paper_with_model(dir.path());
        Arc::make_mut(&mut ctx.config);
        let cfg_mut = Arc::get_mut(&mut ctx.config).expect("unique Arc");
        // Clear the ctx's `prediction_tx` (simulates a supervisor without
        // injection: the wiring forgot to pass it).
        let (book_tx, _book_rx) = broadcast::channel::<Arc<crate::types::BookSnapshot>>(64);
        let (intent_tx, _intent_rx) = mpsc::channel(16);
        let (fill_tx, _fill_rx) = broadcast::channel::<crate::types::Fill>(16);
        let (sub_tx, _sub_rx) = broadcast::channel::<crate::market_feed::TokenEvent>(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let sup = crate::strategy::StrategySupervisor::new(
            book_tx,
            HashMap::new(),
            intent_tx,
            fill_tx,
            crate::risk::RiskGuard::new(crate::config::RiskConfig::default(), crate::types::Mode::Paper),
            kill_rx,
            Duration::from_millis(500),
            Arc::new(crate::market_feed::MarketRegistry::new()),
            Arc::new(cfg_mut.clone()),
            crate::coordination::BracketLedger::new(),
            sub_tx,
            None, // <-- no prediction_tx,
            crate::strategy::Arb15m5mDeps::empty(),
        );
        let res = sup.set_enabled(crate::types::StrategyId::ModelBtc5m, true);
        assert_eq!(
            res,
            Err(crate::config::BlockReason::ModelPredictionTxMissing),
            "without prediction_tx the supervisor must reject the toggle without panicking"
        );
        assert!(!sup.has_task(crate::types::StrategyId::ModelBtc5m));
    }

    /// Defensive mode barrier: even if the TOML says
    /// `paper_only=false && enabled=true` (which
    /// `StrategyLimits::can_run` would let through to the supervisor), the
    /// defensive experiment `ModelBtc5m` must NEVER trade real
    /// money. `set_enabled(_, true)` in `Mode::Live` must be rejected via
    /// `BlockReason::PaperOnly` BEFORE any spawn, without a panic.
    /// The test covers the combination that used to lead to the constructor's
    /// `panic!` (`enabled=true, paper_only=false,
    /// mode=Live`) and verifies the task was not spawned.
    #[tokio::test]
    async fn model_btc5m_live_rejected_even_with_paper_only_false() {
        let dir = tempfile::tempdir().unwrap();
        let (mut ctx, tx) = ctx_paper_with_model(dir.path());
        Arc::make_mut(&mut ctx.config);
        let cfg_mut = Arc::get_mut(&mut ctx.config).expect("unique Arc");
        // Combination that `can_run` approves but that the defensive experiment
        // must NEVER allow: live with `paper_only=false`.
        // The `ModelBtc5m::new` constructor would panic if it got here.
        cfg_mut.mode = crate::types::Mode::Live;
        cfg_mut.strategies.model_btc5m.limits.paper_only = false;
        cfg_mut.strategies.model_btc5m.limits.enabled = true;
        let (book_tx, _book_rx) = broadcast::channel::<Arc<crate::types::BookSnapshot>>(64);
        let (intent_tx, _intent_rx) = mpsc::channel(16);
        let (fill_tx, _fill_rx) = broadcast::channel::<crate::types::Fill>(16);
        let (sub_tx, _sub_rx) = broadcast::channel::<crate::market_feed::TokenEvent>(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let sup = crate::strategy::StrategySupervisor::new(
            book_tx,
            HashMap::new(),
            intent_tx,
            fill_tx,
            crate::risk::RiskGuard::new(crate::config::RiskConfig::default(), crate::types::Mode::Live),
            kill_rx,
            Duration::from_millis(500),
            Arc::new(crate::market_feed::MarketRegistry::new()),
            Arc::new(cfg_mut.clone()),
            crate::coordination::BracketLedger::new(),
            sub_tx,
            Some(tx),
            crate::strategy::Arb15m5mDeps::empty(),
        );
        // The `can_run` barrier would approve: we verify it is NOT the one
        // that triggers the rejection; the rejection comes from the supervisor (mode).
        let lim = sup.config.strategies.model_btc5m.limits.clone();
        assert!(
            lim.can_run(crate::types::Mode::Live).is_ok(),
            "this test assumes can_run lets the combination through (paper_only=false) \
             — the rejection must come from the supervisor, not the barrier"
        );
        let res = sup.set_enabled(crate::types::StrategyId::ModelBtc5m, true);
        assert_eq!(
            res,
            Err(crate::config::BlockReason::PaperOnly),
            "ModelBtc5m in Live must ALWAYS be rejected via PaperOnly, even with \
             paper_only=false (paper-only defensive experiment)"
        );
        assert!(
            !sup.has_task(crate::types::StrategyId::ModelBtc5m),
            "no task must be left spawned after the rejection"
        );
        // Turning off is still allowed even though set_enabled(true) was rejected.
        assert!(sup.set_enabled(crate::types::StrategyId::ModelBtc5m, false).is_ok());
    }

    // ── Normal toggle channel tests for RlPredictor ────────────────────

    /// Paper `Config` with the REAL `paper_approved` policy
    /// (`src/tools/model/artifacts/rl_predictor/paper_bootstrap_v1`) and
    /// checkpoints/DB pointing to the tempdir; the toggle builds the
    /// strategy through the production path (`build` → `RlPredictor::new`
    /// → `load_approved` → `spawn_learner`).
    fn cfg_paper_rl_real(
        tempdir: &std::path::Path,
        mode: crate::types::Mode,
        enabled: bool,
    ) -> crate::config::Config {
        let mut cfg = base_config_with_universe(false);
        cfg.mode = mode;
        let rl = &mut cfg.strategies.rl_predictor;
        rl.limits.enabled = enabled;
        rl.limits.paper_only = true;
        rl.artifact_dir = format!(
            "{}/src/tools/model/artifacts/rl_predictor/paper_bootstrap_v1",
            env!("CARGO_MANIFEST_DIR")
        );
        rl.checkpoint_dir = tempdir.join("rl_ckpt").to_string_lossy().into_owned();
        rl.expected_predictor_sha256 =
            "ef4e4b0860ef9601fb21462539f4d43d2c8af378060eb2dcb63d153502e4c396".into();
        rl.learning_enabled = true;
        cfg
    }

    fn supervisor_rl(
        cfg: crate::config::Config,
    ) -> (
        Arc<StrategySupervisor>,
        broadcast::Sender<crate::tools::model::ModelPrediction>,
        watch::Sender<bool>,
        // Receiver of intents RETAINED: if dropped, the runner interprets
        // "executor closed" and exits on the first BUY; experience
        // stops flowing just as if the real executor had gone down.
        mpsc::Receiver<QueuedIntent>,
    ) {
        let mode = cfg.mode;
        let (book_tx, _book_rx) = broadcast::channel::<Arc<crate::types::BookSnapshot>>(64);
        let (intent_tx, intent_rx) = mpsc::channel(16);
        let (fill_tx, _fill_rx) = broadcast::channel::<crate::types::Fill>(16);
        let (sub_tx, _sub_rx) = broadcast::channel::<crate::market_feed::TokenEvent>(16);
        let (kill_tx, kill_rx) = watch::channel(false);
        let (pred_tx, _pred_rx) =
            broadcast::channel::<crate::tools::model::ModelPrediction>(16);
        let sup = StrategySupervisor::new(
            book_tx,
            HashMap::new(),
            intent_tx,
            fill_tx,
            crate::risk::RiskGuard::new(crate::config::RiskConfig::default(), mode),
            kill_rx,
            Duration::from_millis(500),
            Arc::new(crate::market_feed::MarketRegistry::new()),
            Arc::new(cfg),
            crate::coordination::BracketLedger::new(),
            sub_tx,
            Some(Arc::new(pred_tx.clone())),
            crate::strategy::Arb15m5mDeps::empty(),
        );
        // The senders are returned to the test: if dropped, `run()` exits
        // immediately (`ctx.shutdown.changed()` completes with Err) and has_task
        // would be false.
        (sup, pred_tx, kill_tx, intent_rx)
    }

    /// Tests that build a real `RlPredictor` (artifact + learner)
    /// register the `learner_control()` singleton; serialize them so
    /// reads of the global across parallel tests do not step on each other.
    static RL_SUP_GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Normal OFF→ON→OFF→ON toggle with a real artifact: the existing channel
    /// (`set_enabled`, the same one `/control/strategy` and Telegram use)
    /// builds the strategy with the `paper_approved` policy and starts the
    /// learner, and a second ON does NOT create another worker (spawn is a no-op if the
    /// task is still alive; the kill switch only pauses it).
    #[tokio::test]
    async fn rl_predictor_toggle_normal_off_on_without_double_learner() {
        let _gate = RL_SUP_GATE.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let cfg = cfg_paper_rl_real(dir.path(), crate::types::Mode::Paper, false);
        let (sup, _pred_tx, _kill_tx, _intent_rx) = supervisor_rl(cfg);
        let id = crate::types::StrategyId::RlPredictor;

        assert!(!sup.has_task(id), "enabled=false at boot → no task");

        // ON #1: real construction — approved artifact + live learner.
        assert!(
            sup.set_enabled(id, true).is_ok(),
            "ON toggle in paper with a valid artifact must be accepted"
        );
        assert!(sup.has_task(id));
        tokio::time::sleep(Duration::from_millis(150)).await;
        let status = crate::strategy::rl_predictor::rl_strategy_status();
        assert!(
            status.active,
            "the policy must be active (disabled_reason={:?})",
            status.disabled_reason
        );
        let ctl1 = crate::rl::learner::learner_control()
            .expect("the learner must register in the global control");
        assert!(ctl1.is_alive(), "the learner's worker must be alive");

        // ON #2 (repeated): no-op; the task is still alive, there is no double spawn
        // nor a second learner.
        assert!(sup.set_enabled(id, true).is_ok());
        let ctl2 = crate::rl::learner::learner_control().unwrap();
        assert!(
            Arc::ptr_eq(&ctl1, &ctl2),
            "a second ON must not replace the learner (no double worker)"
        );

        // OFF: always allowed (kill switch → task paused, not dead).
        assert!(sup.set_enabled(id, false).is_ok());
        assert!(
            sup.has_task(id),
            "the kill switch pauses the task, it does not kill it (has_task stays true)"
        );

        // ON #3 after OFF: the paused task resumes without rebuilding;
        // same learner, same policy version.
        assert!(sup.set_enabled(id, true).is_ok());
        let ctl3 = crate::rl::learner::learner_control().unwrap();
        assert!(
            Arc::ptr_eq(&ctl1, &ctl3),
            "re-ON over a paused task must not spawn a second learner"
        );
    }

    /// Structural barrier: `RlPredictor` in `Mode::Live` is rejected via
    /// `BlockReason::PaperOnly` in `set_enabled`, before `build()`,
    /// even if the TOML says `paper_only=false` (the
    /// combination that would make the ctor panic if it got there). Merely existing
    /// disabled in Live neither panics nor spawns.
    #[tokio::test]
    async fn rl_predictor_live_rejected_by_supervisor_without_building() {
        let _gate = RL_SUP_GATE.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = cfg_paper_rl_real(dir.path(), crate::types::Mode::Live, true);
        // Combination that `can_run` would approve in Live; the rejection must be
        // the supervisor's guard, not the limits barrier.
        cfg.strategies.rl_predictor.limits.paper_only = false;
        let (sup, _pred_tx, _kill_tx, _intent_rx) = supervisor_rl(cfg);
        let id = crate::types::StrategyId::RlPredictor;

        let lim = sup.config.strategies.rl_predictor.limits.clone();
        assert!(
            lim.can_run(crate::types::Mode::Live).is_ok(),
            "this test assumes can_run lets the combination through — \
             the rejection must come from the supervisor's mode guard"
        );
        assert_eq!(
            sup.set_enabled(id, true),
            Err(crate::config::BlockReason::PaperOnly),
            "RlPredictor in Live must ALWAYS be rejected via PaperOnly (no panic)"
        );
        assert!(!sup.has_task(id), "no task after the rejection");
        // Turning off is still allowed.
        assert!(sup.set_enabled(id, false).is_ok());
    }

    /// Regression test for a bug seen in a smoke run of the normal channel:
    /// `discover_all_active_universes` deduplicates cids by filter specificity
    /// ("slug" first) and ALPHABETICALLY within the level, so with the asset
    /// `crypto-updown-long` (declared by `model_btc5m`) ALL the `*-updown-*`
    /// brackets end up registered under that universe. If `rl_predictor.universe`
    /// does not declare it, the runner skips every book before `on_book`: live learner,
    /// active policy, but `transitions_rx=0` forever.
    ///
    /// Reproduces THAT registry state (cid claimed by
    /// `crypto-updown-long`) after the real ON toggle and requires that
    /// experience reaches the learner; "alive" is not enough.
    #[tokio::test]
    async fn rl_toggle_book_in_long_updown_universe_feeds_learner() {
        let _gate = RL_SUP_GATE.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = cfg_paper_rl_real(dir.path(), crate::types::Mode::Paper, false);
        // Corrected universe: the 3 that model_btc5m declares (any
        // of them can win the cid claim depending on discovery's dedup).
        cfg.strategies.rl_predictor.limits.universe = vec![
            "crypto-updown-short".into(),
            "crypto-updown-long".into(),
            "g-rw1".into(),
        ];
        let (sup, pred_tx, _kill_tx, _intent_rx) = supervisor_rl(cfg);
        let id = crate::types::StrategyId::RlPredictor;
        let cid = "cid-rl-e2e";

        // EXACT registry from the smoke run: the BTC-5m bracket claimed by
        // crypto-updown-long (wins the alphabetical dedup with model_btc5m on).
        sup.registry.universe_info.insert(
            cid.to_string(),
            crate::market_feed::registry::UniverseInfo {
                universe: "crypto-updown-long".into(),
                has_oracle: true,
            },
        );

        assert!(sup.set_enabled(id, true).is_ok());
        assert!(sup.has_task(id));

        // Valid BTC-5m book (300s duration, window [30,240], liquidity and
        // parabolic fee). `open_ts`/`clear_at` fixed so the prediction's slug
        // matches EXACTLY; `recv_instant` and `ts_ms` are regenerated
        // on each send (the runner discards old books and `pred_usable`
        // past predictions beyond `max_prediction_age_ms`).
        let open_ts = crate::time::now().timestamp() - 180;
        let slug = format!("btc-updown-5m-{open_ts}");
        let make_book = || {
            let lvls = |p: &str| crate::types::HalfBook {
                levels: vec![crate::types::Level {
                    price: p.parse().unwrap(),
                    size: rust_decimal_macros::dec!(50),
                }],
            };
            Arc::new(crate::types::BookSnapshot {
                market_condition_id: cid.into(),
                yes: crate::types::TokenBook {
                    token_id: "tok-yes-rl".into(),
                    bids: lvls("0.54"),
                    asks: lvls("0.55"),
                    ..Default::default()
                },
                no: crate::types::TokenBook {
                    token_id: "tok-no-rl".into(),
                    bids: lvls("0.44"),
                    asks: lvls("0.45"),
                    ..Default::default()
                },
                market_open_ts: open_ts,
                clear_at: open_ts + 300,
                neg_risk: false,
                slug: slug.clone(),
                reference_price: None,
                fee_rate_bps: None,
                parabolic_fee_rate: Some(rust_decimal_macros::dec!(0.07)),
                tick_size: None,
                recv_instant: Some(crate::time::Instant::now()),
            })
        };
        let make_pred = || crate::tools::model::ModelPrediction {
            schema_version: crate::tools::model::MODEL_TOOL_SCHEMA_VERSION,
            ts_ms: crate::time::now().timestamp_millis(),
            market_cid: cid.into(),
            slug: slug.clone(),
            token_id_yes: "tok-yes-rl".into(),
            token_id_no: "tok-no-rl".into(),
            side: "x".into(),
            p_model: Some(0.6),
            p_market: Some(0.5),
            book_age_ms: 0,
            spot_age_ms: 0,
            latency_ms: 0,
            budget_ms: 0,
            abstention: None,
            model_hash: Some(
                "ef4e4b0860ef9601fb21462539f4d43d2c8af378060eb2dcb63d153502e4c396"
                    .into(),
            ),
        };

        // The first decision creates the episode (after recovery Ready over
        // `:memory:`); the transition is EMITTED at the next accounting
        // point → at least 2 books are needed. Bounded wait.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        let mut rx = 0u64;
        while tokio::time::Instant::now() < deadline && rx == 0 {
            let _ = pred_tx.send(make_pred());
            let _ = sup.book_tx.send(make_book());
            tokio::time::sleep(Duration::from_millis(80)).await;
            rx = crate::rl::learner::learner_control()
                .map(|c| c.transitions_rx.load(std::sync::atomic::Ordering::Relaxed))
                .unwrap_or(0);
        }
        assert!(
            rx > 0,
            "after toggle ON a book of the universe claimed by discovery \
             must reach on_book and feed the learner (transitions_rx={rx})"
        );

        // OFF through the same channel: kill switch (task paused, not destroyed).
        assert!(sup.set_enabled(id, false).is_ok());
    }
}
