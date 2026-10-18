//! Post-fill markout computation.
//!
//! Subscribes to the fill and book broadcasts, and for each fill captures the
//! token's mid at +5s and +30s. Writes ONE row to `markouts` per fill (when
//! the 30s deadline expires), off the order hot path.
//!
//! The markout answers "did the price move in my favor or against me right
//! after my fill?" — it is the adverse-selection metric required by the
//! gates of MM/PM/ThetaCapture. Sign convention: positive = in favor.
//!
//!   BUY:  markout = (mid − fill_price) / fill_price
//!   SELL: markout = (fill_price − mid) / fill_price

use std::collections::HashMap;
use std::sync::Arc;

use rust_decimal::Decimal;
use tokio::sync::{broadcast, watch};
use tokio::time::{Duration, Instant};
use tracing::{info, warn};

use crate::{
    state::StateStore,
    types::{BookSnapshot, Fill, FillSide},
};

/// Signed fractional markout (positive = in favor). In bps: × 10⁴.
pub fn markout_frac(side: FillSide, fill_price: Decimal, mid: Decimal) -> Decimal {
    if fill_price == Decimal::ZERO {
        return Decimal::ZERO;
    }
    match side {
        FillSide::Buy => (mid - fill_price) / fill_price,
        FillSide::Sell => (fill_price - mid) / fill_price,
        FillSide::Rejected | FillSide::Cancelled | FillSide::GateRejected => Decimal::ZERO,
    }
}

/// Freshness tolerance for the +5s/+30s mids. A mid captured more
/// than this BEFORE the target deadline is considered stale — the real scenario
/// is that after `clear_at` the bracket stops emitting books (registry
/// eviction), so at T+5/T+30 the `mids` map may still hold the last value from
/// several seconds earlier. Persisting it as if it were fresh contaminates
/// `markout@5s`, the step-to-live gate for MM/PM/ThetaCapture.
const MID_FRESHNESS_TOL: Duration = Duration::from_secs(2);

struct Pending {
    fill: Fill,
    /// Instant at which the fill reached this task — a mid seen BEFORE this
    /// cannot be a valid post-fill observation.
    arrived: Instant,
    t5: Instant,
    t30: Instant,
    mid_5s: Option<Decimal>,
    done_5s: bool,
}

/// Filters the known mid of `token` by freshness against the `target`
/// deadline (t5 or t30) — `None` if there is no sufficiently recent
/// observation instead of filling in a stale value. Three conditions, all
/// required: (1) the mid was seen AFTER the fill arrived, (2) it was seen
/// BEFORE the target deadline — `saturating_duration_since` returned 0
/// for `seen > target` and accepted future mids (lookahead, biasing toward more
/// adverse selection in the MM/PM/ThetaCapture gate) — and (3) it is not more than
/// `MID_FRESHNESS_TOL` behind the deadline.
fn fresh_mid(
    mids: &HashMap<String, (Decimal, Instant)>,
    token: &str,
    arrived: Instant,
    target: Instant,
) -> Option<Decimal> {
    mids.get(token).and_then(|(m, seen)| {
        if *seen >= arrived && *seen <= target && target.duration_since(*seen) <= MID_FRESHNESS_TOL {
            Some(*m)
        } else {
            None
        }
    })
}

/// Registers a fill as `Pending` with its t5/t30 horizons. Shared by the
/// receive arm and the exit drain — in the latter
/// `arrived = now`: it is not known how long it had been queued and there is no point
/// inventing an earlier arrival (conservative: no already-seen mid counts
/// as a post-fill observation for it).
fn push_pending(
    pending: &mut Vec<Pending>,
    fill: Fill,
    delay_5s: Duration,
    delay_30s: Duration,
) {
    let now = Instant::now();
    pending.push(Pending {
        fill,
        arrived: now,
        t5: now + delay_5s,
        t30: now + delay_30s,
        mid_5s: None,
        done_5s: false,
    });
}

/// Horizon sweep over `pending`.
///
/// - Captures `mid_5s` for each pending whose t5 has expired at `now` (with the
///   `fresh_mid` tolerance: only samples `arrived <= seen <= target`).
/// - Persists the `markouts` row of each pending whose t30 expired, and removes it
///   from the list — each pending writes AT MOST ONE row, so a
///   timer sweep followed by the terminal flush never duplicates rows.
/// - With `terminal = true` (exit on shutdown or closed channel) it writes
///   ALL remaining pendings: horizons not yet elapsed are
///   persisted as NULL — an observation that did not happen is never feigned, nor
///   are we waiting for them to expire (shutdown cannot sit idle for 30 s).
async fn flush_expired(
    pending: &mut Vec<Pending>,
    mids: &HashMap<String, (Decimal, Instant)>,
    state: &StateStore,
    now: Instant,
    terminal: bool,
) {
    // Capture at +5s for horizons that have already expired.
    for p in pending.iter_mut() {
        if !p.done_5s && now >= p.t5 {
            // Discard stale mids (before the fill or
            // too old relative to the +5s deadline) and mids AFTER the
            // deadline — neither is a valid observation of the horizon.
            p.mid_5s = fresh_mid(mids, &p.fill.token_id, p.arrived, p.t5);
            p.done_5s = true;
            if let Some(mid) = p.mid_5s {
                use rust_decimal::prelude::ToPrimitive;
                let bps = markout_frac(p.fill.side.clone(), p.fill.price, mid) * Decimal::from(10_000);
                metrics::histogram!("hrf_markout_5s_bps", "strategy" => p.fill.strategy.to_string())
                    .record(bps.to_f64().unwrap_or_default());
            }
        }
    }
    // Rows to write: those with t30 expired, or all of them in the terminal flush.
    let mut done: Vec<Pending> = Vec::new();
    let mut keep: Vec<Pending> = Vec::new();
    for p in pending.drain(..) {
        if terminal || now >= p.t30 {
            done.push(p);
        } else {
            keep.push(p);
        }
    }
    *pending = keep;
    for p in done {
        // Same freshness filter against t30; if the
        // horizon has not elapsed (only possible in the terminal flush) the
        // column stays NULL — there is no valid sample to contribute.
        let mid_30s = if now >= p.t30 {
            fresh_mid(mids, &p.fill.token_id, p.arrived, p.t30)
        } else {
            None
        };
        let side = match p.fill.side {
            FillSide::Buy => "BUY",
            FillSide::Sell => "SELL",
            FillSide::Rejected | FillSide::Cancelled | FillSide::GateRejected => continue,
        };
        if let Err(e) = state
            .record_markout(
                p.fill.ts.timestamp_millis(),
                &p.fill.market_cid,
                &p.fill.token_id,
                &p.fill.strategy.to_string(),
                side,
                p.fill.price,
                p.mid_5s,
                mid_30s,
            )
            .await
        {
            warn!("markout: error writing row: {e}");
        }
    }
}

/// Drains the FINITE PREFIX of fills already accepted by the
/// broadcast — only the `len()` messages visible at the instant of
/// exit. The budget is captured ONCE and never reloaded: if live
/// producers keep publishing during the drain, whatever arrives
/// after the boundary is NOT incorporated — there is no delivery guarantee for
/// sends after the drain starts (nor exactly-once across a crash
/// between accepting and persisting). Drained fills are persisted as partial
/// rows in the terminal flush (their horizons have not elapsed → NULL).
fn drain_fills(
    fill_rx: &mut broadcast::Receiver<Fill>,
    pending: &mut Vec<Pending>,
    delay_5s: Duration,
    delay_30s: Duration,
) {
    drain_fills_budget(fill_rx, pending, delay_5s, delay_30s, fill_rx.len());
    let remaining_len = fill_rx.len();
    if remaining_len > 0 {
        // Not a silent loss: these are messages published after the
        // drain boundary (producers still live) — left out by
        // design, not promised.
        info!(remaining_len, "markout: fills after the boundary not drained");
    }
}

/// Consumes at most `budget` positions from the channel — the prefix
/// of messages already buffered when it was captured. Separate from `drain_fills` so
/// tests can fix the boundary deterministically.
///
/// - Each `Ok` consumes 1 even if the fill is a non-economic terminal
///   (Rejected/Cancelled/GateRejected generate no Pending but do spend a
///   position of the prefix).
/// - `Lagged(n)` consumes `min(n, budget)` positions: the lag dropped `n`
///   messages of the prefix. If it exhausts what is left, it ends without incorporating
///   anything — everything still in the queue arrived after the boundary. If
///   budget remains, the originals that survived the lag are drained
///   normally (a partial lag does not abandon a valid prefix).
/// - `Empty`/`Closed` end. No `await`: the loop is bounded by the
///   initial budget, impossible to extend by concurrent producers.
fn drain_fills_budget(
    fill_rx: &mut broadcast::Receiver<Fill>,
    pending: &mut Vec<Pending>,
    delay_5s: Duration,
    delay_30s: Duration,
    mut budget: usize,
) {
    while budget > 0 {
        match fill_rx.try_recv() {
            Ok(f) => {
                budget -= 1;
                if !matches!(
                    f.side,
                    FillSide::Rejected | FillSide::Cancelled | FillSide::GateRejected
                ) {
                    push_pending(pending, f, delay_5s, delay_30s);
                }
            }
            Err(broadcast::error::TryRecvError::Lagged(n)) => {
                let lost = usize::try_from(n).unwrap_or(usize::MAX);
                warn!(
                    lost,
                    remaining = budget,
                    "markout: fills lost to lag during the exit drain"
                );
                if lost >= budget {
                    // The whole remaining prefix was lost: whatever is left in the
                    // queue is after the boundary — not incorporated.
                    break;
                }
                budget -= lost;
            }
            // Empty (queue empty) or Closed (sender dropped): end of the drain.
            Err(_) => break,
        }
    }
}

/// Markout task. `delay_5s`/`delay_30s` are parameters so it can be tested
/// with short deadlines; main passes the real 5s and 30s.
pub async fn run(
    state: Arc<StateStore>,
    mut fill_rx: broadcast::Receiver<Fill>,
    mut book_rx: broadcast::Receiver<Arc<BookSnapshot>>,
    mut shutdown: watch::Receiver<bool>,
    delay_5s: Duration,
    delay_30s: Duration,
) {
    info!("markout: starting");
    // Last known mid per token. Purged along with the pendings: tokens
    // of old brackets stop emitting books and the map is cleaned by retention.
    let mut mids: HashMap<String, (Decimal, Instant)> = HashMap::new();
    let mut pending: Vec<Pending> = Vec::new();

    loop {
        // Next pending deadline; if none, sleep long (events wake it earlier).
        let next_deadline = pending
            .iter()
            .map(|p| if p.done_5s { p.t30 } else { p.t5 })
            .min()
            .unwrap_or_else(|| Instant::now() + Duration::from_secs(3600));

        tokio::select! {
            book = book_rx.recv() => match book {
                Ok(b) => {
                    let now = Instant::now();
                    // BEFORE overwriting the last known mid, the already-expired
                    // horizons must be captured. If a pending's deadline
                    // passed since the earlier book, the valid sample
                    // is the one in `mids` (pre-deadline); this book
                    // arrives AFTER the target and must neither erase that sample nor
                    // sneak in as a valid observation — and if the timer
                    // arm already captured it, this sweep is a no-op.
                    flush_expired(&mut pending, &mids, &state, now, false).await;
                    if let Some(m) = b.yes_mid() { mids.insert(b.yes.token_id.clone(), (m, now)); }
                    if let Some(m) = b.no_mid()  { mids.insert(b.no.token_id.clone(),  (m, now)); }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!(n, "markout: books lost to broadcast lag");
                }
                // Books channel closed — keep the pendings.
                Err(broadcast::error::RecvError::Closed) => {
                    info!("markout: books channel closed — flushing pendings");
                    break;
                }
            },
            fill = fill_rx.recv() => match fill {
                Ok(f) if !matches!(f.side, FillSide::Rejected | FillSide::Cancelled | FillSide::GateRejected) => {
                    push_pending(&mut pending, f, delay_5s, delay_30s);
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!(n, "markout: fills lost to broadcast lag");
                }
                // Fills channel closed — keep the pendings.
                Err(broadcast::error::RecvError::Closed) => {
                    info!("markout: fills channel closed — flushing pendings");
                    break;
                }
            },
            _ = tokio::time::sleep_until(next_deadline) => {
                let now = Instant::now();
                // Capture at +5s and write at +30s (one row per fill).
                flush_expired(&mut pending, &mids, &state, now, false).await;
                // Retention of the mids map: tokens with no book in 10 min no longer matter.
                mids.retain(|_, (_, seen)| now.duration_since(*seen) < Duration::from_secs(600));
            },
            res = shutdown.changed() => match res {
                Ok(()) if *shutdown.borrow() => {
                    info!("markout: shutdown received — flushing pendings");
                    break;
                }
                Ok(()) => {}
                // If the watch sender dropped, `changed()` returns Err
                // forever — exit with a flush, do not busy-loop.
                Err(_) => {
                    warn!("markout: shutdown watch closed — flushing pendings");
                    break;
                }
            },
        }
    }

    // EVERY exit keeps what was already accepted. (1) Fills left
    // queued in the broadcast when the select broke are drained with
    // `try_recv` bounded to the prefix visible then (`len()` captured once)
    // — messages live producers emit afterwards are neither incorporated
    // nor promised, and the drain cannot hang behind them.
    // (2) All pendings are persisted: expired horizons with their valid
    // sample (or NULL if stale), non-elapsed ones always NULL — an observation is
    // never feigned nor do we wait for t30 to expire.
    drain_fills(&mut fill_rx, &mut pending, delay_5s, delay_30s);
    flush_expired(&mut pending, &mids, &state, Instant::now(), true).await;
    info!("markout: stopped");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn markout_buy_in_favor_is_positive() {
        // Buy at 0.50, mid rises to 0.52 → +4% (+400 bps).
        assert_eq!(
            markout_frac(FillSide::Buy, dec!(0.50), dec!(0.52)),
            dec!(0.04)
        );
    }

    #[test]
    fn markout_sell_in_favor_is_positive() {
        // Sell at 0.50, mid falls to 0.48 → sold high → +4%.
        assert_eq!(
            markout_frac(FillSide::Sell, dec!(0.50), dec!(0.48)),
            dec!(0.04)
        );
    }

    #[test]
    fn markout_buy_adverse_is_negative() {
        assert_eq!(
            markout_frac(FillSide::Buy, dec!(0.50), dec!(0.45)),
            dec!(-0.1)
        );
    }

    #[test]
    fn markout_zero_price_does_not_divide() {
        assert_eq!(
            markout_frac(FillSide::Buy, dec!(0), dec!(0.5)),
            Decimal::ZERO
        );
    }

    // ── fresh_mid must discard stale mids ─────────────────────────────────────

    #[test]
    fn fresh_mid_discards_value_before_fill() {
        let now = Instant::now();
        let mut mids = HashMap::new();
        // mid seen BEFORE the fill arrived — it cannot be a valid post-fill
        // observation even if it is "close" in time to the target.
        mids.insert("tok".to_string(), (dec!(0.50), now));
        let arrived = now + Duration::from_secs(1);
        let target = arrived + Duration::from_secs(5);
        assert_eq!(fresh_mid(&mids, "tok", arrived, target), None);
    }

    #[test]
    fn fresh_mid_discards_value_too_old_relative_to_target() {
        // The bracket stopped emitting books after clear_at: the last known mid
        // is far behind the target deadline (+5s/+30s) — it must not be used.
        let now = Instant::now();
        let arrived = now;
        let target = now + Duration::from_secs(5);
        let mut mids = HashMap::new();
        mids.insert(
            "tok".to_string(),
            (dec!(0.50), now + Duration::from_millis(500)),
        );
        assert_eq!(fresh_mid(&mids, "tok", arrived, target), None);
    }

    #[test]
    fn fresh_mid_accepts_value_within_tolerance() {
        let now = Instant::now();
        let arrived = now;
        let target = now + Duration::from_secs(5);
        let mut mids = HashMap::new();
        mids.insert(
            "tok".to_string(),
            (dec!(0.52), target - Duration::from_millis(500)),
        );
        assert_eq!(fresh_mid(&mids, "tok", arrived, target), Some(dec!(0.52)));
    }

    #[test]
    fn fresh_mid_absent_returns_none() {
        let now = Instant::now();
        let mids: HashMap<String, (Decimal, Instant)> = HashMap::new();
        assert_eq!(
            fresh_mid(&mids, "tok-nonexistent", now, now + Duration::from_secs(5)),
            None
        );
    }

    use crate::types::{HalfBook, Level, Mode, StrategyId, TokenBook};

    fn book(token_yes: &str, bid: &str, ask: &str) -> Arc<BookSnapshot> {
        let half = |p: &str| HalfBook {
            levels: vec![Level {
                price: p.parse().unwrap(),
                size: dec!(100),
            }],
        };
        Arc::new(BookSnapshot {
            market_condition_id: "cid-mk".into(),
            yes: TokenBook {
                token_id: token_yes.into(),
                bids: half(bid),
                asks: half(ask),
                ..Default::default()
            },
            no: TokenBook {
                token_id: "tok-no".into(),
                bids: half("0.40"),
                asks: half("0.60"),
                ..Default::default()
            },
            market_open_ts: 0,
            clear_at: 9_999_999_999,
            neg_risk: false,
            slug: "test".into(),
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        })
    }

    fn fill(price: &str) -> Fill {
        Fill {
            hedge_id: None,
            exec_id: None,
            local_order_id: None,
            market_cid: "cid-mk".into(),
            token_id: "tok-yes".into(),
            side: FillSide::Buy,
            price: price.parse().unwrap(),
            size: dec!(10),
            order_id: None,
            ts: chrono::Utc::now(),
            mode: Mode::Paper,
            neg_risk: false,
            strategy: StrategyId::MarketMaking,
            clear_at: 9_999_999_999,
            outcome_label: None,
        }
    }

    #[tokio::test]
    async fn run_writes_markout_with_synthetic_books() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let (fill_tx, fill_rx) = broadcast::channel(16);
        let (book_tx, book_rx) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        tokio::spawn(run(
            state.clone(),
            fill_rx,
            book_rx,
            kill_rx,
            Duration::from_millis(40),
            Duration::from_millis(80),
        ));

        // Fill at 0.50; before the "5s" deadline the mid rises to 0.52 → markout +400bps.
        fill_tx.send(fill("0.50")).unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        book_tx.send(book("tok-yes", "0.51", "0.53")).unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;

        let rows = state.markouts_for_strategy("market_making").await.unwrap();
        assert_eq!(rows.len(), 1, "there must be one markout row");
        let (token, fill_price, mid_5s, mid_30s) = &rows[0];
        assert_eq!(token, "tok-yes");
        assert_eq!(*fill_price, dec!(0.50));
        assert_eq!(*mid_5s, Some(dec!(0.52)));
        assert_eq!(*mid_30s, Some(dec!(0.52)));
    }

    // ── mids after the deadline are not valid observations ────────────────

    #[test]
    fn fresh_mid_rejects_mid_after_deadline() {
        // `saturating_duration_since` returned 0 for `seen > target` and the
        // future mid passed as fresh (lookahead).
        let now = Instant::now();
        let arrived = now;
        let target = now + Duration::from_secs(5);
        let mut mids = HashMap::new();
        mids.insert(
            "tok".to_string(),
            (dec!(0.60), target + Duration::from_millis(400)),
        );
        assert_eq!(fresh_mid(&mids, "tok", arrived, target), None);
    }

    /// Interleaving of a ready deadline vs later books: a book that arrives
    /// AFTER t5 but is processed before the timer arm must not erase
    /// the earlier valid sample (it is captured before overwriting `mids`)
    /// nor sneak in as a +5s observation. Deterministic: with book→timer
    /// order the pre-deadline sample wins; with timer→book, the same.
    #[tokio::test]
    async fn book_after_deadline_does_not_displace_valid_sample() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let (fill_tx, fill_rx) = broadcast::channel(16);
        let (book_tx, book_rx) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let handle = tokio::spawn(run(
            state.clone(),
            fill_rx,
            book_rx,
            kill_rx,
            Duration::from_millis(500),
            Duration::from_millis(1_000),
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;

        fill_tx.send(fill("0.50")).unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        // Valid sample: arrives after the fill and before the t5 deadline.
        book_tx.send(book("tok-yes", "0.51", "0.53")).unwrap();
        // Book AFTER the t5 deadline (~fill+500ms) but before t30.
        tokio::time::sleep(Duration::from_millis(550)).await;
        book_tx.send(book("tok-yes", "0.98", "1.00")).unwrap();
        tokio::time::sleep(Duration::from_millis(600)).await;

        let rows = state.markouts_for_strategy("market_making").await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].2,
            Some(dec!(0.52)),
            "mid_5s must be the pre-deadline sample, never the post-deadline one"
        );
        assert_eq!(
            rows[0].3,
            Some(dec!(0.99)),
            "the same book is a valid sample for the t30 horizon"
        );

        drop(fill_tx);
        drop(book_tx);
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
    }

    /// If the only thing that arrived before the capture is a mid AFTER the
    /// deadline, the markout stays NULL — future data is not reused as
    /// if it were the horizon's observation.
    #[tokio::test]
    async fn without_predeadline_sample_mid_5s_is_null() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let (fill_tx, fill_rx) = broadcast::channel(16);
        let (book_tx, book_rx) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let handle = tokio::spawn(run(
            state.clone(),
            fill_rx,
            book_rx,
            kill_rx,
            Duration::from_millis(200),
            Duration::from_millis(700),
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;

        fill_tx.send(fill("0.50")).unwrap();
        // No book at all before t5: the only one that arrives (~fill+450ms) is
        // after the deadline — whatever the select order, it does not count.
        tokio::time::sleep(Duration::from_millis(450)).await;
        book_tx.send(book("tok-yes", "0.51", "0.53")).unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;

        let rows = state.markouts_for_strategy("market_making").await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].2, None,
            "there was no pre-deadline sample: mid_5s must stay NULL"
        );
        assert_eq!(
            rows[0].3,
            Some(dec!(0.52)),
            "that book is valid for t30 (it arrived before that deadline)"
        );

        drop(fill_tx);
        drop(book_tx);
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
    }

    // ── every exit keeps the accepted pendings ──────────────────────────────

    /// Shutdown before t5: the row is written NOW with both horizons NULL —
    /// we never wait 30 s on shutdown nor feign a future observation.
    #[tokio::test]
    async fn shutdown_pre_t5_persists_partial_row() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let (fill_tx, fill_rx) = broadcast::channel(16);
        let (book_tx, book_rx) = broadcast::channel(16);
        let (kill_tx, kill_rx) = watch::channel(false);
        let handle = tokio::spawn(run(
            state.clone(),
            fill_rx,
            book_rx,
            kill_rx,
            Duration::from_millis(500),
            Duration::from_millis(10_000),
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        fill_tx.send(fill("0.50")).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        kill_tx.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("shutdown must not wait for the horizons")
            .unwrap();

        let rows = state.markouts_for_strategy("market_making").await.unwrap();
        assert_eq!(rows.len(), 1, "the accepted pending is persisted on shutdown");
        assert_eq!(rows[0].2, None, "t5 did not expire → NULL");
        assert_eq!(rows[0].3, None, "t30 did not expire → NULL");
        drop(fill_tx);
        drop(book_tx);
    }

    /// Shutdown between t5 and t30: the expired +5s with a valid sample is kept;
    /// the non-elapsed +30s stays NULL.
    #[tokio::test]
    async fn shutdown_between_t5_and_t30_keeps_mid_5s() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let (fill_tx, fill_rx) = broadcast::channel(16);
        let (book_tx, book_rx) = broadcast::channel(16);
        let (kill_tx, kill_rx) = watch::channel(false);
        let handle = tokio::spawn(run(
            state.clone(),
            fill_rx,
            book_rx,
            kill_rx,
            Duration::from_millis(300),
            Duration::from_millis(10_000),
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        fill_tx.send(fill("0.50")).unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        book_tx.send(book("tok-yes", "0.51", "0.53")).unwrap();
        // Beyond t5 (300ms since the fill): already captured or capturable.
        tokio::time::sleep(Duration::from_millis(500)).await;

        kill_tx.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("shutdown must not wait for t30")
            .unwrap();

        let rows = state.markouts_for_strategy("market_making").await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].2, Some(dec!(0.52)), "t5 expired with a valid sample");
        assert_eq!(rows[0].3, None, "t30 did not elapse → NULL");
        drop(fill_tx);
        drop(book_tx);
    }

    /// Books channel closure: the pendings are flushed and the fills that were
    /// already queued in the broadcast are drained as partial rows — without
    /// waiting for the fills sender (still alive; the drain is `try_recv`).
    #[tokio::test]
    async fn closed_books_channel_persists_pending_and_drains_fills() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let (fill_tx, fill_rx) = broadcast::channel(16);
        let (book_tx, book_rx) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let handle = tokio::spawn(run(
            state.clone(),
            fill_rx,
            book_rx,
            kill_rx,
            Duration::from_millis(500),
            Duration::from_millis(10_000),
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        fill_tx.send(fill("0.50")).unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        // Fill queued "at the same time" as the books channel closes.
        let mut f2 = fill("0.55");
        f2.exec_id = Some("exec-2".into());
        fill_tx.send(f2).unwrap();
        drop(book_tx);

        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("closing the books channel must not hang the task")
            .unwrap();

        let rows = state.markouts_for_strategy("market_making").await.unwrap();
        assert_eq!(
            rows.len(),
            2,
            "pending + fill drained from the queue: both persist as partial"
        );
        assert!(rows.iter().all(|r| r.2.is_none() && r.3.is_none()));
        drop(fill_tx);
    }

    /// Fills channel closure: the accepted pending is persisted and the task
    /// exits without hanging even if the books producer is still alive.
    #[tokio::test]
    async fn closed_fills_channel_persists_pending() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let (fill_tx, fill_rx) = broadcast::channel(16);
        let (_book_tx, book_rx) = broadcast::channel(16);
        let (_kill_tx, kill_rx) = watch::channel(false);
        let handle = tokio::spawn(run(
            state.clone(),
            fill_rx,
            book_rx,
            kill_rx,
            Duration::from_millis(500),
            Duration::from_millis(10_000),
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        fill_tx.send(fill("0.50")).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(fill_tx);

        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("closing the fills channel must end the task")
            .unwrap();

        let rows = state.markouts_for_strategy("market_making").await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].2, None);
        assert_eq!(rows[0].3, None);
    }

    /// Watch sender dropped (`changed()` returns Err): the task must
    /// exit — it used to be an infinite busy loop — flushing the pendings.
    #[tokio::test]
    async fn closed_watch_exits_with_flush() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let (fill_tx, fill_rx) = broadcast::channel(16);
        let (_book_tx, book_rx) = broadcast::channel(16);
        let (kill_tx, kill_rx) = watch::channel(false);
        let handle = tokio::spawn(run(
            state.clone(),
            fill_rx,
            book_rx,
            kill_rx,
            Duration::from_millis(500),
            Duration::from_millis(10_000),
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        fill_tx.send(fill("0.50")).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(kill_tx); // sender dropped → changed() = Err

        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("a watch without a sender must not be a busy loop")
            .unwrap();

        let rows = state.markouts_for_strategy("market_making").await.unwrap();
        assert_eq!(rows.len(), 1, "the pending is also kept on this exit");
        drop(fill_tx);
    }

    /// A row already written by the timer is not duplicated when combined with the
    /// shutdown's terminal flush.
    #[tokio::test]
    async fn already_written_row_is_not_duplicated_on_shutdown() {
        let state = StateStore::open_memory(Mode::Paper).await.unwrap();
        let (fill_tx, fill_rx) = broadcast::channel(16);
        let (book_tx, book_rx) = broadcast::channel(16);
        let (kill_tx, kill_rx) = watch::channel(false);
        let handle = tokio::spawn(run(
            state.clone(),
            fill_rx,
            book_rx,
            kill_rx,
            Duration::from_millis(100),
            Duration::from_millis(250),
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        fill_tx.send(fill("0.50")).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        book_tx.send(book("tok-yes", "0.51", "0.53")).unwrap();
        // Beyond t30: the row was already written by the timer arm.
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert_eq!(
            state.markouts_for_strategy("market_making").await.unwrap().len(),
            1
        );

        kill_tx.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            state.markouts_for_strategy("market_making").await.unwrap().len(),
            1,
            "the terminal flush does not rewrite already persisted rows"
        );
        drop(fill_tx);
        drop(book_tx);
    }

    // ── the drain is bounded to the buffered prefix ───────────────────────

    /// The drain takes only the captured prefix: messages queued AFTER
    /// the boundary (live producers) stay in the queue and do not
    /// become Pending.
    #[test]
    fn drain_takes_only_the_captured_prefix() {
        let (tx, mut rx) = broadcast::channel::<Fill>(16);
        for p in ["0.50", "0.51", "0.52"] {
            tx.send(fill(p)).unwrap();
        }
        // They simulate sends after the budget capture — the helper
        // receives the explicit boundary (3).
        for p in ["0.53", "0.54"] {
            tx.send(fill(p)).unwrap();
        }

        let mut pending = Vec::new();
        drain_fills_budget(&mut rx, &mut pending, Duration::from_secs(5), Duration::from_secs(30), 3);

        assert_eq!(pending.len(), 3, "only the original prefix becomes Pending");
        assert_eq!(pending[0].fill.price, dec!(0.50));
        assert_eq!(pending[2].fill.price, dec!(0.52));
        assert_eq!(rx.len(), 2, "fills after the boundary are left out");
        assert_eq!(rx.try_recv().unwrap().price, dec!(0.53));
        assert_eq!(rx.try_recv().unwrap().price, dec!(0.54));
    }

    /// A lag that takes only part of the prefix consumes budget and the
    /// originals that survived are drained all the same — they are not abandoned.
    #[test]
    fn partial_lagged_consumes_budget_and_drains_remainder() {
        // Capacity 4 and 6 sends with no reads: the receiver loses f0/f1 (Lagged(2))
        // and keeps f2..f5.
        let (tx, mut rx) = broadcast::channel::<Fill>(4);
        for p in ["0.50", "0.51", "0.52", "0.53", "0.54", "0.55"] {
            tx.send(fill(p)).unwrap();
        }

        let mut pending = Vec::new();
        drain_fills_budget(&mut rx, &mut pending, Duration::from_secs(5), Duration::from_secs(30), 6);

        assert_eq!(
            pending.len(),
            4,
            "Lagged(2) consumed 2 of the prefix; the 4 remaining originals drain"
        );
        assert_eq!(pending[0].fill.price, dec!(0.52));
        assert_eq!(pending[3].fill.price, dec!(0.55));
        assert_eq!(rx.len(), 0);
    }

    /// A lag that takes the WHOLE remaining prefix ends the drain without
    /// incorporating anything — whatever is left in the queue is after the boundary.
    #[test]
    fn total_lagged_ends_without_taking_later_ones() {
        // Capacity 2 and 4 sends: the first try_recv returns Lagged(2) and the
        // queue still holds the last 2 messages.
        let (tx, mut rx) = broadcast::channel::<Fill>(2);
        for p in ["0.50", "0.51", "0.52", "0.53"] {
            tx.send(fill(p)).unwrap();
        }

        let mut pending = Vec::new();
        drain_fills_budget(&mut rx, &mut pending, Duration::from_secs(5), Duration::from_secs(30), 2);

        assert!(
            pending.is_empty(),
            "Lagged(2) >= budget(2): it ends without capturing anything"
        );
        assert_eq!(
            rx.len(),
            2,
            "the fills still queued are after the boundary — not incorporated"
        );
    }

    /// Non-economic terminal fills (Rejected/Cancelled/GateRejected)
    /// consume a prefix position even though they generate no Pending — no extra
    /// slot for a fill after the boundary to sneak in.
    #[test]
    fn non_economic_fills_consume_budget() {
        let (tx, mut rx) = broadcast::channel::<Fill>(16);
        let mut rejected = fill("0.50");
        rejected.side = FillSide::Rejected;
        tx.send(rejected).unwrap();
        tx.send(fill("0.51")).unwrap();
        // After the boundary (budget = 2).
        tx.send(fill("0.52")).unwrap();

        let mut pending = Vec::new();
        drain_fills_budget(&mut rx, &mut pending, Duration::from_secs(5), Duration::from_secs(30), 2);

        assert_eq!(
            pending.len(),
            1,
            "the Rejected spent budget without creating a Pending"
        );
        assert_eq!(pending[0].fill.price, dec!(0.51));
        assert_eq!(
            rx.try_recv().unwrap().price,
            dec!(0.52),
            "the third is left out: the Rejected already consumed its position"
        );
    }

    /// The wrapper takes its budget from `len()` at the instant of the drain:
    /// the N buffered fills become Pending and the queue ends up empty.
    #[test]
    fn wrapper_uses_len_as_boundary() {
        let (tx, mut rx) = broadcast::channel::<Fill>(16);
        for p in ["0.50", "0.51", "0.52"] {
            tx.send(fill(p)).unwrap();
        }

        let mut pending = Vec::new();
        drain_fills(&mut rx, &mut pending, Duration::from_secs(5), Duration::from_secs(30));

        assert_eq!(pending.len(), 3);
        assert_eq!(rx.len(), 0);
    }

    /// With a LIVE producer publishing during the drain, it still
    /// ends, bounded by the prefix — before it might never reach
    /// `Empty` and never get to the terminal flush. The producer has stop/join and
    /// the drain an external timeout: a failure does not block the suite.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_ends_with_active_producer() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let (tx, mut rx) = broadcast::channel::<Fill>(8);
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let producer = tokio::spawn(async move {
            while !stop2.load(Ordering::Relaxed) {
                let _ = tx.send(fill("0.50"));
            }
            tx // the sender stays alive until the end of the test
        });
        // Window for the producer to fill the queue before the drain.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // The drain is synchronous: it runs on the blocking pool so the
        // producer keeps publishing on another worker during the whole loop.
        let drained = tokio::task::spawn_blocking(move || {
            let mut pending = Vec::new();
            drain_fills(&mut rx, &mut pending, Duration::from_secs(5), Duration::from_secs(30));
            pending
        });
        let result = tokio::time::timeout(Duration::from_secs(5), drained).await;
        stop.store(true, Ordering::Relaxed);
        let _ = tokio::time::timeout(Duration::from_secs(2), producer).await;

        let pending = result
            .expect("the drain must end even if the producer is still alive")
            .unwrap();
        assert!(
            pending.len() <= 8,
            "only the buffered prefix can become Pending (<= capacity)"
        );
    }
}
