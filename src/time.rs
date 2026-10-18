//! Injectable clock for deterministic replay.
//!
//! In production `now()` falls back to `chrono::Utc::now()`. The `hrf_backtest`
//! binary sets [`set_replay_now_ms`] to the timestamp of the current event, so
//! `BookSnapshot::secs_to_clear` and the strategies' `now` see REPLAY time,
//! not wall-clock time. Without this the replay is empty: every recorded
//! `clear_at` is in the past and each `secs_to_clear() <= 0` yields zero intents.
//!
//! It is global on purpose: the backtest runs in its own binary
//! (`hrf_backtest`), never inside `poly-bot`. Under `cargo test` the
//! `backtest::tests` tests enable it during their replay (a window of ms per
//! test); [`ReplayClockGuard`] releases it on exit even on panic.
//!
//! [`Instant`] also virtualizes the monotonic clock: in replay it returns
//! stamps based on `REPLAY_NOW_MS`, so the strategies' `elapsed()`/TTLs
//! measure REPLAY time between events (with a real `Instant::now()` every
//! deadline would come out ~0 and nothing would ever expire in backtest).

use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

/// Replay timestamp in epoch ms; `i64::MIN` = real clock (production).
static REPLAY_NOW_MS: AtomicI64 = AtomicI64::new(i64::MIN);

/// Sets the replay clock. Only `backtest` (its own binary) should call this.
pub fn set_replay_now_ms(ms: i64) {
    REPLAY_NOW_MS.store(ms, Ordering::Relaxed);
}

/// RAII guard: enables the replay clock on construction and releases it on
/// drop (including panics and early returns).
pub struct ReplayClockGuard;

impl ReplayClockGuard {
    pub fn new(first_event_ms: i64) -> Self {
        set_replay_now_ms(first_event_ms);
        Self
    }
}

impl Drop for ReplayClockGuard {
    fn drop(&mut self) {
        REPLAY_NOW_MS.store(i64::MIN, Ordering::Relaxed);
    }
}

/// Whether the replay clock is active (tests / assertions).
pub fn replay_now_ms() -> Option<i64> {
    let v = REPLAY_NOW_MS.load(Ordering::Relaxed);
    (v != i64::MIN).then_some(v)
}

/// Test lock to serialize the tests that ARM the global replay clock
/// (the `backtest::tests` ones call `run()`, which enables
/// [`ReplayClockGuard`]). `REPLAY_NOW_MS` is shared by the whole lib test
/// binary process: two concurrent runs would step on each other, since one's
/// `ReplayClockGuard::drop` would disarm the clock in the middle of the other's
/// run and its `now()` calls would fall back to wall-clock time (same failure that motivates the `REPLAY_LOCK` of
/// `tests/replay_pipeline.rs`; that lock lives in ANOTHER test binary, and
/// locks are not shared between binaries, so each one needs its own).
///
/// Known limitation: the lock only serializes the tests that arm the clock
/// against each other. Other tests in the same binary that read
/// `crate::time::now()`/`Instant` WHILE a replay is active may
/// observe tape time (ms from 2023); isolating them completely would require
/// injecting the clock into every reader (out of scope). The window is
/// short (each run lasts ms of wall time) and the executor/state tests do not
/// depend on the clock's absolute value.
#[cfg(test)]
pub static REPLAY_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// "Now" for strategy logic: replay time if set, real clock in
/// production. Returns `DateTime<Utc>`, the same API as
/// `chrono::Utc::now()`, so call sites only change the path.
pub fn now() -> chrono::DateTime<chrono::Utc> {
    let v = REPLAY_NOW_MS.load(Ordering::Relaxed);
    if v == i64::MIN {
        chrono::Utc::now()
    } else {
        chrono::DateTime::from_timestamp_millis(v).unwrap_or_else(chrono::Utc::now)
    }
}

/// "Very old" duration for cross-domain comparisons: a `Real` instant
/// observed from the replay clock (or the reverse) belongs to another world;
/// semantically, any stamp taken outside the current domain is
/// "earlier than everything". One year saturates any TTL/deadline in the project without
/// overflow risk in later arithmetic.
const CROSS_DOMAIN_AGE: Duration = Duration::from_secs(31_536_000);

/// Monotonic instant compatible with the replay clock.
///
/// Drop-in for `std::time::Instant`: in production (`REPLAY_NOW_MS` inactive)
/// `now()` wraps `std::time::Instant::now()` and ALL methods delegate, so
/// behavior is identical. In replay, `now()` stamps the timestamp of the
/// current event, so `elapsed()`/`duration_since()` measure
/// replay time between events, not wall-clock microseconds.
///
/// Mixed domain (a `Real` compared with a `Replay`): can only happen
/// with stamps taken before the replay clock was enabled (e.g. `Instant::now()`
/// fields of structures built before the guard). Those
/// stamps are "older than the whole replay", so the resulting durations
/// saturate to [`CROSS_DOMAIN_AGE`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instant {
    Real(std::time::Instant),
    /// Replay clock epoch ms.
    Replay(i64),
}

impl Instant {
    /// Monotonic "now": replay if the clock is set, wall clock otherwise.
    pub fn now() -> Self {
        match replay_now_ms() {
            Some(ms) => Instant::Replay(ms),
            None => Instant::Real(std::time::Instant::now()),
        }
    }

    /// Explicit replay stamp (for stamping historical timestamps,
    /// e.g. the `updated_at` of a snapshot rebuilt from its recorded
    /// `ts_ms` instead of "now").
    pub fn from_replay_ms(ms: i64) -> Self {
        Instant::Replay(ms)
    }

    fn now_pair(&self) -> (Instant, Instant) {
        (Instant::now(), *self)
    }

    /// Like `std::time::Instant::elapsed`.
    pub fn elapsed(&self) -> Duration {
        let (n, t) = self.now_pair();
        n.saturating_duration_since(t)
    }

    /// Like `std::time::Instant::duration_since` but saturating to 0 (never
    /// panics, same choice as `saturating_duration_since`).
    pub fn duration_since(&self, earlier: Instant) -> Duration {
        self.saturating_duration_since(earlier)
    }

    pub fn saturating_duration_since(&self, earlier: Instant) -> Duration {
        match (*self, earlier) {
            (Instant::Real(a), Instant::Real(b)) => a.saturating_duration_since(b),
            (Instant::Replay(a), Instant::Replay(b)) => {
                Duration::from_millis(a.saturating_sub(b) as u64)
            }
            // Different domains: `earlier` is "earlier than everything".
            _ => CROSS_DOMAIN_AGE,
        }
    }

    pub fn checked_duration_since(&self, earlier: Instant) -> Option<Duration> {
        match (*self, earlier) {
            (Instant::Real(a), Instant::Real(b)) => a.checked_duration_since(b),
            (Instant::Replay(a), Instant::Replay(b)) => {
                (a >= b).then(|| Duration::from_millis((a - b) as u64))
            }
            // Replay is "later than everything" real (and vice versa from the other
            // side): Replay−Real has a positive duration; Real−Replay does not.
            (Instant::Replay(_), Instant::Real(_)) => Some(CROSS_DOMAIN_AGE),
            (Instant::Real(_), Instant::Replay(_)) => None,
        }
    }

    pub fn checked_add(&self, d: Duration) -> Option<Instant> {
        match self {
            Instant::Real(t) => t.checked_add(d).map(Instant::Real),
            Instant::Replay(ms) => ms
                .checked_add(d.as_millis().try_into().ok()?)
                .map(Instant::Replay),
        }
    }

    pub fn checked_sub(&self, d: Duration) -> Option<Instant> {
        match self {
            Instant::Real(t) => t.checked_sub(d).map(Instant::Real),
            Instant::Replay(ms) => ms
                .checked_sub(d.as_millis().try_into().ok()?)
                .map(Instant::Replay),
        }
    }
}

impl PartialOrd for Instant {
    fn partial_cmp(&self, other: &Instant) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Instant {
    fn cmp(&self, other: &Instant) -> std::cmp::Ordering {
        match (*self, *other) {
            (Instant::Real(a), Instant::Real(b)) => a.cmp(&b),
            (Instant::Replay(a), Instant::Replay(b)) => a.cmp(&b),
            // Consistent with the durations: any Replay is later than
            // any Real (pre-replay stamps are "older than everything").
            (Instant::Replay(_), Instant::Real(_)) => std::cmp::Ordering::Greater,
            (Instant::Real(_), Instant::Replay(_)) => std::cmp::Ordering::Less,
        }
    }
}

impl std::ops::Add<Duration> for Instant {
    type Output = Instant;
    fn add(self, d: Duration) -> Instant {
        self.checked_add(d).unwrap_or(self)
    }
}

impl std::ops::Sub<Duration> for Instant {
    type Output = Instant;
    fn sub(self, d: Duration) -> Instant {
        self.checked_sub(d).unwrap_or(self)
    }
}

impl std::ops::Sub<Instant> for Instant {
    type Output = Duration;
    fn sub(self, earlier: Instant) -> Duration {
        self.saturating_duration_since(earlier)
    }
}
