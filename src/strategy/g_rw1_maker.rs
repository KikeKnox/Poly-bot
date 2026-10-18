//! Strategy G-RW-1: measures buy markout in paper
//! without position management or FV — just a fixed ticket (`ticket_shares`) at the
//! best bid (GTD, resting) and the best ask (FOK, taker) of each token
//! (YES/NO), in the front bracket of the universes declared in its config
//! (`config/strategies/g_rw1_maker.toml`, intended for btc-updown-5m/15m).
//!
//! There is no "post_only" concept in this codebase (`OrderIntent::resting`):
//! the bid side posts resting=true (maker) and the ask side resting=false
//! (taker) — both assets at once, which is exactly what the gate asks for
//! ("no post_only so as not to distort the fill"). No inventory, no
//! unwind: the gate measures entry markout, not P&L: leaving the fill alive until
//! settlement is the correct behavior in paper.

use crate::time::Instant;
use std::collections::HashMap;
use std::time::Duration;

use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::{
    config::defaults,
    types::{BookSnapshot, OrderIntent, OrderSide, StrategyId, VolSnapshot},
};

use super::Strategy;

/// G-RW-1 strategy config. See module docs.
#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct GRw1MakerConfig {
    #[serde(flatten)]
    pub limits: crate::config::StrategyLimits,
        /// Exact ticket per side (bid and ask, YES and NO). Fixed by gate design:
        /// does not scale with FV or inventory.
    #[serde(default = "defaults::grw1_ticket_shares")]
    pub ticket_shares: Decimal,
        /// Repost if the touch moved more than this since the last ticket.
    #[serde(default = "defaults::grw1_requote_threshold_bps")]
    pub requote_threshold_bps: u32,
        /// TTL before forcing a new attempt even if the price has not moved.
    #[serde(default = "defaults::grw1_order_ttl_secs")]
    pub order_ttl_secs: u64,
}

impl crate::strategy::StrategyConfig for GRw1MakerConfig {
    fn limits(&self) -> &crate::config::StrategyLimits {
        &self.limits
    }
}

struct LegState {
    price: Decimal,
    posted_at: Instant,
}

struct GState {
    clear_at: i64,
    yes_token: String,
    no_token: String,
    yes_bid: Option<LegState>,
    yes_ask: Option<LegState>,
    no_bid: Option<LegState>,
    no_ask: Option<LegState>,
}

pub struct GRw1Maker {
    cfg: GRw1MakerConfig,
    markets: HashMap<String, GState>,
    last_retain: Instant,
}

impl GRw1Maker {
    pub fn new(cfg: GRw1MakerConfig) -> Self {
        Self {
            cfg,
            markets: HashMap::new(),
            last_retain: Instant::now(),
        }
    }
}

fn cancel_intent(market_cid: &str, token_id: &str, clear_at: i64, neg_risk: bool) -> OrderIntent {
    super::cancel_intent(
        market_cid,
        token_id,
        clear_at,
        neg_risk,
        StrategyId::GRw1Maker,
    )
}

/// Reposts (or cancels) one side of a token if the touch price moved
/// more than `requote_threshold` or the TTL expired. The bid side (`resting=true`)
/// cancels its live GTD before reposting; the ask side (`resting=false`) is
/// FOK — there is nothing "live" to cancel, only the retry is rate-limited.
#[allow(clippy::too_many_arguments)]
fn requote_leg(
    state: &mut Option<LegState>,
    desired: Option<Decimal>,
    resting: bool,
    market_cid: &str,
    token_id: &str,
    size: Decimal,
    clear_at: i64,
    neg_risk: bool,
    requote_threshold: Decimal,
    ttl: Duration,
) -> Vec<OrderIntent> {
    let mut out = Vec::new();

    let Some(price) = desired else {
        // No liquidity on this side: if there was a live GTD, cancel it.
        if resting && state.take().is_some() {
            out.push(cancel_intent(market_cid, token_id, clear_at, neg_risk));
        }
        return out;
    };

    let needs_requote = match state.as_ref() {
        None => true,
        Some(leg) => {
            // `requote_threshold` arrives as a fraction (bps/10000) —
            // compare against the relative fraction, not against bps. With the
            // earlier `* 10000` the threshold was 10⁴ times looser and
            // any touch movement triggered cancel+repost.
            let moved =
                !leg.price.is_zero() && ((price - leg.price) / leg.price).abs() > requote_threshold;
            moved || leg.posted_at.elapsed() >= ttl
        }
    };
    if !needs_requote {
        return out;
    }

    if resting && state.is_some() {
        out.push(cancel_intent(market_cid, token_id, clear_at, neg_risk));
    }
    out.push(OrderIntent {
        market_cid: market_cid.to_string(),
        token_id: token_id.to_string(),
        side: OrderSide::Buy,
        price,
        size,
        clear_at,
        signal: Decimal::ZERO,
        neg_risk,
        limit_price: Some(price),
        second_leg: None,
        strategy: StrategyId::GRw1Maker,
        resting,
        real_clear_at: None,
    });
    *state = Some(LegState {
        price,
        posted_at: Instant::now(),
    });
    out
}

impl Strategy for GRw1Maker {
    fn id(&self) -> StrategyId {
        StrategyId::GRw1Maker
    }

    fn requires_oracle(&self) -> bool {
        false
    }

    fn on_book(&mut self, book: &BookSnapshot, _vol: &VolSnapshot) -> Vec<OrderIntent> {
        let now = crate::time::now().timestamp();
        if self.last_retain.elapsed() >= Duration::from_secs(30) {
            self.markets.retain(|_, s| s.clear_at > now);
            self.last_retain = Instant::now();
        }

        let cid = &book.market_condition_id;
        let remaining = book.secs_to_clear();
        if remaining <= 5 {
            // Too close to the close: opening a new ticket makes no sense.
            return Vec::new();
        }

        let size = self.cfg.ticket_shares;
        let requote_threshold = Decimal::from(self.cfg.requote_threshold_bps) / dec!(10000);
        let ttl = Duration::from_secs(self.cfg.order_ttl_secs);

        let state = self.markets.entry(cid.clone()).or_insert_with(|| GState {
            clear_at: book.clear_at,
            yes_token: book.yes.token_id.clone(),
            no_token: book.no.token_id.clone(),
            yes_bid: None,
            yes_ask: None,
            no_bid: None,
            no_ask: None,
        });

        let mut intents = Vec::new();
        intents.extend(requote_leg(
            &mut state.yes_bid,
            book.yes.best_bid(),
            true,
            cid,
            &state.yes_token,
            size,
            book.clear_at,
            book.neg_risk,
            requote_threshold,
            ttl,
        ));
        intents.extend(requote_leg(
            &mut state.yes_ask,
            book.yes.best_ask(),
            false,
            cid,
            &state.yes_token,
            size,
            book.clear_at,
            book.neg_risk,
            requote_threshold,
            ttl,
        ));
        intents.extend(requote_leg(
            &mut state.no_bid,
            book.no.best_bid(),
            true,
            cid,
            &state.no_token,
            size,
            book.clear_at,
            book.neg_risk,
            requote_threshold,
            ttl,
        ));
        intents.extend(requote_leg(
            &mut state.no_ask,
            book.no.best_ask(),
            false,
            cid,
            &state.no_token,
            size,
            book.clear_at,
            book.neg_risk,
            requote_threshold,
            ttl,
        ));
        intents
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HalfBook, Level, TokenBook};

    fn book(remaining: i64) -> BookSnapshot {
        let now = crate::time::now().timestamp();
        BookSnapshot {
            market_condition_id: "cid-1".into(),
            yes: TokenBook {
                token_id: "yes-1".into(),
                bids: HalfBook {
                    levels: vec![Level {
                        price: dec!(0.50),
                        size: dec!(100),
                    }],
                },
                asks: HalfBook {
                    levels: vec![Level {
                        price: dec!(0.52),
                        size: dec!(100),
                    }],
                },
                flow_net: None,
                flow_total: None,
            },
            no: TokenBook {
                token_id: "no-1".into(),
                bids: HalfBook {
                    levels: vec![Level {
                        price: dec!(0.46),
                        size: dec!(100),
                    }],
                },
                asks: HalfBook {
                    levels: vec![Level {
                        price: dec!(0.48),
                        size: dec!(100),
                    }],
                },
                flow_net: None,
                flow_total: None,
            },
            market_open_ts: now - 60,
            clear_at: now + remaining,
            neg_risk: false,
            slug: "btc-updown-5m-test".into(),
            reference_price: None,
            fee_rate_bps: None,
            parabolic_fee_rate: None,
            tick_size: None,
            recv_instant: None,
        }
    }

    fn cfg() -> GRw1MakerConfig {
        GRw1MakerConfig {
            limits: Default::default(),
            ticket_shares: dec!(50),
            requote_threshold_bps: 20,
            order_ttl_secs: 5,
        }
    }

    #[test]
    fn first_book_posts_all_4_legs() {
        let mut s = GRw1Maker::new(cfg());
        let intents = s.on_book(&book(120), &VolSnapshot::default());
        assert_eq!(intents.len(), 4, "yes_bid + yes_ask + no_bid + no_ask");
        assert!(intents.iter().all(|i| i.size == dec!(50)));
        let resting_count = intents.iter().filter(|i| i.resting).count();
        assert_eq!(resting_count, 2, "only the two bids are resting GTD");
    }

    #[test]
    fn without_movement_or_ttl_does_not_repost() {
        let mut s = GRw1Maker::new(cfg());
        s.on_book(&book(120), &VolSnapshot::default());
        let intents = s.on_book(&book(119), &VolSnapshot::default());
        assert!(
            intents.is_empty(),
            "same price, TTL not reached: nothing to resend"
        );
    }

    #[test]
    fn near_close_does_not_open() {
        let mut s = GRw1Maker::new(cfg());
        let intents = s.on_book(&book(3), &VolSnapshot::default());
        assert!(intents.is_empty());
    }

    /// The requote threshold used to be compared in mixed units
    /// (bps on the left, fraction on the right) → any touch movement,
    /// however small, triggered cancel+repost. With
    /// `requote_threshold_bps = 20` a 10 bps move must NOT requote
    /// and a 200 bps move MUST.
    #[test]
    fn move_below_threshold_does_not_repost() {
        let mut s = GRw1Maker::new(cfg());
        s.on_book(&book(120), &VolSnapshot::default());
        // 0.50 → 0.5005 = 10 relative bps (< 20 bps): no requote.
        let mut b = book(119);
        b.yes.bids.levels[0].price = dec!(0.5005);
        let intents = s.on_book(&b, &VolSnapshot::default());
        assert!(
            intents.is_empty(),
            "10bps move < 20bps threshold: there must be no intents, got {intents:?}"
        );
    }

    #[test]
    fn move_above_threshold_reposts() {
        let mut s = GRw1Maker::new(cfg());
        s.on_book(&book(120), &VolSnapshot::default());
        // 0.50 → 0.51 = 200 relative bps (> 20 bps): cancel + repost of the bid.
        let mut b = book(119);
        b.yes.bids.levels[0].price = dec!(0.51);
        let intents = s.on_book(&b, &VolSnapshot::default());
        assert!(
            intents.iter().any(|i| matches!(i.side, OrderSide::Cancel)),
            "move > threshold: must cancel the live GTD"
        );
        assert!(
            intents
                .iter()
                .any(|i| matches!(i.side, OrderSide::Buy) && i.resting),
            "move > threshold: must repost the bid"
        );
    }
}
