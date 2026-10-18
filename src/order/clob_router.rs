//! ClobRouter — live implementation of the `OrderRouter` trait.
//!
//! Translates `OrderRequest` into `clob_sdk` calls and returns `RouterResult`.
//! EIP-712 signing, retries and timeouts all live in `clob_sdk`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use rust_decimal::Decimal;

use crate::{
    clob_sdk::{self, AuthedClobClient, SubmitResult},
    order::router::{
        OrderKind, OrderRequest, OrderRouter, RouterOpenOrder, RouterOrderStatus, RouterResult,
        RouterTrade, TimeInForce,
    },
    types::{Mode, OrderIntent, OrderSide},
};

// ── ClobRouter ────────────────────────────────────────────────────────────────

pub struct ClobRouter {
    bundle: Arc<AuthedClobClient>,
    /// Own address (EOA or proxy depending on wallet_mode) — needed by
    /// `fetch_trades_since` to filter our maker legs.
    own_address: String,
}

impl ClobRouter {
    pub fn new(bundle: Arc<AuthedClobClient>, own_address: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            bundle,
            own_address: own_address.into(),
        })
    }
}

impl OrderRouter for ClobRouter {
    fn submit(&self, req: OrderRequest) -> Pin<Box<dyn Future<Output = RouterResult> + Send + '_>> {
        Box::pin(async move {
            // Only GTD variants leave a resting order on the book — an
            // "unmatched"/no-fill result there is a legitimate state (live order). For the
            // rest (Market/FOK/FAK/Sell) the order is killed if it does not fill immediately:
            // "no fill" means nothing is left alive, not that it is still pending.
            let resting = is_resting_kind(&req.kind);
            let result: anyhow::Result<SubmitResult> = match &req.kind {
                OrderKind::Market { usdc_amount } => {
                    // Build the minimal OrderIntent that build_signed_market_buy needs.
                    let intent = market_intent(&req, *usdc_amount);
                    clob_sdk::submit_order(&self.bundle, &intent).await
                }

                OrderKind::Limit { price, size, tif } => match tif {
                    TimeInForce::Gtd => {
                        if req.clear_at == 0 {
                            return RouterResult::rejected(
                                "GTD without clear_at — order rejected before sending",
                            );
                        }
                        let exp = match resolve_gtd_expiration(req.clear_at) {
                            Ok(exp) => exp,
                            Err(msg) => {
                                // Previously this silently fell back to "now+1h" and the order
                                // was signed and sent with a completely different expiration
                                // than intended, with no log.
                                tracing::error!(
                                    clear_at = req.clear_at,
                                    token    = %req.token_id,
                                    strategy = %req.strategy,
                                    "clob_router: {msg}"
                                );
                                return RouterResult::rejected(msg);
                            }
                        };
                        clob_sdk::submit_limit_buy(
                            &self.bundle,
                            &req.token_id,
                            *price,
                            *size,
                            exp,
                            req.post_only,
                        )
                        .await
                    }
                    TimeInForce::Fak => {
                        let intent = limit_intent(&req, *price, *size);
                        clob_sdk::submit_order_fak(&self.bundle, &intent).await
                    }
                    TimeInForce::Fok => {
                        let intent = limit_intent(&req, *price, *size);
                        clob_sdk::submit_order(&self.bundle, &intent).await
                    }
                },

                OrderKind::Sell { size } => {
                    // `size: None` cannot be resolved — the earlier fallback
                    // used `req.signal`, which is a PROBABILITY
                    // (0.0-1.0), not a position size. The sim already rejects
                    // this scenario; live must do the same (fail-closed).
                    let Some(s) = size.filter(|s| *s > Decimal::ZERO) else {
                        return RouterResult::rejected(
                            "Sell{size: None} not implemented: no router resolves the real position",
                        );
                    };
                    clob_sdk::submit_sell(&self.bundle, &req.token_id, s).await
                }

                // Maker GTD ask (resting limit sell).
                OrderKind::LimitSell { price, size, tif } => match tif {
                    TimeInForce::Gtd => {
                        if req.clear_at == 0 {
                            return RouterResult::rejected(
                                "GTD without clear_at — order rejected before sending",
                            );
                        }
                        let exp = match resolve_gtd_expiration(req.clear_at) {
                            Ok(exp) => exp,
                            Err(msg) => {
                                // See the analogous comment in the Limit/Gtd branch.
                                tracing::error!(
                                    clear_at = req.clear_at,
                                    token    = %req.token_id,
                                    strategy = %req.strategy,
                                    "clob_router: {msg}"
                                );
                                return RouterResult::rejected(msg);
                            }
                        };
                        clob_sdk::submit_limit_sell(
                            &self.bundle,
                            &req.token_id,
                            *price,
                            *size,
                            exp,
                            req.post_only,
                        )
                        .await
                    }
                    // No caller emits LimitSell FOK/FAK today (only the PM unwind,
                    // always GTD) — an unexercised implementation is a future bug.
                    TimeInForce::Fak | TimeInForce::Fok => {
                        return RouterResult::rejected("LimitSell only supports GTD");
                    }
                },

                OrderKind::Cancel { order_id } => {
                    return match clob_sdk::cancel_order(&self.bundle, order_id).await {
                        Ok(_) => RouterResult::no_fill(None),
                        Err(e) => RouterResult::rejected(e.to_string()),
                    };
                }
            };

            match result {
                Err(e) => {
                    // Distinguish "POST got no response" from "CLOB rejection".
                    //
                    // A timeout/5xx/network drop leaves the exchange in an UNKNOWN
                    // STATE: the order may have been accepted (and we only lost the
                    // ACK) or rejected before reaching the matcher. Treating it as
                    // `rejected` would release the claim and, if the exchange did
                    // accept, an identical double buy would go unaccounted.
                    //
                    // Conservative over `clob_sdk::Err`: this arm covers any `Err`
                    // returned by `submit_order`/`submit_limit_*`/`submit_sell`. Those
                    // methods can fail BEFORE the POST (EIP-712 signing, order
                    // validation, serialization, SDK-internal rate limit) or AFTER
                    // (timeout/HTTP/5xx, network drop). The SDK exposes no error
                    // categories, so we cannot tell them apart. The conservative
                    // choice is `Unknown` for every Err: if the failure was pre-POST
                    // we lost no money (no live order on the CLOB) and the `pending`
                    // mark in the journal stays `unresolved` until explicit
                    // reconciliation, which blocks further submits for the same
                    // bracket. If it was post-POST it also stays `unresolved` and the
                    // exposure is kept. In practice a repeated invalid signature
                    // produces a loop of Unknowns until fixed — exactly what we want
                    // (conservative and visible).
                    //
                    // By contract `unknown=true ⟹ rejected=false` (see
                    // `RouterResult::unknown`). The `Executor` decides: with
                    // `unknown` it keeps the exposure and records a local mark.
                    //
                    // The metric below uses a heuristic error category — it does not
                    // change behavior (every Err is still Unknown), it only lets a
                    // dashboard tell "lost POST" from "the SDK never sent".
                    metrics::counter!(
                        "hrf_clob_err_category_total",
                        "endpoint" => "submit",
                        "category" => classify_clob_err(&e),
                    )
                    .increment(1);
                    tracing::error!(
                        token = %req.token_id,
                        strategy = %req.strategy,
                        "clob_router: POST got no response (Unknown — the exchange may have accepted): {e}"
                    );
                    RouterResult::unknown(e.to_string())
                }
                Ok(sr) => {
                    if !sr.success {
                        // The CLOB may answer success:false without an `error` field
                        // — an empty reason is indistinguishable from "no reason" in logs.
                        return RouterResult::rejected(sr.error.unwrap_or_else(|| {
                            "CLOB rejected the order with no error detail (success=false)".to_string()
                        }));
                    }
                    if !resting && sr.executed_size.is_none() {
                        // FOK/FAK/Market/Sell killed with no fill — the CLOB accepts it
                        // (success:true) but nothing is left resting on the book. It used to
                        // return no_fill(Some(order_id)), indistinguishable from a live GTD.
                        tracing::info!(
                            order_id = %sr.order_id,
                            token = %req.token_id,
                            strategy = %req.strategy,
                            "clob_router: order with no immediate fill and not resting — treated as rejection"
                        );
                        return RouterResult::rejected(
                            "order not filled (unmatched, no counterparty)",
                        );
                    }
                    // Live fills arrive via user_ws (WS trade UUID ≠ REST order_id hex).
                    // The SubmitResult trade_ids stay in the caller's DEBUG logs.
                    RouterResult::no_fill(Some(sr.order_id))
                }
            }
        })
    }

    fn submit_pair(
        &self,
        req1: OrderRequest,
        req2: OrderRequest,
    ) -> Pin<Box<dyn Future<Output = (RouterResult, RouterResult)> + Send + '_>> {
        Box::pin(async move {
            let (p1, s1) = match &req1.kind {
                OrderKind::Limit { price, size, .. } => (*price, *size),
                _ => {
                    let e = "submit_pair: both legs must be Limit FOK";
                    return (RouterResult::rejected(e), RouterResult::rejected(e));
                }
            };
            let (p2, s2) = match &req2.kind {
                OrderKind::Limit { price, size, .. } => (*price, *size),
                _ => {
                    let e = "submit_pair: leg must NOT be Limit FOK";
                    return (RouterResult::rejected(e), RouterResult::rejected(e));
                }
            };

            let i1 = limit_intent(&req1, p1, s1);
            let i2 = limit_intent(&req2, p2, s2);

            match clob_sdk::submit_order_pair(&self.bundle, &i1, &i2).await {
                Err(e) => {
                    // See the analogous comment in `submit`. `submit_order_pair` makes
                    // ONE batch POST; if the transport fails, both legs are in state
                    // Unknown — the exchange may have accepted 0, 1 or 2.
                    // Keeping all the claims and registering both
                    // `attempt_id`s is the `Executor`'s job.
                    // Same per-category metric as in `submit` — heuristic on the
                    // error text, behavior unchanged (both legs stay Unknown).
                    metrics::counter!(
                        "hrf_clob_err_category_total",
                        "endpoint" => "submit_pair",
                        "category" => classify_clob_err(&e),
                    )
                    .increment(1);
                    tracing::error!(
                        yes = %req1.token_id, no = %req2.token_id,
                        "clob_router: submit_pair POST got no response (Unknown — the exchange may have accepted): {e}"
                    );
                    let r = e.to_string();
                    (RouterResult::unknown(&r), RouterResult::unknown(&r))
                }
                Ok((sr1, sr2)) => {
                    // trade_ids = WS MATCHED UUIDs. Logged to correlate with user_ws.
                    if !sr1.trade_ids.is_empty() || !sr2.trade_ids.is_empty() {
                        tracing::debug!(
                            yes_trade_ids = ?sr1.trade_ids,
                            no_trade_ids  = ?sr2.trade_ids,
                            "clob_router: trade_ids of immediate fills"
                        );
                    }
                    let to_r = |sr: SubmitResult| {
                        // Explicit message instead of reason="" when the CLOB sends
                        // no `error` field alongside success:false.
                        if !sr.success {
                            return RouterResult::rejected(sr.error.unwrap_or_else(|| {
                                "CLOB rejected the order with no error detail (success=false)"
                                    .to_string()
                            }));
                        }
                        // Both submit_pair legs are Limit FOK (not resting) — with no
                        // immediate fill nothing is left alive on the book: it is a
                        // rejection, not a pending GTD.
                        if sr.executed_size.is_none() {
                            tracing::info!(
                                order_id = %sr.order_id,
                                "clob_router: submit_pair — leg with no immediate fill and not resting — treated as rejection"
                            );
                            return RouterResult::rejected(
                                "order not filled (unmatched, no counterparty)",
                            );
                        }
                        RouterResult::no_fill(Some(sr.order_id))
                    };
                    (to_r(sr1), to_r(sr2))
                }
            }
        })
    }

    fn cancel(
        &self,
        _market_cid: &str,
        order_id: &str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send + '_>> {
        let order_id = order_id.to_string();
        Box::pin(async move {
            clob_sdk::cancel_order(&self.bundle, &order_id)
                .await
                .map(|_| ())
        })
    }

    /// Unknown reconciliation: queries `data/order/{id}` on the real CLOB.
    /// An SDK `Err` (timeout, 5xx, unknown status) maps to
    /// `None` — indeterminate, the reconciler keeps the claim.
    fn order_status(
        &self,
        order_id: &str,
    ) -> Pin<Box<dyn Future<Output = Option<RouterOrderStatus>> + Send + '_>> {
        let order_id = order_id.to_string();
        Box::pin(async move {
            match clob_sdk::fetch_order_status(&self.bundle, &order_id).await {
                Ok(clob_sdk::ClobOrderStatus::Live { remaining }) => {
                    Some(RouterOrderStatus::Live { remaining })
                }
                Ok(clob_sdk::ClobOrderStatus::Gone { matched }) => {
                    Some(RouterOrderStatus::Gone { matched })
                }
                Ok(clob_sdk::ClobOrderStatus::NotFound) => Some(RouterOrderStatus::NotFound),
                Err(e) => {
                    tracing::warn!(order_id, "clob_router: order_status indeterminate: {e}");
                    None
                }
            }
        })
    }

    /// Unknown reconciliation: live GTD orders on the CLOB with attributes
    /// to match intents whose `order_id` was lost in the POST.
    fn open_gtd_orders(
        &self,
    ) -> Pin<Box<dyn Future<Output = Option<Vec<RouterOpenOrder>>> + Send + '_>> {
        Box::pin(async move {
            match clob_sdk::fetch_open_gtd_orders(&self.bundle).await {
                Ok(orders) => Some(
                    orders
                        .into_iter()
                        .map(|o| RouterOpenOrder {
                            order_id: o.order_id,
                            market_cid: o.market_cid,
                            token_id: o.token_id,
                            // `Side::Unknown` from the wire → Unknown
                            // (indeterminate, never BUY by default).
                            side: (&o.side).into(),
                            price: o.price,
                            remaining: o.remaining,
                            clear_at: o.clear_at,
                        })
                        .collect(),
                ),
                Err(e) => {
                    tracing::warn!("clob_router: open_gtd_orders indeterminate: {e}");
                    None
                }
            }
        })
    }

    /// Unknown taker reconciliation: `GET /data/trades` on the real CLOB. The
    /// `taker_order_id`/`maker order_id` of each trade gives the identity
    /// of the order the unanswered POST took with it.
    /// `Err` → `None` (indeterminate: the reconciler releases nothing).
    fn trades_since(
        &self,
        after_ts_secs: i64,
    ) -> Pin<Box<dyn Future<Output = Option<Vec<RouterTrade>>> + Send + '_>> {
        Box::pin(async move {
            match clob_sdk::fetch_trades_since(&self.bundle, after_ts_secs, &self.own_address).await
            {
                Ok(trades) => Some(
                    trades
                        .into_iter()
                        .map(|t| RouterTrade {
                            // exec_id already carries the :order_id suffix on
                            // maker legs — synthesized fills deduplicate
                            // per leg, not per aggregated trade.
                            trade_id: t.exec_id,
                            order_id: t.own_order_id,
                            token_id: t.token_id,
                            is_buy: t.is_buy,
                            price: t.price,
                            size: t.size,
                            match_ts_ms: t.match_ts.saturating_mul(1000),
                        })
                        .collect(),
                ),
                Err(e) => {
                    tracing::warn!("clob_router: trades_since indeterminate: {e}");
                    None
                }
            }
        })
    }

    fn mode(&self) -> Mode {
        Mode::Live
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Resolves the expiration timestamp of a GTD from `clear_at`,
/// applying Polymarket's 60s safety threshold (effective_expiry =
/// sent_timestamp - 60s; for the order to expire at `clear_at` we must send
/// `clear_at + 60s`). Returns `Err` if `clear_at` is invalid instead of silently
/// falling back to "now+1h" — a real GTD order signed with an invented
/// expiration is worse than losing the cycle.
fn resolve_gtd_expiration(clear_at: i64) -> Result<chrono::DateTime<chrono::Utc>, &'static str> {
    let exp = chrono::DateTime::from_timestamp(clear_at, 0)
        .ok_or("GTD with invalid clear_at — order rejected before sending")?;
    Ok(exp + chrono::Duration::seconds(60))
}

/// Kinds that place a resting GTD order on the book. A "no fill"
/// for these is a legitimate state (live order awaiting a counterparty); for the
/// rest (Market/FOK/FAK/Sell) it means the order died leaving nothing alive.
fn is_resting_kind(kind: &OrderKind) -> bool {
    matches!(
        kind,
        OrderKind::Limit {
            tif: TimeInForce::Gtd,
            ..
        } | OrderKind::LimitSell {
            tif: TimeInForce::Gtd,
            ..
        }
    )
}

/// Builds a minimal `OrderIntent` for market buy orders.
/// `build_signed_market_buy` only needs token_id, price*size and limit_price.
fn market_intent(req: &OrderRequest, usdc_amount: Decimal) -> OrderIntent {
    let price = req
        .signal
        .max(Decimal::ONE / rust_decimal_macros::dec!(100));
    OrderIntent {
        market_cid: req.market_cid.clone(),
        token_id: req.token_id.clone(),
        side: OrderSide::Buy,
        price,
        size: if price > Decimal::ZERO {
            usdc_amount / price
        } else {
            Decimal::ZERO
        },
        limit_price: None,
        signal: req.signal,
        clear_at: req.clear_at,
        neg_risk: req.neg_risk,
        second_leg: None,
        strategy: req.strategy,
        resting: false,
        real_clear_at: None,
    }
}

/// Builds a minimal `OrderIntent` for FOK/FAK limit buy orders.
fn limit_intent(req: &OrderRequest, price: Decimal, size: Decimal) -> OrderIntent {
    OrderIntent {
        market_cid: req.market_cid.clone(),
        token_id: req.token_id.clone(),
        side: OrderSide::Buy,
        price,
        size,
        limit_price: Some(price),
        signal: req.signal,
        clear_at: req.clear_at,
        neg_risk: req.neg_risk,
        second_leg: None,
        strategy: req.strategy,
        resting: false,
        real_clear_at: None,
    }
}

/// Loose heuristic to label an SDK `Err` by WHERE it probably failed:
/// `pre_post` (EIP-712 signing, validation, serialization:
/// the order almost surely did not go out) vs `post_post` (timeout/5xx/network:
/// the order may have been accepted). The SDK exposes no categories, so this is the
/// best signal available; whatever it cannot classify falls into `indeterminate`.
///
/// Observability ONLY — behavior is unchanged (every `Err` is still
/// `Unknown` and conservative). Lets the operator tell in metrics a
/// "lost ACK" from an invalid-signature loop.
fn classify_clob_err(e: &anyhow::Error) -> &'static str {
    let m = e.to_string().to_lowercase();
    if [
        "signature",
        "invalid",
        "validation",
        "serializ",
        "parse",
    ]
    .iter()
    .any(|s| m.contains(s))
    {
        return "suspicious_pre_post";
    }
    if [
        "timeout",
        "timed out",
        "network",
        "connection",
        "connect",
        "5xx",
        "502",
        "503",
        "504",
        "eof",
        "reset",
        "refused",
        "dns",
    ]
    .iter()
    .any(|s| m.contains(s))
    {
        return "post_post";
    }
    "indeterminate"
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    // classify_clob_err — pure, heuristic over the error text.
    #[test]
    fn classify_err_clob_distinguishes_pre_post_from_post_post() {
        assert_eq!(
            classify_clob_err(&anyhow::anyhow!("invalid signature")),
            "suspicious_pre_post"
        );
        assert_eq!(
            classify_clob_err(&anyhow::anyhow!("order validation failed")),
            "suspicious_pre_post"
        );
        assert_eq!(
            classify_clob_err(&anyhow::anyhow!("connection timeout")),
            "post_post"
        );
        assert_eq!(
            classify_clob_err(&anyhow::anyhow!("HTTP 502 bad gateway")),
            "post_post"
        );
        assert_eq!(
            classify_clob_err(&anyhow::anyhow!("something odd with no hint")),
            "indeterminate"
        );
    }

    // resolve_gtd_expiration — pure, no network or bundle.
    #[test]
    fn resolve_gtd_expiration_applies_60s_threshold() {
        let clear_at = 1_700_000_000_i64;
        let exp = resolve_gtd_expiration(clear_at).unwrap();
        assert_eq!(exp.timestamp(), clear_at + 60);
    }

    #[test]
    fn resolve_gtd_expiration_rejects_invalid_timestamp() {
        // Outside the range representable by chrono::DateTime — before this fix
        // it silently fell back to "now+1h"; now it is rejected explicitly.
        let r = resolve_gtd_expiration(i64::MAX);
        assert!(r.is_err());
    }

    // is_resting_kind — pure.
    #[test]
    fn is_resting_kind_distinguishes_gtd_from_fok_fak() {
        assert!(is_resting_kind(&OrderKind::Limit {
            price: dec!(0.5),
            size: dec!(1),
            tif: TimeInForce::Gtd
        }));
        assert!(is_resting_kind(&OrderKind::LimitSell {
            price: dec!(0.5),
            size: dec!(1),
            tif: TimeInForce::Gtd
        }));
        assert!(!is_resting_kind(&OrderKind::Limit {
            price: dec!(0.5),
            size: dec!(1),
            tif: TimeInForce::Fok
        }));
        assert!(!is_resting_kind(&OrderKind::Limit {
            price: dec!(0.5),
            size: dec!(1),
            tif: TimeInForce::Fak
        }));
        assert!(!is_resting_kind(&OrderKind::Market {
            usdc_amount: dec!(10)
        }));
        assert!(!is_resting_kind(&OrderKind::Sell {
            size: Some(dec!(1))
        }));
        assert!(!is_resting_kind(&OrderKind::Cancel {
            order_id: "0xabc".into()
        }));
    }
}
