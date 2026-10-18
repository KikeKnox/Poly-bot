//! Submit evidence: uses the existing tracing, never writes the ledger.
use super::router::{OrderRequest, OrderRouter, RouterResult};

#[derive(Debug, Clone)]
struct Event {
    phase: &'static str,
    request: OrderRequest,
    order_id: Option<String>,
    exec_id: Option<String>,
    observed_at_ms: i64,
    elapsed_us: u128,
    fill: Option<crate::types::Fill>,
    mode: crate::types::Mode,
}

fn started(request: OrderRequest, mode: crate::types::Mode) -> Event {
    Event {
        phase: "submit_started",
        request,
        order_id: None,
        exec_id: None,
        observed_at_ms: chrono::Utc::now().timestamp_millis(),
        elapsed_us: 0,
        fill: None,
        mode,
    }
}

fn completed(mut event: Event, result: &RouterResult, start: std::time::Instant) -> Event {
    event.phase = if result.unknown {
        "unknown"
    } else if result.rejected {
        "rejected"
    } else if result.fill.is_some() {
        "accepted_with_fill"
    } else {
        "accepted_no_fill"
    };
    event.order_id = result
        .order_id
        .clone()
        .or_else(|| result.fill.as_ref().and_then(|f| f.order_id.clone()));
    event.exec_id = result.fill.as_ref().and_then(|f| f.exec_id.clone());
    event.fill = result.fill.clone();
    event.observed_at_ms = chrono::Utc::now().timestamp_millis();
    event.elapsed_us = start.elapsed().as_micros();
    event
}

fn emit(event: Event) {
    use super::router::OrderKind;
    let (limit_price, shares, usdc) = match event.request.kind {
        OrderKind::Limit { price, size, .. } | OrderKind::LimitSell { price, size, .. } => {
            (Some(price), Some(size), None)
        }
        OrderKind::Market { usdc_amount } => (None, None, Some(usdc_amount)),
        OrderKind::Sell { size } => (None, size, None),
        OrderKind::Cancel { .. } => (None, None, None),
    };
    tracing::info!(target: "decision_audit", schema = "order_attempt/v1", event = event.phase,
        observed_at_ms = event.observed_at_ms, elapsed_us = %event.elapsed_us,
        mode = ?event.mode, strategy = %event.request.strategy,
        cid = %event.request.market_cid, token_id = %event.request.token_id,
        local_order_id = ?event.request.local_order_id, order_id = ?event.order_id, exec_id = ?event.exec_id,
        signal = %event.request.signal, kind = ?event.request.kind,
        limit_price = ?limit_price, requested_shares = ?shares, requested_usdc = ?usdc,
        fill_price = ?event.fill.as_ref().map(|f| f.price),
        fill_size = ?event.fill.as_ref().map(|f| f.size),
        fill_ts = ?event.fill.as_ref().map(|f| f.ts),
        "submit evidence; it does not prove fill persistence");
}

pub(super) async fn submit(router: &dyn OrderRouter, request: OrderRequest) -> RouterResult {
    submit_observed(router, request, emit).await
}

async fn submit_observed(
    router: &dyn OrderRouter,
    request: OrderRequest,
    mut record: impl FnMut(Event) + Send,
) -> RouterResult {
    let event = started(request.clone(), router.mode());
    record(event.clone());
    let start = std::time::Instant::now();
    let result = router.submit(request).await;
    record(completed(event, &result, start));
    result
}

/// Paired variant: wraps `submit_pair` preserving the local identity
/// of EACH leg (the pair can end with different outcomes, e.g. YES
/// rejected and NO unknown, and each is audited with its `local_order_id`).
pub(super) async fn submit_pair(
    router: &dyn OrderRouter,
    a: OrderRequest,
    b: OrderRequest,
) -> (RouterResult, RouterResult) {
    submit_pair_observed(router, a, b, emit).await
}

async fn submit_pair_observed(
    router: &dyn OrderRouter,
    a: OrderRequest,
    b: OrderRequest,
    mut record: impl FnMut(Event) + Send,
) -> (RouterResult, RouterResult) {
    let ea = started(a.clone(), router.mode());
    let eb = started(b.clone(), router.mode());
    record(ea.clone());
    record(eb.clone());
    let start = std::time::Instant::now();
    let (ra, rb) = router.submit_pair(a, b).await;
    record(completed(ea, &ra, start));
    record(completed(eb, &rb, start));
    (ra, rb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        order::router::OrderKind,
        types::{Mode, StrategyId},
    };
    type BoxFut<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;
    use rust_decimal_macros::dec;
    use std::sync::Mutex;

    struct Fake(Mutex<Option<RouterResult>>);
    impl OrderRouter for Fake {
        fn submit(&self, _req: OrderRequest) -> BoxFut<'_, RouterResult> {
            let result = self.0.lock().unwrap().take().unwrap();
            Box::pin(async move { result })
        }
        fn cancel(&self, _: &str, _: &str) -> BoxFut<'_, anyhow::Result<()>> {
            Box::pin(async { Ok(()) })
        }
        fn mode(&self) -> Mode {
            Mode::Paper
        }
    }
    fn request(local: &str) -> OrderRequest {
        OrderRequest {
            local_order_id: Some(local.into()),
            market_cid: "cid".into(),
            token_id: "token".into(),
            kind: OrderKind::Market {
                usdc_amount: dec!(2),
            },
            signal: dec!(0.03),
            neg_risk: false,
            clear_at: 99,
            strategy: StrategyId::Arb,
            post_only: false,
            real_clear_at: None,
        }
    }
    #[tokio::test]
    async fn paired_submission_keeps_both_local_ids_and_unknown() {
        struct Pair(Fake);
        impl OrderRouter for Pair {
            fn submit(&self, _: OrderRequest) -> BoxFut<'_, RouterResult> {
                panic!("must preserve overridden batch submit")
            }
            fn submit_pair(
                &self,
                _: OrderRequest,
                _: OrderRequest,
            ) -> BoxFut<'_, (RouterResult, RouterResult)> {
                Box::pin(async {
                    (
                        RouterResult::rejected("no"),
                        RouterResult::unknown("timeout"),
                    )
                })
            }
            fn cancel(&self, c: &str, o: &str) -> BoxFut<'_, anyhow::Result<()>> {
                self.0.cancel(c, o)
            }
            fn mode(&self) -> Mode {
                Mode::Paper
            }
        }
        let router = Pair(Fake(Mutex::new(None)));
        let mut events = Vec::new();
        let (a, b) =
            submit_pair_observed(&router, request("a"), request("b"), |e| events.push(e)).await;
        assert!(a.rejected && b.unknown && !b.rejected);
        assert_eq!(events.len(), 4);
        assert_eq!(events[2].phase, "rejected");
        assert_eq!(events[3].phase, "unknown");
        assert_eq!(events[2].request.local_order_id.as_deref(), Some("a"));
        assert_eq!(events[3].request.local_order_id.as_deref(), Some("b"));
    }

    #[tokio::test]
    async fn submit_outcomes_are_observed_without_inventing_fills() {
        for (result, expected) in [
            (
                RouterResult::no_fill(Some("exchange".into())),
                "accepted_no_fill",
            ),
            (
                RouterResult::rejected("raw potentially sensitive diagnostic"),
                "rejected",
            ),
            (RouterResult::unknown("timeout"), "unknown"),
        ] {
            let fake = Fake(Mutex::new(Some(result)));
            let mut events = Vec::new();
            let result = submit_observed(&fake, request("local-1"), |e| events.push(e)).await;
            assert_eq!(
                events.len(),
                2,
                "start + terminal outcome, no synthetic fills"
            );
            assert_eq!(events[0].phase, "submit_started");
            assert_eq!(events[1].phase, expected);
            assert!(events
                .iter()
                .all(|e| e.request.local_order_id.as_deref() == Some("local-1")
                    && e.exec_id.is_none()));
            assert_eq!(events[1].order_id, result.order_id);
            assert_eq!(result.rejected, expected == "rejected");
            assert_eq!(result.unknown, expected == "unknown");
            assert!(result.fill.is_none());
        }
    }
}
