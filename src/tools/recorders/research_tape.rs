//! Adapter for `ToolId::ResearchEvents`: lifecycle of the events recorder
//! Research tape (`crate::research::events`) as a toggleable tool.
//!
//! No loop of its own: the real writer lives in its OS thread, started by
//! `ResearchRecorder::start` on the shared [`Hub`] — this task
//! only waits for the stop signal (individual toggle or global kill forwarded
//! by `ToolSupervisor`) and closes the session with `ResearchRecorder::stop`, which
//! calls `detach()` on the hub and joins the writer thread (drains what is
//! queued and flushes). The `Hub` itself outlives this task — a
//! next `start` on the same hub resumes without re-wiring the
//! producers spread across the rest of the bot.
//!
//! IMPORTANT: `ResearchRecorder::start` (the `attach()` on the hub) is NOT
//! done here — `ToolSupervisor::spawn_recorder_task` does it, in a
//! SYNCHRONOUS way, before spawning this task. See the comment at that
//! call site: the WS tape subscription evaluates `hub.is_active()` once
//! per session, so the `attach` must have happened before the
//! `oracle::price`/`market_feed::run` start — a `tokio::spawn` of this
//! full function (with `start()` included) would not give that guarantee, because
//! the order in which a spawned task runs is decided by the scheduler, not the
//! textual order in `main.rs`.

use tokio::sync::watch;

use crate::research::events::ResearchRecorder;

/// Waits for `stop` and closes the session that the caller already opened with
/// `ResearchRecorder::start`.
pub async fn run(recorder: ResearchRecorder, mut stop: watch::Receiver<bool>) {
    while stop.changed().await.is_ok() {
        if *stop.borrow() {
            break;
        }
    }

    let totals = recorder.stop().await;
    tracing::info!(
        enqueued = totals.enqueued,
        written = totals.written,
        deduped = totals.deduped,
        discarded = totals.discarded,
        "research_events: recorder stopped"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::events::{Hub, ResearchRecorderConfig};

    /// Full lifecycle: the caller starts on a new `Hub` (hub already
    /// `active` BEFORE the task even exists — that is the guarantee of
    /// ordering that the `main.rs` wiring requires), an event recorded through
    /// `hub.producer()` (as any hot-path hook would do) ends up
    /// written, and `stop` closes without leaving the writer thread alive — verified
    /// by reopening the same `dir` in a second session on the SAME hub without
    /// error (if `stop()` had not joined the writer thread, the second
    /// `start` would race with the first).
    #[tokio::test]
    async fn starts_records_one_event_and_closes_on_stop() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = ResearchRecorderConfig {
            dir: dir.path().to_path_buf(),
            channel_capacity: 64,
            dedup_capacity: 64,
        };
        let hub = Hub::new();

        // Synchronous start — as `ToolSupervisor::spawn_recorder_task` does
        // — the hub is already active BEFORE `run` is spawned.
        let recorder = ResearchRecorder::start(&cfg, &hub).expect("start must work");
        assert!(hub.is_active(), "the hub must be active after start, before spawning the task");

        let (stop_tx, stop_rx) = watch::channel(false);
        let handle = tokio::spawn(run(recorder, stop_rx));

        let producer = hub.producer();
        let ok = producer.record(crate::research::events::EventInput::new(
            crate::research::contract::EventSource::BinanceWs,
            crate::research::contract::EventKind::Trade,
            serde_json::json!({"price": "50000"}),
            chrono::Utc::now().timestamp_millis(),
        ));
        assert!(ok, "record() should have enqueued with the hub active");

        stop_tx.send(true).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), handle)
            .await
            .expect("run() did not finish in time")
            .unwrap();
        assert!(!hub.is_active(), "stop() must leave the hub inactive");

        // Second session on the SAME hub: if the first had not
        // truly joined the writer thread, this second `start` would clash
        // (or the stale writer would keep writing to the same dir).
        let recorder2 = ResearchRecorder::start(&cfg, &hub).expect("resume must work");
        let totals = recorder2.stop().await;
        // `enqueued`/`discarded` are reported PER
        // SESSION (baseline taken in `start()`) — this second session did not
        // enqueue anything of its own, so its delta is 0, even though the
        // internal counter of the `Hub` (accumulated since its creation) is still 1 because of the
        // event of the first session.
        assert_eq!(
            totals.enqueued, 0,
            "the delta of the new session must not carry over what was enqueued in the earlier session"
        );
    }
}
