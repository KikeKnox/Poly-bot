//! Circular buffer of log lines exposed via Telegram (/logs).
//!
//! `LogLayer` is a `tracing_subscriber` layer that stores the last N
//! events (level INFO+) in a `VecDeque` protected by a `std::sync::Mutex`.
//! It uses `std::sync::Mutex` (not tokio's) because `on_event` is called in a synchronous context.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use tracing::field::{Field, Visit};

pub type LogBuffer = Arc<Mutex<VecDeque<String>>>;

pub fn new(cap: usize) -> LogBuffer {
    Arc::new(Mutex::new(VecDeque::with_capacity(cap)))
}

pub struct LogLayer {
    buf: LogBuffer,
    cap: usize,
}

impl LogLayer {
    pub fn new(buf: LogBuffer, cap: usize) -> Self {
        Self { buf, cap }
    }
}

// ── Visitor de fields ─────────────────────────────────────────────────────────

struct FieldCollector {
    msg: String,
    extra: String,
}

impl Visit for FieldCollector {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.msg = value.to_string();
        } else {
            if !self.extra.is_empty() {
                self.extra.push(' ');
            }
            self.extra.push_str(&format!("{}={}", field.name(), value));
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let s = format!("{value:?}");
            // Debug wraps &str in quotes — strip them for readability
            self.msg = s.trim_matches('"').to_string();
        } else {
            if !self.extra.is_empty() {
                self.extra.push(' ');
            }
            self.extra
                .push_str(&format!("{}={:?}", field.name(), value));
        }
    }
}

// ── Layer ─────────────────────────────────────────────────────────────────────

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for LogLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let level = match *event.metadata().level() {
            tracing::Level::ERROR => "ERR",
            tracing::Level::WARN => "WRN",
            tracing::Level::INFO => "INF",
            tracing::Level::DEBUG => "DBG",
            tracing::Level::TRACE => return,
        };

        let mut c = FieldCollector {
            msg: String::new(),
            extra: String::new(),
        };
        event.record(&mut c);

        // Shorten the target by stripping the crate prefix
        let target = event
            .metadata()
            .target()
            .strip_prefix("poly_bot::")
            .unwrap_or(event.metadata().target());
        let now = chrono::Local::now().format("%H:%M:%S");

        let line = if c.extra.is_empty() {
            format!("{now} {level} {target}: {}", c.msg)
        } else {
            format!("{now} {level} {target}: {} [{}]", c.msg, c.extra)
        };

        let mut buf = self.buf.lock().unwrap_or_else(|e| e.into_inner());
        if buf.len() >= self.cap {
            buf.pop_front();
        }
        buf.push_back(line);
    }
}
