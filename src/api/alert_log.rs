//! In-memory alert log: a simple ring buffer, not persisted.
//!
//! Bug fixed: `/alerts` did not exist in the backend even though the UI already called it
//! (the error was swallowed by a `.catch()` on the JS side), so the "Recent alerts"
//! panel never showed anything real. This log fixes that with the minimum needed: no
//! SQLite persistence, since "recent" meaning whatever has happened since the last
//! process startup is enough for the use case (live monitoring).

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Serialize)]
pub struct AlertEntry {
    pub ts_ms: i64,
    pub severity: &'static str,
    pub message: String,
}

pub struct AlertLog {
    capacity: usize,
    entries: Mutex<VecDeque<AlertEntry>>,
}

impl AlertLog {
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            entries: Mutex::new(VecDeque::with_capacity(capacity)),
        })
    }

    pub fn push(&self, severity: &'static str, message: impl Into<String>) {
        let mut entries = self.entries.lock();
        if entries.len() >= self.capacity {
            entries.pop_front();
        }
        entries.push_back(AlertEntry {
            ts_ms: chrono::Utc::now().timestamp_millis(),
            severity,
            message: message.into(),
        });
    }

    /// The `limit` most recent entries, newest first.
    pub fn recent(&self, limit: usize) -> Vec<AlertEntry> {
        self.entries
            .lock()
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recent_returns_newest_first() {
        let log = AlertLog::new(10);
        log.push("critical", "one");
        log.push("warning", "two");
        log.push("info", "three");

        let recent = log.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].message, "three");
        assert_eq!(recent[1].message, "two");
    }

    #[test]
    fn respects_capacity_discarding_the_oldest() {
        let log = AlertLog::new(2);
        log.push("info", "one");
        log.push("info", "two");
        log.push("info", "three");

        let all = log.recent(10);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].message, "three");
        assert_eq!(all[1].message, "two");
    }
}
