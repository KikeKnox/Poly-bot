#[derive(Debug, Clone)]
pub enum TokenEvent {
    /// Dynamic subscribe: `{"operation":"subscribe"}` over an already
    /// established connection (verified live; the earlier failure came from the
    /// `type:subscribe` message format, see `ws::dynamic_subscribe_probe`).
    Subscribe { token_ids: Vec<String> },
    /// Dynamic unsubscribe: `{"operation":"unsubscribe"}` frees tokens of evicted
    /// or expired markets and of universes that moved to REST polling. Keeping
    /// the total subscribed bounded reduces venue throttling (proportional to total tokens).
    Unsubscribe { token_ids: Vec<String> },
}
