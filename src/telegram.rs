//! Telegram bot.
//!
//! Long-poll getUpdates + fill notifications + periodic summary.
//! Starts only if `telegram.token` is non-empty.

use std::{collections::HashSet, sync::Arc};

use anyhow::Context;
use rust_decimal::Decimal;
use serde::Deserialize;
use tokio::sync::{broadcast, watch, Mutex};
use tracing::{debug, info, warn};

use crate::{
    config::TelegramConfig,
    log_buffer::LogBuffer,
    risk::RiskGuard,
    state::StateStore,
    tools::{ToolId, ToolSupervisor},
    types::{Fill, FillSide, StrategyId},
};

// ── Bot API types ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct TgResponse<T> {
    ok: bool,
    result: Option<T>,
}

#[derive(Deserialize)]
struct Update {
    update_id: i64,
    message: Option<Message>,
}

#[derive(Deserialize)]
struct Message {
    chat: Chat,
    from: Option<User>,
    text: Option<String>,
}

#[derive(Deserialize)]
struct Chat {
    id: i64,
}

#[derive(Deserialize)]
struct User {
    id: i64,
}

// ── Bot core ────────────────────────────────────────────────────────────

struct Bot {
    client: reqwest::Client,
    base: String, // https://api.telegram.org/bot<token>
    allowed: HashSet<i64>,
    /// Active chat_ids: filled when an authorized user sends a command.
    subs: Arc<Mutex<HashSet<i64>>>,
    /// Mutes fill notifications when true.
    mute: Arc<Mutex<bool>>,
}

impl Bot {
    fn new(token: &str, allowed: Vec<i64>) -> Arc<Self> {
        Arc::new(Self {
            client: reqwest::Client::builder()
                // timeout > long-poll so reqwest doesn't cut off before Telegram
                .timeout(std::time::Duration::from_secs(45))
                .build()
                .expect("reqwest client"),
            base: format!("https://api.telegram.org/bot{token}"),
            allowed: allowed.into_iter().collect(),
            subs: Arc::new(Mutex::new(HashSet::new())),
            mute: Arc::new(Mutex::new(false)),
        })
    }

    /// Registers the bot's commands with BotFather, replacing any previous list.
    async fn set_commands(&self) {
        let url = format!("{}/setMyCommands", self.base);
        let body = serde_json::json!({ "commands": [
            {"command": "state",      "description": "Equity, cash, PnL and strategy state"},
            {"command": "positions",  "description": "Open positions with time remaining"},
            {"command": "pnl",         "description": "Realized PnL per strategy"},
            {"command": "limits",      "description": "Operating limits [strategy]"},
            {"command": "exposure",  "description": "Global exposure [new value]"},
            {"command": "enable",      "description": "Enable strategy <name>"},
            {"command": "disable",     "description": "Disable strategy <name>"},
            {"command": "kill",        "description": "Kill switch [strategy]"},
            {"command": "reset",       "description": "Reset kill switch [strategy]"},
            {"command": "logs",        "description": "Last 50 log records"},
            {"command": "mute",        "description": "Mute fills on|off"},
            {"command": "help",        "description": "Command list"},
        ]});
        if let Err(e) = self.client.post(&url).json(&body).send().await {
            warn!("telegram: setMyCommands: {e}");
        }
    }

    async fn send_msg(&self, chat_id: i64, text: &str, silent: bool) {
        let url = format!("{}/sendMessage", self.base);
        let body = serde_json::json!({
            "chat_id":              chat_id,
            "text":                 text,
            "parse_mode":           "HTML",
            "disable_notification": silent,
        });
        if let Err(e) = self.client.post(&url).json(&body).send().await {
            warn!("telegram: sendMessage a {chat_id}: {e}");
        }
    }

    async fn broadcast_msg(&self, text: &str, silent: bool) {
        let subs = self.subs.lock().await.clone();
        for id in subs {
            self.send_msg(id, text, silent).await;
        }
    }

    async fn get_updates(&self, offset: i64) -> anyhow::Result<Vec<Update>> {
        let url = format!("{}/getUpdates", self.base);
        let res: TgResponse<Vec<Update>> = self
            .client
            .get(&url)
            .query(&[
                ("offset", &offset.to_string()),
                ("timeout", &"30".to_string()),
            ])
            .send()
            .await
            .context("getUpdates")?
            .json()
            .await
            .context("getUpdates parse")?;
        Ok(if res.ok {
            res.result.unwrap_or_default()
        } else {
            vec![]
        })
    }
}

// ── Public task ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub async fn run(
    token: String,
    allowed_ids: Vec<i64>,
    cfg: TelegramConfig,
    state: Arc<StateStore>,
    risk: Arc<RiskGuard>,
    fill_rx: broadcast::Receiver<Fill>,
    shutdown: watch::Receiver<bool>,
    log_buf: LogBuffer,
    tools: Arc<ToolSupervisor>,
    // On-demand spawn of a strategy with `enabled=false` at
    // boot — same mechanism used by `api::control_strategy`.
    strategies: Arc<crate::strategy::StrategySupervisor>,
) {
    if token.is_empty() {
        debug!("telegram: empty token — bot disabled");
        return;
    }
    info!("telegram: bot starting");

    let bot = Bot::new(&token, allowed_ids);

    // Pre-register all authorized IDs as notification destinations.
    // In private chats user_id == chat_id, so the bot sends without waiting for /start.
    {
        let mut subs = bot.subs.lock().await;
        for &id in &bot.allowed {
            subs.insert(id);
        }
    }

    // Update the "/" menu in Telegram clients (removes commands of the old bot).
    bot.set_commands().await;

    // Welcome message with the command list at startup.
    bot.broadcast_msg(&help_text(), true).await;

    let mut tasks = tokio::task::JoinSet::<()>::new();

    // 1. Command poller (long-poll getUpdates)
    {
        let bot = bot.clone();
        let risk = risk.clone();
        let state = state.clone();
        let log_buf = log_buf.clone();
        let tools = tools.clone();
        let strategies = strategies.clone();
        let mut sd = shutdown.clone();
        tasks.spawn(async move {
            command_loop(bot, risk, state, log_buf, tools, strategies, &mut sd).await
        });
    }

    // 2. Fill notifier
    if cfg.fill_alerts {
        let bot = bot.clone();
        let st = state.clone();
        let mut rx = fill_rx;
        let mut sd = shutdown.clone();
        tasks.spawn(async move { fill_loop(bot, st, &mut rx, &mut sd).await });
    } else {
        drop(fill_rx);
    }

    // 3. Monitor: periodic summary + kill switch alert
    {
        let bot = bot.clone();
        let risk = risk.clone();
        let state = state.clone();
        let hours = cfg.periodic_summary_hours;
        let tools = tools.clone();
        let mut sd = shutdown.clone();
        tasks.spawn(async move { monitor_loop(bot, risk, state, hours, tools, &mut sd).await });
    }

    // Finish when any task exits (normally shutdown).
    tasks.join_next().await;
    tasks.abort_all();
    info!("telegram: bot stopped");
}

// ── Command loop ──────────────────────────────────────────────────────────

async fn command_loop(
    bot: Arc<Bot>,
    risk: Arc<RiskGuard>,
    state: Arc<StateStore>,
    log_buf: LogBuffer,
    tools: Arc<ToolSupervisor>,
    strategies: Arc<crate::strategy::StrategySupervisor>,
    shutdown: &mut watch::Receiver<bool>,
) {
    let mut offset = 0i64;
    loop {
        let result = tokio::select! {
            biased;
            _ = shutdown.changed() => break,
            r = bot.get_updates(offset) => r,
        };

        let updates = match result {
            Ok(v) => v,
            Err(e) => {
                warn!("telegram: getUpdates error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        for upd in updates {
            offset = upd.update_id + 1;
            let Some(msg) = upd.message else { continue };
            let Some(text) = msg.text else { continue };
            let from_id = msg.from.as_ref().map(|u| u.id).unwrap_or(0);
            let chat_id = msg.chat.id;

            if !bot.allowed.contains(&from_id) {
                debug!("telegram: message from unauthorized ID {from_id}");
                continue;
            }
            // Register the chat as a destination for future broadcasts.
            bot.subs.lock().await.insert(chat_id);
            handle_command(
                &bot,
                &risk,
                &state,
                &log_buf,
                &tools,
                &strategies,
                chat_id,
                text.trim(),
            )
            .await;
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_command(
    bot: &Bot,
    risk: &Arc<RiskGuard>,
    state: &Arc<StateStore>,
    log_buf: &LogBuffer,
    tools: &Arc<ToolSupervisor>,
    strategies: &Arc<crate::strategy::StrategySupervisor>,
    chat_id: i64,
    text: &str,
) {
    let parts: Vec<&str> = text.split_whitespace().collect();
    let cmd = parts.first().copied().unwrap_or("").to_lowercase();
    let arg = parts.get(1).copied();

    match cmd.as_str() {
        "/state" | "/start" => {
            bot.send_msg(chat_id, &build_state(state, risk, tools).await, true)
                .await;
        }
        "/positions" => {
            bot.send_msg(chat_id, &build_positions(state).await, true)
                .await;
        }
        "/pnl" => {
            bot.send_msg(chat_id, &build_pnl(state).await, true).await;
        }
        "/logs" => {
            bot.send_msg(chat_id, &build_logs(log_buf), true).await;
        }
        "/mute" => {
            let on = arg.map(|s| s == "on").unwrap_or(true);
            *bot.mute.lock().await = on;
            let msg = if on {
                "🔇 Fills muted."
            } else {
                "🔔 Fills enabled."
            };
            bot.send_msg(chat_id, msg, true).await;
        }
        "/kill" => match arg {
            None => {
                risk.kill_global("manual kill via Telegram");
                bot.send_msg(chat_id, "🔴 <b>Global kill switch activated.</b>", false)
                    .await;
            }
            // ToolId first: `StrategyId::from_str` does fail on
            // unknown names (the Arb-tolerant deserialization is only
            // for historical SQLite reads), but without this check
            // "/kill vol_surface" would fall into the strategy arm and answer
            // "unknown" instead of saying a tool has no kill switch.
            Some(s) if s.parse::<ToolId>().is_ok() => {
                bot.send_msg(
                    chat_id,
                    "🔧 This is a tool, it has no kill switch — use /disable.",
                    true,
                )
                .await;
            }
            Some(s) => match s.parse::<StrategyId>() {
                Ok(id) => {
                    risk.kill_strategy(id, "manual kill via Telegram");
                    bot.send_msg(
                        chat_id,
                        &format!("🔴 Kill activated: <code>{s}</code>"),
                        true,
                    )
                    .await;
                }
                Err(_) => {
                    bot.send_msg(chat_id, "❌ Unknown strategy.", true)
                        .await;
                }
            },
        },
        "/reset" => match arg {
            None => {
                risk.reset_global_kill();
                bot.send_msg(chat_id, "✅ Global kill switch reset.", true)
                    .await;
            }
            Some(s) if s.parse::<ToolId>().is_ok() => {
                bot.send_msg(
                    chat_id,
                    "🔧 This is a tool, it has no kill switch — use /enable.",
                    true,
                )
                .await;
            }
            Some(s) => match s.parse::<StrategyId>() {
                Ok(id) => {
                    risk.reset_strategy_kill(id);
                    bot.send_msg(
                        chat_id,
                        &format!("✅ Kill reset: <code>{s}</code>"),
                        true,
                    )
                    .await;
                }
                Err(_) => {
                    bot.send_msg(chat_id, "❌ Unknown strategy.", true)
                        .await;
                }
            },
        },
        "/enable" => match arg {
            Some(s) if s.parse::<ToolId>().is_ok() => {
                let id = s.parse::<ToolId>().unwrap();
                // `set_enabled` may reject (Stopping, missing
                // deps) — as with strategies, a rejection
                // persists nothing.
                match tools.set_enabled(id, true).await {
                    Ok(()) => {
                        // Only update the desired state reported by the API/UI
                        // if persistence really succeeded — see
                        // `set_desired_enabled`.
                        let msg = match persist_tool_toggle(id, true).await {
                            Ok(()) => {
                                tools.set_desired_enabled(id, true);
                                format!("🔧 <code>{id}</code> enabled.")
                            }
                            Err(e) => format!(
                                "🔧 <code>{id}</code> enabled, but it could NOT be persisted \
                                 ⚠ a bot restart will revert this: {e}"
                            ),
                        };
                        bot.send_msg(chat_id, &msg, true).await;
                    }
                    Err(reason) => {
                        bot.send_msg(
                            chat_id,
                            &format!("⛔ <code>{id}</code> NOT enabled: {reason}"),
                            true,
                        )
                        .await;
                    }
                }
            }
            Some(s) => match s.parse::<StrategyId>().ok() {
                Some(id) => {
                    // `set_enabled` now spawns the task on demand
                    // if it never ran, besides resetting the kill switch.
                    // It may reject it (in live `paper_only` strategies
                    // are blocked); if it rejects, nothing is persisted.
                    if let Err(reason) = strategies.set_enabled(id, true) {
                        bot.send_msg(
                            chat_id,
                            &format!("⛔ <code>{id}</code> NOT enabled: {reason}"),
                            true,
                        )
                        .await;
                        return;
                    }
                    // Previously only the in-memory kill switch was touched — a
                    // bot restart silently reverted it (unlike the
                    // equivalent API endpoint, which persisted from the
                    // start). Same pattern as `persist_tool_toggle` above.
                    let persisted = persist_strategy_toggle(id, true)
                        .await
                        .map_err(|e| e.to_string());
                    bot.send_msg(chat_id, &strategy_toggle_message(id, true, persisted), true)
                        .await;
                }
                None => {
                    bot.send_msg(chat_id, "❌ Unknown strategy.", true)
                        .await;
                }
            },
            None => {
                bot.send_msg(
                    chat_id,
                    "Usage: /enable &lt;strategy|tool&gt;",
                    true,
                )
                .await;
            }
        },
        "/disable" => match arg {
            Some(s) if s.parse::<ToolId>().is_ok() => {
                let id = s.parse::<ToolId>().unwrap();
                // Unlike a strategy, stopping a tool CAN
                // be rejected (it is already stopping) — there is no fallback kill
                // switch to leave it in a known state.
                match tools.set_enabled(id, false).await {
                    Ok(()) => {
                        let msg = match persist_tool_toggle(id, false).await {
                            Ok(()) => {
                                tools.set_desired_enabled(id, false);
                                format!("🔧 <code>{id}</code> disabled.")
                            }
                            Err(e) => format!(
                                "🔧 <code>{id}</code> disabled, but it could NOT be persisted \
                                 ⚠ a bot restart will revert this: {e}"
                            ),
                        };
                        bot.send_msg(chat_id, &msg, true).await;
                    }
                    Err(reason) => {
                        bot.send_msg(
                            chat_id,
                            &format!("⛔ <code>{id}</code> NOT disabled: {reason}"),
                            true,
                        )
                        .await;
                    }
                }
            }
            Some(s) => match s.parse::<StrategyId>().ok() {
                Some(id) => {
                    // Turning off is never rejected (see `set_enabled`).
                    let _ = strategies.set_enabled(id, false);
                    // Same as /enable — persist so it survives a restart.
                    let persisted = persist_strategy_toggle(id, false)
                        .await
                        .map_err(|e| e.to_string());
                    bot.send_msg(
                        chat_id,
                        &strategy_toggle_message(id, false, persisted),
                        true,
                    )
                    .await;
                }
                None => {
                    bot.send_msg(chat_id, "❌ Unknown strategy.", true)
                        .await;
                }
            },
            None => {
                bot.send_msg(
                    chat_id,
                    "Usage: /disable &lt;strategy|tool&gt;",
                    true,
                )
                .await;
            }
        },
        "/limits" => {
            let msg = match arg {
                Some(s) if s.parse::<ToolId>().is_ok() => {
                    "🔧 This is a tool, with no risk limits (it does not go through RiskGuard)."
                        .to_string()
                }
                Some(s) => match s.parse::<StrategyId>().ok() {
                    Some(id) => {
                        let (rate, usdc) = risk.get_strategy_limits(id);
                        let icon = if risk.is_strategy_killed(id) {
                            "⏸"
                        } else {
                            "▶"
                        };
                        format!(
                            "{icon} <b>{id}</b>\n\
                             Max per order: <code>${usdc:.2}</code>\n\
                             Max orders/h: <code>{rate}</code>"
                        )
                    }
                    None => "❌ Unknown strategy.".to_string(),
                },
                None => {
                    let lines: Vec<String> = StrategyId::ALL
                        .iter()
                        .map(|&id| {
                            let (rate, usdc) = risk.get_strategy_limits(id);
                            let icon = if risk.is_strategy_killed(id) {
                                "⏸"
                            } else {
                                "▶"
                            };
                            format!("{icon} <code>{id}</code>  ${usdc:.0}/order · {rate}/h")
                        })
                        .collect();
                    format!("<b>Limits per strategy:</b>\n{}", lines.join("\n"))
                }
            };
            bot.send_msg(chat_id, &msg, true).await;
        }
        "/exposure" => match arg {
            None => {
                let v = risk.max_total_exposure();
                bot.send_msg(
                    chat_id,
                    &format!("💰 Current global exposure: <code>${v:.2}</code>"),
                    true,
                )
                .await;
            }
            Some(s) => match crate::config::parse_exposure_limit(s) {
                Ok(v) => {
                    risk.set_max_total_exposure(v);
                    let v_save = v.to_string();
                    let r = tokio::task::spawn_blocking(move || {
                        crate::config::persist_max_total_exposure(&v_save)
                    })
                    .await;
                    match r {
                        Ok(Ok(())) => {
                            let mut msg = format!("✅ Global exposure → <code>${v:.2}</code>");
                            if crate::config::exposure_env_override_present() {
                                msg.push_str("\n⚠ HRF_RISK__MAX_TOTAL_EXPOSURE_USDC is defined in the environment — a restart will revert this value.");
                            }
                            bot.send_msg(chat_id, &msg, true).await;
                        }
                        Ok(Err(e)) => {
                            bot.send_msg(chat_id, &format!("❌ Error while persisting: {e}"), true)
                                .await;
                        }
                        Err(e) => {
                            warn!("telegram: /exposure persist: {e}");
                            bot.send_msg(chat_id, "❌ Internal error while persisting.", true)
                                .await;
                        }
                    }
                }
                Err(e) => {
                    bot.send_msg(chat_id, &format!("❌ {e}"), true).await;
                }
            },
        },
        "/help" => {
            bot.send_msg(chat_id, &help_text(), true).await;
        }
        _ => {} // unknown commands silently ignored
    }
}

/// Persists a tool's toggle in its TOML (same pattern as
/// `control_strategy` in the API: the runtime effect already happened via
/// `ToolSupervisor::set_enabled`, this only ensures it survives a
/// restart). Returns `Result` (it used to only log) — the caller
/// needs to know whether it really persisted in order to
/// decide whether to update `ToolSupervisor::set_desired_enabled`.
async fn persist_tool_toggle(id: ToolId, enabled: bool) -> Result<(), String> {
    let name = id.to_string();
    match tokio::task::spawn_blocking(move || crate::config::persist_strategy_toggle(&name, enabled))
        .await
    {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e.to_string()),
        Err(e) => Err(format!("join: {e}")),
    }
}

/// Persists a STRATEGY's toggle in its TOML — same pattern as
/// `persist_tool_toggle`, but returning the error to the caller instead of just
/// logging it, because here the operator needs to know in the Telegram message
/// itself whether the toggle will survive a restart (before: /enable and
/// /disable only touched the in-memory kill switch, and the operator never
/// found out it reverted on its own).
async fn persist_strategy_toggle(id: StrategyId, enabled: bool) -> anyhow::Result<()> {
    let name = id.to_string();
    tokio::task::spawn_blocking(move || crate::config::persist_strategy_toggle(&name, enabled))
        .await
        .map_err(|e| anyhow::anyhow!("join: {e}"))?
}

/// Final message after enabling/disabling a strategy via Telegram, depending on
/// whether persistence to TOML succeeded. Extracted as a pure function (instead of
/// inline in `handle_command`) so the warning wording can be tested without
/// writing to real disk or depending on the Telegram API.
fn strategy_toggle_message(id: StrategyId, enabled: bool, persisted: Result<(), String>) -> String {
    let (icon, verb) = if enabled {
        ("▶", "enabled")
    } else {
        ("⏸", "disabled")
    };
    match persisted {
        Ok(()) => format!("{icon} <code>{id}</code> {verb}."),
        Err(e) => format!(
            "{icon} <code>{id}</code> {verb}, but it could NOT be persisted \
             ⚠ a bot restart will revert this: {e}"
        ),
    }
}

fn help_text() -> String {
    let strategies = StrategyId::ALL
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(" · ");
    let tools = ToolId::ALL
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(" · ");
    format!(
        "\
🤖 <b>Poly-bot</b>

<b>📊 Queries</b>
/state — portfolio, PnL and strategy/tool state
/positions — open positions with time remaining
/pnl — realized PnL per strategy
/limits [strategy] — current operating limits
/logs — last 50 log records

<b>🎛 Control</b>
/enable &lt;strategy|tool&gt; — remove kill switch / start tool
/disable &lt;strategy|tool&gt; — set kill switch / stop tool
/kill [strategy] — kill switch (global or per strategy; not applicable to tools)
/reset [strategy] — reset kill switch (not applicable to tools)
/exposure [new value] — view or change the global exposure limit

<b>🔔 Notifications</b>
/mute [on|off] — mute/unmute fill alerts

<i>Strategies: {strategies}</i>
<i>🔧 Tools (scan-only, no kill switch): {tools}</i>"
    )
}

// ── Fill loop ─────────────────────────────────────────────────────────────

async fn fill_loop(
    bot: Arc<Bot>,
    state: Arc<StateStore>,
    fill_rx: &mut broadcast::Receiver<Fill>,
    shutdown: &mut watch::Receiver<bool>,
) {
    loop {
        let fill = tokio::select! {
            biased;
            _ = shutdown.changed() => break,
            r = fill_rx.recv() => r,
        };

        match fill {
            Err(broadcast::error::RecvError::Closed) => break,
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Ok(fill) => {
                if matches!(
                    fill.side,
                    FillSide::Rejected | FillSide::Cancelled | FillSide::GateRejected
                ) {
                    continue;
                }
                if *bot.mute.lock().await {
                    continue;
                }
                let slug = state
                    .get_market_slug(&fill.market_cid)
                    .await
                    .unwrap_or_default();
                bot.broadcast_msg(&format_fill(&fill, &slug), true).await;
            }
        }
    }
}

/// Converts slug `btc-updown-5m-1719172500` → `BTC 5m 19:55–20:00`.
fn slug_to_label(slug: &str) -> String {
    let parts: Vec<&str> = slug.splitn(4, '-').collect();
    if parts.len() != 4 || parts[1] != "updown" {
        return slug.to_string();
    }
    let asset = parts[0].to_uppercase();
    let dur_mins: i64 = parts[2]
        .strip_suffix('m')
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let start_ts: i64 = parts[3].parse().unwrap_or(0);
    if dur_mins == 0 || start_ts == 0 {
        return slug.to_string();
    }
    let fmt = |ts: i64| -> String {
        chrono::DateTime::from_timestamp(ts, 0)
            .map(|dt| dt.format("%H:%M").to_string())
            .unwrap_or_default()
    };
    format!(
        "{asset} {dur_mins}m {}–{}",
        fmt(start_ts),
        fmt(start_ts + dur_mins * 60)
    )
}

fn format_fill(fill: &Fill, slug: &str) -> String {
    let (icon, side_label) = match fill.side {
        FillSide::Buy => ("🟢", "BUY"),
        FillSide::Sell => ("🔴", "SELL"),
        FillSide::Rejected | FillSide::Cancelled | FillSide::GateRejected => {
            unreachable!("Rejected/Cancelled/GateRejected filtered out before format_fill")
        }
    };
    let market = if slug.is_empty() {
        fill.market_cid.chars().take(12).collect::<String>()
    } else {
        slug_to_label(slug)
    };
    let secs = fill.clear_at - chrono::Utc::now().timestamp();
    let time = fmt_time(secs);
    let usdc = fill.price * fill.size;
    format!(
        "{icon} <b>{side_label}</b>  ·  <code>{strat}</code>\n\
         📌 <b>{market}</b>\n\
         \n\
         <code>{size:.2}</code> shares @ <code>${price:.4}</code>\n\
         💵 <b>${usdc:.2}</b>  ·  ⏱ {time}",
        strat = fill.strategy,
        size = fill.size,
        price = fill.price,
    )
}

// ── Monitor loop ───────────────────────────────────────────────────────────

async fn monitor_loop(
    bot: Arc<Bot>,
    risk: Arc<RiskGuard>,
    state: Arc<StateStore>,
    hours: u32,
    tools: Arc<ToolSupervisor>,
    shutdown: &mut watch::Receiver<bool>,
) {
    // If hours=0: summary disabled — summary_ticker never fires.
    let mut summary_ticker = (hours > 0).then(|| {
        let mut i = tokio::time::interval(std::time::Duration::from_secs(hours as u64 * 3600));
        i.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        i
    });

    let mut alert_ticker = tokio::time::interval(std::time::Duration::from_secs(10));
    let mut was_killed = false;

    loop {
        tokio::select! {
            biased;
            _ = shutdown.changed()    => break,
            _ = alert_ticker.tick()   => {
                let killed = risk.is_killed();
                if killed && !was_killed {
                    bot.broadcast_msg("🔴 <b>GLOBAL KILL SWITCH ACTIVATED</b>", false).await;
                }
                was_killed = killed;
            }
            _ = async {
                match &mut summary_ticker {
                    Some(t) => { t.tick().await; }
                    None    => std::future::pending::<()>().await,
                }
            } => {
                bot.broadcast_msg(&build_state(&state, &risk, &tools).await, true).await;
            }
        }
    }
}

// ── Message builders ──────────────────────────────────────────────────────

async fn build_state(state: &StateStore, risk: &RiskGuard, tools: &ToolSupervisor) -> String {
    let (deposits, cost, pnl) = tokio::join!(
        state.deposits_total(),
        state.total_open_cost(),
        state.realized_pnl_total(),
    );
    // Previously a DB failure in any of the three queries was silenced with
    // unwrap_or_default() and /state showed cash/PnL fabricated from
    // zeros without warning — in a bot with real capital that can read as "free
    // cash = $X" when the data is actually unknown. build_pnl/
    // build_positions in this same file already return the error to the user;
    // here there was not even a log.
    let deposits = deposits.unwrap_or_else(|e| {
        tracing::error!("build_state: deposits_total failed: {e}");
        Decimal::ZERO
    });
    let cost = cost.unwrap_or_else(|e| {
        tracing::error!("build_state: total_open_cost failed: {e}");
        Decimal::ZERO
    });
    let pnl = pnl.unwrap_or_else(|e| {
        tracing::error!("build_state: realized_pnl_total failed: {e}");
        Decimal::ZERO
    });
    let cash = deposits - cost + pnl;

    let global_icon = if risk.is_killed() { "🔴" } else { "🟢" };
    let global_txt = if risk.is_killed() {
        "<b>KILL ACTIVE</b>"
    } else {
        "OK"
    };

    let strat_lines: Vec<String> = StrategyId::ALL
        .iter()
        .map(|&id| {
            let icon = if risk.is_strategy_killed(id) {
                "⏸"
            } else {
                "▶"
            };
            format!("{icon} <code>{id}</code>")
        })
        .collect();

    // Tools in a separate section: 🔧 instead of ▶/⏸ (they don't go through the kill switch,
    // the state that matters is whether the task is running or not).
    let tool_lines: Vec<String> = ToolId::ALL
        .iter()
        .map(|&id| {
            let running = tools.is_running(id);
            format!(
                "🔧 <code>{id}</code>  {}",
                if running { "running" } else { "stopped" }
            )
        })
        .collect();

    format!(
        "🤖 <b>HRF V3</b>  {global_icon} {global_txt}\n\
         \n\
         <b>💰 Portfolio</b>\n\
         Deposits       <code>${deposits:.2}</code>\n\
         Open cost      <code>${cost:.2}</code>\n\
         Free cash      <code>${cash:.2}</code>\n\
         Realized PnL   <code>{pnl_sign}${pnl_abs:.2}</code>\n\
         \n\
         <b>📊 Strategies</b>\n\
         {strats}\n\
         \n\
         <b>🔧 Tools</b>\n\
         {tools_}",
        pnl_sign = if pnl < Decimal::ZERO { "−" } else { "+" },
        pnl_abs = pnl.abs(),
        strats = strat_lines.join("\n"),
        tools_ = tool_lines.join("\n"),
    )
}

async fn build_positions(state: &StateStore) -> String {
    match state.list_open_positions().await {
        Ok(pos) if pos.is_empty() => "📭 <i>No open positions.</i>".to_string(),
        Ok(pos) => {
            let now = chrono::Utc::now().timestamp();
            let lines: Vec<String> = pos.iter().map(|p| {
                let label = slug_to_label(&p.slug);
                let t     = fmt_time(p.clear_at - now);
                let cost  = p.avg_price * p.size;
                format!(
                    "📌 <b>{label}</b>  <i>{strat}</i>\n\
                     <code>{size:.2}</code> shares @ <code>${avg:.4}</code>  ·  <b>${cost:.2}</b>  ·  ⏱ {t}",
                    strat = p.strategy,
                    size  = p.size,
                    avg   = p.avg_price,
                )
            }).collect();
            format!(
                "<b>Open positions ({})</b>\n\n{}",
                pos.len(),
                lines.join("\n\n")
            )
        }
        Err(e) => format!("❌ Error: {e}"),
    }
}

async fn build_pnl(state: &StateStore) -> String {
    match state.realized_pnl_by_strategy().await {
        Ok(map) if map.is_empty() => "<i>No PnL recorded yet.</i>".to_string(),
        Ok(map) => {
            let total: Decimal = map.values().sum();
            let mut pairs: Vec<(&String, &Decimal)> = map.iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(a.1));
            let lines: Vec<String> = pairs
                .iter()
                .map(|(k, v)| {
                    let sign = if **v >= Decimal::ZERO { "+" } else { "−" };
                    let icon = if **v >= Decimal::ZERO { "🟢" } else { "🔴" };
                    format!(
                        "{icon} <code>{k}</code>  <code>{sign}${:.4}</code>",
                        v.abs()
                    )
                })
                .collect();
            let total_sign = if total >= Decimal::ZERO { "+" } else { "−" };
            format!(
                "<b>PnL per strategy</b>\n\n{}\n\n<b>Total:  <code>{total_sign}${:.4}</code></b>",
                lines.join("\n"),
                total.abs()
            )
        }
        Err(e) => format!("❌ Error: {e}"),
    }
}

fn build_logs(log_buf: &LogBuffer) -> String {
    let lines: Vec<String> = {
        let buf = log_buf.lock().unwrap_or_else(|e| e.into_inner());
        // Last 50 lines
        buf.iter()
            .rev()
            .take(50)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    };

    if lines.is_empty() {
        return "<i>No records yet.</i>".to_string();
    }

    // Build the <pre> block respecting Telegram's 4096-char limit
    let header = format!("📋 <b>Log</b> ({} lines)\n<pre>", lines.len());
    let footer = "</pre>";
    let budget = 4090usize.saturating_sub(header.len() + footer.len());

    let mut body = String::new();
    for line in &lines {
        // Escape < > & inside the pre block for valid HTML
        let escaped = line
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");
        let candidate = format!("{escaped}\n");
        if body.len() + candidate.len() > budget {
            body.push_str("…\n");
            break;
        }
        body.push_str(&candidate);
    }

    format!("{header}{body}{footer}")
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn fmt_time(secs: i64) -> String {
    if secs > 3600 {
        format!("{:.1}h", secs as f64 / 3600.0)
    } else if secs > 0 {
        format!("{secs}s")
    } else {
        "expired".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_includes_post_close_sweep() {
        assert!(help_text().contains("post_close_sweep"));
    }

    /// /enable and /disable must explicitly warn if
    /// persistence to TOML failed (before, a persistence failure was
    /// invisible to the operator — the in-memory kill switch was applied
    /// anyway and the message sounded permanent even if it wasn't).
    #[test]
    fn strategy_toggle_message_warns_if_not_persisted() {
        let ok = strategy_toggle_message(StrategyId::Arb, true, Ok(()));
        assert_eq!(ok, "▶ <code>arb</code> enabled.");
        assert!(!ok.contains('⚠'));

        let fail = strategy_toggle_message(StrategyId::Arb, false, Err("disk full".to_string()));
        assert!(fail.contains("⏸ <code>arb</code> disabled"));
        assert!(fail.contains('⚠'), "must warn that it was not persisted: {fail}");
        assert!(fail.contains("disk full"));
    }
}
