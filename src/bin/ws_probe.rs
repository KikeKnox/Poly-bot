//! One-off probe: checks live whether the Polymarket market WS accepts
//! `{"operation":"subscribe"}` / `{"operation":"unsubscribe"}` on an
//! already-established connection — the documented format for dynamic
//! subscribe/unsubscribe, unlike the `{"type":"subscribe"}` format tried earlier.
//!
//! Usage: `cargo run --release --bin ws_probe -- <token_A> <token_B>`

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Same as main.rs: install the TLS provider before any connect_async.
    rustls::crypto::CryptoProvider::install_default(rustls::crypto::aws_lc_rs::default_provider())
        .expect("crypto provider");

    let args: Vec<String> = std::env::args().collect();
    let token_a = args.get(1).cloned().unwrap_or_else(|| {
        "100119495087428376848600374130039433860169587688250545542125705489373509206852".into()
    });
    let token_b = args.get(2).cloned().unwrap_or_else(|| {
        "100475956345878524985075624356854356332013866849010352532056817449346496198989".into()
    });

    let (mut ws, _) = connect_async("wss://ws-subscriptions-clob.polymarket.com/ws/market").await?;
    println!("connected");

    // Initial subscription using the bot's current format.
    let init = serde_json::json!({
        "type": "subscribe",
        "channel": "book",
        "assets_ids": [token_a],
    });
    ws.send(Message::Text(init.to_string())).await?;
    println!("initial subscribe sent for A={}", &token_a[..12]);

    let mut got_a = false;
    let mut sent_b = false;
    let mut got_b = false;
    let mut unsub_a = false;
    let mut a_after_unsub = 0u32;
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(45);
    let mut last_ping = tokio::time::Instant::now();

    while tokio::time::Instant::now() < deadline {
        if last_ping.elapsed() > std::time::Duration::from_secs(10) {
            ws.send(Message::Text("ping".into())).await?;
            last_ping = tokio::time::Instant::now();
        }
        // At 8s send the dynamic subscribe for B; at 30s unsubscribe A.
        if got_a && !sent_b {
            sent_b = true;
            let dyn_sub = serde_json::json!({
                "operation": "subscribe",
                "assets_ids": [token_b],
            });
            ws.send(Message::Text(dyn_sub.to_string())).await?;
            println!(
                "[t=+8s] operation:subscribe sent for B={}",
                &token_b[..12]
            );
        }
        if got_b && !unsub_a {
            unsub_a = true;
            let dyn_unsub = serde_json::json!({
                "operation": "unsubscribe",
                "assets_ids": [token_a],
            });
            ws.send(Message::Text(dyn_unsub.to_string())).await?;
            println!("[t≈+25s] operation:unsubscribe sent for A");
        }

        let msg = match tokio::time::timeout(std::time::Duration::from_secs(3), ws.next()).await {
            Ok(Some(Ok(m))) => m,
            Ok(Some(Err(e))) => {
                println!("WS error: {e}");
                break;
            }
            Ok(None) => {
                println!("WS closed");
                break;
            }
            Err(_) => continue,
        };
        let Message::Text(t) = msg else { continue };
        if t.contains(&token_a) {
            if !got_a {
                println!("first event for A received");
            }
            got_a = true;
            if unsub_a {
                a_after_unsub += 1;
            }
        }
        if t.contains(&token_b) {
            if !got_b {
                println!("*** FIRST EVENT FOR B after operation:subscribe ***");
                println!("sample: {}", &t[..t.len().min(300)]);
            }
            got_b = true;
        }
    }

    println!("---");
    println!("RESULT: got_A={got_a} got_B={got_b} events_of_A_after_unsub={a_after_unsub}");
    println!(
        "{}",
        if got_b {
            "operation:subscribe WORKS → dynamic subscriptions without reconnecting are possible"
        } else {
            "operation:subscribe IGNORED → keep the epoch restart"
        }
    );
    Ok(())
}
