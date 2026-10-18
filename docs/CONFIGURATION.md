# Configuration

Configuration has two layers: environment variables (secrets and mode switches, usually from `.env`) and TOML files under `config/` (operating parameters). The bot must be started from the repository root, because it looks for `config/` and `.env` in the working directory and exits with an error if `config/` is missing.

## 1. Layers and precedence

Loading is implemented in `src/config.rs` (`Config::load`) with [figment](https://docs.rs/figment). Later sources override earlier ones:

1. `config/general.toml`
2. `config/telegram.toml`
3. every `config/strategies/*.toml`, merged in alphabetical order
4. environment variables with the `HRF_` prefix

`.env` is loaded into the process environment at startup (via `dotenvy`) unless `HRF_NO_DOTENV` is set. Values already exported in the shell win over `.env`.

`.env.example` documents every variable the code reads. Copy it to `.env`; the file is git-ignored and must never be committed.

### Environment override rule

The prefix is `HRF_`, matching is case-insensitive, and a **double underscore** (`__`) becomes a nesting dot:

| Variable | Overrides |
|---|---|
| `HRF_MODE=paper` | `mode` in `config/general.toml` |
| `HRF_DB_PATH=...` | `db_path` |
| `HRF_API__TOKEN=xxx` | `token` in `[api]` |
| `HRF_API__PORT=8766` | `port` in `[api]` |
| `HRF_RISK__MAX_TOTAL_EXPOSURE_USDC=20` | `max_total_exposure_usdc` in `[risk]` |
| `HRF_ORACLE__FUTURES_LIQ_ENABLED=true` | `futures_liq_enabled` in `[oracle]` |
| `HRF_ARB__ENABLED=true` | `enabled` in `[arb]` (`config/strategies/arb.toml`) |
| `HRF_REDEEM__RPC_URL=https://...` | `rpc_url` in `[redeem]` |

Single underscores belong to the key name; only `__` separates levels. Anything that can be written in a TOML file can be overridden this way, including per-strategy limits.

### Variables that are not `HRF_` overrides

| Variable | Used for |
|---|---|
| `POLY_PRIVATE_KEY` | Wallet key; required in live mode |
| `POLY_SIGNATURE_TYPE`, `POLY_PROXY_ADDRESS` | Signature type (0 EOA default, 1 proxy, 2 Gnosis Safe, 3 Poly1271) and funder address |
| `POLY_API_KEY`, `POLY_API_SECRET`, `POLY_API_PASSPHRASE` | CLOB L2 credentials. In live they are derived and written back to `.env` if missing; in paper they are used only by `arb_15m5m` (Chainlink TWAP feed) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_IDS` | Telegram control; the bot only starts if the token is set; IDs are comma-separated |
| `RUST_LOG`, `LOG_FORMAT=json` | Log filter and structured output |
| `HRF_UI_DIST_DIR` | Directory of the built UI (default `src/api/static/dist`) |
| `HRF_NO_DOTENV` | Skip loading `.env` |

## 2. `config/general.toml`

Each section is commented in the file itself. The most relevant keys:

| Key | Meaning |
|---|---|
| `mode` | `"paper"` (simulator) or `"live"` (real orders). Default `"paper"` |
| `db_path` | SQLite file for the state store. The mode is appended to the file name (`hrf_state_paper.sqlite`) so paper and live never share a file |
| `clob_api_url`, `gamma_api_url` | Polymarket REST endpoints |
| `[oracle]` | Binance WebSocket URL, staleness and reconnect timers, outlier filter, optional futures-liquidation feed |
| `[market_feed]` | Slug patterns to trade, discovery window, REST resync tuning, `max_book_age_ms` (books older than this are not traded), taker fee lookup |
| `[universes.<name>]` | Market discovery filters. A strategy's `universe` list decides where it trades; universes with `scan = true` are discovered and logged but no strategy trades them |
| `[risk]` | `max_total_exposure_usdc`, `max_bracket_usdc`, `daily_loss_usdc_stop`, loss cooldowns, and the `paper_*` simulator knobs (`paper_slippage_bps`, `paper_fok_fill_rate`, `paper_fok_fill_rate_per_strategy`, `paper_pair_fill_correlation`, `paper_unknown_rate`, `paper_initial_deposit_usdc`, `paper_disable_kill_switch`) |
| `[api]` | `bind` (default `127.0.0.1`), `port` (default `8766`), `token` |
| `[analytics]`, `[spot_stream]`, `[research_events]`, `[resolution_fetcher]`, `[post_close]` | Recorder and observer settings (paths, retention, intervals). Their `enabled` flag lives in `config/strategies/<tool>.toml` so the live toggle can persist it |
| `[redeem]` | Polygon RPC URL for on-chain redemption (live with `risk.auto_redeem = false`) |

`config/telegram.toml` holds `fill_alerts` and `periodic_summary_hours`.

The `[model_shadow]` block (shadow evaluation of the numeric model) is commented out; uncomment it to enable that observer.

## 3. `config/strategies/<name>.toml`

One file per strategy or tool. The section name matches the file (`[arb]`, `[arb_15m5m]`, ...). Every strategy shares these keys:

| Key | Meaning | Default |
|---|---|---|
| `enabled` | Run the strategy | `false` |
| `paper_only` | Never allowed to touch real money: blocked in live mode | `false` |
| `max_orders_per_hour` | Sliding one-hour order limit; `0` disables it | `20` |
| `max_order_usdc` | Cap per order | `50` |
| `daily_loss_usdc_stop` | Realized-loss stop for the day | `100` |
| `universe` | Universes where it trades | the two crypto up/down universes |

Strategy-specific parameters follow, each documented in its file. Toggling a strategy from the API, UI or Telegram rewrites `enabled` in its file in place, keeping comments.

Tools (`analytics`, `spot_stream`, `research_events`, `resolution_fetcher`, `model_tool`, `vol_surface`, `ladder_arb`) use the same directory; for them `enabled` is the only thing the toggle persists. In the repository, `analytics`, `spot_stream`, `resolution_fetcher` and `model_tool` are enabled; all strategies are disabled.

## 4. Paper versus live: the safety barriers

Paper is the default mode; live trading is selected explicitly. Several independent conditions apply, and each fails closed:

1. **`mode = "live"`** in `config/general.toml`. `./launch.sh` exports `HRF_MODE=paper` unless you pass `--live`, so a copied `.env` cannot switch modes by accident.
2. **No mode conflict.** If `HRF_MODE=live` disagrees with `mode` in `config/general.toml`, startup aborts. The reverse (environment says paper, file says live) only logs an error and runs in paper.
3. **`enabled = true`** in the strategy's own TOML (or toggled on at runtime through the API, UI or Telegram).
4. **`paper_only = false`.** Strategies marked `paper_only` (`model_btc5m`, `rl_predictor`, `g_rw1_maker`, `theta_capture`) cannot run in live.
5. **A daily loss stop.** In live mode a strategy with `daily_loss_usdc_stop <= 0` cannot run: live startup aborts if such a strategy is enabled, and enabling it at runtime is refused.
6. **Credentials.** Live requires `POLY_PRIVATE_KEY`; with `risk.auto_redeem = false` it also requires `[redeem].rpc_url`.

In live mode a strategy runs if and only if the global mode is `live`, it is `enabled` and it is not `paper_only`. Anyone who runs live is responsible for complying with Polymarket's terms and local law.

In paper mode, `risk.paper_disable_kill_switch = true` (the default) keeps strategies running through simulated losses, and the simulated top-ups are deposits, not profits.

## 5. API token and ports

- The HTTP API, the UI (`/ui/`) and the Prometheus endpoint (`/metrics`) share one port: `api.port`, default **8766**, bound to `api.bind`, default `127.0.0.1`.
- Binding anything other than loopback requires `api.token` (or `HRF_API__TOKEN`); startup fails otherwise.
- With a token set, clients send `Authorization: Bearer <token>`. `/healthz` and `/metrics` stay open and the static UI files are public. With an empty token, paper mode is open and live mode returns 503 for everything except `/paper/*`.
- Generate a token with `openssl rand -base64 32`.
