-- calibration_analysis.sql
-- Run automatically at the end of calibrate.sh.
-- Requires: sqlite3 >= 3.35 (for .param set). Recent Debian/Ubuntu releases include it.
-- Manual usage: sqlite3 hrf_state_paper.sqlite ".param set :start_ms 0" < tools/research_tools/calibration_analysis.sql

.mode column
.headers on

-- ── 1. Summary per strategy ──────────────────────────────────────────────

.print ""
.print "=== 1. SUMMARY PER STRATEGY ==="
.print "WR computed over already-resolved positions (outcome_won IS NOT NULL)."

SELECT
    strategy,
    count(*)                                                                    n,
    sum(CASE WHEN outcome_won = 1                      THEN 1 ELSE 0 END)      won,
    sum(CASE WHEN outcome_won = 0                      THEN 1 ELSE 0 END)      lost,
    sum(CASE WHEN bail_sold   = 1                      THEN 1 ELSE 0 END)      bails,
    sum(CASE WHEN outcome_won IS NULL AND bail_sold = 0 THEN 1 ELSE 0 END)     pending,
    CASE WHEN sum(CASE WHEN outcome_won IS NOT NULL THEN 1 ELSE 0 END) > 0
         THEN printf('%.1f%%',
              100.0 * sum(CASE WHEN outcome_won = 1 THEN 1 ELSE 0 END)
              / sum(CASE WHEN outcome_won IS NOT NULL THEN 1 ELSE 0 END))
         ELSE '—' END                                                           WR,
    printf('%.1f%%',
        100.0 * sum(CASE WHEN bail_sold = 1 THEN 1 ELSE 0 END)
        / MAX(count(*), 1))                                                     bail_rate
FROM positions
WHERE opened_ts_ms >= :start_ms
GROUP BY strategy
ORDER BY strategy;

-- ── 2. PnL per strategy ──────────────────────────────────────────────────

.print ""
.print "=== 2. PNL PER STRATEGY (SELL/REDEEM closes) ==="

SELECT
    strategy,
    count(*)                                n_closes,
    printf('$%.4f', sum(pnl_um)  / 1e6)    pnl_total,
    printf('$%.4f', avg(pnl_um)  / 1e6)    pnl_avg,
    printf('$%.4f', min(pnl_um)  / 1e6)    pnl_min,
    printf('$%.4f', max(pnl_um)  / 1e6)    pnl_max
FROM transactions
WHERE ts_ms >= :start_ms
  AND pnl_um IS NOT NULL
  AND kind IN ('SELL', 'REDEEM')
GROUP BY strategy
ORDER BY strategy;

-- ── 3. Fills per strategy ────────────────────────────────────────────────

.print ""
.print "=== 3. FILLS PER STRATEGY ==="
.print "Compare n_fills(BUY) with Prometheus signals to compute the fill rate per strategy."

SELECT
    strategy,
    side,
    count(*)                                           n_fills,
    printf('$%.4f', sum(size_um * price_um) / 1e12)   volume_usdc
FROM fills
WHERE ts_ms >= :start_ms
GROUP BY strategy, side
ORDER BY strategy, side;

-- ── 4. ARB — pair completeness (both legs) ───────────────────────────

.print ""
.print "=== 4. ARB: PAIR COMPLETENESS (hedge_id) ==="
.print "Re-enable gate: complete_pairs/total >= 90% and EV > 0 with n >= 50 pairs."

SELECT
    total_hedges,
    complete_pairs,
    loose_legs,
    printf('%.1f%%', 100.0 * complete_pairs / MAX(total_hedges, 1))  pct_complete
FROM (
    SELECT
        count(*)                                                        total_hedges,
        sum(CASE WHEN legs = 2 THEN 1 ELSE 0 END)                      complete_pairs,
        sum(CASE WHEN legs = 1 THEN 1 ELSE 0 END)                      loose_legs
    FROM (
        SELECT hedge_id, count(*) legs
        FROM fills
        WHERE ts_ms >= :start_ms
          AND strategy = 'arb'
          AND side = 'BUY'
          AND hedge_id IS NOT NULL
        GROUP BY hedge_id
    )
);

-- ── 4b. ARB — P&L of orphan legs ─────────────────────────────────────

.print ""
.print "=== 4b. ARB: P&L OF ORPHAN LEGS ==="
.print "Calibrates whether the orphan bail-sell is optimal vs. hold-to-expiry."
.print "  bail+   : sold above the buy price (correct)."
.print "  bail-   : sold below it (waiting would have been worth more)."
.print "  expiry+ : held and won (bail-sell would have been suboptimal)."
.print "  expiry- : held and lost (bail-sell would have saved capital)."

SELECT
    orphan.market_cid,
    printf('%.4f', orphan.buy_price)                                              buy_price,
    COALESCE(printf('%.4f', bail.sell_price), '—')                               sell_price,
    CASE
        WHEN bail.sell_price IS NOT NULL AND bail.sell_price > orphan.buy_price   THEN 'bail+'
        WHEN bail.sell_price IS NOT NULL                                           THEN 'bail-'
        WHEN p.outcome_won = 1                                                     THEN 'expiry+'
        WHEN p.outcome_won = 0                                                     THEN 'expiry-'
        ELSE                                                                            'pending'
    END                                                                            result,
    printf('%+.4f',
        COALESCE(bail.sell_price,
            CASE WHEN p.outcome_won = 1 THEN 1.0
                 WHEN p.outcome_won = 0 THEN 0.0
                 ELSE orphan.buy_price END)
        - orphan.buy_price)                                                        pnl_per_share
FROM (
    SELECT f.market_cid, f.token_id, avg(f.price_um) / 1e6 AS buy_price
    FROM fills f
    WHERE f.ts_ms       >= :start_ms
      AND f.strategy     = 'arb'
      AND f.side         = 'BUY'
      AND f.hedge_id IS NOT NULL
      AND f.hedge_id IN (
          SELECT hedge_id FROM fills
          WHERE ts_ms >= :start_ms AND strategy='arb' AND side='BUY' AND hedge_id IS NOT NULL
          GROUP BY hedge_id HAVING count(*) = 1
      )
    GROUP BY f.market_cid, f.token_id
) orphan
LEFT JOIN (
    SELECT market_cid, token_id, avg(price_um) / 1e6 AS sell_price
    FROM fills
    WHERE ts_ms >= :start_ms AND strategy = 'arb' AND side = 'SELL'
    GROUP BY market_cid, token_id
) bail ON bail.market_cid = orphan.market_cid AND bail.token_id = orphan.token_id
LEFT JOIN positions p
       ON p.market_cid = orphan.market_cid AND p.token_id = orphan.token_id
ORDER BY orphan.market_cid;

-- ── 5. NC / NE — WR per entry price bucket ───────────────────────

.print ""
.print "=== 5. NC / NE: WR PER ENTRY PRICE BUCKET ==="
.print "NC gate: WR > entry_ask to be profitable. Calibrate z_threshold with this table."
.print "Example: bucket 0.86-0.88 requires WR > 87% for positive EV."

SELECT
    p.strategy,
    CASE
        WHEN f.price < 0.84 THEN 'A <0.84'
        WHEN f.price < 0.86 THEN 'B 0.84-0.86'
        WHEN f.price < 0.88 THEN 'C 0.86-0.88'
        WHEN f.price < 0.90 THEN 'D 0.88-0.90'
        ELSE                      'E >=0.90'
    END                                                                         bucket,
    count(*)                                                                    n,
    printf('%.4f', avg(f.price))                                               avg_price,
    sum(CASE WHEN p.outcome_won = 1 THEN 1 ELSE 0 END)                         won,
    sum(CASE WHEN p.outcome_won IS NOT NULL THEN 1 ELSE 0 END)                  resolved_count,
    CASE WHEN sum(CASE WHEN p.outcome_won IS NOT NULL THEN 1 ELSE 0 END) > 0
         THEN printf('%.1f%%',
              100.0 * sum(CASE WHEN p.outcome_won = 1 THEN 1 ELSE 0 END)
              / sum(CASE WHEN p.outcome_won IS NOT NULL THEN 1 ELSE 0 END))
         ELSE '—' END                                                           WR
FROM positions p
JOIN (
    SELECT token_id, strategy, avg(price_um) / 1e6 AS price
    FROM fills
    WHERE ts_ms >= :start_ms AND side = 'BUY'
    GROUP BY token_id, strategy
) f ON p.token_id = f.token_id AND p.strategy = f.strategy
WHERE p.strategy IN ('near_certain', 'near_expiry')
  AND p.opened_ts_ms >= :start_ms
GROUP BY p.strategy, bucket
ORDER BY p.strategy, bucket;

-- ── 6. DriftBox — phase 2 and bail rate ──────────────────────────────────────

.print ""
.print "=== 6. DRIFTBOX: ENTRY PRICE AND BAIL RATE ==="
.print "Gate: WR > 50% over resolved_count (since it is a directional bet)."

SELECT
    printf('%.4f', avg(f.price))                                           avg_price,
    count(p.token_id)                                                       n,
    sum(CASE WHEN p.outcome_won = 1 THEN 1 ELSE 0 END)                     won,
    sum(CASE WHEN p.outcome_won IS NOT NULL THEN 1 ELSE 0 END)              resolved_count,
    CASE WHEN sum(CASE WHEN p.outcome_won IS NOT NULL THEN 1 ELSE 0 END) > 0
         THEN printf('%.1f%%',
              100.0 * sum(CASE WHEN p.outcome_won = 1 THEN 1 ELSE 0 END)
              / sum(CASE WHEN p.outcome_won IS NOT NULL THEN 1 ELSE 0 END))
         ELSE '—' END                                                       WR,
    printf('%.1f%%',
        100.0 * sum(p.bail_sold) / MAX(count(p.token_id), 1))              bail_rate
FROM positions p
LEFT JOIN (
    SELECT token_id, strategy, avg(price_um) / 1e6 AS price
    FROM fills WHERE ts_ms >= :start_ms AND side = 'BUY'
    GROUP BY token_id, strategy
) f ON p.token_id = f.token_id AND p.strategy = f.strategy
WHERE p.strategy = 'drift_box'
  AND p.opened_ts_ms >= :start_ms;

-- ── 7. OpenMomentum — entry price ───────────────────────────────────

.print ""
.print "=== 7. OPEN MOMENTUM: ENTRY PRICE ==="
.print "Gate: avg(fill_price) < avg(FV at entry). FV is not in the DB — compare manually."
.print "Benchmark: avg_price < 0.55 with fv > 0.60 = edge confirmed."

SELECT
    count(p.token_id)                                                       n,
    printf('%.4f', avg(f.price))                                           avg_entry_price,
    printf('%.4f', min(f.price))                                           min_price,
    printf('%.4f', max(f.price))                                           max_price,
    sum(CASE WHEN p.outcome_won = 1 THEN 1 ELSE 0 END)                     won,
    sum(CASE WHEN p.outcome_won IS NOT NULL THEN 1 ELSE 0 END)              resolved_count,
    CASE WHEN sum(CASE WHEN p.outcome_won IS NOT NULL THEN 1 ELSE 0 END) > 0
         THEN printf('%.1f%%',
              100.0 * sum(CASE WHEN p.outcome_won = 1 THEN 1 ELSE 0 END)
              / sum(CASE WHEN p.outcome_won IS NOT NULL THEN 1 ELSE 0 END))
         ELSE '—' END                                                       WR
FROM positions p
LEFT JOIN (
    SELECT token_id, strategy, avg(price_um) / 1e6 AS price
    FROM fills WHERE ts_ms >= :start_ms AND side = 'BUY'
    GROUP BY token_id, strategy
) f ON p.token_id = f.token_id AND p.strategy = f.strategy
WHERE p.strategy = 'open_momentum'
  AND p.opened_ts_ms >= :start_ms;

-- ── 8. ThetaCapture — maker fill price ───────────────────────────────

.print ""
.print "=== 8. THETACAPTURE: MAKER FILL PRICE ==="
.print "Gate fill_rate (bids_placed/bids_filled) is available only in Prometheus."
.print "Here: average fill price and WR of the positions obtained."

SELECT
    count(p.token_id)                                                       n_fills,
    printf('%.4f', avg(f.price))                                           avg_fill_price,
    sum(CASE WHEN p.outcome_won = 1 THEN 1 ELSE 0 END)                     won,
    sum(CASE WHEN p.outcome_won IS NOT NULL THEN 1 ELSE 0 END)              resolved_count,
    CASE WHEN sum(CASE WHEN p.outcome_won IS NOT NULL THEN 1 ELSE 0 END) > 0
         THEN printf('%.1f%%',
              100.0 * sum(CASE WHEN p.outcome_won = 1 THEN 1 ELSE 0 END)
              / sum(CASE WHEN p.outcome_won IS NOT NULL THEN 1 ELSE 0 END))
         ELSE '—' END                                                       WR,
    printf('%.1f%%',
        100.0 * sum(p.bail_sold) / MAX(count(p.token_id), 1))              bail_rate
FROM positions p
LEFT JOIN (
    SELECT token_id, strategy, avg(price_um) / 1e6 AS price
    FROM fills WHERE ts_ms >= :start_ms AND side = 'BUY'
    GROUP BY token_id, strategy
) f ON p.token_id = f.token_id AND p.strategy = f.strategy
WHERE p.strategy = 'theta_capture'
  AND p.opened_ts_ms >= :start_ms;

-- ── Markouts — gate for MM/PM/ThetaCapture: n≥100 AND mk5≥0 ─────────────

.print ""
.print "=== MARKOUT PER STRATEGY (bps; positive = in favor; gate: n>=100 AND mk5>=0) ==="
SELECT
    strategy,
    side,
    COUNT(*)                                                              AS n,
    COUNT(mid_5s_um)                                                      AS with_mid5,
    ROUND(AVG(CASE side
        WHEN 'BUY'  THEN (mid_5s_um  - fill_price_um) * 10000.0 / fill_price_um
        WHEN 'SELL' THEN (fill_price_um - mid_5s_um)  * 10000.0 / fill_price_um
    END), 1)                                                              AS mk5_bps,
    ROUND(AVG(CASE side
        WHEN 'BUY'  THEN (mid_30s_um - fill_price_um) * 10000.0 / fill_price_um
        WHEN 'SELL' THEN (fill_price_um - mid_30s_um) * 10000.0 / fill_price_um
    END), 1)                                                              AS mk30_bps
FROM markouts
WHERE ts_ms >= :start_ms
GROUP BY strategy, side
ORDER BY strategy, side;

-- ── Scans — gate for LadderArb/VolSurface: >=1 week with confirmed edge ──

.print ""
.print "=== SCAN OBSERVATIONS PER DAY (ladder_arb: edge in USDC; vol_surface: ratio) ==="
SELECT
    strategy,
    DATE(ts_ms / 1000, 'unixepoch')      AS day,
    COUNT(*)                             AS n,
    ROUND(AVG(edge_um / 1e6), 4)         AS mean_edge,
    ROUND(MAX(edge_um / 1e6), 4)         AS edge_max
FROM scan_observations
WHERE ts_ms >= :start_ms
GROUP BY strategy, day
ORDER BY strategy, day;

-- ── Binance/Chainlink agreement per |rel_move| bucket ─────────────────────────
-- Lower min_rel_move from 0.001 to 0.0005 (config/strategies/post_close_sweep.toml)
-- ONLY if the agreement stays ~100% in the new bucket (<0.0005-0.001, currently
-- Indeterminate below the current threshold). A high agreement in the
-- >=0.001 bucket (where both sources already decide) is the reference for "how
-- often they disagree" in the current regime; if the new bucket does not
-- match that level, the threshold stays where it is.

.print ""
.print "=== BINANCE/CHAINLINK AGREEMENT PER |REL_MOVE| BUCKET ==="
SELECT
    CASE
        WHEN ABS(rel_move_um) < 500       THEN '< 0.0005'
        WHEN ABS(rel_move_um) < 1000      THEN '0.0005 - 0.001'
        ELSE                                   '>= 0.001'
    END                                                                    AS bucket_rel_move,
    COUNT(*)                                                              AS n,
    SUM(CASE WHEN cl_verdict IS NOT NULL AND cl_verdict = verdict
             THEN 1 ELSE 0 END)                                           AS agreements,
    SUM(CASE WHEN cl_verdict IS NOT NULL THEN 1 ELSE 0 END)               AS with_cl,
    CASE WHEN SUM(CASE WHEN cl_verdict IS NOT NULL THEN 1 ELSE 0 END) > 0
         THEN printf('%.1f%%',
              100.0 * SUM(CASE WHEN cl_verdict IS NOT NULL AND cl_verdict = verdict THEN 1 ELSE 0 END)
              / SUM(CASE WHEN cl_verdict IS NOT NULL THEN 1 ELSE 0 END))
         ELSE '—' END                                                     AS agreement_pct
FROM post_close_observations
WHERE ts_ms >= :start_ms
  AND rel_move_um IS NOT NULL
GROUP BY bucket_rel_move
ORDER BY MIN(ABS(rel_move_um));
