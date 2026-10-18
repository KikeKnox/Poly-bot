#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Effective inventory and safe export of the bot's historical DBs (T1).

Stdlib only (`sqlite3`, `argparse`, `tomllib`, `json`). Resolves the
EFFECTIVE paths of the bot's three databases the same way the binary does:

- `state`       — `db_path` (general.toml) with `HRF_DB_PATH` override;
                  goes through `StateStore::resolve_path` (src/state.rs):
                  `{stem}_{mode}.sqlite` (paper/live never share a file).
- `analytics`   — `analytics.db_path` with `HRF_ANALYTICS__DB_PATH` override;
                  `analytics_recorder::run` applies the SAME `resolve_path`
                  to it.
- `spot_stream` — `spot_stream.db_path` with `HRF_SPOT_STREAM__DB_PATH` override;
                  `main` applies `resolve_path` to it.

Relative paths are resolved against the repo root (launch.sh does
`cd` to the script's directory before exec'ing the binary).

Guarantees:

- NEVER creates a missing DB: it only reports it with `exists=false`.
- NEVER opens an original in write mode: everything goes through URI `mode=ro` +
  `PRAGMA query_only=ON`.
- Export uses SQLite's backup API (`Connection.backup`) over the
  read-only connection — a consistent snapshot that INCLUDES what lives only in
  the -wal. Never a file copy of a main DB with an associated WAL, never a
  checkpoint nor repair over originals. (`VACUUM INTO` is ruled out:
  SQLite rejects it over read-only connections.)
- The exported dataset goes to an EXTERNAL research directory
  (`--out-dir`), never inside the tree nor over the active DB.

Coverage per table with a known spec: one pass ordered by
(group, ts) — decodes ALL rows before filtering — that emits per group: rows, time
range, gaps (discontinuities > threshold), timestamp duplicates and
domain violations. Bounded with `--max-scan-rows`.

Usage:
    python3 tools/research_tools/research_inventory.py inventory [--repo-root DIR] [--out manifest.json]
    python3 tools/research_tools/research_inventory.py export --out-dir EXTERNAL_DIR [...]
"""

import argparse
import json
import os
import sqlite3
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

VERSION = 1

# Mode suffixes applied by StateStore::resolve_path (src/state.rs).
MODE_SUFFIXES = ("_paper", "_live")

# Environment keys (HRF_ prefix, double underscore for nesting — see the
# header of config/general.toml and .env).
ENV_KEYS = {
    "state": "HRF_DB_PATH",
    "analytics": "HRF_ANALYTICS__DB_PATH",
    "spot_stream": "HRF_SPOT_STREAM__DB_PATH",
}

# config.rs defaults when the key is missing from the TOML.
TOML_KEYS = {
    "state": ("db_path", "hrf_state.sqlite"),
    "analytics": ("analytics", "db_path", "hrf_market_analytics.sqlite"),
    "spot_stream": ("spot_stream", "db_path", "hrf_spot_stream.sqlite"),
}

# Coverage specs per known table:
#   group: columns that identify the series (asset/market/strategy)
#   ts:    epoch-ms time column (or epoch-s where the schema says so)
#   domain: allowed values of tag columns — rows that violate the
#            domain are COUNTED instead of being silently filtered.
SPECS_TABLES = {
    # hrf_market_analytics_*.sqlite — book_snapshots (analytics_recorder.rs).
    # The group is only market_condition_id so the ORDER BY uses the
    # UNIQUE index (market_condition_id, ts_ms) — the slug is a per-group label
    # (1 indexed lookup per group, `label_col`).
    # Note: the recorder dedups by content, so a long gap may
    # just be a static book — gaps are informational.
    "book_snapshots": {
        "group": ["market_condition_id"],
        "ts": "ts_ms",
        "label_col": "slug",
    },
    # hrf_spot_stream_*.sqlite — records only when the spot changes; a long gap
    # = static spot or collector down (cf. spot_stream_stats).
    "spot_samples": {"group": ["asset", "source"], "ts": "source_ts_ms"},
    # hrf_state_*.sqlite — state.rs.
    "fills": {
        "group": ["market_cid"],
        "ts": "ts_ms",
        "domain": {"side": ("BUY", "SELL")},
    },
    "transactions": {
        "group": ["market_cid"],
        "ts": "ts_ms",
        "domain": {"kind": ("BUY", "SELL", "REDEEM")},
    },
    "positions": {"group": ["market_cid"], "ts": "opened_ts_ms"},
    "order_index": {"ts": "ts_ms"},
    "deposits": {"ts": "ts_ms"},
    # first_seen_ts is epoch-SECONDS (not ms) — see the schema in state.rs.
    "markets_observed": {"ts": "first_seen_ts", "ts_unit": "s"},
    "markouts": {
        "group": ["strategy"],
        "ts": "ts_ms",
        "domain": {"side": ("BUY", "SELL")},
    },
    "scan_observations": {"group": ["strategy"], "ts": "ts_ms"},
    "post_close_observations": {
        "group": ["market_cid"],
        "ts": "ts_ms",
        "domain": {"verdict": ("yes", "no", "indeterminate")},
    },
    "gtd_orders": {"group": ["market_cid"], "ts": "ts_ms"},
    "risk_state": {"ts": "updated_ms"},
    "equity_snapshots": {"ts": "ts_ms"},
    "dead_letter_fills": {"ts": "ts_ms", "domain": {"side": ("BUY", "SELL")}},
    "unknown_attempts": {"ts": "ts_ms"},
}

# Default cap of rows read in the per-table coverage scan —
# the real DBs reach ~9 GiB; the scan stays bounded and the report
# marks `truncated=true` when it is cut off.
MAX_SCAN_ROWS_DEFAULT = 2_000_000
# Gap = discontinuity larger than this threshold between consecutive
# timestamps of the same group (1 min by default).
GAP_THRESHOLD_MS_DEFAULT = 60_000
# The N largest gaps per group are listed explicitly.
TOP_GAPS = 5


# ── Resolution of effective paths ────────────────────────────────────────────


def resolve_db_path(raw: str, mode: str) -> str:
    """Mirror of `StateStore::resolve_path` (src/state.rs): applies the
    `{stem}_{mode}.sqlite` suffix and strips a mode suffix already present
    so as not to produce `_paper_paper`."""
    if raw == ":memory:":
        return raw
    p = Path(raw)
    stem = p.stem if p.stem else "hrf_state"
    for suf in MODE_SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    parent = str(p.parent) if str(p.parent) else "."
    return str(Path(parent) / f"{stem}_{mode}.sqlite")


def load_env(env_path: Path, process: dict | None = None) -> dict:
    """Minimal .env parser: only extracts the HRF_* keys that resolve the
    mode/DB paths. Never reads nor exports secrets.

    The PROCESS ENVIRONMENT is overlaid AFTER the .env file: the binary
    reads these keys with figment `Env::prefixed("HRF_")`, which looks at
    `os.environ` — an `HRF_DB_PATH` exported in the shell diverges from .env and
    wins, so the effective resolution must reflect it the same way."""
    keys = {"HRF_MODE", *ENV_KEYS.values()}
    out = {}
    try:
        for line in Path(env_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            if k in keys:
                out[k] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    environment = os.environ if process is None else process
    for k in keys:
        if environment.get(k):
            out[k] = environment[k]
    return out


def load_toml(config_path: Path) -> dict:
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}


def _raw_and_origin(name_str: str, config: dict, env: dict, cli_overrides: dict):
    """Priority: CLI > env HRF_* > TOML > config.rs default."""
    if name_str in cli_overrides:
        return cli_overrides[name_str], "cli"
    ek = ENV_KEYS[name_str]
    if env.get(ek):
        return env[ek], f"env:{ek}"
    spec = TOML_KEYS[name_str]
    val = config
    for k in spec[:-1]:
        val = val.get(k, {}) if isinstance(val, dict) else {}
    raw = val if isinstance(val, str) else None
    if raw:
        toml_key = ".".join(spec[:-1])
        return raw, f"toml:{toml_key}"
    return spec[-1], "default"


def effective_sources(
    repo_root: Path,
    config_path: Path,
    env_path: Path,
    cli_overrides: dict,
    extra: dict | None = None,
) -> dict:
    """Resolves the effective paths of the bot's DBs.

    Returns {name_str: {configured, resolved_path, origin, mode}}. Relative
    paths are anchored to `repo_root` (the bot's effective CWD via launch.sh).
    `extra` allows inventorying additional historical/external DBs by
    LITERAL path (no mode suffix): {name_str: path}.
    """
    repo_root = Path(repo_root)
    config = load_toml(Path(config_path))
    env = load_env(Path(env_path))
    mode = cli_overrides.get("mode") or env.get("HRF_MODE") or config.get("mode") or "paper"

    sources = {"mode": mode}
    for name_str in ("state", "analytics", "spot_stream"):
        raw, origin = _raw_and_origin(name_str, config, env, cli_overrides)
        resolved_path = Path(resolve_db_path(raw, mode))
        if not resolved_path.is_absolute():
            resolved_path = repo_root / resolved_path
        sources[name_str] = {
            "configured": raw,
            "resolved_path": str(resolved_path),
            "origin": origin,
            "mode": mode,
        }
    for name_str, path in (extra or {}).items():
        p = Path(path)
        if not p.is_absolute():
            p = repo_root / p
        sources[name_str] = {
            "configured": path,
            "resolved_path": str(p),
            "origin": "cli:extra",
            "mode": mode,
        }
    return sources


# ── Read-only opening ────────────────────────────────────────────────────────


def open_readonly(path: Path) -> sqlite3.Connection:
    """Opens the DB with URI `mode=ro` + `PRAGMA query_only=ON`. Never creates
    the file (mode=ro fails if it does not exist) nor writes to it, to the -wal
    or to the -shm. SQLite ≥3.22 reads WAL DBs from read-only connections."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


# ── Inventory of one DB ─────────────────────────────────────────────────────


def _iso(ts: int | None, unit: str = "ms") -> str | None:
    if ts is None:
        return None
    try:
        seg = ts / 1000 if unit == "ms" else ts
        return datetime.fromtimestamp(seg, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _columns(conn: sqlite3.Connection, table: str) -> list:
    return [r[0] for r in conn.execute(f"SELECT name FROM pragma_table_info('{table}')")]


def _domain_violations(conn: sqlite3.Connection, table: str, domain: dict) -> dict:
    """Counts rows whose `col` value falls OUTSIDE the known domain —
    an explicit anomaly (corruption/schema drift), never silently filtered
    by a WHERE in the main scan."""
    out = {}
    for col, allowed in domain.items():
        ph = ",".join("?" for _ in allowed)
        try:
            rows = conn.execute(
                f"SELECT COALESCE(CAST({col} AS TEXT), '<NULL>'), COUNT(*) "
                f"FROM {table} WHERE {col} IS NULL OR {col} NOT IN ({ph}) "
                f"GROUP BY {col}",
                allowed,
            ).fetchall()
        except sqlite3.Error as e:
            out[col] = {"error": str(e)}
            continue
        if rows:
            out[col] = {str(v): n for v, n in rows}
    return out


def _table_coverage(
    conn: sqlite3.Connection,
    table: str,
    spec: dict,
    gap_threshold_ms: int,
    max_scan_rows: int,
) -> dict:
    """One `ORDER BY (group, ts)` pass — ALL rows, no WHERE over
    the tag column — aggregating per group: time range, gaps over
    the threshold, exact timestamp duplicates and the largest gaps.
    Bounded by `max_scan_rows` (marks `truncated` if cut off)."""
    gcols = list(spec.get("group", []))
    ts = spec["ts"]
    unit = spec.get("ts_unit", "ms")
    # The threshold arrives in ms; if the ts column is in seconds it is converted.
    threshold = gap_threshold_ms if unit == "ms" else gap_threshold_ms / 1000
    cols = gcols + [ts]
    sel = ", ".join(cols)
    sql = f"SELECT {sel} FROM {table} ORDER BY {sel} LIMIT ?"
    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    groups: dict = {}
    order: list = []  # preserves order of appearance (alphabetical by the ORDER BY)
    prev_key = prev_ts = None
    rows_read = 0
    truncated = False
    for row_entry in conn.execute(sql, (max_scan_rows + 1,)):
        if rows_read >= max_scan_rows:
            truncated = True
            break
        rows_read += 1
        key = tuple(row_entry[:-1]) if gcols else ()
        ts_v = row_entry[-1]
        g = groups.get(key)
        if g is None:
            g = {
                "group": list(key),
                "rows": 0,
                "ts_min": None,
                "ts_max": None,
                "gaps": 0,
                "gap_max_ms": 0,
                "top_gaps": [],
                "ts_duplicate": 0,
            }
            groups[key] = g
            order.append(key)
            prev_ts = None
        g["rows"] += 1
        if ts_v is None:
            prev_ts = None
            continue
        if g["ts_min"] is None or ts_v < g["ts_min"]:
            g["ts_min"] = ts_v
        if g["ts_max"] is None or ts_v > g["ts_max"]:
            g["ts_max"] = ts_v
        if prev_ts is not None:
            if ts_v == prev_ts:
                g["ts_duplicate"] += 1
            elif ts_v - prev_ts > threshold:
                g["gaps"] += 1
                # Gaps are always reported in ms even if the ts column
                # is in seconds (e.g. markets_observed.first_seen_ts).
                gap = (ts_v - prev_ts) * (1000 if unit == "s" else 1)
                if gap > g["gap_max_ms"]:
                    g["gap_max_ms"] = gap
                top = g["top_gaps"]
                top.append({"from_ts": prev_ts, "until_ts": ts_v, "gap_ms": gap})
                top.sort(key=lambda h: h["gap_ms"], reverse=True)
                del top[TOP_GAPS:]
        prev_ts = ts_v

    # Label per group (e.g. market slug): 1 indexed lookup per
    # group — kept out of the ORDER BY so as not to break the index usage.
    label_name = spec.get("label_col")
    if label_name and gcols:
        g0 = gcols[0]
        for key in order:
            row_entry = conn.execute(
                f"SELECT {label_name} FROM {table} WHERE {g0} = ? AND {label_name} IS NOT NULL LIMIT 1",
                (key[0],),
            ).fetchone()
            groups[key]["label"] = row_entry[0] if row_entry else None

    by_group = []
    for key in order:
        g = groups[key]
        g["ts_min_iso"] = _iso(g["ts_min"], unit)
        g["ts_max_iso"] = _iso(g["ts_max"], unit)
        by_group.append(g)

    ts_min, ts_max = conn.execute(
        f"SELECT MIN({ts}), MAX({ts}) FROM {table}"
    ).fetchone()

    return {
        "rows": total,
        "columns": _columns(conn, table),
        "ts": {"column": ts, "unit": unit, "min": ts_min, "max": ts_max,
               "min_iso": _iso(ts_min, unit), "max_iso": _iso(ts_max, unit)},
        "group_by": gcols,
        "n_groups": len(by_group),
        "by_group": by_group,
        "gap_threshold_ms": gap_threshold_ms,
        "scanned_rows": rows_read,
        "truncated": truncated,
        "domain_violations": _domain_violations(conn, table, spec.get("domain", {})),
    }


def inventory_db(
    path: Path,
    gap_threshold_ms: int = GAP_THRESHOLD_MS_DEFAULT,
    max_scan_rows: int = MAX_SCAN_ROWS_DEFAULT,
) -> dict:
    """Read-only inventory of ONE DB: contiguous files (-wal/-shm), journal
    mode, and per table: columns, rows and coverage if there is a known spec."""
    path = Path(path)
    entry = {"resolved_path": str(path), "exists": path.is_file()}
    if not entry["exists"]:
        return entry

    st = path.stat()
    entry["bytes"] = st.st_size
    entry["mtime_iso"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    for suf, key in (("-wal", "wal_bytes"), ("-shm", "shm_bytes")):
        side = Path(str(path) + suf)
        entry[key] = side.stat().st_size if side.exists() else None

    try:
        conn = open_readonly(path)
    except sqlite3.Error as e:
        entry["open_error"] = str(e)
        return entry
    try:
        entry["journal_mode"] = conn.execute("PRAGMA journal_mode").fetchone()[0]
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        out_tables = {}
        for t in tables:
            spec = SPECS_TABLES.get(t)
            if spec is None:
                out_tables[t] = {
                    "rows": conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0],
                    "columns": _columns(conn, t),
                }
            else:
                out_tables[t] = _table_coverage(
                    conn, t, spec, gap_threshold_ms, max_scan_rows
                )
        entry["tables"] = out_tables
    except sqlite3.Error as e:
        entry["inventory_error"] = str(e)
    finally:
        conn.close()
    return entry


def take_inventory(
    sources: dict,
    gap_threshold_ms: int = GAP_THRESHOLD_MS_DEFAULT,
    max_scan_rows: int = MAX_SCAN_ROWS_DEFAULT,
) -> dict:
    """Inventory of all the effective sources. Missing ones are reported
    explicitly (`exists=false`) and are never created."""
    result = {
        "tool": "research_inventory",
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": sources.get("mode"),
        "sources": {},
    }
    for name_str, f in sources.items():
        if name_str == "mode":
            continue
        entry = inventory_db(Path(f["resolved_path"]), gap_threshold_ms, max_scan_rows)
        entry.update(
            {"configured": f["configured"], "origin": f["origin"], "mode": f["mode"]}
        )
        result["sources"][name_str] = entry
    return result


# ── Safe export ──────────────────────────────────────────────────────────────


def _validate_out_dir(out_dir: Path, sources: dict, repo_root: Path | None) -> Path:
    """Guard for "the dataset NEVER inside the tree nor over the active DB":

    - Rejects an `out_dir` that resolves inside `repo_root`.
    - Rejects if `out_dir/<name_str>` collides with the RESOLVED path of
      any source — without this check `sqlite3.connect(dst)` would open
      the live DB in write mode and the backup would take it as the destination.
    Returns `out_dir` already resolved."""
    out = Path(out_dir).resolve()
    if repo_root is not None:
        root = Path(repo_root).resolve()
        if out == root or root in out.parents:
            raise ValueError(
                f"out_dir {out} is inside the repo tree ({root}) — "
                "the dataset must go to an EXTERNAL research directory"
            )
    for name_str, f in sources.items():
        if name_str == "mode":
            continue
        orig = Path(f["resolved_path"])
        if orig.name == ":memory:":
            continue
        dst = (out / orig.name).resolve()
        if dst == orig.resolve():
            raise ValueError(
                f"out_dir {out} collides with source '{name_str}' "
                f"({orig}) — the snapshot would overwrite the active DB"
            )
    return out


def export_snapshot(src: Path, dst: Path) -> None:
    """Consistent snapshot of a (possibly) live WAL DB via the backup
    API over a read-only connection: includes what exists only in the -wal and
    does not touch the original (no checkpoint, no repair, no writes)."""
    src = Path(src)
    dst = Path(dst)
    # Defense in depth: the destination can never be the origin itself.
    if dst.resolve() == src.resolve():
        raise ValueError(f"export_snapshot: destination == origin ({src})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    origin = open_readonly(src)
    try:
        destination = sqlite3.connect(dst)
        try:
            origin.backup(destination)
        finally:
            destination.close()
    finally:
        origin.close()


def export_cmd(
    sources: dict,
    out_dir: Path,
    gap_threshold_ms: int = GAP_THRESHOLD_MS_DEFAULT,
    max_scan_rows: int = MAX_SCAN_ROWS_DEFAULT,
    repo_root: Path | None = None,
) -> dict:
    """Exports each existing DB to `out_dir/<file_name>` via a consistent
    snapshot, inventories the exported SNAPSHOT (the coverage documents what
    actually ended up in the dataset) and writes `manifest.json`.

    `out_dir` is validated before touching anything (`_validate_out_dir`): outside
    the repo tree and without collision with any live source."""
    out_dir = _validate_out_dir(out_dir, sources, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tool": "research_inventory",
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": sources.get("mode"),
        "out_dir": str(out_dir),
        "sources": {},
    }
    dst_names: dict = {}  # basename → source name, to avoid overwriting snapshots
    for name_str, f in sources.items():
        if name_str == "mode":
            continue
        orig = Path(f["resolved_path"])
        entry = {
            "configured": f["configured"],
            "resolved_path": str(orig),
            "origin": f["origin"],
            "mode": f["mode"],
            "exists": orig.is_file(),
        }
        if entry["exists"]:
            dst = out_dir / orig.name
            if dst.name in dst_names:
                # Two sources with the same basename: disambiguate by name.
                dst = out_dir / f"{name_str}__{orig.name}"
            dst_names[dst.name] = name_str
            export_snapshot(orig, dst)
            entry["exported_to"] = str(dst)
            snap = inventory_db(dst, gap_threshold_ms, max_scan_rows)
            snap.pop("resolved_path", None)
            entry.update(snap)
        manifest["sources"][name_str] = entry

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_extra(kv_pairs: list | None) -> dict:
    """`--extra-db name_str=path`, repeatable — historical/external DBs at a
    literal path (no mode suffix)."""
    out = {}
    for pair in kv_pairs or []:
        name_str, _, path = pair.partition("=")
        if not name_str or not path:
            raise SystemExit(f"--extra-db malformed (expected name=path): {pair!r}")
        out[name_str.strip()] = path.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Effective inventory and safe export of the bot's DBs (read-only).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--repo-root", default=".",
                        help="Repo root (the bot's effective CWD; relative paths are anchored here).")
        sp.add_argument("--config", default=None,
                        help="Path to general.toml (default: <repo-root>/config/general.toml).")
        sp.add_argument("--env", default=None,
                        help="Path to .env (default: <repo-root>/.env).")
        sp.add_argument("--mode", default=None, choices=["paper", "live"],
                        help="Mode for the _paper/_live suffix (default: HRF_MODE > toml > paper).")
        sp.add_argument("--state-db", default=None, help="Override of the state DB.")
        sp.add_argument("--analytics-db", default=None, help="Override of the analytics DB.")
        sp.add_argument("--spot-db", default=None, help="Override of the spot_stream DB.")
        sp.add_argument("--extra-db", action="append", default=None, metavar="NAME=PATH",
                        help="Additional historical/external DB at a literal path (repeatable).")
        sp.add_argument("--gap-threshold-ms", type=int, default=GAP_THRESHOLD_MS_DEFAULT,
                        help="Temporal discontinuity threshold per group (ms).")
        sp.add_argument("--max-scan-rows", type=int, default=MAX_SCAN_ROWS_DEFAULT,
                        help="Cap of rows scanned per table (bounds the coverage scan).")

    spi = sub.add_parser("inventory", help="Read-only inventory → JSON on stdout or --out.")
    common(spi)
    spi.add_argument("--out", default=None, help="Write the inventory JSON to this path.")

    spe = sub.add_parser("export", help="Consistent snapshots + manifest.json in an EXTERNAL dir.")
    common(spe)
    spe.add_argument("--out-dir", required=True,
                     help="EXTERNAL research directory (never inside the tree nor over the active DB).")
    return p


def _sources_from_args(args) -> dict:
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config) if args.config else repo_root / "config" / "general.toml"
    env_path = Path(args.env) if args.env else repo_root / ".env"
    overrides = {"mode": args.mode}
    for key, val in (("state", args.state_db), ("analytics", args.analytics_db),
                       ("spot_stream", args.spot_db)):
        if val:
            overrides[key] = val
    return effective_sources(
        repo_root=repo_root,
        config_path=config_path,
        env_path=env_path,
        cli_overrides=overrides,
        extra=_parse_extra(args.extra_db),
    )


def _exit_code(result: dict) -> int:
    """1 if any source is missing or failed to open/inventory —
    an inventory with gaps must not look like a clean success."""
    for f in result["sources"].values():
        if not f.get("exists") or f.get("open_error") or f.get("inventory_error"):
            return 1
    return 0


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    sources = _sources_from_args(args)
    try:
        if args.cmd == "inventory":
            res = take_inventory(sources, args.gap_threshold_ms, args.max_scan_rows)
            text = json.dumps(res, indent=2, ensure_ascii=False)
            if args.out:
                Path(args.out).write_text(text + "\n", encoding="utf-8")
            else:
                print(text)
        else:  # export
            res = export_cmd(
                sources,
                Path(args.out_dir),
                args.gap_threshold_ms,
                args.max_scan_rows,
                repo_root=repo_root,
            )
            n_ok = sum(1 for f in res["sources"].values() if f.get("exported_to"))
            n_out = sum(1 for f in res["sources"].values() if not f.get("exists"))
            print(
                f"export: {n_ok} DBs exported, {n_out} missing → "
                f"{res['out_dir']}/manifest.json",
                file=sys.stderr,
            )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return _exit_code(res)


if __name__ == "__main__":
    sys.exit(main())
