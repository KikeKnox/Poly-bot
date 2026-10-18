# -*- coding: utf-8 -*-
"""Tests for tools/research_tools/research_inventory.py.

Fixture: a temporary tree with a "repo" (config/general.toml + .env) and an
external runtime/ directory, as in a real deployment, where HRF_DB_PATH and
HRF_ANALYTICS__DB_PATH point outside the tree. The DBs are created in WAL
mode (wal_autocheckpoint=0 and a live writer connection) so the test shows
that inventory/export reads the WAL without modifying it or
checkpointing the original.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import research_inventory as ri  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_fingerprint(d: Path) -> dict:
    """Name → sha256 of each DATA file in the directory (non-recursive).

The `-shm` (shared wal-index) is excluded on purpose: it is not DB data and
its content can change when a WAL DB is read, even if read-only. The
invariant that matters is that the main DB + -wal stay byte-for-byte intact.
    """
    return {
        p.name: (_sha256(p) if not p.name.endswith("-shm") else "<shm>")
        for p in sorted(d.iterdir())
        if p.is_file()
    }


def _db_wal(path: Path) -> sqlite3.Connection:
    """Create a DB in WAL mode and return the OPEN writer connection, so the
    -wal/-shm persist for the whole test (live DB)."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    # No autocheckpoint: the pages stay ONLY in the -wal; copying the main
    # DB file would lose them.
    conn.execute("PRAGMA wal_autocheckpoint=0")
    return conn


def _create_state_db(path: Path) -> sqlite3.Connection:
    conn = _db_wal(path)
    conn.executescript(
        """
        CREATE TABLE fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            market_cid TEXT NOT NULL,
            token_id TEXT NOT NULL,
            side TEXT NOT NULL,
            price_um INTEGER NOT NULL,
            size_um INTEGER NOT NULL,
            order_id TEXT,
            strategy TEXT NOT NULL DEFAULT 'arb'
        );
        CREATE TABLE positions (
            token_id TEXT NOT NULL,
            strategy TEXT NOT NULL DEFAULT '',
            market_cid TEXT NOT NULL,
            size_um INTEGER NOT NULL DEFAULT 0,
            opened_ts_ms INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (token_id, strategy)
        );
        """
    )
    conn.executemany(
        "INSERT INTO fills (ts_ms, market_cid, token_id, side, price_um, size_um) "
        "VALUES (?,?,?,?,?,?)",
        [
            (1_000, "0xaaa", "tok1", "BUY", 500_000, 10_000_000),
            (2_000, "0xaaa", "tok1", "SELL", 600_000, 10_000_000),
            # A value outside the domain must be REPORTED, not silently filtered out
            # with a WHERE clause.
            (3_000, "0xaaa", "tok1", "bogus", 1, 1),
        ],
    )
    conn.execute(
        "INSERT INTO positions (token_id, strategy, market_cid, size_um, opened_ts_ms) "
        "VALUES ('tok1','arb','0xaaa',100,1000)"
    )
    conn.commit()
    return conn


def _create_spot_db(path: Path) -> sqlite3.Connection:
    conn = _db_wal(path)
    conn.executescript(
        """
        CREATE TABLE spot_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            source TEXT NOT NULL,
            price TEXT NOT NULL,
            source_ts_ms INTEGER NOT NULL,
            recv_ts_ms INTEGER NOT NULL,
            avail_ts_ms INTEGER NOT NULL
        );
        CREATE INDEX idx_spot_samples_asset_source ON spot_samples(asset, source);
        CREATE TABLE spot_stream_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            enqueued INTEGER NOT NULL,
            written_count INTEGER NOT NULL,
            deduplicated INTEGER NOT NULL,
            discarded_count INTEGER NOT NULL
        );
        """
    )
    rows = []
    # BTC: 3 samples every ~1s, then a 120s gap, then 2 more.
    base = 1_700_000_000_000
    for i, ts in enumerate([0, 1_000, 2_000, 122_000, 123_000]):
        rows.append(("BTC/USDT", "binance", "60000.1", base + ts, base + ts + 5, base + ts + 5))
    # Exact duplicate of (asset, source, source_ts_ms).
    rows.append(("BTC/USDT", "binance", "60000.1", base + 2_000, base + 2_005, base + 2_005))
    # ETH: short continuous coverage, a different group.
    for ts in [0, 1_000]:
        rows.append(("ETH/USDT", "binance", "3000.5", base + ts, base + ts + 5, base + ts + 5))
    conn.executemany(
        "INSERT INTO spot_samples (asset, source, price, source_ts_ms, recv_ts_ms, avail_ts_ms) "
        "VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.execute(
        "INSERT INTO spot_stream_stats (ts_ms, enqueued, written_count, deduplicated, discarded_count) "
        "VALUES (?,7,7,0,0)",
        (base,),
    )
    conn.commit()
    return conn


class FixtureMixin(unittest.TestCase):
    """Build the temporary tree in setUp and keep the original fingerprints."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="t1_inv_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = self.tmp / "repo"
        (self.repo / "config").mkdir(parents=True)
        self.runtime = self.tmp / "runtime" / "paper"
        self.runtime.mkdir(parents=True)
        self.out_dir = self.tmp / "research_out"  # outside the repo, does not exist yet

        (self.repo / "config" / "general.toml").write_text(
            'mode = "paper"\n'
            'db_path = "hrf_state.sqlite"\n'
            "[analytics]\n"
            "enabled = true\n"
            'db_path = "hrf_market_analytics.sqlite"\n'
            "[spot_stream]\n"
            "enabled = true\n"
            'db_path = "hrf_spot_stream.sqlite"\n',
            encoding="utf-8",
        )
        (self.repo / ".env").write_text(
            "HRF_MODE=paper\n"
            f"HRF_DB_PATH={self.runtime}/hrf_state.sqlite\n"
            f"HRF_ANALYTICS__DB_PATH={self.runtime}/hrf_market_analytics.sqlite\n",
            encoding="utf-8",
        )

        self.state_db = self.runtime / "hrf_state_paper.sqlite"
        self.spot_db = self.repo / "hrf_spot_stream_paper.sqlite"
        self.analytics_db = self.runtime / "hrf_market_analytics_paper.sqlite"  # ABSENT on purpose

        self._conns = [
            _create_state_db(self.state_db),
            _create_spot_db(self.spot_db),
        ]
        self.addCleanup(self._close)
        # Fingerprints of ALL data files before any inventory call.
        self.runtime_fingerprint = _dir_fingerprint(self.runtime)
        self.repo_fingerprint = _dir_fingerprint(self.repo)

    def _close(self):
        for c in self._conns:
            c.close()

    def assert_originals_intact(self):
        """Not a single byte changed, and no data files added or removed next to the DBs."""
        self.assertEqual(self.runtime_fingerprint, _dir_fingerprint(self.runtime))
        self.assertEqual(self.repo_fingerprint, _dir_fingerprint(self.repo))


class TestPathResolution(FixtureMixin):
    def test_resolve_db_path_replica_rust(self):
        # Mirror of StateStore::resolve_path (src/state.rs:323).
        self.assertEqual(
            ri.resolve_db_path("hrf_state.sqlite", "paper"),
            str(Path(".") / "hrf_state_paper.sqlite"),
        )
        self.assertEqual(
            ri.resolve_db_path("hrf_state_paper.sqlite", "live"),
            str(Path(".") / "hrf_state_live.sqlite"),  # no double suffix
        )
        self.assertEqual(
            ri.resolve_db_path("/a/b/x.sqlite", "paper"),
            str(Path("/a/b") / "x_paper.sqlite"),
        )
        self.assertEqual(ri.resolve_db_path(":memory:", "paper"), ":memory:")

    def test_effective_sources_priority_and_suffixes(self):
        sources = ri.effective_sources(
            repo_root=self.repo,
            config_path=self.repo / "config" / "general.toml",
            env_path=self.repo / ".env",
            cli_overrides={},
        )
        self.assertEqual(sources["state"]["mode"], "paper")
        # the HRF_DB_PATH env var wins over the db_path in the TOML.
        self.assertEqual(sources["state"]["resolved_path"], str(self.state_db))
        self.assertEqual(sources["analytics"]["resolved_path"], str(self.analytics_db))
        # spot_stream without an env override → TOML path relative to the repo root.
        self.assertEqual(sources["spot_stream"]["resolved_path"], str(self.spot_db))
        # Documented origin (traceability).
        self.assertEqual(sources["state"]["origin"], "env:HRF_DB_PATH")
        self.assertEqual(sources["spot_stream"]["origin"], "toml:spot_stream.db_path")

    def test_cli_override_wins_over_env(self):
        sources = ri.effective_sources(
            repo_root=self.repo,
            config_path=self.repo / "config" / "general.toml",
            env_path=self.repo / ".env",
            cli_overrides={"state": "/other/path/state_paper.sqlite"},
        )
        self.assertEqual(
            sources["state"]["resolved_path"],
            str(Path("/other/path/state_paper.sqlite")),
        )
        self.assertEqual(sources["state"]["origin"], "cli")

    def test_process_environment_wins_over_env_file(self):
        # The binary reads HRF_* from os.environ (figment Env::prefixed) — an
        # HRF_DB_PATH exported in the shell diverges from .env and must win.
        with mock.patch.dict(
            os.environ,
            {"HRF_DB_PATH": str(self.tmp / "proc" / "state.sqlite"),
             "HRF_MODE": "live"},
        ):
            sources = ri.effective_sources(
                repo_root=self.repo,
                config_path=self.repo / "config" / "general.toml",
                env_path=self.repo / ".env",
                cli_overrides={},
            )
        self.assertEqual(
            sources["state"]["resolved_path"],
            str(self.tmp / "proc" / "state_live.sqlite"),
        )
        self.assertEqual(sources["state"]["mode"], "live")
        self.assertEqual(sources["state"]["origin"], "env:HRF_DB_PATH")


class TestInventory(FixtureMixin):
    def _inventory(self):
        return ri.take_inventory(
            ri.effective_sources(
                repo_root=self.repo,
                config_path=self.repo / "config" / "general.toml",
                env_path=self.repo / ".env",
                cli_overrides={},
            ),
            gap_threshold_ms=10_000,
        )

    def test_absent_reported_without_creating_it(self):
        inv = self._inventory()
        a = inv["sources"]["analytics"]
        self.assertFalse(a["exists"])
        self.assertEqual(a["resolved_path"], str(self.analytics_db))
        self.assertFalse(self.analytics_db.exists(), "inventory created the absent DB")

    def test_does_not_modify_originals(self):
        self._inventory()
        self.assert_originals_intact()

    def test_spot_coverage_gaps_and_duplicates(self):
        inv = self._inventory()
        spot = inv["sources"]["spot_stream"]
        self.assertTrue(spot["exists"])
        table = spot["tables"]["spot_samples"]
        self.assertEqual(table["rows"], 8)
        # Per group: BTC has 6 rows with a 120s gap; ETH has 2 rows.
        groups = {tuple(g["group"]): g for g in table["by_group"]}
        btc = groups[("BTC/USDT", "binance")]
        self.assertEqual(btc["rows"], 6)
        self.assertGreaterEqual(btc["gaps"], 1)
        self.assertEqual(btc["gap_max_ms"], 120_000)
        self.assertEqual(btc["ts_duplicate"], 1)
        eth = groups[("ETH/USDT", "binance")]
        self.assertEqual(eth["rows"], 2)
        self.assertEqual(eth["gaps"], 0)

    def test_domain_violations_reported(self):
        # startup-scan-decode-everything: 'bogus' in fills.side comes out as
        # explicit anomaly, not silently discarded.
        inv = self._inventory()
        fills = inv["sources"]["state"]["tables"]["fills"]
        self.assertEqual(fills["rows"], 3)
        self.assertIn("side", fills["domain_violations"])
        self.assertEqual(fills["domain_violations"]["side"]["bogus"], 1)

    def test_table_without_spec_only_counts_and_columns(self):
        inv = self._inventory()
        stats = inv["sources"]["spot_stream"]["tables"]["spot_stream_stats"]
        self.assertEqual(stats["rows"], 1)
        self.assertIn("ts_ms", stats["columns"])


class TestExport(FixtureMixin):
    def test_consistent_snapshot_includes_wal_only_data(self):
        # The state rows live ONLY in the -wal (autocheckpoint=0): a plain file
        # copy would lose them; the backup API would not.
        wal = self.state_db.with_suffix(self.state_db.suffix + "-wal")
        self.assertTrue(wal.exists() and wal.stat().st_size > 0)
        destination = self.out_dir / "state_snapshot.sqlite"
        ri.export_snapshot(self.state_db, destination)
        snap = sqlite3.connect(destination)
        self.assertEqual(snap.execute("SELECT COUNT(*) FROM fills").fetchone()[0], 3)
        snap.close()
        self.assert_originals_intact()

    def test_full_export_generates_manifest(self):
        res = ri.export_cmd(
            ri.effective_sources(
                repo_root=self.repo,
                config_path=self.repo / "config" / "general.toml",
                env_path=self.repo / ".env",
                cli_overrides={},
            ),
            out_dir=self.out_dir,
            gap_threshold_ms=10_000,
            repo_root=self.repo,
        )
        manifest_path = self.out_dir / "manifest.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(res, manifest)
        # Snapshots of the existing DBs; the missing one is documented.
        self.assertTrue((self.out_dir / "hrf_state_paper.sqlite").is_file())
        self.assertTrue((self.out_dir / "hrf_spot_stream_paper.sqlite").is_file())
        a = manifest["sources"]["analytics"]
        self.assertFalse(a["exists"])
        self.assertNotIn("exported_to", a)
        s = manifest["sources"]["state"]
        self.assertEqual(s["exported_to"], str(self.out_dir / "hrf_state_paper.sqlite"))
        self.assertEqual(s["tables"]["fills"]["rows"], 3)
        # The dataset stays OUTSIDE the repo tree.
        self.assertNotIn(str(self.repo), str(self.out_dir))
        self.assert_originals_intact()

    def _sources(self):
        return ri.effective_sources(
            repo_root=self.repo,
            config_path=self.repo / "config" / "general.toml",
            env_path=self.repo / ".env",
            cli_overrides={},
        )

    def test_export_rejects_outdir_inside_repo(self):
        # The dataset must NEVER be inside the tree: an out_dir under repo_root is
        # rejected BEFORE anything is created or written.
        within = self.repo / "research_out"
        with self.assertRaises(ValueError):
            ri.export_cmd(self._sources(), out_dir=within, repo_root=self.repo)
        self.assertFalse(within.exists())
        self.assert_originals_intact()

    def test_export_rejects_outdir_that_overwrites_live_source(self):
        # out_dir = the source directory → out_dir/orig.name == origin:
        # without this guard, sqlite3.connect(dst) would open the live DB for
        # writing and the backup would use it as the destination.
        with self.assertRaises(ValueError):
            ri.export_cmd(
                self._sources(), out_dir=self.runtime, repo_root=self.repo
            )
        self.assert_originals_intact()

    def test_export_snapshot_rejects_destination_equal_to_origin(self):
        with self.assertRaises(ValueError):
            ri.export_snapshot(self.state_db, self.state_db)
        self.assert_originals_intact()


class TestCli(FixtureMixin):
    def test_exit_code_1_with_absent_source(self):
        # The analytics DB is missing from the fixture → inventory is correct
        # but incomplete: exit 1, not 0.
        rc = ri.main(
            ["inventory", "--repo-root", str(self.repo),
             "--out", str(self.tmp / "inv.json")]
        )
        self.assertEqual(rc, 1)
        self.assertTrue((self.tmp / "inv.json").is_file())

    def test_exit_code_2_with_invalid_out_dir(self):
        rc = ri.main(
            ["export", "--repo-root", str(self.repo),
             "--out-dir", str(self.repo / "within")]
        )
        self.assertEqual(rc, 2)
        self.assertFalse((self.repo / "within").exists())


if __name__ == "__main__":
    unittest.main()
