"""Selective snapshot of research code -> migration_stage/
(transition phase).

- Regular copies (not hardlinks), sources preserved intact.
- MANIFEST.json with source/dest/SHA256/bytes per file, verifiable.
- No large data or secrets: exclusions by file name + content scan for
  credential patterns; a positive ABORTS the copy of
  that file and is recorded.
- The final product CANNOT depend on the BOT working tree: this
  snapshot exists so the lab can reproduce from versioned sources
  while the integrator extracts dependencies (second wave).
"""

import hashlib
import json
import os
import re
import shutil
import stat
import time

MANIFEST_SCHEMA = "rl_predictor.migration_stage.v1"
MANIFEST_NAME = "MANIFEST.json"
INVENTORY_NAME = "DEPENDENCIES.md"

# Explicit selection (paths relative to the BOT root).
SOURCE_GLOBS = (
    "research/numeric_model/*.py",
    "research/numeric_model/*.md",
    "research/numeric_model/pyproject.toml",
    "research/numeric_model/tests/*.py",
    "research/numeric_model/tests/*.txt",
    "tools/*.py",
    "tools/tests/*.py",
    "src/research/*.rs",
)

# Never copy: caches, data, states, secrets by file name.
EXCLUDE_NAME_PATTERNS = (
    re.compile(r"__pycache__"),
    re.compile(r"\.pyc$"),
    re.compile(r"\.env", re.IGNORECASE),
    re.compile(r"\.jsonl$"),
    re.compile(r"\.sqlite"),
    re.compile(r"\.db$"),
    re.compile(r"(secret|credential|private_key|token)[^/]*$",
               re.IGNORECASE),
)

# Content that never leaves the BOT in a copy.
SECRET_CONTENT_PATTERNS = (
    re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    re.compile(r"POLY_PRIVATE_KEY\s*=\s*\S+"),
    re.compile(r"api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{30,}"),
)


def _excluded(rel):
    base = os.path.basename(rel)
    return any(p.search(base) or p.search(rel)
               for p in EXCLUDE_NAME_PATTERNS)


def _scan_secret(data):
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return "binary, not inspectable"
    hits = [p.pattern for p in SECRET_CONTENT_PATTERNS if p.search(text)]
    return "; ".join(hits) if hits else None


def select_sources(bot_root):
    """List (src_abs, rel) selected, without touching anything yet."""
    import glob as _glob
    out = []
    for pat in SOURCE_GLOBS:
        for path in sorted(_glob.glob(os.path.join(bot_root, pat))):
            rel = os.path.relpath(path, bot_root)
            if os.path.isfile(path) and not _excluded(rel):
                out.append((path, rel))
    return out


def build_snapshot(bot_root, stage_dir):
    """Copies the selection to stage_dir and writes MANIFEST.json + inventory.

    Returns (manifest_dict, warnings[list])."""
    entries, warnings = [], []
    for src, rel in select_sources(bot_root):
        with open(src, "rb") as f:
            data = f.read()
        leak = _scan_secret(data)
        if leak:
            warnings.append(f"EXCLUDED by sensitive pattern: {rel} ({leak})")
            continue
        dest = os.path.join(stage_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # Normal copy preserving content; no hardlinks.
        with open(dest, "wb") as f:
            f.write(data)
        shutil.copymode(src, dest)  # mode metadata, not content
        st = os.lstat(dest)
        if stat.S_ISLNK(st.st_mode):
            raise RuntimeError(f"unexpected symlink dest: {dest}")
        entries.append({
            "src": rel,
            "dest": rel,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        })
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_root": os.path.abspath(bot_root),
        "copy_mode": "normal-bytes (no hardlink)",
        "entries": entries,
        "excluded_secret_like": warnings,
    }
    os.makedirs(stage_dir, exist_ok=True)
    mpath = os.path.join(stage_dir, MANIFEST_NAME)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_inventory(stage_dir, manifest)
    return manifest, warnings


def verify_snapshot(stage_dir):
    """Re-hashes each dest against the manifest. Returns (ok, problems)."""
    mpath = os.path.join(stage_dir, MANIFEST_NAME)
    problems = []
    try:
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)
    except OSError as e:
        return False, [f"unreadable manifest: {e}"]
    for e in manifest.get("entries", []):
        dest = os.path.join(stage_dir, e["dest"])
        try:
            with open(dest, "rb") as f:
                h = hashlib.sha256()
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        except OSError as ex:
            problems.append(f"{e['dest']}: unreadable ({ex})")
            continue
        if h.hexdigest() != e["sha256"]:
            problems.append(f"{e['dest']}: sha256 != manifest")
    return not problems, problems


def write_inventory(stage_dir, manifest):
    """Dependency inventory for the external Rust migration
    (next wave): what each piece copied from the BOT runtime needs."""
    lines = [
        "# Dependency inventory — migration_stage (transition phase)",
        "",
        "Separation rule: the final product does NOT",
        "depend on the BOT working tree. This snapshot is the transition",
        "phase with versioned sources + hashes; the real extraction",
        "is done by the integrator in the second wave.",
        "",
        "## Copied pieces and external dependencies",
        "",
        "| file | external dependency | note |",
        "|---|---|---|",
    ]
    dep_map = {
        "environment.py":
            ("compiled hrf_research_env binary from the BOT "
             "(src/bin/hrf_research_env.rs + src/research/* + pipeline "
             "order/state)", "JSONL client; without the binary there is no env"),
        "dataset.py":
            ("ResearchEvent schema of src/research/contract.rs; tapes "
             "JSONL in research_runs (external)", "predictor features"),
        "dataset_tape.py":
            ("ResearchEvent JSONL", "tape merge/order"),
        "export_sqlite.py":
            ("BOT SQLite schema (src/state.rs)", "offline export"),
        "train.py": ("dataset.py + tapes", "baseline predictor"),
        "evaluate.py": ("model.json + dataset", "eval OOS predictor"),
        "run_env_episodes.py": ("environment.py + driver binary", ""),
        "shadow_eval_t9.py": ("dataset + model", ""),
        "analyze_t8b.py": ("dataset/labels", ""),
        "contract.rs": ("serde/rust_decimal crates of the BOT workspace",
                        "driver wire contract"),
        "events.rs": ("contract.rs + BOT internal types", ""),
        "labels.rs": ("contract.rs", ""),
        "mod.rs": ("contract/events/labels", ""),
        "calibrate_paper.py": ("paper DB + BOT config", "research tool"),
        "research_inventory.py": ("research_runs layout", ""),
        "verify_research_environment.py": ("BOT binaries/config", ""),
    }
    for e in manifest["entries"]:
        base = os.path.basename(e["src"])
        dep, note = dep_map.get(base, ("(support/test)", ""))
        lines.append(f"| {e['src']} | {dep} | {note} |")
    lines += [
        "",
        "## Rules",
        "",
        "- No absolute BOT paths in the final product; those in",
        "  environment.py (DEFAULT_BIN) are for the development host and",
        "  must be replaced by CLI/config when packaging the driver.",
        "- Tapes/model live in research_runs (outside the repo) and are",
        "  referenced by manifest+sha256, never copied here.",
        "- The .rs files are copied as a reference for the wire contract; the crate",
        "  externally re-implements them or imports them as a package, a decision",
        "  for the integrator.",
    ]
    path = os.path.join(stage_dir, INVENTORY_NAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
