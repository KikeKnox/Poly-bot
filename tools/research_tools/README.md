# research_tools

Python tools for the bot's research environment (numeric environment verification, paper simulator calibration,
DB inventory). Stdlib only. Tests in `tests/`.

| Script | Purpose | Usage |
|---|---|---|
| `verify_research_environment.py` | Acceptance verifier: gates G1–G9 (`pass`/`pending`/`fail`) over an external run directory | `python3 verify_research_environment.py --run-dir DIR [--repo R] [--model R] [--strict] [--out J]` |
| `calibrate_paper.py` | Measures the paper simulator's fidelity against recorded fills and proposes a calibrated profile | `python3 calibrate_paper.py run [--repo-root .] [--since-ms N --until-ms N] [--replay-strategies arb,near_expiry]` |
| `research_inventory.py` | Inventory of the bot's three effective DBs and safe export | `python3 research_inventory.py inventory_state [--repo-root DIR] [--out manifest.json]` / `export --out-dir DIR` |
| `calibration_analysis.sql` | Calibration queries over `hrf_state_paper.sqlite` | `sqlite3 hrf_state_paper.sqlite ".param set :start_ms 0" < calibration_analysis.sql` |

Tests: `python3 -m unittest discover -s tests` (from this folder).
