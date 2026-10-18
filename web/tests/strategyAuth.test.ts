/**
 * Pure-logic tests for `web/src/lib/strategyAuth.ts`.
 *
 * No external framework: only `node:test` (built into Node ≥18) and
 * `node:assert`. No server, no DOM, no browser, no dependency
 * installation. Run with:
 *
 *     node --experimental-strip-types --test web/tests/strategyAuth.test.ts
 *
 * (Node 26 already strips TS by default; the flag is kept to start
 * older versions.)
 *
 * Coverage:
 *
 * 1. `computeModePermission` REPLICATES the backend contract:
 *    - paper + paper_only=true → `ok` (paper_only only blocks in live).
 *    - live + paper_only=false → `ok`.
 *    - live + paper_only=true → `blocked_paper_only`.
 * 2. The `enabled` flag does NOT take part in the decision: `enabled=false`
 *    with OK barrier flags must still return `ok`. This is the key
 *    behavior: the supervisor's "Enable" toggle forces `enabled=true`
 *    before consulting `can_run`, so returning `blocked_disabled` from here
 *    would misrepresent what the backend will accept.
 * 3. If ANY flag is missing, it returns `unknown` and the UI must disable
 *    the "Enable" button (so as not to expose a toggle that might be
 *    rejected without our knowing).
 * 4. `canActivateFromUi`: only `ok` allows pressing.
 * 5. `isCurrentlyRunning`: prefers `s.running`; if missing, falls back to
 *    `!killed`. It neither assumes false (which would break interactions)
 *    nor true (which would hide backend errors).
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  canActivateFromUi,
  computeModePermission,
  flagLabel,
  isCurrentlyRunning,
  permissionExplain,
  permissionLabel,
  permissionVariant,
} from "../src/lib/strategyAuth.ts";

import type { UiStrategyRow } from "../src/lib/types.ts";

const flags = (
  enabled: boolean,
  paper_only: boolean,
) => ({ enabled, paper_only });

test("computeModePermission: paper is never blocked, not even by paper_only", () => {
  assert.equal(computeModePermission(flags(false, false), "paper"), "ok");
  assert.equal(computeModePermission(flags(true,  false), "paper"), "ok");
  assert.equal(computeModePermission(flags(false, true ), "paper"), "ok",
    "paper_only=true does NOT block in paper");
});

test("computeModePermission: live without paper_only is ok", () => {
  assert.equal(computeModePermission(flags(true, false), "live"), "ok");
});

test("computeModePermission: enabled does NOT take part in the decision (fix for the state≠authorization bug)", () => {
  // Core case of the fix: `enabled=false` with OK barrier flags must keep
  // returning `ok` so the "Activate" button can be enabled.
  assert.equal(computeModePermission(flags(false, false), "paper"), "ok",
    "enabled=false in paper: the toggle must be able to activate it");
  assert.equal(computeModePermission(flags(false, false), "live"),  "ok",
    "enabled=false in live with a clean barrier: the supervisor will force enabled=true before consulting the barrier");

  // It must never return `blocked_disabled` for any combination (that branch
  // was the state/authorization confusion this fix removes).
  for (const enabled of [true, false]) {
    for (const po of [true, false]) {
      for (const mode of ["paper", "live"] as const) {
        const result = computeModePermission(flags(enabled, po), mode);
        assert.notEqual(result, "blocked_disabled",
          `enabled=${enabled} po=${po} mode=${mode}: the client must never return blocked_disabled`);
      }
    }
  }
});

test("computeModePermission: live with paper_only ALWAYS blocks by paper_only", () => {
  // This is what the barrier does and what `mode_permission_for_toggle` in the backend says.
  assert.equal(computeModePermission(flags(true,  true), "live"), "blocked_paper_only");
  assert.equal(computeModePermission(flags(false, true), "live"), "blocked_paper_only",
    "paper_only blocks in live even when enabled=false");
});

test("computeModePermission: any missing flag → unknown", () => {
  // If the backend did not expose one of the flags, the UI must not
  // claim anything: a hot toggle could be accepted or rejected without us knowing.
  const cases: Array<{ name: string; f: ReturnType<typeof flags> | Record<string, null | undefined>; expected: string }> = [
    { name: "enabled missing",  f: { enabled: undefined, paper_only: false }, expected: "unknown" },
    { name: "enabled null",     f: { enabled: null,      paper_only: false }, expected: "unknown" },
    { name: "paper_only null",  f: { enabled: true,      paper_only: null },  expected: "unknown" },
    { name: "paper_only undef", f: { enabled: true,      paper_only: undefined }, expected: "unknown" },
    { name: "all null",         f: { enabled: null,      paper_only: null },  expected: "unknown" },
  ];
  for (const c of cases) {
    assert.equal(computeModePermission(c.f as any, "live"), c.expected,
      `${c.name} → unknown (no claim without all flags)`);
  }
});

test("canActivateFromUi: only 'ok' allows pressing Activate", () => {
  assert.equal(canActivateFromUi("ok"),                   true,  "only ok");
  assert.equal(canActivateFromUi("blocked_disabled"),     false);
  assert.equal(canActivateFromUi("blocked_paper_only"),   false);
  assert.equal(canActivateFromUi("unknown"),              false,
    "unknown disables the button: the backend could reject the toggle");
});

test("permissionLabel / permissionVariant / permissionExplain: exhaustive coverage", () => {
  // No throw or fallthrough (TypeScript `string | undefined` due to exhaustiveness).
  const allCases = ["ok", "blocked_disabled", "blocked_paper_only", "unknown"];
  for (const p of allCases) {
    const lab = permissionLabel(p as any);
    const var_ = permissionVariant(p as any);
    const exp  = permissionExplain(p as any, "paper");
    assert.ok(typeof lab === "string" && lab.length > 0, `permissionLabel(${p}) returns a non-empty string`);
    assert.ok(["default", "secondary", "destructive", "outline"].includes(var_), `permissionVariant(${p}) returns a valid variant`);
    assert.ok(typeof exp === "string" && exp.length > 0, `permissionExplain(${p}) returns a non-empty string`);
  }
});

test("flagLabel: ? for missing/null", () => {
  assert.equal(flagLabel(true),  "yes");
  assert.equal(flagLabel(false), "no");
  assert.equal(flagLabel(null),      "?");
  assert.equal(flagLabel(undefined), "?");
});

test("isCurrentlyRunning: prefers the backend's s.running over the !killed fallback", () => {
  // Backend exposes running=true → reflects the truth even when killed=true
  // (an unlikely scenario, but the backend is the source: if it says running, it is running).
  assert.equal(isCurrentlyRunning({ strategy: "x", killed: true,  running: true } as UiStrategyRow),  true,
    "running=true from the backend wins, even when killed=true");
  assert.equal(isCurrentlyRunning({ strategy: "x", killed: false, running: false } as UiStrategyRow), false);

  // running missing → fallback to the previous behavior (!killed).
  assert.equal(isCurrentlyRunning({ strategy: "x", killed: false } as UiStrategyRow), true,
    "fallback: do not expose new data, keep what already worked");
  assert.equal(isCurrentlyRunning({ strategy: "x", killed: true } as UiStrategyRow), false);

  // running=null (the backend exposes it but the value is null) → fallback, same as undefined.
  assert.equal(isCurrentlyRunning({ strategy: "x", killed: false, running: null } as UiStrategyRow), true);
  assert.equal(isCurrentlyRunning({ strategy: "x", killed: true,  running: null } as UiStrategyRow), false);
});

test("isCurrentlyRunning: explicit distinction among the three cases so backend errors are not hidden", () => {
  // If running=true: returns true (what the backend saw).
  // If running=false: returns false (even if !killed → the backend knows
  //   there is no task, perhaps because the global kill switch is active).
  // If running=null/undefined: fallback !killed, without inventing state.
  assert.equal(isCurrentlyRunning({ strategy: "x", killed: true,  running: true  } as UiStrategyRow), true);
  assert.equal(isCurrentlyRunning({ strategy: "x", killed: false, running: false } as UiStrategyRow), false);
  assert.equal(isCurrentlyRunning({ strategy: "x", killed: false } as UiStrategyRow), true,
    "killed=false + no running field = fallback 'assumes running' (backward compat)");
  assert.equal(isCurrentlyRunning({ strategy: "x", killed: true  } as UiStrategyRow), false,
    "killed=true + no running field = fallback 'assumes stopped' (backward compat)");
});

test("cross-check coverage: every flag combination returns what the backend returns", () => {
  // This matrix MUST match `mode_permission_for_toggle` in
  // `src/api/handlers.rs`. If the backend changes its logic, this test
  // breaks → both sides must be updated at the same time.
  for (const enabled of [true, false]) {
    for (const po of [true, false]) {
      for (const mode of ["paper", "live"] as const) {
        const got = computeModePermission(flags(enabled, po), mode);
        // The client IGNORES `enabled` (the toggle's view).
        const expected = mode === "live" && po ? "blocked_paper_only" : "ok";
        assert.equal(got, expected,
          `enabled=${enabled} po=${po} mode=${mode}: client=${got} vs expected=${expected}`);
      }
    }
  }
});
