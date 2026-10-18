/**
 * Anchors the local reducer of the Enable/Disable toggle.
 *
 * The critical UX behavior lives in `StrategyCard.tsx` as a
 * `useState<ToggleState>` with three variants (`idle | pending | error`).
 * These tests do NOT render React (no component harness is installed:
 * vitest/jsdom would add weight and dependencies to pin down a 3-state
 * logic). Instead, the pure function `step()` reproduces the component's
 * exact state machine, and the tests verify the two invariants the UI
 * promises to the operator:
 *
 *   1. **A rejection (HTTP 4xx/5xx) does NOT roll the state over to "success".**
 *      The operator sees the real state (its `s.running`) and a
 *      `role="alert"` banner with the backend message. The next poll
 *      re-synchronizes.
 *
 *   2. **`pending` prevents a double request.** Any click while
 *      `kind === "pending"` is ignored. Without this guard, a 403 could
 *      accidentally be followed by another POST before the banner
 *      is painted.
 *
 * If a real React component test is wanted later (with jsdom +
 * testing-library), the logic to pin down is this same state machine;
 * only the harness changes.
 */

import test from "node:test";
import assert from "node:assert/strict";

// ── State machine (mirror of StrategyCard.tsx) ────────────────────────

type ToggleState =
  | { kind: "idle" }
  | { kind: "pending"; target: "on" | "off" }
  | { kind: "error"; message: string; attempted: "on" | "off" };

interface UiStrategyRow {
  strategy: string;
  killed: boolean;
  running?: boolean | null;
}

/**
 * One reducer transition. Models the call to `toggleEnabled` in
 * `StrategyCard.tsx`, in order:
 *   - `pending` guard (does not start another transition)
 *   - `perm` guard for Enable (if the UI knows it will fail, it does not
 *     start; disabling is never blocked)
 *   - API call -> success or error
 */
async function step(
  state: ToggleState,
  row: UiStrategyRow,
  perm: "ok" | "blocked_paper_only" | "unknown",
  mode: string,
  apiPost: (path: string, body: unknown) => Promise<{ ok: boolean; status?: number; message?: string }>,
): Promise<ToggleState> {
  // Mirror of the guard `if (toggle.kind === "pending") return;`
  if (state.kind === "pending") return state;

  const isOn = row.running ?? !row.killed;
  const target: "on" | "off" = isOn ? "off" : "on";
  const activateAllowed = perm === "ok";

  // Mirror of the Enable guard (disabling is never blocked).
  if (target === "on" && !activateAllowed) return state;

  try {
    // Mirror of setToggle({kind:"pending", target}) — the pending state
    // only lives inside the try until the result; the component
    // observes it on each render, so we do not need to materialize it here.
    await apiPost("/control/strategy", { strategy: row.strategy, enabled: target === "on" });
    // Mirror of setToggle({kind:"idle"}) + onChanged() on success.
    return { kind: "idle" };
  } catch (e) {
    // Mirror of the catch: ApiError with a message; the state stays as
    // `error` with `attempted = target` (does not roll over to "success").
    const message =
      e && typeof e === "object" && "status" in e
        ? (e as { status?: number }).status === 403
          ? `Rejected by the paper/live barrier: ${(e as { message?: string }).message ?? ""}`
          : `Error ${(e as { status?: number }).status ?? "?"}: ${(e as { message?: string }).message ?? ""}`
        : `Error changing state of ${row.strategy}: ${String(e)}`;
    return { kind: "error", message, attempted: target };
  }
}

// Helper: `apiPost` simulator that returns success or throws an error
// with `status` + `message`, like `ApiError` in `lib/api.ts`.
function fakeApi(
  on: (path: string, body: unknown) => { status: number; message?: string },
): (path: string, body: unknown) => Promise<{ ok: boolean; status?: number; message?: string }> {
  return async (path, body) => {
    const r = on(path, body);
    if (r.status >= 200 && r.status < 300) return { ok: true };
    throw Object.assign(new Error(r.message ?? ""), { status: r.status });
  };
}

// ── Tests ────────────────────────────────────────────────────────────────

test("403 rejection leaves state as 'error', does NOT roll over to 'idle' (phantom success)", async () => {
  // Strategy running + operator presses "Disable" (off) — but the
  // backend returns 403 because the asymmetry is wired wrong.
  // The UI must show `error`, not "success".
  const state: ToggleState = { kind: "idle" };
  const row: UiStrategyRow = { strategy: "arb", killed: false, running: true };
  const next = await step(
    state, row, "ok", "live",
    fakeApi(() => ({ status: 403, message: "LiveNotAuthorized" })),
  );
  assert.equal(next.kind, "error",
    "after a 403 the UI must show `error`, not `idle` (no silent roll-over to success)");
  if (next.kind === "error") {
    assert.equal(next.attempted, "off");
    assert.match(next.message, /Rejected by the paper\/live barrier/);
    assert.match(next.message, /LiveNotAuthorized/);
  }
});

test("pending prevents a double request while the first one is in flight", async () => {
  // Quick double click on "Enable" — the second result must be ignored
  // because `kind === "pending"`. Without this guard, the backend would
  // receive two POSTs with the same body.
  let postCount = 0;
  const api = fakeApi(() => {
    postCount++;
    return { status: 200 };
  });
  // Force `pending` by hand to simulate "the first request is still
  // in flight" (without needing a real setTimeout).
  const pending: ToggleState = { kind: "pending", target: "on" };
  const row: UiStrategyRow = { strategy: "arb", killed: false, running: false };

  const after = await step(pending, row, "ok", "paper", api);
  assert.equal(after.kind, "pending",
    "pending state must be preserved — the guard returns the state without calling the API");
  assert.equal(postCount, 0,
    "the pending guard must prevent a second POST while the first one is in flight");
});

test("disabling is never blocked by permission (hard asymmetry)", async () => {
  // Even if `perm` says "blocked_paper_only", turning off must go through.
  let postedBody: unknown = null;
  const api = fakeApi((_path, body) => {
    postedBody = body;
    return { status: 200 };
  });
  const row: UiStrategyRow = { strategy: "arb", killed: false, running: true };
  const next = await step(
    { kind: "idle" }, row, "blocked_paper_only", "live", api,
  );
  assert.equal(next.kind, "idle", "disabling is never blocked: even if perm is blocked, the off toggle goes through");
  assert.deepEqual(postedBody, { strategy: "arb", enabled: false },
    "the sent body must have enabled=false (off), never enabled=true");
});

test("enabling is blocked when perm !== 'ok' (no request is started)", async () => {
  // Central case: if the UI KNOWS the backend will reject it
  // (`perm` ∈ {blocked_*, unknown}), it does not start the toggle.
  // This avoids a useless round-trip plus a visual error banner that
  // the operator already expected (the mode badge warned about it).
  let postCount = 0;
  const api = fakeApi(() => {
    postCount++;
    return { status: 200 };
  });
  const row: UiStrategyRow = { strategy: "arb", killed: true, running: false };

  for (const perm of ["blocked_paper_only", "unknown"] as const) {
    const next = await step({ kind: "idle" }, row, perm, "live", api);
    assert.equal(next.kind, "idle",
      `enabling with perm=${perm} must not start the toggle (state stays idle, neither pending nor error)`);
  }
  assert.equal(postCount, 0,
    "none of the blocking perm values may generate a POST (the UX guard cuts them off first)");
});

test("enabling with perm='ok' sends a POST and returns to idle on success", async () => {
  let postedBody: unknown = null;
  const api = fakeApi((_path, body) => {
    postedBody = body;
    return { status: 200 };
  });
  const row: UiStrategyRow = { strategy: "arb", killed: true, running: false };
  const next = await step(
    { kind: "idle" }, row, "ok", "live", api,
  );
  assert.equal(next.kind, "idle");
  assert.deepEqual(postedBody, { strategy: "arb", enabled: true });
});

test("error state can recover to idle with a later successful attempt", async () => {
  // If the UI showed an error and the operator retries, the transition
  // error -> pending -> idle must work (the component does
  // setToggle({kind:"pending", target}), which replaces the error state).
  let attempts = 0;
  const api = fakeApi((_path, _body) => {
    attempts++;
    return attempts === 1 ? { status: 500, message: "boom" } : { status: 200 };
  });
  const row: UiStrategyRow = { strategy: "arb", killed: false, running: true };

  const first = await step(
    { kind: "idle" }, row, "ok", "paper", api,
  );
  assert.equal(first.kind, "error");

  const second = await step(first, row, "ok", "paper", api);
  assert.equal(second.kind, "idle", "after a successful attempt following an error, it returns to idle");
});

test("non-403 4xx error shows a generic message with the code", async () => {
  const row: UiStrategyRow = { strategy: "arb", killed: false, running: true };
  const next = await step(
    { kind: "idle" }, row, "ok", "paper",
    fakeApi(() => ({ status: 400, message: "bad json" })),
  );
  assert.equal(next.kind, "error");
  if (next.kind === "error") {
    assert.match(next.message, /Error 400/);
    assert.match(next.message, /bad json/);
  }
});

test("network error (no status) shows the exception message", async () => {
  const row: UiStrategyRow = { strategy: "arb", killed: false, running: true };
  const api = fakeApi(() => { throw new Error("network unreachable"); });
  const next = await step({ kind: "idle" }, row, "ok", "paper", api);
  assert.equal(next.kind, "error");
  if (next.kind === "error") {
    assert.match(next.message, /Error changing state of arb/);
    assert.match(next.message, /network unreachable/);
  }
});
