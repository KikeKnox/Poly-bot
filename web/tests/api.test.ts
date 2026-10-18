/**
 * Tests for `web/src/lib/api.ts`, specifically the retry on 401.
 *
 * `api.ts` uses three browser globals (`localStorage`, `window.prompt`,
 * `fetch`). Node does not have them, so mocks are injected into `globalThis`
 * before each test. No external framework: `node:test` + `node:assert`.
 *
 * Coverage (a bad token stayed stored in localStorage and the UI never
 * asked for it again):
 *
 * 1. A 401 clears `hrf_tok`, asks again (window.prompt) and retries the
 *    request ONCE with the new Bearer token.
 * 2. If the retry is also a 401 → ApiError HTTP 401, without a loop
 *    (fetch is called exactly 2 times).
 * 3. With a valid token already stored, nothing is asked and there is a single fetch.
 * 4. A network error (fetch rejects) propagates ApiError `network:` without retrying.
 */

import test from "node:test";
import assert from "node:assert/strict";

import { apiGet, ApiError } from "../src/lib/api.ts";

// ── Browser mocks ──────────────────────────────────────────────────────

type FetchCall = { path: string; headers: Record<string, string> };

interface BrowserMocks {
  storage: Map<string, string>;
  fetchCalls: FetchCall[];
  promptCalls: number;
}

/**
 * Installs fake `localStorage`, `window.prompt` and `fetch`.
 *
 * `statuses` is the queue of HTTP codes that `fetch` will return, in order
 * (the JSON body is always `{ ok: true }`). `prompt` is what the user
 * "types" in the token dialog.
 */
function installBrowserMocks(
  statuses: number[],
  prompt: () => string | null,
): BrowserMocks {
  const storage = new Map<string, string>();
  const fetchCalls: FetchCall[] = [];
  const mocks: BrowserMocks = { storage, fetchCalls, promptCalls: 0 };

  (globalThis as any).localStorage = {
    getItem: (k: string) => storage.get(k) ?? null,
    setItem: (k: string, v: string) => void storage.set(k, v),
    removeItem: (k: string) => void storage.delete(k),
  };
  (globalThis as any).window = {
    prompt: () => {
      mocks.promptCalls += 1;
      return prompt();
    },
  };
  (globalThis as any).fetch = async (path: string, init: any) => {
    fetchCalls.push({ path, headers: init?.headers ?? {} });
    const status = statuses.shift() ?? 500;
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => ({ ok: true }),
    } as Response;
  };
  return mocks;
}

// ── Tests ───────────────────────────────────────────────────────────────────

test("401 clears the stored token, asks again and retries with the new one", async () => {
  const m = installBrowserMocks([401, 200], () => "tok-fresh");
  m.storage.set("hrf_tok", "tok-bad");

  const out = await apiGet<{ ok: boolean }>("/ui/strategies");

  assert.equal(out.ok, true);
  assert.equal(m.fetchCalls.length, 2, "one attempt + one retry");
  assert.equal(
    m.fetchCalls[0].headers["Authorization"],
    "Bearer tok-bad",
    "the first attempt uses the stored token",
  );
  assert.equal(m.promptCalls, 1, "the token is asked for again");
  assert.equal(
    m.fetchCalls[1].headers["Authorization"],
    "Bearer tok-fresh",
    "the retry uses the newly entered token",
  );
  assert.equal(m.storage.get("hrf_tok"), "tok-fresh");
});

test("double 401 → ApiError HTTP 401 without a loop (fetch x2)", async () => {
  const m = installBrowserMocks([401, 401], () => "tok-also-bad");
  m.storage.set("hrf_tok", "tok-bad");

  await assert.rejects(apiGet("/ui/strategies"), (e: unknown) => {
    assert.ok(e instanceof ApiError);
    assert.equal((e as ApiError).status, 401);
    return true;
  });
  assert.equal(m.fetchCalls.length, 2, "a single retry — never a loop");
});

test("valid token already stored: no prompt and no retry", async () => {
  const m = installBrowserMocks([200], () => {
    throw new Error("prompt should not be called");
  });
  m.storage.set("hrf_tok", "tok-good");

  const out = await apiGet<{ ok: boolean }>("/ui/strategies");

  assert.equal(out.ok, true);
  assert.equal(m.fetchCalls.length, 1);
  assert.equal(m.fetchCalls[0].headers["Authorization"], "Bearer tok-good");
  assert.equal(m.promptCalls, 0);
});

test("network error propagates ApiError `network:` without retrying", async () => {
  const storage = new Map<string, string>([["hrf_tok", "tok"]]);
  (globalThis as any).localStorage = {
    getItem: (k: string) => storage.get(k) ?? null,
    setItem: () => {},
    removeItem: () => {},
  };
  let fetchCalls = 0;
  (globalThis as any).fetch = async () => {
    fetchCalls += 1;
    throw new Error("conn refused");
  };

  await assert.rejects(apiGet("/ui/strategies"), (e: unknown) => {
    assert.ok(e instanceof ApiError);
    assert.match((e as ApiError).message, /^network: /);
    return true;
  });
  assert.equal(fetchCalls, 1, "a network error does not trigger the 401 retry");
});

test("pasting the .env line (`HRF_API__TOKEN=abc`) normalizes to the token", async () => {
  const m = installBrowserMocks([401, 200], () => "HRF_API__TOKEN=tok-real");
  m.storage.set("hrf_tok", "tok-bad");

  const out = await apiGet<{ ok: boolean }>("/ui/strategies");

  assert.equal(out.ok, true);
  assert.equal(
    m.fetchCalls[1].headers["Authorization"],
    "Bearer tok-real",
    "the .env line prefix must not leak into the header",
  );
});

test("pasting `Bearer abc` or with quotes also normalizes", async () => {
  for (const entered of ['"Bearer tok-q"', "'tok-q'", "  Bearer   tok-q  "]) {
    const m = installBrowserMocks([401, 200], () => entered);
    m.storage.set("hrf_tok", "tok-bad");
    await apiGet("/ui/strategies");
    assert.equal(
      m.fetchCalls[1].headers["Authorization"],
      "Bearer tok-q",
      `entry ${JSON.stringify(entered)}`,
    );
  }
});

test("a POST retry keeps Content-Type", async () => {
  const { apiPost } = await import("../src/lib/api.ts");
  const m = installBrowserMocks([401, 200], () => "tok-fresh");
  m.storage.set("hrf_tok", "tok-bad");

  await apiPost("/control/strategy", { strategy: "arb", enabled: true });

  assert.equal(m.fetchCalls.length, 2);
  assert.equal(
    m.fetchCalls[1].headers["Content-Type"],
    "application/json",
    "the retry must not lose the POST's Content-Type",
  );
  assert.equal(
    m.fetchCalls[1].headers["Authorization"],
    "Bearer tok-fresh",
  );
});

test("localStorage that throws: the in-memory fallback avoids asking again", async () => {
  // Broken storage (sandbox): every access throws SecurityError.
  (globalThis as any).localStorage = {
    getItem: () => {
      throw new Error("SecurityError");
    },
    setItem: () => {
      throw new Error("SecurityError");
    },
    removeItem: () => {
      throw new Error("SecurityError");
    },
  };
  let promptCalls = 0;
  (globalThis as any).window = {
    prompt: () => {
      promptCalls += 1;
      return "tok-memoria";
    },
  };
  let fetchCalls = 0;
  (globalThis as any).fetch = async (_p: string, init: any) => {
    fetchCalls += 1;
    const auth = init?.headers?.["Authorization"];
    const ok = auth === "Bearer tok-memoria";
    return {
      ok,
      status: ok ? 200 : 401,
      json: async () => ({ ok: true }),
    } as Response;
  };

  const a = await apiGet<{ ok: boolean }>("/ui/strategies");
  const b = await apiGet<{ ok: boolean }>("/ui/data");

  assert.equal(a.ok, true);
  assert.equal(b.ok, true);
  assert.equal(promptCalls, 1, "a single prompt despite broken storage");
  assert.equal(fetchCalls, 3, "1st request 401 + retry + 2nd request OK");
});
