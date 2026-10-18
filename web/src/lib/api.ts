const TOKEN_KEY = "hrf_tok";

// In-memory fallback for when localStorage is unavailable
// (sandboxed contexts may throw SecurityError on access). Lasts for the
// page lifetime — enough to avoid asking for the token on every request.
// localStorage (not sessionStorage): the token survives closing the
// browser — this is a local UI on loopback, asking every time is only
// friction. If the token changes, the 401 from `request` clears it and retries.
let memoryToken: string | null = null;

function safeGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return memoryToken;
  }
}

function safeSet(key: string, value: string): void {
  memoryToken = value;
  try {
    localStorage.setItem(key, value);
  } catch {
    // storage unavailable — the in-memory value already covers this session.
  }
}

function safeRemove(key: string): void {
  memoryToken = null;
  try {
    localStorage.removeItem(key);
  } catch {
    // same here.
  }
}

/**
 * Normalizes what the user pastes into the prompt: tolerates pasting the
 * whole .env line (`HRF_API__TOKEN=abc`), a `Bearer abc` or surrounding
 * quotes: any of those formats would fail the exact match
 * in the backend with a 401 on every request (prompt loop).
 */
function normalizeEntered(entered: string | null): string {
  // Surrounding quotes first — `"Bearer abc"` must still be able to
  // normalize the prefix once the quotes are stripped.
  let t = (entered ?? "")
    .trim()
    .replace(/^["']+|["']+$/g, "")
    .trim();
  t = t.replace(/^HRF_API__TOKEN\s*=\s*/i, "");
  t = t.replace(/^Bearer\s+/i, "");
  return t;
}

function getToken(): string {
  let tok = safeGet(TOKEN_KEY);
  if (tok === null) {
    tok = normalizeEntered(
      window.prompt("Poly-bot access token (Cancel if there is no token):"),
    );
    safeSet(TOKEN_KEY, tok);
  }
  return tok;
}

function clearToken(): void {
  safeRemove(TOKEN_KEY);
}

function authHeaders(): Record<string, string> {
  const tok = getToken();
  return tok ? { Authorization: `Bearer ${tok}` } : {};
}

export class ApiError extends Error {
  public status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.status = status;
  }
}

async function doFetch(path: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(path, init);
  } catch (e) {
    throw new ApiError(`network: ${(e as Error).message}`);
  }
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let res = await doFetch(path, init);
  if (res.status === 401) {
    // Stored token invalid or expired: clear it and retry ONCE
    // — getToken() asks the user again. Without this a bad token
    // stayed pinned in localStorage and the UI never asked again.
    // The merge keeps the previous headers (POST Content-Type) and
    // only replaces Authorization.
    clearToken();
    res = await doFetch(path, {
      ...init,
      headers: { ...(init.headers as Record<string, string>), ...authHeaders() },
    });
  }
  if (!res.ok) throw new ApiError(`HTTP ${res.status}`, res.status);
  return res.json() as Promise<T>;
}

export async function apiGet<T>(path: string): Promise<T> {
  return request(path, { headers: authHeaders() });
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders() };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  return request(path, {
    method: "POST",
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
}
