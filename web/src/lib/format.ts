/** Shared formatters — previously duplicated verbatim in Summary.tsx
 * and Money.tsx (and loose variants in other pages). Single source. */

export function fmtUsd(v: string | number | null | undefined): string {
  const n = typeof v === "string" ? parseFloat(v) : v;
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)} $`;
}

export function pnlTone(n: number): "gain" | "danger" | "muted" {
  return n > 0 ? "gain" : n < 0 ? "danger" : "muted";
}

// secs_to_clear<=0 does not mean settled: the bracket closed but the
// position stays open until orphan.rs resolves it (redemption or
// bail). Showing it as a normal countdown would suggest it already
// closed on its own: 42 of 48 follower positions looked closed
// without ever settling, because the UI did not distinguish this state.
export function fmtClear(secs: number): string {
  if (secs <= 0) return "pending settlement";
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.round(secs / 60)}m`;
  return `${(secs / 3600).toFixed(1)}h`;
}
