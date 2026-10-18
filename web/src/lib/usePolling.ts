import { useEffect, useRef, useState } from "react";
import { ApiError } from "./api";

interface PollingState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Generic polling: calls `fetcher` immediately and then every `intervalMs`.
 *  Skips ticks while the user has focus on an <input> (avoids overwriting an edit in progress). */
export function usePolling<T>(fetcher: () => Promise<T>, intervalMs: number): PollingState<T> {
  const [state, setState] = useState<PollingState<T>>({ data: null, error: null, loading: true });
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    let cancelled = false;

    async function tick() {
      if (document.activeElement?.tagName === "INPUT") return;
      try {
        const data = await fetcherRef.current();
        if (!cancelled) setState({ data, error: null, loading: false });
      } catch (e) {
        if (!cancelled) {
          const msg = e instanceof ApiError ? e.message : String(e);
          setState((prev) => ({ ...prev, error: msg, loading: false }));
        }
      }
    }

    tick();
    const id = setInterval(tick, intervalMs);
    return () => { cancelled = true; clearInterval(id); };
  }, [intervalMs]);

  return state;
}
