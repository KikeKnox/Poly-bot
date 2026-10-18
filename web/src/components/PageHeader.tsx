import type { ReactNode } from "react";

/** Page header: nav group + name, as the Sidebar itself announces it — so each
 * screen is anchored to the same vocabulary as the menu instead of starting
 * cold with the cards. */
export function PageHeader({ group, title, action }: { group: string; title: string; action?: ReactNode }) {
  return (
    <div className="mb-1 flex items-baseline justify-between gap-3">
      <div className="flex items-baseline gap-2 font-mono text-[11px] tracking-widest text-muted-foreground">
        <span className="uppercase">{group}</span>
        <span aria-hidden>/</span>
        <span className="text-sm font-semibold uppercase tracking-wide text-foreground">{title}</span>
      </div>
      {action}
    </div>
  );
}
