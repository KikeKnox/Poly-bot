import { NAV_GROUPS } from "@/pages";
import type { PageId } from "@/pages";
import { cn } from "@/lib/utils";

export function Sidebar({ current, onNavigate }: { current: PageId; onNavigate: (p: PageId) => void }) {
  return (
    <nav className="w-52 shrink-0 border-r border-border bg-sidebar p-3">
      <div className="mb-6 flex items-center gap-2 px-1">
        <span className="relative flex h-2 w-2 shrink-0">
          <span className="absolute inline-flex h-full w-full animate-led-pulse rounded-full bg-primary" />
        </span>
        <div>
          <div className="font-mono text-sm font-semibold tracking-tight text-foreground">
            HRF<span className="text-primary">·V4</span>
          </div>
          <div className="font-mono text-[10px] tracking-wide text-muted-foreground">panel de control</div>
        </div>
      </div>
      {NAV_GROUPS.map((group) => (
        <div key={group.label} className="mb-5">
          <div className="mb-1.5 px-2 font-mono text-[10px] font-medium tracking-widest text-muted-foreground/70">
            {group.label}
          </div>
          {group.items.map((item) => {
            const active = current === item.id;
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={cn(
                  "flex w-full items-center gap-2 border-l-2 py-1.5 pl-2.5 pr-2 text-left font-mono text-[13px] transition-colors",
                  active
                    ? "border-l-primary bg-primary/10 text-primary"
                    : "border-l-transparent text-foreground/70 hover:border-l-border hover:bg-white/5 hover:text-foreground"
                )}
              >
                <Icon className="size-3.5 shrink-0" strokeWidth={2} />
                {item.label}
              </button>
            );
          })}
        </div>
      ))}
    </nav>
  );
}
