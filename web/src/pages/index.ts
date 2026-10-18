import {
  LayoutDashboard,
  ShieldAlert,
  Activity,
  Boxes,
  Wrench,
  Wallet,
  BarChart3,
  History,
  Cpu,
  type LucideIcon,
} from "lucide-react";

export type PageId = "summary" | "risk" | "market" | "strategies" | "tools" | "money" | "metrics" | "history" | "system";

export const NAV_GROUPS: { label: string; items: { id: PageId; label: string; icon: LucideIcon }[] }[] = [
  {
    label: "MONITORING",
    items: [
      { id: "summary", label: "Summary", icon: LayoutDashboard },
      { id: "risk", label: "Risk", icon: ShieldAlert },
      { id: "market", label: "Market", icon: Activity },
    ],
  },
  {
    label: "MANAGEMENT",
    items: [
      { id: "strategies", label: "Strategies", icon: Boxes },
      { id: "tools", label: "Tools", icon: Wrench },
    ],
  },
  {
    label: "ANALYSIS",
    items: [
      { id: "money", label: "Money", icon: Wallet },
      { id: "metrics", label: "Metrics", icon: BarChart3 },
      { id: "history", label: "History", icon: History },
    ],
  },
  {
    label: "SYSTEM",
    items: [
      { id: "system", label: "System", icon: Cpu },
    ],
  },
];

/** Group + label of each page, derived from NAV_GROUPS so that PageHeader
 * can never drift out of sync with the menu it comes from. */
export const PAGE_META: Record<PageId, { group: string; label: string }> = Object.fromEntries(
  NAV_GROUPS.flatMap((g) => g.items.map((it) => [it.id, { group: g.label, label: it.label }])),
) as Record<PageId, { group: string; label: string }>;
