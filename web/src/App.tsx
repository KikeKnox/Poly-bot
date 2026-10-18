import { useState } from "react";
import { Shell } from "@/components/layout/Shell";
import type { PageId } from "@/pages";
import { Summary } from "@/pages/Summary";
import { Risk } from "@/pages/Risk";
import { Market } from "@/pages/Market";
import { Strategies } from "@/pages/Strategies";
import { Tools } from "@/pages/Tools";
import { Money } from "@/pages/Money";
import { Metrics } from "@/pages/Metrics";
import { History } from "@/pages/History";
import { System } from "@/pages/System";

export default function App() {
  const [page, setPage] = useState<PageId>("summary");

  return (
    <Shell current={page} onNavigate={setPage}>
      {page === "summary" && <Summary />}
      {page === "risk" && <Risk />}
      {page === "market" && <Market />}
      {page === "strategies" && <Strategies />}
      {page === "tools" && <Tools />}
      {page === "money" && <Money />}
      {page === "metrics" && <Metrics />}
      {page === "history" && <History />}
      {page === "system" && <System />}
    </Shell>
  );
}
