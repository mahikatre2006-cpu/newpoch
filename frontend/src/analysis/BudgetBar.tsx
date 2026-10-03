export interface BudgetEntry {
  key: string;
  label: string;
  pct: number; // share of attention
  delta: number | null; // change in percentage points since the previous analysis
  color: string;
}

/** The attention budget: one stacked bar, a segment per layer, with how each share moved since the last run. */
export default function BudgetBar({ entries, updating }: { entries: BudgetEntry[]; updating: boolean }) {
  const sorted = [...entries].sort((a, b) => b.pct - a.pct);
  return (
    <section aria-labelledby="budget-h" className={`rounded-xl border border-stone-200 bg-white p-4 transition-opacity dark:border-stone-800 dark:bg-stone-900 ${updating ? "opacity-70" : ""}`}>
      <div className="flex items-baseline justify-between">
        <h2 id="budget-h" className="text-base font-semibold">Attention budget</h2>
        <span role="status" className="text-xs text-stone-500 dark:text-stone-400">{updating ? "Updating…" : "Up to date"}</span>
      </div>
      <div className="mt-2 flex h-7 overflow-hidden rounded-md ring-1 ring-black/10" role="img" aria-label={`Attention per layer: ${sorted.map((e) => `${e.label} ${e.pct.toFixed(1)}%`).join(", ")}`}>
        {sorted.map((e) => (
          <div key={e.key} data-budget-key={e.key} title={`${e.label}: ${e.pct.toFixed(1)}%`} className="flex items-center justify-center overflow-hidden text-[11px] font-semibold text-white transition-[width] duration-500"
            style={{ width: `${e.pct}%`, background: e.color, textShadow: "0 1px 2px rgba(0,0,0,.6)" }}>
            {e.pct >= 8 ? `${e.pct.toFixed(0)}%` : ""}
          </div>
        ))}
      </div>
      <ul className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm sm:grid-cols-3">
        {sorted.map((e) => (
          <li key={e.key} className="flex items-center gap-1.5" data-budget-row={e.key}>
            <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: e.color }} aria-hidden="true" />
            <span className="min-w-0 flex-1 truncate">{e.label}</span>
            <span className="tabular-nums">{e.pct.toFixed(1)}%</span>
            {e.delta !== null && Math.abs(e.delta) >= 0.1 && (
              <span className={`text-xs tabular-nums ${e.delta > 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`} aria-label={`${e.delta > 0 ? "up" : "down"} ${Math.abs(e.delta).toFixed(1)} points`}>
                {e.delta > 0 ? "▲" : "▼"}{Math.abs(e.delta).toFixed(1)}
              </span>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
