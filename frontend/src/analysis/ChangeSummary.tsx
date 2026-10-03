import type { ChangeRow } from "../api/types";

export interface ChangeInfo {
  title: string; // e.g. the fix name
  message: string;
  rows: ChangeRow[];
  orderBefore?: string[];
  orderAfter?: string[];
}

const pct = (v: number | null) => (v === null ? "–" : `${v.toFixed(1)}%`);

/** What a fix (or a comparison) changed, measured by re-analysing: per element before, after and the difference. */
export default function ChangeSummary({ info, onDismiss }: { info: ChangeInfo; onDismiss?: () => void }) {
  const max = Math.max(1, ...info.rows.map((r) => Math.abs(r.delta_pct ?? 0)));
  return (
    <section aria-labelledby="chg-h" className="rounded-xl border border-emerald-300 bg-emerald-50 p-4 dark:border-emerald-800 dark:bg-emerald-950/30">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 id="chg-h" className="text-base font-semibold">{info.title}</h2>
          <p className="mt-0.5 text-sm" role="status">{info.message}</p>
        </div>
        {onDismiss && <button type="button" onClick={onDismiss} className="rounded-md border border-emerald-400 px-2 py-1 text-sm hover:bg-emerald-100 focus-visible:outline-2 focus-visible:outline-amber-500 dark:border-emerald-700 dark:hover:bg-emerald-900/40">Dismiss</button>}
      </div>
      <table className="mt-3 w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-stone-600 dark:text-stone-400">
            <th className="pb-1 font-medium">Element</th>
            <th className="pb-1 text-right font-medium">Before</th>
            <th className="pb-1 text-right font-medium">After</th>
            <th className="w-36 pb-1 pl-3 font-medium">Change</th>
          </tr>
        </thead>
        <tbody>
          {info.rows.map((r, i) => (
            <tr key={`${r.a_id}-${r.b_id}-${i}`} className="border-t border-emerald-200 dark:border-emerald-900">
              <td className="py-1.5 pr-2">
                {r.label}
                {r.status !== "matched" && <span className="ml-1 text-xs text-stone-500">({r.status === "new" ? "new" : "no longer found"})</span>}
              </td>
              <td className="py-1.5 text-right tabular-nums">{pct(r.a_pct)}</td>
              <td className="py-1.5 text-right tabular-nums">{pct(r.b_pct)}</td>
              <td className="py-1.5 pl-3">
                {r.delta_pct === null ? <span className="text-stone-400">–</span> : (
                  <div className="flex items-center gap-2">
                    <div className="h-2 flex-1 rounded-full bg-white/70 dark:bg-stone-800" aria-hidden="true">
                      <div className={`h-2 rounded-full ${r.delta_pct >= 0 ? "bg-emerald-500" : "bg-rose-500"}`} style={{ width: `${Math.min(100, (Math.abs(r.delta_pct) / max) * 100)}%` }} />
                    </div>
                    <span className={`w-12 text-right text-xs tabular-nums ${r.delta_pct >= 0 ? "text-emerald-700 dark:text-emerald-300" : "text-rose-700 dark:text-rose-300"}`}>{r.delta_pct >= 0 ? "+" : ""}{r.delta_pct.toFixed(1)}</span>
                  </div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {info.orderBefore && info.orderAfter && (
        <dl className="mt-3 space-y-1 text-xs text-stone-700 dark:text-stone-300">
          <div><dt className="inline font-medium">Viewing order before: </dt><dd className="inline">{info.orderBefore.join(" → ")}</dd></div>
          <div><dt className="inline font-medium">After: </dt><dd className="inline">{info.orderAfter.join(" → ")}</dd></div>
        </dl>
      )}
    </section>
  );
}
