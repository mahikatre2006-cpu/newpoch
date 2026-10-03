import { Fragment } from "react";
import type { Element } from "../api/types";

const TYPE_NAME: Record<string, string> = { face: "Face", text: "Text", subject: "Subject", background: "Background", layer: "Layer" };
const pct = (v: number) => `${v.toFixed(1)}%`;

interface Props {
  elements: Element[];
  onHover: (id: string | null) => void;
  onAblate?: (id: string) => void; // 'what if this were removed?'
  ablatingId?: string | null;
}

/** Attention breakdown: who gets how much attention, how big they are, how hard they pull for their size, and why. */
export default function Breakdown({ elements, onHover, onAblate, ablatingId }: Props) {
  const rows = [...elements].sort((a, b) => a.predicted_rank - b.predicted_rank);
  const maxDensity = Math.max(2, ...rows.map((r) => r.density ?? 0));
  return (
    <section aria-labelledby="breakdown-h" className="rounded-xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900">
      <h2 id="breakdown-h" className="text-base font-semibold">Attention breakdown</h2>
      <p className="mt-0.5 text-xs text-stone-500 dark:text-stone-400">
        In the order viewers see them. Density is attention share divided by area share: above 1.0 means the element pulls more than its size.
      </p>
      <table className="mt-3 w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-stone-500 dark:text-stone-400">
            <th className="w-8 pb-1 font-medium">#</th>
            <th className="pb-1 font-medium">Element</th>
            <th className="pb-1 text-right font-medium">Attention</th>
            <th className="pb-1 text-right font-medium">Area</th>
            <th className="w-32 pb-1 pl-3 font-medium">Density</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((e) => (
            <Fragment key={e.id}>
              <tr
                tabIndex={0}
                onMouseEnter={() => onHover(e.id)}
                onMouseLeave={() => onHover(null)}
                onFocus={() => onHover(e.id)}
                onBlur={() => onHover(null)}
                className="border-t border-stone-100 hover:bg-stone-50 focus-visible:outline-2 focus-visible:outline-amber-500 dark:border-stone-800 dark:hover:bg-stone-800/60"
              >
                <td className="pt-2 align-top tabular-nums text-stone-500">{e.predicted_rank}</td>
                <td className="pt-2 pr-2 align-top">
                  <div className="font-medium">{e.label}</div>
                  <div className="text-xs text-stone-500 dark:text-stone-400">
                    {TYPE_NAME[e.type] ?? e.type}
                    {e.intent_rank !== null && e.intent_rank !== e.predicted_rank && <span className="ml-1 text-amber-700 dark:text-amber-400">· you want #{e.intent_rank}</span>}
                  </div>
                </td>
                <td className="pt-2 text-right align-top tabular-nums">{pct(e.attention_pct)}</td>
                <td className="pt-2 text-right align-top tabular-nums text-stone-600 dark:text-stone-400">{pct(e.area_pct)}</td>
                <td className="pt-2 pl-3 align-top">
                  {e.density === null ? <span className="text-stone-400">n/a</span> : (
                    <div className="flex items-center gap-2">
                      <div className="h-2 flex-1 rounded-full bg-stone-200 dark:bg-stone-700" role="img" aria-label={`density ${e.density.toFixed(1)}`}>
                        <div className={`h-2 rounded-full ${e.density >= 1 ? "bg-amber-500" : "bg-stone-400"}`} style={{ width: `${Math.min(100, (e.density / maxDensity) * 100)}%` }} />
                      </div>
                      <span className="w-10 text-right text-xs tabular-nums">{e.density.toFixed(1)}×</span>
                    </div>
                  )}
                </td>
              </tr>
              <tr onMouseEnter={() => onHover(e.id)} onMouseLeave={() => onHover(null)}>
                <td />
                <td colSpan={4} className="pb-2 pr-1">
                  {onAblate && e.type !== "background" && (
                    <button type="button" disabled={!!ablatingId} onClick={() => onAblate(e.id)} className="mb-1 rounded border border-stone-300 px-2 py-0.5 text-xs hover:bg-stone-100 focus-visible:outline-2 focus-visible:outline-amber-500 disabled:opacity-50 dark:border-stone-700 dark:hover:bg-stone-800">
                      {ablatingId === e.id ? "Removing…" : `What if ${e.label.length > 22 ? "it" : e.label} were removed?`}
                    </button>
                  )}
                  {e.explanations && e.explanations.length > 0 && (
                    <ul className="list-disc space-y-0.5 pl-4 text-xs text-stone-700 dark:text-stone-300">
                      {e.explanations.map((m, i) => <li key={i}>{m}</li>)}
                    </ul>
                  )}
                </td>
              </tr>
            </Fragment>
          ))}
        </tbody>
      </table>
    </section>
  );
}
