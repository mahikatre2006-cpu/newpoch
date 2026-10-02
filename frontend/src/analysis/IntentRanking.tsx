import { useState } from "react";
import type { Element, IntentCheck } from "../api/types";

interface Props {
  elements: Element[]; // the non-background elements
  order: string[]; // element ids, the order the creator wants them seen
  setOrder: (o: string[]) => void;
  check: IntentCheck | null;
  busy: boolean;
  onCheck: () => void;
  onClear: () => void;
}

/** Intent: drag the elements into the order you want viewers to see them, then check it against the prediction. */
export default function IntentRanking({ elements, order, setOrder, check, busy, onCheck, onClear }: Props) {
  const [dragging, setDragging] = useState<number | null>(null);
  const byId = new Map(elements.map((e) => [e.id, e]));
  const mismatch = new Map((check?.mismatches ?? []).map((m) => [m.element_id, m]));

  function move(from: number, to: number) {
    if (to < 0 || to >= order.length || from === to) return;
    const next = [...order];
    next.splice(to, 0, next.splice(from, 1)[0]);
    setOrder(next);
  }

  return (
    <section aria-labelledby="intent-h" className="rounded-xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900">
      <h2 id="intent-h" className="text-base font-semibold">Your intended order</h2>
      <p className="mt-0.5 text-xs text-stone-500 dark:text-stone-400">Drag (or use the arrows) so the element you want seen first is on top, then check it against what viewers will do.</p>
      <ol className="mt-3 space-y-1">
        {order.map((id, i) => {
          const e = byId.get(id);
          if (!e) return null;
          const m = mismatch.get(id);
          return (
            <li
              key={id}
              draggable
              onDragStart={() => setDragging(i)}
              onDragOver={(ev) => ev.preventDefault()}
              onDrop={() => { if (dragging !== null) move(dragging, i); setDragging(null); }}
              onDragEnd={() => setDragging(null)}
              className={`flex items-center gap-2 rounded-lg border px-2 py-1.5 text-sm ${m ? "border-amber-400 bg-amber-50 dark:border-amber-700 dark:bg-amber-950/30" : check ? "border-emerald-300 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/20" : "border-stone-200 dark:border-stone-700"} ${dragging === i ? "opacity-50" : ""}`}
            >
              <span aria-hidden="true" className="cursor-grab select-none text-stone-400">⠿</span>
              <span className="w-5 tabular-nums text-stone-500">{i + 1}</span>
              <span className="min-w-0 flex-1 truncate">{e.label}</span>
              {m && <span className="shrink-0 text-xs text-amber-800 dark:text-amber-300">viewers: #{m.predicted}</span>}
              {check && !m && <span className="shrink-0 text-xs text-emerald-700 dark:text-emerald-300">as predicted</span>}
              <button type="button" aria-label={`Move ${e.label} up`} disabled={i === 0} onClick={() => move(i, i - 1)} className="rounded px-1.5 py-0.5 hover:bg-stone-100 disabled:opacity-30 dark:hover:bg-stone-800">↑</button>
              <button type="button" aria-label={`Move ${e.label} down`} disabled={i === order.length - 1} onClick={() => move(i, i + 1)} className="rounded px-1.5 py-0.5 hover:bg-stone-100 disabled:opacity-30 dark:hover:bg-stone-800">↓</button>
            </li>
          );
        })}
      </ol>
      <div className="mt-3 flex items-center gap-2">
        <button type="button" disabled={busy} onClick={onCheck} className="rounded-lg bg-stone-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-stone-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-500 disabled:opacity-50 dark:bg-amber-400 dark:text-stone-950">
          {busy ? "Checking…" : "Check my order"}
        </button>
        {check && <button type="button" onClick={onClear} className="rounded-lg border border-stone-300 px-3 py-1.5 text-sm hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800">Clear</button>}
        {check && <span role="status" className="text-sm">{check.matches ? "Matches the prediction." : `${check.mismatches.length} element${check.mismatches.length === 1 ? "" : "s"} differ.`}</span>}
      </div>
    </section>
  );
}
