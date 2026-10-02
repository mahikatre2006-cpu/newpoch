import type { Frame } from "../api/types";

function side(v: number, neg: string, pos: string): string {
  return `${Math.abs(v * 100).toFixed(1)}% ${v < 0 ? neg : pos}`;
}

/** Whole-frame measurements: palette, where the attention's centre of mass sits, focal points and empty space. */
export default function FramePanel({ frame }: { frame: Frame }) {
  const [bx, by] = frame.balance_offset ?? [0, 0];
  return (
    <section aria-labelledby="frame-h" className="rounded-xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900">
      <h2 id="frame-h" className="text-base font-semibold">Whole frame</h2>
      <div className="mt-3 flex h-6 overflow-hidden rounded-md ring-1 ring-black/10" role="img" aria-label={`Palette: ${frame.palette.map((c) => `${c.name} ${(c.share * 100).toFixed(1)}%`).join(", ")}`}>
        {frame.palette.map((c, i) => <div key={i} style={{ background: c.hex, width: `${c.share * 100}%` }} title={`${c.name} ${(c.share * 100).toFixed(1)}%`} />)}
      </div>
      <ul className="mt-1 flex flex-wrap gap-x-3 text-xs text-stone-600 dark:text-stone-400">
        {frame.palette.map((c, i) => <li key={i}>{c.name} {(c.share * 100).toFixed(1)}%</li>)}
      </ul>
      <dl className="mt-3 grid grid-cols-3 gap-2 text-sm">
        {frame.balance_offset && (
          <div>
            <dt className="text-xs text-stone-500 dark:text-stone-400">Attention centre</dt>
            <dd className="tabular-nums">{side(bx, "left", "right")}, {side(by, "up", "down")}</dd>
          </div>
        )}
        {frame.focal_points !== undefined && (
          <div>
            <dt className="text-xs text-stone-500 dark:text-stone-400">Focal points</dt>
            <dd className="tabular-nums">{frame.focal_points}</dd>
          </div>
        )}
        {frame.negative_space_pct !== undefined && (
          <div>
            <dt className="text-xs text-stone-500 dark:text-stone-400">Empty space</dt>
            <dd className="tabular-nums">{frame.negative_space_pct.toFixed(1)}%</dd>
          </div>
        )}
      </dl>
      {frame.explanations.length > 0 && (
        <ul className="mt-3 list-disc space-y-0.5 pl-4 text-xs text-stone-700 dark:text-stone-300">
          {frame.explanations.map((m, i) => <li key={i}>{m}</li>)}
        </ul>
      )}
    </section>
  );
}
