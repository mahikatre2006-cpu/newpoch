import type { AblationResult, AnalysisResult } from "../api/types";

interface Props {
  result: AnalysisResult; // the analysis the element was removed from
  ablation: AblationResult;
  onClose: () => void;
}

const Frame = ({ image, heatmap, caption }: { image: string; heatmap: string | null; caption: string }) => (
  <figure className="min-w-0">
    <div className="relative aspect-video overflow-hidden rounded-lg bg-stone-900 ring-1 ring-black/10">
      <img src={`data:image/jpeg;base64,${image}`} alt="" className="absolute inset-0 h-full w-full object-fill" />
      {heatmap && <img src={`data:image/png;base64,${heatmap}`} alt="" className="absolute inset-0 h-full w-full object-fill opacity-70" />}
    </div>
    <figcaption className="mt-1 text-xs text-stone-600 dark:text-stone-400">{caption}</figcaption>
  </figure>
);

/** What happens to the attention when one element is removed: the edited picture, its heatmap and where the share went. */
export default function AblationPanel({ result, ablation, onClose }: Props) {
  const maxDelta = Math.max(1, ...ablation.flow.map((f) => Math.abs(f.delta_pct)));
  return (
    <section aria-labelledby="abl-h" className="rounded-xl border border-amber-300 bg-white p-4 dark:border-amber-700 dark:bg-stone-900">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 id="abl-h" className="text-base font-semibold">Without {ablation.removed.label}</h2>
          <p className="mt-0.5 text-sm text-stone-700 dark:text-stone-300">{ablation.message}</p>
        </div>
        <button type="button" onClick={onClose} className="rounded-md border border-stone-300 px-2 py-1 text-sm hover:bg-stone-100 focus-visible:outline-2 focus-visible:outline-amber-500 dark:border-stone-700 dark:hover:bg-stone-800">Close</button>
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <Frame image={result.image_jpg} heatmap={result.heatmap_png} caption={`Before: ${ablation.removed.label} holds ${ablation.removed.attention_pct.toFixed(1)}%`} />
        <Frame image={ablation.ablated_image_jpg} heatmap={ablation.ablated_heatmap_png} caption={`After removal (${ablation.removed.method === "inpaint" ? "inpainted" : "filled with the blurred surroundings"})`} />
      </div>

      <h3 className="mt-4 text-sm font-medium">Where its {ablation.removed.attention_pct.toFixed(1)}% went</h3>
      <ul className="mt-2 space-y-1.5">
        {ablation.flow.map((f) => (
          <li key={f.element_id} className="grid grid-cols-[minmax(0,10rem)_1fr_auto] items-center gap-2 text-sm">
            <span className="truncate">{f.label}</span>
            <div className="h-2 rounded-full bg-stone-200 dark:bg-stone-700" aria-hidden="true">
              <div className={`h-2 rounded-full ${f.delta_pct >= 0 ? "bg-emerald-500" : "bg-rose-500"}`} style={{ width: `${Math.min(100, (Math.abs(f.delta_pct) / maxDelta) * 100)}%` }} />
            </div>
            <span className="tabular-nums text-xs">
              {f.before_pct.toFixed(1)}% → {f.after_pct.toFixed(1)}% <span className={f.delta_pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}>({f.delta_pct >= 0 ? "+" : ""}{f.delta_pct.toFixed(1)})</span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
