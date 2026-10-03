import { useEffect, useState } from "react";
import { compare } from "../api/client";
import type { CompareResult, Version } from "../api/types";
import ChangeSummary from "../analysis/ChangeSummary";

interface Props {
  versions: Version[];
  initial?: [string, string];
  onBack: () => void;
}

const KIND: Record<string, string> = { upload: "upload", edit: "editor", fix: "fix" };

/** Compare (M11): two versions side by side with their heatmaps and the change in each element's attention. */
export default function ComparePage({ versions, initial, onBack }: Props) {
  const [a, setA] = useState(initial?.[0] ?? versions[0]?.version_id ?? "");
  const [b, setB] = useState(initial?.[1] ?? versions[versions.length - 1]?.version_id ?? "");
  const [data, setData] = useState<CompareResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [heat, setHeat] = useState(true);

  useEffect(() => {
    if (!a || !b) return;
    let live = true;
    setError(null);
    compare(a, b).then((d) => live && setData(d)).catch((e) => live && setError(e instanceof Error ? e.message : "Comparison failed"));
    return () => { live = false; };
  }, [a, b]);

  const select = (value: string, set: (v: string) => void, label: string) => (
    <label className="flex items-center gap-2 text-sm">
      {label}
      <select value={value} onChange={(e) => set(e.target.value)} className="rounded border border-stone-300 bg-transparent px-2 py-1 dark:border-stone-700">
        {versions.map((v, i) => <option key={v.version_id} value={v.version_id} className="text-stone-900">{i + 1}. {v.label} ({KIND[v.kind] ?? v.kind})</option>)}
      </select>
    </label>
  );

  const side = (s: CompareResult["a"] | undefined, title: string) => s && (
    <figure className="min-w-0">
      <div className="relative aspect-video overflow-hidden rounded-xl bg-stone-900 shadow ring-1 ring-black/10">
        <img src={`data:image/jpeg;base64,${s.image_jpg}`} alt={`${title}: ${s.version.label}`} className="absolute inset-0 h-full w-full object-fill" />
        {heat && <img src={`data:image/png;base64,${s.heatmap_png}`} alt="" className="absolute inset-0 h-full w-full object-fill opacity-70" />}
      </div>
      <figcaption className="mt-1 text-sm"><span className="font-medium">{title}:</span> {s.version.label}</figcaption>
    </figure>
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-semibold">Compare versions</h1>
        <button type="button" onClick={onBack} className="rounded-lg border border-stone-300 px-4 py-2 text-sm font-medium hover:bg-stone-100 focus-visible:outline-2 focus-visible:outline-amber-500 dark:border-stone-700 dark:hover:bg-stone-800">Back</button>
      </div>
      {versions.length < 2 ? (
        <p className="text-sm text-stone-600 dark:text-stone-400">Save at least two versions to compare them: apply a fix, or save an edit in the editor.</p>
      ) : (
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
          {select(a, setA, "From")}
          {select(b, setB, "To")}
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={heat} onChange={(e) => setHeat(e.target.checked)} /> Heatmaps</label>
        </div>
      )}
      {error && <div role="alert" className="rounded-lg border border-red-300 bg-red-50 px-4 py-2 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200">{error}</div>}
      {data && (
        <>
          <div className="grid gap-4 sm:grid-cols-2">{side(data.a, "From")}{side(data.b, "To")}</div>
          <ChangeSummary info={{ title: "How attention moved", message: `${data.a.version.label} → ${data.b.version.label}`, rows: data.rows, orderBefore: data.hierarchy_a, orderAfter: data.hierarchy_b }} />
        </>
      )}
    </div>
  );
}
