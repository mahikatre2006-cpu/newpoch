import { useState } from "react";
import type { AnalysisResult } from "../api/types";
import Breakdown from "../analysis/Breakdown";
import { ElementKey } from "../analysis/ElementLayer";
import FramePanel from "../analysis/FramePanel";
import HeatmapOverlay from "../analysis/HeatmapOverlay";
import IntentRanking from "../analysis/IntentRanking";
import Legend from "../analysis/Legend";
import type { ViewToggles } from "../analysis/LayerViews";

const MODEL_LABEL: Record<string, string> = {
  deepgaze_iie: "DeepGaze IIE",
  msinet_tf: "MSI-Net (TensorFlow)",
  opencv_fine_grained: "OpenCV fallback (DeepGaze did not run)",
  none: "none (saliency failed)",
};

const VIEWS: { key: keyof ViewToggles; label: string; hint: string }[] = [
  { key: "composition", label: "Composition", hint: "Thirds grid, attention centre of mass and focal points" },
  { key: "text", label: "Text", hint: "Text contrast and whether it survives at phone size" },
  { key: "contrast", label: "Contrast", hint: "Local luminance contrast map" },
  { key: "safeZone", label: "Safe zone", hint: "Where YouTube's timestamp and progress bar cover the thumbnail" },
];

interface Props {
  result: AnalysisResult;
  filename: string;
  busy: boolean;
  onNew: () => void;
  onIntent: (intent: string[] | null) => void;
}

/** Studio (main screen): heatmap and layer views in the centre; breakdown, explanations, intent and frame stats beside it. */
export default function Studio({ result, filename, busy, onNew, onIntent }: Props) {
  const [opacity, setOpacity] = useState(0.7);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showElements, setShowElements] = useState(true);
  const [showPath, setShowPath] = useState(true);
  const [views, setViews] = useState<ViewToggles>({ composition: false, text: false, contrast: false, safeZone: false });
  const [replayKey, setReplayKey] = useState(0);
  const [hover, setHover] = useState<string | null>(null);

  const elements = result.elements ?? [];
  const ranked = [...elements].filter((e) => e.type !== "background").sort((a, b) => a.predicted_rank - b.predicted_rank);
  const [order, setOrder] = useState<string[]>(ranked.map((e) => e.id));
  const fallback = result.saliency_model !== "deepgaze_iie";
  const byId = new Map(elements.map((e) => [e.id, e]));
  const hierarchy = result.hierarchy ?? [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h1 className="truncate text-xl font-semibold">{filename}</h1>
          <p className="text-sm text-stone-600 dark:text-stone-400">
            Model: {MODEL_LABEL[result.saliency_model] ?? result.saliency_model} · {result.timing_ms.total} ms{result.cached ? " (cached)" : ""}
          </p>
        </div>
        <button type="button" onClick={onNew} className="rounded-lg border border-stone-300 px-4 py-2 text-sm font-medium hover:bg-stone-100 focus-visible:outline-2 focus-visible:outline-amber-500 dark:border-stone-700 dark:hover:bg-stone-800">
          Analyse another
        </button>
      </div>

      {fallback && (
        <div role="status" className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-100">
          The main model did not run, so this heatmap comes from a simpler classical method and is less accurate.
        </div>
      )}
      {result.errors.length > 0 && (
        <div role="status" className="rounded-lg border border-stone-300 bg-stone-100 px-4 py-2 text-xs text-stone-700 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-300">
          Notes: {result.errors.join(" · ")}
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)]">
        <div className="space-y-3">
          <HeatmapOverlay result={result} opacity={opacity} showHeatmap={showHeatmap} showElements={showElements} views={views} showPath={showPath} replayKey={replayKey} highlightId={hover} />

          <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} /> Heatmap</label>
            <label className="flex items-center gap-2 text-sm">
              Opacity
              <input type="range" min={0} max={100} value={Math.round(opacity * 100)} disabled={!showHeatmap} onChange={(e) => setOpacity(Number(e.target.value) / 100)} aria-valuetext={`${Math.round(opacity * 100)} percent`} />
            </label>
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={showElements} onChange={(e) => setShowElements(e.target.checked)} /> Elements</label>
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={showPath} onChange={(e) => setShowPath(e.target.checked)} /> Gaze path</label>
            <button type="button" disabled={!showPath} onClick={() => setReplayKey((k) => k + 1)} className="rounded-md border border-stone-300 px-2 py-1 text-sm hover:bg-stone-100 disabled:opacity-50 dark:border-stone-700 dark:hover:bg-stone-800">Replay</button>
          </div>

          {result.layer_views && (
            <fieldset className="flex flex-wrap items-center gap-x-5 gap-y-2">
              <legend className="sr-only">Layer views</legend>
              <span className="text-sm font-medium">Layer views</span>
              {VIEWS.map((v) => (
                <label key={v.key} title={v.hint} className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={views[v.key]} onChange={(e) => setViews({ ...views, [v.key]: e.target.checked })} /> {v.label}
                </label>
              ))}
            </fieldset>
          )}

          <div className="flex flex-wrap items-start gap-x-6 gap-y-2">
            <Legend />
            {showElements && <ElementKey elements={elements} />}
          </div>

          {hierarchy.length > 0 && (
            <p className="text-sm text-stone-700 dark:text-stone-300">
              <span className="font-medium">Viewing order: </span>
              {hierarchy.map((id) => byId.get(id)?.label ?? id).join(" → ")}
            </p>
          )}
        </div>

        <div className="space-y-4">
          {ranked.length > 0 && (
            <IntentRanking
              elements={ranked} order={order} setOrder={setOrder} check={result.intent_check} busy={busy}
              onCheck={() => onIntent(order)} onClear={() => onIntent(null)}
            />
          )}
          {result.frame && <FramePanel frame={result.frame} />}
        </div>
      </div>

      {result.elements && <Breakdown elements={result.elements} onHover={setHover} />}
    </div>
  );
}
