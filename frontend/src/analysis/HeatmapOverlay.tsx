import { useEffect, useRef, useState } from "react";
import type { AnalysisResult } from "../api/types";
import ElementLayer from "./ElementLayer";
import LayerViewsOverlay, { type ViewToggles } from "./LayerViews";

interface Props {
  result: AnalysisResult;
  opacity: number; // 0..1
  showHeatmap: boolean;
  showElements: boolean;
  views: ViewToggles;
  showPath: boolean;
  replayKey: number; // bump to restart the gaze animation
  highlightId?: string | null; // element a hovered row is about
}

const STEP_MS = 650;
const FRAME_W = 1280;

/** The analysed frame with the heatmap and the predicted gaze path on top. Everything is in 1280x720 frame units,
 *  so it shares the server's pixels exactly. */
export default function HeatmapOverlay({ result, opacity, showHeatmap, showElements, views, showPath, replayKey, highlightId }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(FRAME_W);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  // On a small display the frame shrinks, so scale markers up to stay readable.
  const k = Math.max(1, 640 / Math.max(width, 1));

  const path = [...(result.scanpath ?? [])].sort((a, b) => a.order - b.order);
  const hl = highlightId ? result.elements?.find((e) => e.id === highlightId) : null;

  return (
    <div ref={ref} className="relative aspect-video w-full overflow-hidden rounded-xl bg-stone-900 shadow-lg ring-1 ring-black/10">
      <img src={`data:image/jpeg;base64,${result.image_jpg}`} alt="Analysed thumbnail" className="absolute inset-0 h-full w-full object-fill" />
      {showElements && result.elements && <ElementLayer elements={result.elements} scale={k} />}
      {result.layer_views && <LayerViewsOverlay views={result.layer_views} on={views} k={k} />}
      {showHeatmap && result.heatmap_png && (
        <img
          src={`data:image/png;base64,${result.heatmap_png}`}
          alt="Attention heatmap: brighter areas are more likely to be looked at"
          className="absolute inset-0 h-full w-full object-fill"
          style={{ opacity }}
        />
      )}
      <svg viewBox={`0 0 ${FRAME_W} 720`} className="absolute inset-0 h-full w-full" role="img" aria-label="Predicted gaze order">
        {hl && hl.type !== "background" && (
          <g pointerEvents="none">
            <rect x={hl.box[0]} y={hl.box[1]} width={hl.box[2] - hl.box[0]} height={hl.box[3] - hl.box[1]} fill="rgba(255,255,255,0.1)" stroke="#000" strokeWidth={9 * k} rx={8} />
            <rect x={hl.box[0]} y={hl.box[1]} width={hl.box[2] - hl.box[0]} height={hl.box[3] - hl.box[1]} fill="none" stroke="#fff" strokeWidth={5 * k} rx={8} />
          </g>
        )}
        {showPath && path.length > 0 && (
          <g key={replayKey}>
            {path.slice(1).map((p, i) => (
              <line
                key={`s${p.order}`}
                x1={path[i].x} y1={path[i].y} x2={p.x} y2={p.y}
                pathLength={1}
                className="gaze-seg"
                stroke="#fff" strokeWidth={5 * k} strokeLinecap="round"
                style={{ ["--d" as string]: `${i * STEP_MS + 300}ms`, filter: "drop-shadow(0 0 3px rgba(0,0,0,.9))" }}
              />
            ))}
            {path.map((p, i) => (
              <g key={`d${p.order}`} className="gaze-dot" style={{ ["--d" as string]: `${i * STEP_MS}ms` }}>
                <circle cx={p.x} cy={p.y} r={28 * k} fill="#0c0a09" stroke="#fff" strokeWidth={4 * k} />
                <text x={p.x} y={p.y + 10 * k} textAnchor="middle" fontSize={30 * k} fontWeight={800} fill="#fff">{p.order}</text>
              </g>
            ))}
          </g>
        )}
      </svg>
    </div>
  );
}
