import { useEffect, useRef } from "react";
import type { Element } from "../api/types";
import { ELEMENT_COLORS, paintRle, rgb } from "./elementColors";

const W = 1280;
const H = 720;

/** The detected elements as coloured masks (faces, text blocks, subjects) with their attention share. */
export default function ElementLayer({ elements, scale }: { elements: Element[]; scale: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const img = new ImageData(W, H);
    for (const e of elements) {
      const color = ELEMENT_COLORS[e.type];
      if (color) paintRle(e.mask_rle, img.data, color, 105);
    }
    c.getContext("2d")?.putImageData(img, 0, 0);
  }, [elements]);

  const labelled = elements.filter((e) => e.type !== "background" && e.attention_pct >= 1);
  return (
    <>
      <canvas ref={canvas} width={W} height={H} className="absolute inset-0 h-full w-full" aria-hidden="true" />
      <svg viewBox={`0 0 ${W} ${H}`} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
        {labelled.map((e) => (
          <text key={e.id} x={e.box[0] + 8} y={Math.min(H - 8, e.box[1] + 30 * scale)} fontSize={26 * scale} fontWeight={700}
            fill="#fff" stroke="#000" strokeWidth={4 * scale} paintOrder="stroke">
            {e.type} {e.attention_pct.toFixed(0)}%
          </text>
        ))}
      </svg>
    </>
  );
}

export function ElementKey({ elements }: { elements: Element[] }) {
  const types = [...new Set(elements.map((e) => e.type))].filter((t) => ELEMENT_COLORS[t]);
  return (
    <ul className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-stone-600 dark:text-stone-400" aria-label="Element colours">
      {types.map((t) => (
        <li key={t} className="flex items-center gap-1">
          <span className="inline-block h-3 w-3 rounded-sm ring-1 ring-black/20" style={{ background: rgb(ELEMENT_COLORS[t]) }} />
          {t}
        </li>
      ))}
    </ul>
  );
}
