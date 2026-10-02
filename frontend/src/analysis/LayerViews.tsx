import type { LayerViews } from "../api/types";

export interface ViewToggles {
  composition: boolean;
  text: boolean;
  contrast: boolean;
  safeZone: boolean;
}

const W = 1280;
const H = 720;

/** The M5 layer views drawn over the frame: composition grid, text checks, contrast map and the YouTube safe zone. */
export default function LayerViewsOverlay({ views, on, k }: { views: LayerViews; on: ViewToggles; k: number }) {
  const comp = views.composition;
  return (
    <>
      {on.contrast && (
        <img src={`data:image/png;base64,${views.contrast_png}`} alt="Local contrast map: brighter means a sharper change in luminance" className="absolute inset-0 h-full w-full object-fill opacity-80" />
      )}
      <svg viewBox={`0 0 ${W} ${H}`} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
        <defs>
          <pattern id="safe-hatch" width="16" height="16" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="16" height="16" fill="rgba(239,68,68,0.18)" />
            <line x1="0" y1="0" x2="0" y2="16" stroke="rgba(239,68,68,0.9)" strokeWidth="5" />
          </pattern>
        </defs>

        {on.safeZone && views.safe_zone.boxes.map((b, i) => (
          <g key={i}>
            <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill="url(#safe-hatch)" stroke="#ef4444" strokeWidth={3 * k} />
            {i === 0 && <text x={b[0] - 8} y={b[1] - 10 * k} textAnchor="end" fontSize={22 * k} fontWeight={700} fill="#fff" stroke="#000" strokeWidth={4 * k} paintOrder="stroke">YouTube timestamp</text>}
          </g>
        ))}

        {on.composition && comp && (
          <g>
            {[1, 2].map((i) => (
              <g key={i} stroke="rgba(255,255,255,0.75)" strokeWidth={2.5 * k} strokeDasharray={`${14 * k} ${10 * k}`}>
                <line x1={(W * i) / 3} y1={0} x2={(W * i) / 3} y2={H} />
                <line x1={0} y1={(H * i) / 3} x2={W} y2={(H * i) / 3} />
              </g>
            ))}
            {comp.thirds_points.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={7 * k} fill="none" stroke="#fff" strokeWidth={3 * k} />)}
            {comp.focal_points.map((p, i) => (
              <g key={i}>
                <circle cx={p.x} cy={p.y} r={20 * k} fill="none" stroke="#22d3ee" strokeWidth={4 * k} />
                <text x={p.x + 26 * k} y={p.y + 8 * k} fontSize={22 * k} fontWeight={700} fill="#22d3ee" stroke="#000" strokeWidth={4 * k} paintOrder="stroke">focal {i + 1}</text>
              </g>
            ))}
            <g stroke="#f472b6" strokeWidth={5 * k}>
              <line x1={comp.centre_of_mass[0] - 22 * k} y1={comp.centre_of_mass[1]} x2={comp.centre_of_mass[0] + 22 * k} y2={comp.centre_of_mass[1]} />
              <line x1={comp.centre_of_mass[0]} y1={comp.centre_of_mass[1] - 22 * k} x2={comp.centre_of_mass[0]} y2={comp.centre_of_mass[1] + 22 * k} />
            </g>
            <text x={comp.centre_of_mass[0] + 28 * k} y={comp.centre_of_mass[1] - 14 * k} fontSize={22 * k} fontWeight={700} fill="#f472b6" stroke="#000" strokeWidth={4 * k} paintOrder="stroke">attention centre</text>
          </g>
        )}

        {on.text && views.text.map((t) => {
          const color = t.mobile_legible === false ? "#ef4444" : t.mobile_legible === true ? "#22c55e" : "#a8a29e";
          const phone = t.mobile_legible === true ? "phone ✓" : t.mobile_legible === false ? "phone ✗" : "phone ?";
          const lowContrast = t.contrast_ratio !== null && t.contrast_ratio < 3;
          const label = `${t.contrast_ratio === null ? "?" : t.contrast_ratio.toFixed(1)}:1${lowContrast ? " ⚠" : ""} · ${phone}`;
          return (
            <g key={t.element_id}>
              <rect x={t.box[0]} y={t.box[1]} width={t.box[2] - t.box[0]} height={t.box[3] - t.box[1]} fill="none" stroke={color} strokeWidth={4 * k} strokeDasharray={`${12 * k} ${7 * k}`} rx={6} />
              <text x={t.box[0] + 6} y={Math.max(24 * k, t.box[1] - 8)} fontSize={22 * k} fontWeight={700} fill="#fff" stroke="#000" strokeWidth={4 * k} paintOrder="stroke">{label}</text>
            </g>
          );
        })}
      </svg>
    </>
  );
}
