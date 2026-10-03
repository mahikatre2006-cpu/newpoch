import type { Layer, LayerType } from "./types";

const TYPES: LayerType[] = ["image", "subject", "text", "shape"];
const KIND_ICON: Record<string, string> = { image: "▣", text: "T", rect: "▭", ellipse: "◯", arrow: "➜" };

interface Props {
  layers: Layer[]; // bottom to top
  selectedId: string | null;
  colors: Record<string, string>;
  onSelect: (id: string) => void;
  onChange: (id: string, patch: Partial<Layer>) => void;
  onMove: (id: string, dir: 1 | -1) => void;
  onDelete: (id: string) => void;
}

/** Layer stack, top layer first. Names are editable and flow into the attention breakdown. */
export default function LayerList({ layers, selectedId, colors, onSelect, onChange, onMove, onDelete }: Props) {
  const top = [...layers].reverse();
  return (
    <section aria-labelledby="layers-h" className="rounded-xl border border-stone-200 bg-white p-3 dark:border-stone-800 dark:bg-stone-900">
      <h2 id="layers-h" className="px-1 text-sm font-semibold">Layers</h2>
      <ul className="mt-2 space-y-1">
        {top.map((l, i) => (
          <li key={l.id} className={`flex items-center gap-1.5 rounded-lg border px-1.5 py-1 text-sm ${l.id === selectedId ? "border-amber-500 bg-amber-50 dark:bg-amber-950/30" : "border-stone-200 dark:border-stone-700"}`}>
            <span className="h-3 w-3 shrink-0 rounded-sm" style={{ background: colors[l.id] }} aria-hidden="true" />
            <span aria-hidden="true" className="w-4 text-center text-xs text-stone-500">{KIND_ICON[l.kind]}</span>
            <input
              value={l.name} aria-label={`Name of layer ${l.name}`} maxLength={60}
              onFocus={() => onSelect(l.id)} onChange={(e) => onChange(l.id, { name: e.target.value })}
              className="min-w-0 flex-1 rounded bg-transparent px-1 py-0.5 focus:bg-white focus-visible:outline-2 focus-visible:outline-amber-500 dark:focus:bg-stone-800"
            />
            <select value={l.type} aria-label={`Type of ${l.name}`} onChange={(e) => onChange(l.id, { type: e.target.value as LayerType })} className="rounded border border-stone-200 bg-transparent px-1 text-xs dark:border-stone-700">
              {TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
            <button type="button" aria-label={`Move ${l.name} up`} disabled={i === 0} onClick={() => onMove(l.id, 1)} className="rounded px-1 hover:bg-stone-100 disabled:opacity-30 dark:hover:bg-stone-800">↑</button>
            <button type="button" aria-label={`Move ${l.name} down`} disabled={i === top.length - 1} onClick={() => onMove(l.id, -1)} className="rounded px-1 hover:bg-stone-100 disabled:opacity-30 dark:hover:bg-stone-800">↓</button>
            <button type="button" aria-label={`Delete ${l.name}`} onClick={() => onDelete(l.id)} className="rounded px-1 text-red-600 hover:bg-red-50 dark:hover:bg-red-950/40">✕</button>
          </li>
        ))}
      </ul>
    </section>
  );
}
