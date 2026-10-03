import type { ReactNode } from "react";
import { FONTS, type Layer } from "./types";

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex items-center justify-between gap-2 text-sm">
      <span className="text-stone-600 dark:text-stone-400">{label}</span>
      {children}
    </label>
  );
}

const num = "w-20 rounded border border-stone-300 bg-transparent px-1.5 py-0.5 text-right tabular-nums dark:border-stone-700";

function Num({ label, value, onChange, step = 1, min }: { label: string; value: number; onChange: (v: number) => void; step?: number; min?: number }) {
  return (
    <Field label={label}>
      <input type="number" aria-label={label} className={num} value={Number.isFinite(value) ? Math.round(value * 100) / 100 : 0} step={step} min={min}
        onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) onChange(v); }} />
    </Field>
  );
}

function Color({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return <Field label={label}><input type="color" aria-label={label} value={value} onChange={(e) => onChange(e.target.value)} className="h-7 w-12 cursor-pointer rounded border border-stone-300 bg-transparent dark:border-stone-700" /></Field>;
}

/** Style and position of the selected layer. Moving or resizing here changes its attention share after the next analysis. */
export default function Properties({ layer, onChange }: { layer: Layer | null; onChange: (patch: Partial<Layer>) => void }) {
  if (!layer) {
    return (
      <section className="rounded-xl border border-stone-200 bg-white p-3 text-sm text-stone-500 dark:border-stone-800 dark:bg-stone-900 dark:text-stone-400">
        Select a layer on the canvas or in the list to edit it.
      </section>
    );
  }
  const set = (p: Record<string, unknown>) => onChange(p as Partial<Layer>);
  return (
    <section aria-labelledby="props-h" className="space-y-2 rounded-xl border border-stone-200 bg-white p-3 dark:border-stone-800 dark:bg-stone-900">
      <h2 id="props-h" className="text-sm font-semibold">{layer.name}</h2>

      {layer.kind === "text" && (
        <>
          <textarea aria-label="Text" value={layer.text} rows={2} onChange={(e) => set({ text: e.target.value })} className="w-full rounded border border-stone-300 bg-transparent px-2 py-1 text-sm dark:border-stone-700" />
          <Field label="Font">
            <select aria-label="Font" value={layer.fontFamily} onChange={(e) => set({ fontFamily: e.target.value })} className="rounded border border-stone-300 bg-transparent px-1 py-0.5 dark:border-stone-700">
              {FONTS.map((f) => <option key={f}>{f}</option>)}
            </select>
          </Field>
          <Num label="Size" value={layer.fontSize} min={10} onChange={(v) => set({ fontSize: v })} />
          <Field label="Bold"><input type="checkbox" aria-label="Bold" checked={layer.bold} onChange={(e) => set({ bold: e.target.checked })} /></Field>
          <Color label="Fill" value={layer.fill} onChange={(v) => set({ fill: v })} />
          <Color label="Outline" value={layer.stroke} onChange={(v) => set({ stroke: v })} />
          <Num label="Outline width" value={layer.strokeWidth} min={0} onChange={(v) => set({ strokeWidth: v })} />
          <Field label="Shadow"><input type="checkbox" aria-label="Shadow" checked={layer.shadow} onChange={(e) => set({ shadow: e.target.checked })} /></Field>
        </>
      )}
      {(layer.kind === "rect" || layer.kind === "ellipse") && (
        <>
          <Color label="Fill" value={layer.fill} onChange={(v) => set({ fill: v })} />
          <Color label="Outline" value={layer.stroke} onChange={(v) => set({ stroke: v })} />
          <Num label="Outline width" value={layer.strokeWidth} min={0} onChange={(v) => set({ strokeWidth: v })} />
          <Num label="Width" value={layer.width} min={1} onChange={(v) => set({ width: v })} />
          <Num label="Height" value={layer.height} min={1} onChange={(v) => set({ height: v })} />
        </>
      )}
      {layer.kind === "arrow" && (
        <>
          <Color label="Colour" value={layer.stroke} onChange={(v) => set({ stroke: v })} />
          <Num label="Thickness" value={layer.strokeWidth} min={2} onChange={(v) => set({ strokeWidth: v })} />
          <Num label="Length" value={layer.length} min={20} onChange={(v) => set({ length: v })} />
        </>
      )}

      <div className="border-t border-stone-100 pt-2 dark:border-stone-800">
        <Num label="X" value={layer.x} onChange={(v) => set({ x: v })} />
        <Num label="Y" value={layer.y} onChange={(v) => set({ y: v })} />
        <Num label="Rotation" value={layer.rotation} onChange={(v) => set({ rotation: v })} />
        <Num label="Scale %" value={layer.scaleX * 100} min={5} onChange={(v) => set({ scaleX: v / 100, scaleY: v / 100 })} />
      </div>
    </section>
  );
}
