import { useCallback, useEffect, useRef, useState } from "react";
import type Konva from "konva";
import { analyze } from "../api/client";
import type { AnalysisResult, FixLayer } from "../api/types";
import BudgetBar, { type BudgetEntry } from "../analysis/BudgetBar";
import Breakdown from "../analysis/Breakdown";
import FramePanel from "../analysis/FramePanel";
import IntentRanking from "../analysis/IntentRanking";
import Legend from "../analysis/Legend";
import EditorCanvas from "../editor/EditorCanvas";
import { exportForAnalysis, exportJpeg, type Payload } from "../editor/exportForAnalysis";
import LayerList from "../editor/LayerList";
import Properties from "../editor/Properties";
import { FRAME_H, FRAME_W, loadImageSize, makeArrow, makeBackground, makeImage, makeImageAt, makeShape, makeText, type Layer, type LayerType, type TextLayer } from "../editor/types";

const DEBOUNCE_MS = 600;
const colorFor = (i: number) => `hsl(${(i * 67 + 20) % 360} 62% 48%)`;

interface Analysed {
  result: AnalysisResult;
  ids: string[]; // editor layer id behind `layer_0`, `layer_1`, ... of this result
  seq: number; // which analysis request produced it, so a fix can tell the result that came after it
}

interface Props {
  initialImage?: File | null;
  initialLayers?: FixLayer[]; // e.g. the parts from "Split into layers"
  sessionId?: string; // the Studio session this editor continues, so versions line up
  onBack: () => void;
}

/** What a fix is waiting to verify: the measurements before it, compared once the next analysis lands. */
interface Pending {
  title: string;
  seq: number;
  ids: string[];
  before: Record<string, { contrast: number | null; legible: boolean | null; safe: number; share: number }>;
  kind: "text" | "safe";
}

const SAFE_MARGIN = 6;
const luminance = (hex: string) => {
  const n = parseInt(hex.slice(1), 16);
  return (0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255;
};

/** Attention-aware editor (M9): edit layers; after every change (debounced) the flattened image and one mask per layer
 *  are analysed, so each layer's share of attention stays visible in the budget bar while you work. */
export default function EditorPage({ initialImage, initialLayers, sessionId: startSession, onBack }: Props) {
  const stageRef = useRef<Konva.Stage | null>(null);
  const [layers, setLayers] = useState<Layer[]>(() =>
    initialLayers && initialLayers.length > 0
      ? initialLayers.map((l) => makeImageAt(`data:${l.mime};base64,${l.data}`, l.x, l.y, l.width, l.height, l.name, l.type))
      : [makeBackground()],
  );
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [analysed, setAnalysed] = useState<Analysed | null>(null);
  const [prevShares, setPrevShares] = useState<Record<string, number> | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageTick, setImageTick] = useState(0);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [opacity, setOpacity] = useState(0.65);
  const [order, setOrder] = useState<string[]>([]);
  const [intentBusy, setIntentBusy] = useState(false);

  const seq = useRef(0);
  const sessionId = useRef<string | undefined>(startSession);
  const [pending, setPending] = useState<Pending | null>(null);
  const [fixNote, setFixNote] = useState<{ title: string; lines: string[] } | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const saves = useRef(0);
  const lastPayload = useRef<Payload | null>(null);
  const colorIndex = useRef<Record<string, number>>({});
  const fileInput = useRef<HTMLInputElement>(null);
  const subjectInput = useRef<HTMLInputElement>(null);

  const colors: Record<string, string> = {};
  for (const l of layers) {
    if (colorIndex.current[l.id] === undefined) colorIndex.current[l.id] = Object.keys(colorIndex.current).length;
    colors[l.id] = colorFor(colorIndex.current[l.id]);
  }

  // ---- layer operations ----
  const update = useCallback((id: string, patch: Partial<Layer>) => setLayers((ls) => ls.map((l) => (l.id === id ? ({ ...l, ...patch } as Layer) : l))), []);
  const add = (l: Layer) => { setLayers((ls) => [...ls, l]); setSelectedId(l.id); };
  const remove = (id: string) => { setLayers((ls) => ls.filter((l) => l.id !== id)); setSelectedId((s) => (s === id ? null : s)); };
  const move = (id: string, dir: 1 | -1) => setLayers((ls) => {
    const i = ls.findIndex((l) => l.id === id);
    const j = i + dir;
    if (i < 0 || j < 0 || j >= ls.length) return ls;
    const next = [...ls];
    [next[i], next[j]] = [next[j], next[i]];
    return next;
  });

  async function addPicture(file: File | undefined, type: LayerType) {
    if (!file) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) return setError("Please use a JPG, PNG or WEBP image.");
    setError(null);
    const src = await new Promise<string>((res) => { const r = new FileReader(); r.onload = () => res(String(r.result)); r.readAsDataURL(file); });
    try {
      const { w, h } = await loadImageSize(src);
      add(makeImage(src, w, h, file.name.replace(/\.[^.]+$/, "").slice(0, 40) || "Image", type));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not read that image");
    }
  }

  // an image sent from the Studio opens as the first picture layer
  const started = useRef(false);
  useEffect(() => {
    if (started.current || !initialImage) return;
    started.current = true;
    void addPicture(initialImage, "image");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialImage]);

  // ---- analysis ----
  const runAnalysis = useCallback(async () => {
    const stage = stageRef.current;
    if (!stage) return;
    const mine = ++seq.current;
    setBusy(true);
    setError(null);
    try {
      const payload = await exportForAnalysis(stage, layers);
      lastPayload.current = payload;
      const result = await analyze({ image: payload.image, filename: "edit.png", layers: payload.layers, sessionId: sessionId.current });
      if (mine !== seq.current) return; // a newer edit superseded this one
      sessionId.current = result.session_id;
      setAnalysed((cur) => {
        setPrevShares(cur ? sharesOf(cur) : null);
        return { result, ids: payload.ids, seq: mine };
      });
    } catch (e) {
      if (mine === seq.current) setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      if (mine === seq.current) setBusy(false);
    }
  }, [layers]);

  useEffect(() => {
    const t = window.setTimeout(() => void runAnalysis(), DEBOUNCE_MS);
    return () => window.clearTimeout(t);
  }, [runAnalysis, imageTick]);

  // ---- derived: budget entries and the intent list, keyed by editor layer id ----
  const sharesOf = (a: Analysed): Record<string, number> => {
    const out: Record<string, number> = {};
    for (const e of a.result.elements ?? []) {
      const m = /^layer_(\d+)$/.exec(e.id);
      if (m && a.ids[Number(m[1])]) out[a.ids[Number(m[1])]] = e.attention_pct;
      else if (e.id === "background") out["__uncovered"] = e.attention_pct;
    }
    return out;
  };
  const nameOf = (id: string) => layers.find((l) => l.id === id)?.name ?? "Deleted layer";
  const entries: BudgetEntry[] = [];
  if (analysed) {
    const cur = sharesOf(analysed);
    for (const [id, pct] of Object.entries(cur)) {
      const prev = prevShares?.[id];
      if (pct < 0.05 && !(prev && prev >= 0.05)) continue; // nothing to show for an empty segment
      entries.push({
        key: id, pct, color: id === "__uncovered" ? "#78716c" : colors[id] ?? "#78716c",
        label: id === "__uncovered" ? "Uncovered" : nameOf(id),
        delta: prevShares ? pct - (prev ?? 0) : null,
      });
    }
  }

  const layerElements = (analysed?.result.elements ?? [])
    .filter((e) => e.type === "layer" && analysed!.ids[Number(e.id.split("_")[1])])
    .map((e) => ({ ...e, id: analysed!.ids[Number(e.id.split("_")[1])] }));
  const rankedIds = [...layerElements].sort((a, b) => a.predicted_rank - b.predicted_rank).map((e) => e.id);
  const orderNow = [...order.filter((id) => rankedIds.includes(id)), ...rankedIds.filter((id) => !order.includes(id))];

  async function checkIntent(intent: string[] | null) {
    if (!analysed || !lastPayload.current) return;
    setIntentBusy(true);
    try {
      const elementIds = intent?.map((id) => `layer_${analysed.ids.indexOf(id)}`);
      const result = await analyze({ image: lastPayload.current.image, filename: "edit.png", layers: lastPayload.current.layers, intent: elementIds, sessionId: sessionId.current });
      setAnalysed((cur) => ({ result, ids: lastPayload.current!.ids, seq: cur?.seq ?? 0 }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Check failed");
    } finally {
      setIntentBusy(false);
    }
  }

  // ---- verified fixes that need the layers: the editor owns them, so it restyles/moves them, then re-analyses ----
  const elementFor = (layerId: string) => {
    if (!analysed) return undefined;
    const i = analysed.ids.indexOf(layerId);
    return analysed.result.elements?.find((e) => e.id === `layer_${i}`);
  };
  const snapshot = (id: string) => {
    const e = elementFor(id);
    const f = e?.features;
    return { contrast: f?.text_contrast_ratio ?? null, legible: f?.mobile_legible ?? null, safe: f?.safe_zone_overlap_pct ?? 0, share: e?.attention_pct ?? 0 };
  };

  /** One step per click: outline + shadow first, then bigger text. The next analysis says whether it worked. */
  function fixText() {
    if (!analysed) return;
    const failing = layers.filter((l): l is TextLayer => l.kind === "text").filter((l) => {
      const s = snapshot(l.id);
      return (s.contrast !== null && s.contrast < 4.5) || s.legible === false;
    });
    if (failing.length === 0) {
      setFixNote({ title: "Fix text legibility", lines: ["Every text layer already has at least 4.5:1 contrast and is readable at phone size."] });
      return;
    }
    setPending({ title: "Fix text legibility", kind: "text", seq: seq.current, ids: failing.map((l) => l.id), before: Object.fromEntries(failing.map((l) => [l.id, snapshot(l.id)])) });
    for (const l of failing) {
      const styled = l.shadow && l.strokeWidth >= Math.round(l.fontSize * 0.05);
      const fontSize = styled ? Math.min(400, Math.round(l.fontSize * 1.15)) : l.fontSize;
      update(l.id, { fontSize, shadow: true, stroke: luminance(l.fill) > 0.5 ? "#000000" : "#ffffff", strokeWidth: Math.max(6, Math.round(fontSize * 0.06)) } as Partial<Layer>);
    }
    setFixNote(null);
  }

  /** Move every layer that sits under the YouTube timestamp or progress bar up or left, just out of the way. */
  function clearSafeZone() {
    const stage = stageRef.current;
    const boxes = analysed?.result.layer_views?.safe_zone.boxes;
    if (!stage || !boxes) return;
    const moves: { id: string; dx: number; dy: number }[] = [];
    for (const l of layers) {
      const node = stage.findOne(`#${l.id}`);
      if (!node) continue;
      const r = node.getClientRect({ relativeTo: node.getParent() as Konva.Container });
      if (r.width * r.height > 0.8 * FRAME_W * FRAME_H) continue; // a background or full-frame picture is meant to sit there
      let dx = 0, dy = 0;
      for (const b of boxes) {
        const ox = Math.min(r.x + dx + r.width, b[2]) - Math.max(r.x + dx, b[0]);
        const oy = Math.min(r.y + dy + r.height, b[3]) - Math.max(r.y + dy, b[1]);
        if (ox <= 0 || oy <= 0) continue;
        const up = r.y + dy + r.height - b[1] + SAFE_MARGIN;
        const left = r.x + dx + r.width - b[0] + SAFE_MARGIN;
        if (b[2] - b[0] > FRAME_W * 0.5 || up <= left) dy -= up; // the progress bar spans the width: only up helps
        else dx -= left;
      }
      if (dx !== 0 || dy !== 0) moves.push({ id: l.id, dx: Math.round(dx), dy: Math.round(dy) });
    }
    if (moves.length === 0) {
      setFixNote({ title: "Clear safe zone", lines: ["No layer overlaps the YouTube timestamp or progress bar."] });
      return;
    }
    setPending({ title: "Clear safe zone", kind: "safe", seq: seq.current, ids: moves.map((m) => m.id), before: Object.fromEntries(moves.map((m) => [m.id, snapshot(m.id)])) });
    for (const m of moves) {
      const l = layers.find((x) => x.id === m.id)!;
      update(m.id, { x: l.x + m.dx, y: l.y + m.dy });
    }
    setFixNote(null);
  }

  // once the analysis that follows a fix lands, report the measured change
  useEffect(() => {
    if (!pending || !analysed || analysed.seq <= pending.seq) return;
    const lines = pending.ids.map((id) => {
      const name = layers.find((l) => l.id === id)?.name ?? "Layer";
      const b = pending.before[id];
      const a = snapshot(id);
      const mark = (v: boolean | null) => (v === null ? "?" : v ? "✓" : "✗");
      const num = (v: number | null) => (v === null ? "?" : v.toFixed(1));
      return pending.kind === "text"
        ? `${name}: contrast ${num(b.contrast)}:1 → ${num(a.contrast)}:1, readable on phones ${mark(b.legible)} → ${mark(a.legible)}, attention ${b.share.toFixed(1)}% → ${a.share.toFixed(1)}%.`
        : `${name}: ${b.safe.toFixed(1)}% under the timestamp or progress bar → ${a.safe.toFixed(1)}%, attention ${b.share.toFixed(1)}% → ${a.share.toFixed(1)}%.`;
    });
    setFixNote({ title: pending.title, lines });
    setPending(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysed]);

  async function saveVersion() {
    const payload = lastPayload.current;
    if (!payload) return;
    saves.current += 1;
    const label = `Editor version ${saves.current}`;
    try {
      const r = await analyze({ image: payload.image, filename: "edit.png", layers: payload.layers, sessionId: sessionId.current, saveVersion: true, label });
      sessionId.current = r.session_id;
      setSaved(label);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save the version");
    }
  }

  async function download() {
    if (!stageRef.current) return;
    try {
      const blob = await exportJpeg(stageRef.current, layers);
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "thumbnail.jpg";
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Export failed");
    }
  }

  const selected = layers.find((l) => l.id === selectedId) ?? null;
  const result = analysed?.result;
  const shown = (result?.elements ?? []).filter((e) => !(e.id === "background" && e.area_pct === 0)); // nothing is uncovered
  const btn = "rounded-lg border border-stone-300 px-3 py-1.5 text-sm font-medium hover:bg-stone-100 focus-visible:outline-2 focus-visible:outline-amber-500 dark:border-stone-700 dark:hover:bg-stone-800";

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-semibold">Editor</h1>
        <div className="flex gap-2">
          <button type="button" onClick={() => void download()} className="rounded-lg bg-stone-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-stone-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-500 dark:bg-amber-400 dark:text-stone-950">Download JPG</button>
          <button type="button" onClick={onBack} className={btn}>Back to start</button>
        </div>
      </div>
      {error && <div role="alert" className="rounded-lg border border-red-300 bg-red-50 px-4 py-2 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200">{error}</div>}

      {analysed && <BudgetBar entries={entries} updating={busy} />}

      <div className="flex flex-wrap items-center gap-2" role="toolbar" aria-label="Add layer">
        <button type="button" className={btn} onClick={() => add(makeText())}>+ Text</button>
        <button type="button" className={btn} onClick={() => add(makeShape("rect"))}>+ Rectangle</button>
        <button type="button" className={btn} onClick={() => add(makeShape("ellipse"))}>+ Circle</button>
        <button type="button" className={btn} onClick={() => add(makeArrow())}>+ Arrow</button>
        <button type="button" className={btn} onClick={() => fileInput.current?.click()}>+ Image</button>
        <button type="button" className={btn} title="A picture of a person or product that is already cut out (transparent background)" onClick={() => subjectInput.current?.click()}>+ Cut-out subject</button>
        <input ref={fileInput} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(e) => { void addPicture(e.target.files?.[0], "image"); e.target.value = ""; }} />
        <input ref={subjectInput} type="file" accept="image/png,image/webp" hidden onChange={(e) => { void addPicture(e.target.files?.[0], "subject"); e.target.value = ""; }} />
      </div>

      {analysed && (
        <div className="flex flex-wrap items-center gap-2" role="toolbar" aria-label="Fixes and versions">
          <span className="mr-1 text-sm font-medium">Fixes</span>
          <button type="button" className={btn} disabled={busy || !!pending} title="Outline and shadow for text that is low-contrast or unreadable at phone size; click again to make it bigger" onClick={fixText}>Fix text legibility</button>
          <button type="button" className={btn} disabled={busy || !!pending} title="Move layers out from under YouTube's timestamp and progress bar" onClick={clearSafeZone}>Clear safe zone</button>
          <span className="mx-1 h-5 border-l border-stone-300 dark:border-stone-700" aria-hidden="true" />
          <button type="button" className={btn} disabled={busy} title="Keep this state as a version you can compare later" onClick={() => void saveVersion()}>Save version</button>
          {saved && <span role="status" className="text-sm text-emerald-700 dark:text-emerald-300">Saved “{saved}”</span>}
        </div>
      )}
      {fixNote && (
        <section className="rounded-xl border border-emerald-300 bg-emerald-50 p-3 text-sm dark:border-emerald-800 dark:bg-emerald-950/30" aria-labelledby="fixnote-h">
          <div className="flex items-start justify-between gap-3">
            <h2 id="fixnote-h" className="font-semibold">{fixNote.title}</h2>
            <button type="button" onClick={() => setFixNote(null)} className="rounded-md border border-emerald-400 px-2 py-0.5 text-xs hover:bg-emerald-100 dark:border-emerald-700 dark:hover:bg-emerald-900/40">Dismiss</button>
          </div>
          <ul role="status" className="mt-1 list-disc space-y-0.5 pl-4">{fixNote.lines.map((l, i) => <li key={i}>{l}</li>)}</ul>
        </section>
      )}
      {pending && <p role="status" className="text-sm text-stone-600 dark:text-stone-400">Applying “{pending.title}” and measuring the result…</p>}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,22rem)]">
        <div className="space-y-3">
          <EditorCanvas layers={layers} selectedId={selectedId} onSelect={setSelectedId} onChange={update} onImageLoaded={() => setImageTick((n) => n + 1)} stageRef={stageRef}>
            {showHeatmap && result?.heatmap_png && (
              <img src={`data:image/png;base64,${result.heatmap_png}`} alt="" className={`absolute inset-0 h-full w-full object-fill transition-opacity ${busy ? "opacity-30" : ""}`} style={{ opacity: busy ? undefined : opacity }} />
            )}
          </EditorCanvas>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} /> Heatmap</label>
            <label className="flex items-center gap-2 text-sm">
              Opacity
              <input type="range" min={0} max={100} value={Math.round(opacity * 100)} disabled={!showHeatmap} onChange={(e) => setOpacity(Number(e.target.value) / 100)} />
            </label>
            <Legend />
          </div>
          {result && result.hierarchy && (
            <p className="text-sm text-stone-700 dark:text-stone-300">
              <span className="font-medium">Viewing order: </span>
              {result.hierarchy.filter((h) => shown.some((e) => e.id === h)).map((h) => (h === "background" ? "Uncovered" : shown.find((e) => e.id === h)?.label ?? h)).join(" → ")}
            </p>
          )}
        </div>

        <div className="space-y-4">
          <LayerList layers={layers} selectedId={selectedId} colors={colors} onSelect={setSelectedId} onChange={update} onMove={move} onDelete={remove} />
          <Properties layer={selected} onChange={(p) => selected && update(selected.id, p)} />
        </div>
      </div>

      {result && (
        <div className="grid gap-5 lg:grid-cols-2">
          {layerElements.length > 0 && (
            <IntentRanking elements={layerElements} order={orderNow} setOrder={setOrder} check={result.intent_check} busy={intentBusy} onCheck={() => void checkIntent(orderNow)} onClear={() => void checkIntent(null)} />
          )}
          {result.frame && <FramePanel frame={result.frame} />}
        </div>
      )}
      {shown.length > 0 && <Breakdown elements={shown.map((e) => (e.id === "background" ? { ...e, label: "Uncovered" } : e))} onHover={() => {}} />}
    </div>
  );
}
