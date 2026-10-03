import Konva from "konva";
import { FRAME_H, FRAME_W, type Layer } from "./types";

export interface LayerPayload {
  name: string;
  type: string;
  mask: string; // base64 PNG, alpha channel = where the layer is
  text?: string; // text layers: the exact string, so the server needs no OCR to know what it says
}

export interface Payload {
  image: Blob; // the flattened 1280x720 frame
  layers: LayerPayload[]; // bottom to top
  ids: string[]; // the editor layer id behind each entry of `layers`
}

/** Draw clones of the given live nodes on a fresh, unscaled 1280x720 stage: exact pixels, no selection handles. */
function renderStage(nodes: Konva.Node[]): Konva.Stage {
  const stage = new Konva.Stage({ container: document.createElement("div"), width: FRAME_W, height: FRAME_H });
  const layer = new Konva.Layer();
  stage.add(layer);
  nodes.forEach((n) => layer.add(n.clone()));
  layer.draw();
  return stage;
}

function render(nodes: Konva.Node[]): string {
  const stage = renderStage(nodes);
  const url = stage.toDataURL({ mimeType: "image/png", pixelRatio: 1 });
  stage.destroy();
  return url;
}

/** Where the layer is, as a PNG whose alpha channel is the layer's coverage and whose colour is zeroed. The server reads
 *  the alpha channel; a PNG with the layer's real colours (a photo!) would be megabytes and overflow the form field,
 *  while constant colour compresses to almost nothing. */
function renderMask(node: Konva.Node): string {
  const stage = renderStage([node]);
  const src = stage.toCanvas({ pixelRatio: 1 });
  const ctx = src.getContext("2d")!;
  const px = ctx.getImageData(0, 0, FRAME_W, FRAME_H);
  for (let i = 0; i < px.data.length; i += 4) px.data[i] = px.data[i + 1] = px.data[i + 2] = 0;
  ctx.putImageData(px, 0, 0);
  const url = src.toDataURL("image/png");
  stage.destroy();
  return url;
}

async function toBlob(dataUrl: string): Promise<Blob> {
  return (await fetch(dataUrl)).blob();
}

/** The flattened frame plus one alpha mask per layer, which is what /analyze takes: the masks replace detection. */
export async function exportForAnalysis(stage: Konva.Stage, layers: Layer[]): Promise<Payload> {
  const nodes = layers.map((l) => stage.findOne(`#${l.id}`)).filter((n): n is Konva.Node => !!n);
  if (nodes.length === 0) throw new Error("Nothing to analyse yet: add a layer.");
  const flat = render(nodes);
  const kept = layers.filter((l) => stage.findOne(`#${l.id}`));
  const masks = kept.map((_, i) => renderMask(nodes[i]));
  return {
    image: await toBlob(flat),
    layers: kept.map((l, i) => ({ name: l.name, type: l.type, mask: masks[i].replace(/^data:image\/png;base64,/, ""), ...(l.kind === "text" ? { text: l.text } : {}) })),
    ids: kept.map((l) => l.id),
  };
}

/** The finished thumbnail as a 1280x720 JPG under YouTube's 2 MB limit: the quality is lowered until it fits. */
export async function exportJpeg(stage: Konva.Stage, layers: Layer[], maxBytes = 2 * 1024 * 1024 - 1024): Promise<Blob> {
  const nodes = layers.map((l) => stage.findOne(`#${l.id}`)).filter((n): n is Konva.Node => !!n);
  if (nodes.length === 0) throw new Error("Nothing to export yet: add a layer.");
  const st = renderStage(nodes);
  const canvas = st.toCanvas({ pixelRatio: 1 });
  st.destroy();
  for (let q = 0.95; q >= 0.4; q -= 0.05) {
    const blob = await new Promise<Blob | null>((res) => canvas.toBlob(res, "image/jpeg", q));
    if (blob && blob.size <= maxBytes) return blob;
  }
  throw new Error("Could not get the image under 2 MB; try a simpler picture.");
}
