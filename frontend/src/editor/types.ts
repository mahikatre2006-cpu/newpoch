// The editor's layer model. Coordinates are in the 1280x720 frame; the canvas only scales them for display.

export const FRAME_W = 1280;
export const FRAME_H = 720;

export type LayerKind = "image" | "text" | "rect" | "ellipse" | "arrow";
/** What the layer is, in the creator's words: shows up in the breakdown next to its name. */
export type LayerType = "image" | "subject" | "text" | "shape";

interface Base {
  id: string;
  name: string;
  type: LayerType;
  x: number;
  y: number;
  rotation: number;
  scaleX: number;
  scaleY: number;
}

export interface ImageLayer extends Base { kind: "image"; src: string; width: number; height: number }
export interface TextLayer extends Base {
  kind: "text"; text: string; fontFamily: string; fontSize: number; bold: boolean;
  fill: string; stroke: string; strokeWidth: number; shadow: boolean;
}
export interface ShapeLayer extends Base { kind: "rect" | "ellipse"; width: number; height: number; fill: string; stroke: string; strokeWidth: number }
export interface ArrowLayer extends Base { kind: "arrow"; length: number; stroke: string; strokeWidth: number }

export type Layer = ImageLayer | TextLayer | ShapeLayer | ArrowLayer;

export const FONTS = ["Impact", "Arial Black", "Arial", "Verdana", "Georgia", "Courier New", "Comic Sans MS"];

let counter = 0;
export const newId = () => `l${Date.now().toString(36)}${(counter++).toString(36)}`;

const base = (name: string, type: LayerType, x: number, y: number): Base => ({ id: newId(), name, type, x, y, rotation: 0, scaleX: 1, scaleY: 1 });

export function makeBackground(fill = "#1f2937"): ShapeLayer {
  return { ...base("Background", "shape", 0, 0), kind: "rect", width: FRAME_W, height: FRAME_H, fill, stroke: "#000000", strokeWidth: 0 };
}

export function makeText(text = "YOUR TITLE"): TextLayer {
  return { ...base("Title", "text", 120, 260), kind: "text", text, fontFamily: "Impact", fontSize: 150, bold: false, fill: "#ffffff", stroke: "#000000", strokeWidth: 8, shadow: false };
}

export function makeShape(kind: "rect" | "ellipse"): ShapeLayer {
  return { ...base(kind === "rect" ? "Rectangle" : "Circle", "shape", 760, 160), kind, width: 360, height: kind === "rect" ? 240 : 360, fill: "#f59e0b", stroke: "#000000", strokeWidth: 0 };
}

export function makeArrow(): ArrowLayer {
  return { ...base("Arrow", "shape", 300, 560), kind: "arrow", length: 420, stroke: "#ef4444", strokeWidth: 26 };
}

/** An uploaded picture becomes an image layer, scaled to fit the frame (cover = false keeps all of it visible). */
export function makeImage(src: string, w: number, h: number, name: string, type: LayerType = "image"): ImageLayer {
  const s = Math.min(FRAME_W / w, FRAME_H / h);
  return { ...base(name, type, (FRAME_W - w * s) / 2, (FRAME_H - h * s) / 2), kind: "image", src, width: w * s, height: h * s };
}

/** A picture placed exactly (position and size in frame pixels), e.g. the parts from "Split into layers". */
export function makeImageAt(src: string, x: number, y: number, width: number, height: number, name: string, type: LayerType): ImageLayer {
  return { ...base(name, type, x, y), kind: "image", src, width, height };
}

export function loadImageSize(src: string): Promise<{ w: number; h: number }> {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res({ w: im.naturalWidth, h: im.naturalHeight });
    im.onerror = () => rej(new Error("Could not read that image"));
    im.src = src;
  });
}
