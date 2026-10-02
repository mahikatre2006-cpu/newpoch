// One colour per element type, shared by the mask layer, the labels and the legend.
export const ELEMENT_COLORS: Record<string, [number, number, number]> = {
  face: [0, 200, 255],
  text: [255, 220, 0],
  subject: [170, 90, 255],
  layer: [0, 255, 120],
};

export const ELEMENT_NAMES: Record<string, string> = { face: "Face", text: "Text", subject: "Subject", layer: "Editor layer" };

export const rgb = (c: [number, number, number]) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

/** Paint a run-length-encoded mask ("zeros,ones,zeros,..." in row-major order) straight into RGBA pixels. */
export function paintRle(rle: string, pixels: Uint8ClampedArray, color: [number, number, number], alpha: number): void {
  let pos = 0;
  let on = false;
  for (const part of rle.split(",")) {
    const run = Number(part);
    if (on) {
      for (let i = pos; i < pos + run; i++) {
        const o = i * 4;
        pixels[o] = color[0];
        pixels[o + 1] = color[1];
        pixels[o + 2] = color[2];
        pixels[o + 3] = alpha;
      }
    }
    pos += run;
    on = !on;
  }
}
