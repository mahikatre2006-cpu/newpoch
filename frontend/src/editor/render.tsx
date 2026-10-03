import { useEffect, useState } from "react";
import { Arrow, Ellipse, Image as KImage, Rect, Text } from "react-konva";
import type Konva from "konva";
import type { Layer } from "./types";

export function useImage(src: string, onLoaded?: () => void): HTMLImageElement | null {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!src) return;
    const im = new window.Image();
    im.onload = () => {
      setImg(im);
      onLoaded?.(); // the editor re-analyses once a picture is actually on the canvas
    };
    im.src = src;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);
  return img;
}

interface Props {
  layer: Layer;
  draggable: boolean;
  onSelect: () => void;
  onChange: (patch: Partial<Layer>) => void;
  onLoaded?: () => void;
}

/** One layer as a Konva node. The same attributes are used for the live canvas and for the offscreen export. */
export default function LayerNode({ layer, draggable, onSelect, onChange, onLoaded }: Props) {
  const common = {
    id: layer.id, name: "layer-node", x: layer.x, y: layer.y, rotation: layer.rotation, scaleX: layer.scaleX, scaleY: layer.scaleY,
    draggable, onClick: onSelect, onTap: onSelect, onMouseDown: onSelect,
    onDragEnd: (e: Konva.KonvaEventObject<DragEvent>) => onChange({ x: Math.round(e.target.x()), y: Math.round(e.target.y()) }),
    onTransformEnd: (e: Konva.KonvaEventObject<Event>) => {
      const n = e.target;
      onChange({ x: Math.round(n.x()), y: Math.round(n.y()), rotation: Math.round(n.rotation()), scaleX: n.scaleX(), scaleY: n.scaleY() });
    },
  };
  const img = useImage(layer.kind === "image" ? layer.src : "", onLoaded);
  switch (layer.kind) {
    case "image":
      return img ? <KImage {...common} image={img} width={layer.width} height={layer.height} /> : null;
    case "text":
      return (
        <Text {...common} text={layer.text} fontFamily={layer.fontFamily} fontSize={layer.fontSize} fontStyle={layer.bold ? "bold" : "normal"}
          fill={layer.fill} stroke={layer.strokeWidth > 0 ? layer.stroke : undefined} strokeWidth={layer.strokeWidth} fillAfterStrokeEnabled lineJoin="round"
          shadowEnabled={layer.shadow} shadowColor="#000000" shadowBlur={14} shadowOffset={{ x: 6, y: 6 }} shadowOpacity={0.8} />
      );
    case "rect":
      return <Rect {...common} width={layer.width} height={layer.height} fill={layer.fill} stroke={layer.strokeWidth > 0 ? layer.stroke : undefined} strokeWidth={layer.strokeWidth} />;
    case "ellipse":
      // Konva draws an ellipse around its centre; the offset makes x, y its top-left corner like every other layer
      return <Ellipse {...common} radiusX={layer.width / 2} radiusY={layer.height / 2} offsetX={-layer.width / 2} offsetY={-layer.height / 2}
        fill={layer.fill} stroke={layer.strokeWidth > 0 ? layer.stroke : undefined} strokeWidth={layer.strokeWidth} />;
    case "arrow":
      return <Arrow {...common} points={[0, 0, layer.length, 0]} stroke={layer.stroke} fill={layer.stroke} strokeWidth={layer.strokeWidth}
        pointerLength={layer.strokeWidth * 1.6} pointerWidth={layer.strokeWidth * 1.9} lineCap="round" />;
  }
}
