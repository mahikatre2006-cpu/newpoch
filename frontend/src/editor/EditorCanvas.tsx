import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import { Layer as KLayer, Stage, Transformer } from "react-konva";
import type Konva from "konva";
import LayerNode from "./render";
import { FRAME_H, FRAME_W, type Layer } from "./types";

interface Props {
  layers: Layer[]; // bottom to top
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onChange: (id: string, patch: Partial<Layer>) => void;
  onImageLoaded: () => void;
  stageRef: RefObject<Konva.Stage | null>;
  children?: ReactNode; // overlays (heatmap), drawn above the canvas without catching the pointer
}

/** react-konva canvas at 1280x720, scaled to fit its container. Layers are drawn bottom to top. */
export default function EditorCanvas({ layers, selectedId, onSelect, onChange, onImageLoaded, stageRef, children }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const tr = useRef<Konva.Transformer>(null);
  const [width, setWidth] = useState(FRAME_W);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(320, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // attach the resize/rotate handles to the selected layer
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !tr.current) return;
    const node = selectedId ? stage.findOne(`#${selectedId}`) : null;
    tr.current.nodes(node ? [node] : []);
    tr.current.getLayer()?.batchDraw();
  }, [selectedId, layers, stageRef]);

  const scale = width / FRAME_W;
  return (
    <div ref={box} className="relative w-full overflow-hidden rounded-xl bg-stone-900 shadow-lg ring-1 ring-black/10" style={{ height: Math.round(FRAME_H * scale) }}>
      <Stage
        ref={stageRef} width={width} height={Math.round(FRAME_H * scale)} scaleX={scale} scaleY={scale}
        onMouseDown={(e) => { if (e.target === e.target.getStage()) onSelect(null); }}
      >
        <KLayer>
          {layers.map((l) => (
            <LayerNode key={l.id} layer={l} draggable onSelect={() => onSelect(l.id)} onChange={(p) => onChange(l.id, p)} onLoaded={onImageLoaded} />
          ))}
        </KLayer>
        <KLayer>
          <Transformer ref={tr} rotateEnabled keepRatio={false} borderStroke="#f59e0b" anchorStroke="#f59e0b" anchorFill="#fff" anchorSize={11} rotationSnaps={[0, 45, 90, 135, 180, 225, 270, 315]} />
        </KLayer>
      </Stage>
      <div className="pointer-events-none absolute inset-0">{children}</div>
    </div>
  );
}
