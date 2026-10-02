// Mirrors backend/app/api/schemas.py (PRD section 9). Parts owned by later modules are null until those exist.

export interface Element {
  id: string;
  type: "face" | "text" | "subject" | "background" | "layer";
  label: string;
  text?: string;
  box: [number, number, number, number]; // x0, y0, x1, y1 in the 1280x720 frame
  mask_rle: string;
  area_pct: number;
  attention_pct: number;
  density: number | null; // attention share / area share; above 1 = pulls more than its size
  predicted_rank: number;
  intent_rank: number | null;
  features: ElementFeatures | null; // M5
  top_driver: string | null;
  explanations: string[] | null; // M7: each cites a measured number
}

export interface ElementFeatures {
  local_contrast?: number | null;
  saturation_pop?: number | null;
  colour_distinctness?: number | null;
  dist_to_thirds?: number;
  dist_to_center?: number;
  safe_zone_overlap_pct?: number;
  text_contrast_ratio?: number | null;
  mobile_legible?: boolean | null;
  mobile_match_pct?: number;
}

export interface Swatch {
  hex: string;
  name: string;
  share: number;
}

export interface Frame {
  palette: Swatch[];
  balance_offset?: [number, number]; // -1..1, positive = right / bottom
  focal_points?: number;
  negative_space_pct?: number;
  text_area_pct: number;
  explanations: string[];
}

export interface LayerViews {
  composition?: {
    thirds_points: [number, number][];
    grid_attention_pct: number[][];
    centre_of_mass: [number, number];
    focal_points: { x: number; y: number; strength: number }[];
  };
  text: { element_id: string; box: [number, number, number, number]; text: string | null; contrast_ratio: number | null; mobile_legible: boolean | null; safe_zone_overlap_pct: number | null }[];
  colour: { palette: Swatch[]; frame_saturation: number };
  contrast_png: string;
  safe_zone: { boxes: [number, number, number, number][] };
}

export interface IntentCheck {
  matches: boolean;
  mismatches: { element_id: string; intended: number; predicted: number; gap: number }[];
}

export interface Fixation {
  x: number;
  y: number;
  order: number;
  element_id: string;
}

export interface AnalysisResult {
  analysis_id: string;
  session_id: string;
  saliency_model: string; // "deepgaze_iie" | "opencv_fine_grained"
  heatmap_png: string | null; // base64, transparent RGBA; null if saliency failed entirely
  image_jpg: string; // base64, the normalised 16:9 frame the models saw
  image_info: Record<string, unknown>;
  elements: Element[] | null;
  hierarchy: string[] | null;
  scanpath: Fixation[] | null;
  frame: Frame | null;
  intent_check: IntentCheck | null;
  layer_views: LayerViews | null;
  timing_ms: Record<string, number>;
  errors: string[];
  cached: boolean;
}

export interface Health {
  status: "ok" | "degraded";
  saliency: { loaded: string[]; unavailable: Record<string, string>; fallback_active: boolean; elements: { loaded: string[]; unavailable: Record<string, string> }; scanpath: { loaded: string[]; unavailable: Record<string, string> } };
}
