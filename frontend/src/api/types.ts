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
  version_id?: string | null;
}

export interface Health {
  status: "ok" | "degraded";
  saliency: { loaded: string[]; unavailable: Record<string, string>; fallback_active: boolean; elements: { loaded: string[]; unavailable: Record<string, string> }; scanpath: { loaded: string[]; unavailable: Record<string, string> } };
}

// ---- M8 ablation, M10 fixes, M11 versions ----
export interface AblationFlow {
  element_id: string;
  label: string;
  before_pct: number;
  after_pct: number;
  delta_pct: number;
}

export interface AblationResult {
  removed: { element_id: string; label: string; attention_pct: number; method: "inpaint" | "blur_fill" };
  flow: AblationFlow[];
  ablated_heatmap_png: string;
  ablated_image_jpg: string;
  message: string;
  saliency_model: string;
}

export interface Version {
  version_id: string;
  session_id: string;
  analysis_id: string;
  parent_id: string | null;
  label: string;
  kind: "upload" | "edit" | "fix";
  created_at: string;
}

export interface ChangeRow {
  label: string;
  type: string;
  a_id: string | null;
  b_id: string | null;
  a_pct: number | null;
  b_pct: number | null;
  delta_pct: number | null;
  status: "matched" | "removed" | "new";
}

export type FixName = "focus_subject" | "fix_text" | "enhance" | "separate_layers";

export interface FixLayer {
  name: string;
  type: "image" | "subject";
  x: number;
  y: number;
  width: number;
  height: number;
  mime: string;
  data: string; // base64
}

export interface FixResult {
  fix: FixName;
  label: string;
  version?: Version;
  analysis: AnalysisResult | null;
  changes?: ChangeRow[];
  hierarchy_before?: string[];
  hierarchy_after?: string[];
  message?: string;
  layers?: FixLayer[];
}

export interface CompareResult {
  a: { version: Version; image_jpg: string; heatmap_png: string; saliency_model: string };
  b: { version: Version; image_jpg: string; heatmap_png: string; saliency_model: string };
  rows: ChangeRow[];
  hierarchy_a: string[];
  hierarchy_b: string[];
}
