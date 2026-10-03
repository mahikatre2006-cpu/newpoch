import type { AblationResult, AnalysisResult, CompareResult, FixName, FixResult, Health, Version } from "./types";

const BASE = import.meta.env.VITE_API_URL ?? "/api";
const UNREACHABLE = "Cannot reach the analysis server. Is the backend running on port 8000?";

async function failure(res: Response, what: string): Promise<Error> {
  const detail = await res.json().then((j) => j.detail).catch(() => null);
  return new Error(typeof detail === "string" ? detail : `${what} failed (HTTP ${res.status})`);
}

export interface AnalyzeInput {
  image?: Blob;
  filename?: string;
  videoId?: string;
  layers?: unknown;
  intent?: string[];
  sessionId?: string;
  saveVersion?: boolean; // keep this analysis as a named version
  label?: string;
}

export async function analyze(input: AnalyzeInput): Promise<AnalysisResult> {
  const body = new FormData();
  if (input.image) body.append("image", input.image, input.filename ?? "thumbnail.png");
  if (input.videoId) body.append("video_id", input.videoId);
  if (input.layers) body.append("layers", JSON.stringify(input.layers));
  if (input.intent) body.append("intent", JSON.stringify(input.intent));
  if (input.sessionId) body.append("session_id", input.sessionId);
  if (input.saveVersion) body.append("save_version", "true");
  if (input.label) body.append("label", input.label);
  let res: Response;
  try {
    res = await fetch(`${BASE}/analyze`, { method: "POST", body });
  } catch {
    throw new Error(UNREACHABLE);
  }
  if (!res.ok) throw await failure(res, "Analysis");
  return res.json();
}

export async function health(): Promise<Health> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/health`);
  } catch {
    throw new Error(UNREACHABLE);
  }
  if (!res.ok) throw await failure(res, "Health check");
  return res.json();
}

async function json<T>(path: string, init: RequestInit | undefined, what: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new Error(UNREACHABLE);
  }
  if (!res.ok) throw await failure(res, what);
  return res.json();
}

const post = (body: unknown): RequestInit => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/** Remove one element and see where its attention goes. */
export const ablate = (analysisId: string, elementId: string) => json<AblationResult>("/ablate", post({ analysis_id: analysisId, element_id: elementId }), "Ablation");

/** Apply a fix; pixel fixes come back re-analysed as a new version with the measured change. */
export const fix = (analysisId: string, name: FixName, sessionId?: string) => json<FixResult>("/fix", post({ analysis_id: analysisId, fix: name, session_id: sessionId }), "Fix");

export const versions = (sessionId: string) => json<{ versions: Version[] }>(`/versions/${encodeURIComponent(sessionId)}`, undefined, "Loading versions").then((r) => r.versions);

export const compare = (a: string, b: string) => json<CompareResult>(`/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`, undefined, "Comparison");

export const getAnalysis = (analysisId: string, intent?: string[] | null) =>
  json<AnalysisResult>(`/analysis/${encodeURIComponent(analysisId)}${intent ? `?intent=${encodeURIComponent(JSON.stringify(intent))}` : ""}`, undefined, "Loading analysis");

/** The stored 1280x720 frame of an analysis, e.g. to open a fixed version in the editor. */
export async function fetchImage(analysisId: string): Promise<Blob> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/image/${encodeURIComponent(analysisId)}`);
  } catch {
    throw new Error(UNREACHABLE);
  }
  if (!res.ok) throw await failure(res, "Loading the image");
  return res.blob();
}
