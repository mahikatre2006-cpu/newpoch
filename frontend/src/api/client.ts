import type { AnalysisResult, Health } from "./types";

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
}

export async function analyze(input: AnalyzeInput): Promise<AnalysisResult> {
  const body = new FormData();
  if (input.image) body.append("image", input.image, input.filename ?? "thumbnail.png");
  if (input.videoId) body.append("video_id", input.videoId);
  if (input.layers) body.append("layers", JSON.stringify(input.layers));
  if (input.intent) body.append("intent", JSON.stringify(input.intent));
  if (input.sessionId) body.append("session_id", input.sessionId);
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
