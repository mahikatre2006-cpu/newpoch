import { useState } from "react";
import { analyze } from "./api/client";
import type { AnalysisResult } from "./api/types";
import Start from "./pages/Start";
import Studio from "./pages/Studio";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(f: File) {
    setBusy(true);
    setError(null);
    try {
      setResult(await analyze({ image: f, filename: f.name }));
      setFile(f);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  /** Same image with the creator's intended order: the server answers from its cache, only the intent check is new. */
  async function checkIntent(intent: string[] | null) {
    if (!file || !result) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await analyze({ image: file, filename: file.name, intent: intent ?? undefined, sessionId: result.session_id }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Check failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-8 text-sm font-semibold tracking-wide text-stone-500 dark:text-stone-400">Thumbnail Attention Studio</header>
      {error && <div role="alert" className="mb-4 rounded-lg border border-red-300 bg-red-50 px-4 py-2 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200">{error}</div>}
      <main>
        {result && file ? (
          <Studio key={result.analysis_id} result={result} filename={file.name} busy={busy} onNew={() => { setResult(null); setFile(null); }} onIntent={checkIntent} />
        ) : (
          <Start busy={busy} onFile={run} onError={setError} />
        )}
      </main>
    </div>
  );
}
