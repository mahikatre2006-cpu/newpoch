import { useState } from "react";
import { ablate, analyze, fetchImage, fix, getAnalysis, versions as loadVersions } from "./api/client";
import type { AblationResult, AnalysisResult, FixLayer, FixName, Version } from "./api/types";
import type { ChangeInfo } from "./analysis/ChangeSummary";
import ErrorBoundary from "./ErrorBoundary";
import ComparePage from "./pages/ComparePage";
import EditorPage from "./pages/EditorPage";
import Start from "./pages/Start";
import Studio from "./pages/Studio";

interface EditorStart {
  image: File | null;
  layers?: FixLayer[];
  sessionId?: string;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [versions, setVersions] = useState<Version[]>([]);
  const [change, setChange] = useState<ChangeInfo | null>(null);
  const [ablation, setAblation] = useState<AblationResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [fixBusy, setFixBusy] = useState<FixName | null>(null);
  const [ablateBusy, setAblateBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editor, setEditor] = useState<EditorStart | null>(null);
  const [comparing, setComparing] = useState(false);

  const fail = (e: unknown, fallback: string) => setError(e instanceof Error ? e.message : fallback);
  const refreshVersions = (sessionId: string) => loadVersions(sessionId).then(setVersions).catch(() => {});

  async function run(f: File) {
    setBusy(true);
    setError(null);
    try {
      const r = await analyze({ image: f, filename: f.name, saveVersion: true, label: f.name.replace(/\.[^.]+$/, "").slice(0, 60) });
      setResult(r);
      setFile(f);
      setChange(null);
      setAblation(null);
      await refreshVersions(r.session_id);
    } catch (e) {
      fail(e, "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  /** Re-check the creator's intended order on the stored analysis (so it is right for a fixed version too). */
  async function checkIntent(intent: string[] | null) {
    if (!result) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await getAnalysis(result.analysis_id, intent));
    } catch (e) {
      fail(e, "Check failed");
    } finally {
      setBusy(false);
    }
  }

  async function applyFix(name: FixName) {
    if (!result) return;
    setFixBusy(name);
    setError(null);
    try {
      const r = await fix(result.analysis_id, name, result.session_id);
      if (name === "separate_layers" && r.layers) {
        setEditor({ image: null, layers: r.layers, sessionId: result.session_id });
        return;
      }
      if (r.analysis) {
        setResult(r.analysis);
        setAblation(null);
        setChange({ title: r.label, message: r.message ?? "", rows: r.changes ?? [], orderBefore: r.hierarchy_before, orderAfter: r.hierarchy_after });
        await refreshVersions(r.analysis.session_id);
      }
    } catch (e) {
      fail(e, "The fix failed");
    } finally {
      setFixBusy(null);
    }
  }

  async function runAblation(elementId: string) {
    if (!result) return;
    setAblateBusy(elementId);
    setError(null);
    try {
      setAblation(await ablate(result.analysis_id, elementId));
    } catch (e) {
      fail(e, "Ablation failed");
    } finally {
      setAblateBusy(null);
    }
  }

  async function selectVersion(versionId: string) {
    const v = versions.find((x) => x.version_id === versionId);
    if (!v) return;
    setBusy(true);
    try {
      setResult(await getAnalysis(v.analysis_id));
      setChange(null);
      setAblation(null);
    } catch (e) {
      fail(e, "Could not open that version");
    } finally {
      setBusy(false);
    }
  }

  /** Open the current version, at full size, in the editor as a picture layer. */
  async function editCurrent() {
    if (!result) return;
    try {
      const blob = await fetchImage(result.analysis_id);
      const name = versions.find((v) => v.analysis_id === result.analysis_id)?.label ?? "thumbnail";
      setEditor({ image: new File([blob], `${name}.png`, { type: "image/png" }), sessionId: result.session_id });
    } catch (e) {
      fail(e, "Could not open the editor");
    }
  }

  const leaveEditor = () => {
    const sid = editor?.sessionId ?? result?.session_id;
    setEditor(null);
    if (sid) void refreshVersions(sid);
  };

  const title = result ? versions.find((v) => v.analysis_id === result.analysis_id)?.label ?? file?.name ?? "Thumbnail" : "";

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-8 text-sm font-semibold tracking-wide text-stone-500 dark:text-stone-400">Thumbnail Attention Studio</header>
      {error && <div role="alert" className="mb-4 rounded-lg border border-red-300 bg-red-50 px-4 py-2 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200">{error}</div>}
      <main>
        <ErrorBoundary>
          {editor ? (
            <EditorPage initialImage={editor.image} initialLayers={editor.layers} sessionId={editor.sessionId} onBack={leaveEditor} />
          ) : comparing ? (
            <ComparePage versions={versions} onBack={() => setComparing(false)} />
          ) : result ? (
            <Studio
              key={result.analysis_id} result={result} title={title} busy={busy} versions={versions} change={change} ablation={ablation}
              fixBusy={fixBusy} ablateBusy={ablateBusy}
              onNew={() => { setResult(null); setFile(null); setVersions([]); setChange(null); setAblation(null); }}
              onIntent={checkIntent} onEdit={editCurrent} onFix={applyFix} onAblate={runAblation} onCloseAblation={() => setAblation(null)}
              onDismissChange={() => setChange(null)} onSelectVersion={selectVersion} onCompare={() => setComparing(true)}
            />
          ) : (
            <Start busy={busy} onFile={run} onError={setError} onOpenEditor={() => setEditor({ image: null })} />
          )}
        </ErrorBoundary>
      </main>
    </div>
  );
}
