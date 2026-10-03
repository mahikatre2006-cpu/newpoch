import { Component, type ErrorInfo, type ReactNode } from "react";

/** A rendering error must never leave a blank page: say so, keep the message for the bug report, offer a reload. */
export default class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("UI error:", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-900 dark:border-red-800 dark:bg-red-950/40 dark:text-red-100">
        <p className="font-medium">Something went wrong while showing this result.</p>
        <p className="mt-1 break-words text-xs opacity-80">{this.state.error.message}</p>
        <button type="button" onClick={() => window.location.reload()} className="mt-3 rounded-lg border border-red-400 px-3 py-1.5 font-medium hover:bg-red-100 dark:hover:bg-red-900/40">Reload</button>
      </div>
    );
  }
}
