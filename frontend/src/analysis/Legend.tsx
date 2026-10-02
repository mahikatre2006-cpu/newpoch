// Inferno stops, the same colour map the server uses, so the legend matches the overlay.
const INFERNO = "linear-gradient(to right, #000004, #420a68, #932667, #dd513a, #fca50a, #fcffa4)";

export default function Legend() {
  return (
    <div className="w-64 max-w-full text-xs text-stone-600 dark:text-stone-400" aria-label="Heatmap colour scale">
      <div className="h-2.5 rounded-full ring-1 ring-black/10" style={{ background: INFERNO }} />
      <div className="mt-1 flex justify-between">
        <span>Less attention</span>
        <span>More attention</span>
      </div>
    </div>
  );
}
