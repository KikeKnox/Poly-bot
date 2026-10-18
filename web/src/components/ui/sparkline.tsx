/**
 * Inline trend stroke: no axes, no tooltip, no charting library. It is a
 * figure with memory, not a chart to analyze. It is used in the StatTile
 * of the header and of Summary/Money so that a point value (equity, PnL)
 * carries the shape of how it got there.
 */
export function Sparkline({
  points,
  tone = "up",
  width = 72,
  height = 24,
}: {
  points: number[];
  tone?: "up" | "down" | "gain" | "danger";
  width?: number;
  height?: number;
}) {
  if (points.length < 2) return <svg width={width} height={height} aria-hidden />;

  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const pad = 2;
  const step = (width - pad * 2) / (points.length - 1);
  const coords = points.map((v, i) => {
    const x = pad + i * step;
    const y = pad + (1 - (v - min) / span) * (height - pad * 2);
    return [x, y] as const;
  });
  const path = coords.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area = `${path} L${coords[coords.length - 1][0].toFixed(1)},${height} L${coords[0][0].toFixed(1)},${height} Z`;
  const color =
    tone === "danger" ? "var(--danger)" : tone === "gain" ? "var(--gain)" : tone === "down" ? "var(--cyan)" : "var(--amber)";

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible" aria-hidden>
      <path d={area} fill={color} opacity={0.12} />
      <path d={path} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}
