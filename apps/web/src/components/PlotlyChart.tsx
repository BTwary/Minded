"use client";

import React, { useMemo } from "react";
import { BarChart3, LineChart, ScatterChart } from "lucide-react";

interface Props {
  spec?: any;
  height?: number;
}

type Point = { x: unknown; y: number; series: string; ciLow?: number | null; ciHigh?: number | null; startValue?: number | null; endValue?: number | null };
type BoxSummary = { label: string; p10: number; p25: number; p50: number; p75: number; p90: number };

function object(value: unknown): Record<string, any> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, any>
    : null;
}

function finiteNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value);
}

export default function PlotlyChart({ spec, height = 300 }: Props) {
  const normalized = useMemo(() => {
    const cfg = object(spec);
    if (!cfg) return null;
    const plotly = object(cfg.plotly_figure_json);
    const traces: any[] = Array.isArray(plotly?.data) ? plotly!.data : [];
    const rawData: any[] = Array.isArray(cfg.data) ? cfg.data : [];
    const rawRows: any[] = Array.isArray(cfg.data_rows) ? cfg.data_rows : [];
    const xField = cfg.x_axis?.field || cfg.x_field || cfg.xField;
    const yField = cfg.y_axis?.field || cfg.y_field || cfg.yField;
    const xLabel = cfg.x_axis?.label || cfg.x_axis?.title || cfg.x_label || xField || "Category";
    const yLabel = cfg.y_axis?.label || cfg.y_axis?.title || cfg.y_label || yField || "Value";
    const seriesField = cfg.series_field || cfg.seriesField;
    const chartType = String(cfg.chart_type || cfg.type || traces[0]?.type || (rawData.length ? "bar" : "table")).toLowerCase();
    const metadata = object(cfg.metadata) || {};
    const ciLowField = String(metadata.ci_low_field || "ci_low");
    const ciHighField = String(metadata.ci_high_field || "ci_high");
    const hasPersistedIntervals = [...rawData, ...rawRows].some((row) =>
      finiteNumber(row?.[ciLowField]) !== null && finiteNumber(row?.[ciHighField]) !== null
    );
    const showCI = metadata.shows_ci === true || hasPersistedIntervals;
    const makePoint = (x: unknown, y: number, series: string, row?: Record<string, any>): Point => ({
      x, y, series,
      ciLow: showCI && row ? finiteNumber(row[ciLowField]) : null,
      ciHigh: showCI && row ? finiteNumber(row[ciHighField]) : null,
      startValue: row ? finiteNumber(row.start) : null,
      endValue: row ? finiteNumber(row.end) : null,
    });
    const points: Point[] = [];

    if (rawData.length && typeof rawData[0] === "object" && !Array.isArray(rawData[0])) {
      const keys = Object.keys(rawData[0]);
      const xf = xField || keys[0];
      const yf = yField || keys.find((k) => k !== xf && rawData.some((r) => finiteNumber(r?.[k]) !== null));
      if (xf && yf) {
        for (const row of rawData) {
          const y = finiteNumber(row?.[yf]);
          if (y !== null && row?.[xf] !== undefined && row?.[xf] !== null) {
            points.push(makePoint(row[xf], y, seriesField ? String(row?.[seriesField] ?? "Series 1") : "Series 1", row));
          }
        }
      }
    }

    // Backwards compatibility with Plotly's {data: [{x: [], y: [], name: ...}]} shape.
    if (!points.length && traces.length) {
      for (const trace of traces) {
        const xs = Array.isArray(trace?.x) ? trace.x : [];
        const ys = Array.isArray(trace?.y) ? trace.y : [];
        const name = String(trace?.name || `Series ${points.length + 1}`);
        for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
          const y = finiteNumber(ys[i]);
          if (y !== null && xs[i] !== null && xs[i] !== undefined) points.push(makePoint(xs[i], y, name));
        }
      }
    }

    // Generic row-based fallback for older chart payloads.
    if (!points.length && rawRows.length && typeof rawRows[0] === "object" && !Array.isArray(rawRows[0])) {
      const keys = Object.keys(rawRows[0]);
      const xf = xField || keys[0];
      const yf = yField || keys.find((k) => k !== xf && rawRows.some((r) => finiteNumber(r?.[k]) !== null));
      if (xf && yf) {
        for (const row of rawRows) {
          const y = finiteNumber(row?.[yf]);
          if (y !== null && row?.[xf] !== null && row?.[xf] !== undefined) {
            points.push(makePoint(row[xf], y, seriesField ? String(row?.[seriesField] ?? "Series 1") : "Series 1", row));
          }
        }
      }
    }

    const boxSummaries: BoxSummary[] = chartType.includes("box") ? rawData.flatMap((row) => {
      const p10 = finiteNumber(row?.p10);
      const p25 = finiteNumber(row?.p25);
      const p50 = finiteNumber(row?.p50);
      const p75 = finiteNumber(row?.p75);
      const p90 = finiteNumber(row?.p90);
      if ([p10, p25, p50, p75, p90].every((v) => v !== null) && p10! <= p25! && p25! <= p50! && p50! <= p75! && p75! <= p90!) {
        return [{ label: String(row?.label ?? row?.category ?? "Distribution"), p10: p10!, p25: p25!, p50: p50!, p75: p75!, p90: p90! }];
      }
      return [];
    }) : [];
    const xAxisType = String(cfg.x_axis?.type || "").toLowerCase();
    const yAxisFormat = String(cfg.y_axis?.format || "").toLowerCase();
    return {
      chartType,
      title: String(cfg.title || cfg.name || "Analytical Breakdown"),
      explanation: typeof cfg.explanation === "string" ? cfg.explanation : "",
      xLabel: String(xLabel),
      yLabel: String(yLabel),
      points,
      tableRows: rawRows,
      showCI,
      boxSummaries,
      xAxisType,
      yAxisFormat,
    };
  }, [spec]);

  if (!normalized) {
    return <div className="flex h-48 items-center justify-center rounded-xl border border-slate-800 bg-slate-950/40 text-xs text-slate-400">No valid visualization specification available.</div>;
  }

  const { chartType, title, explanation, xLabel, yLabel, points: rawPoints, tableRows, showCI, boxSummaries, xAxisType, yAxisFormat } = normalized;
  const points: Point[] = Array.isArray(rawPoints) ? rawPoints as Point[] : [];
  const boxes: BoxSummary[] = Array.isArray(boxSummaries) ? boxSummaries as BoxSummary[] : [];
  const isLine = chartType.includes("line") || chartType === "trend";
  const isScatter = chartType.includes("scatter");
  const isHistogram = chartType === "histogram";
  const isDonut = chartType === "donut" || chartType === "pie";
  const isBoxplot = chartType.includes("box");
  const isWaterfall = chartType === "waterfall";
  const isStackedBar = chartType === "stacked_bar";
  const isBar = !isDonut && !isBoxplot && !isWaterfall && (chartType.includes("bar") || chartType === "ranking" || isHistogram || (!isLine && !isScatter && points.length > 0));
  const seriesNames: string[] = Array.from(new Set<string>(points.map((p: Point) => p.series)));
  const xValues: string[] = Array.from(new Set<string>(points.map((p: Point) => String(p.x))));
  const parsedDates = xAxisType !== "nominal" && points.length > 0 && points.every((p) => typeof p.x === "string" && Number.isFinite(Date.parse(p.x)) && /[-/]/.test(p.x));
  const numericX = xAxisType !== "nominal" && points.length > 0 && points.every((p) => finiteNumber(p.x) !== null);
  const stackExtents = isStackedBar ? xValues.flatMap((category) => {
    const values = points.filter((point) => String(point.x) === category).map((point) => point.y);
    return [values.filter((value) => value > 0).reduce((a, b) => a + b, 0),
      values.filter((value) => value < 0).reduce((a, b) => a + b, 0)];
  }) : [];
  const chartValues = [
    ...points.flatMap((p) => [
      p.y,
      ...(p.ciLow !== null && p.ciLow !== undefined ? [p.ciLow] : []),
      ...(p.ciHigh !== null && p.ciHigh !== undefined ? [p.ciHigh] : []),
      ...(isWaterfall ? [p.startValue ?? 0, p.endValue ?? p.y] : []),
    ]),
    ...stackExtents,
    ...(isBoxplot ? boxes.flatMap((b) => [b.p10, b.p25, b.p50, b.p75, b.p90]) : []),
  ];
  const minY = Math.min(...chartValues, (isBar || isWaterfall) ? 0 : Number.POSITIVE_INFINITY);
  const maxY = Math.max(...chartValues, (isBar || isWaterfall) ? 0 : Number.NEGATIVE_INFINITY);
  const rangeY = maxY - minY || 1;
  const width = 820;
  const svgHeight = Math.max(180, height);
  const margin = { top: 20, right: 24, bottom: 54, left: 62 };
  const plotW = width - margin.left - margin.right;
  const plotH = svgHeight - margin.top - margin.bottom;
  const xPositions = new Map<string, number>();
  if (numericX || parsedDates) {
    const vals = points.map((p) => numericX ? Number(p.x) : Date.parse(String(p.x)));
    const minX = Math.min(...vals);
    const maxX = Math.max(...vals);
    points.forEach((p) => {
      const n = numericX ? Number(p.x) : Date.parse(String(p.x));
      xPositions.set(String(p.x), margin.left + ((maxX === minX ? 0.5 : (n - minX) / (maxX - minX)) * plotW));
    });
  } else {
    xValues.forEach((x, i) => xPositions.set(x, margin.left + (xValues.length <= 1 ? plotW / 2 : (i / (xValues.length - 1)) * plotW)));
  }
  const yPos = (v: number) => margin.top + ((maxY - v) / rangeY) * plotH;
  const colors = ["#22d3ee", "#a78bfa", "#34d399", "#fb7185", "#fbbf24", "#60a5fa", "#f472b6", "#a3e635"];
  const grouped = new Map<string, Point[]>();
  seriesNames.forEach((name) => grouped.set(name, points.filter((p) => p.series === name)));

  const tableData = tableRows.length ? tableRows : points.map((p) => ({ [xLabel]: p.x, [yLabel]: p.y, ...(seriesNames.length > 1 ? { Series: p.series } : {}) }));
  const donutPoints = points.filter((p) => p.y >= 0);
  const donutTotal = donutPoints.reduce((sum, p) => sum + p.y, 0);
  const donutEligible = points.length > 0 && points.every((p) => p.y >= 0) && donutTotal > 0;
  const renderTable = (!points.length && !boxes.length) || (!isDonut && !isBoxplot && !isWaterfall && !isLine && !isScatter && !isBar) || (isDonut && !donutEligible);
  const fmt = (value: number) => yAxisFormat === "percent" ? `${(value * 100).toFixed(1)}%` : formatNumber(value);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-5">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h4 className="flex items-center gap-2 text-sm font-semibold text-white">
            {isLine ? <LineChart className="h-4 w-4 text-cyan-400" /> : isScatter ? <ScatterChart className="h-4 w-4 text-cyan-400" /> : <BarChart3 className="h-4 w-4 text-violet-400" />}
            <span>{title}</span>
          </h4>
          {explanation && <p className="mt-1 text-xs text-slate-400">{explanation}</p>}
        </div>
        <span className="rounded border border-slate-700 bg-slate-800 px-2 py-0.5 font-mono text-[10px] uppercase text-slate-300">{chartType}</span>
      </div>

      {isDonut && donutEligible ? (
        <div className="grid grid-cols-1 items-center gap-4 sm:grid-cols-[220px_1fr]">
          <svg role="img" aria-label={`${title}. Composition of ${fmt(donutTotal)} total.`} viewBox="0 0 220 220" className="mx-auto h-[220px] w-[220px]">
            <circle cx="110" cy="110" r="69" fill="none" stroke="#334155" strokeWidth="28" />
            {(() => {
              const circumference = 2 * Math.PI * 69;
              let offset = 0;
              return donutPoints.map((point, index) => {
                const length = circumference * (point.y / donutTotal);
                const circle = <circle key={`${String(point.x)}-${index}`} cx="110" cy="110" r="69" fill="none" stroke={colors[index % colors.length]} strokeWidth="28" strokeDasharray={`${length} ${Math.max(0, circumference - length)}`} strokeDashoffset={-offset} transform="rotate(-90 110 110)"><title>{`${String(point.x)}: ${fmt(point.y)} (${(point.y / donutTotal * 100).toFixed(1)}%)`}</title></circle>;
                offset += length;
                return circle;
              });
            })()}
            <text x="110" y="105" textAnchor="middle" fontSize="11" fill="#94a3b8">Returned total</text>
            <text x="110" y="124" textAnchor="middle" fontSize="15" fontWeight="600" fill="#f8fafc">{fmt(donutTotal)}</text>
          </svg>
          <div className="space-y-2">
            {donutPoints.map((point, index) => <div key={`${String(point.x)}-${index}`} className="flex items-center justify-between gap-3 text-xs">
              <span className="inline-flex min-w-0 items-center gap-2 text-slate-300"><span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ backgroundColor: colors[index % colors.length] }} />{String(point.x)}</span>
              <span className="shrink-0 font-mono text-slate-100">{fmt(point.y)} · {(point.y / donutTotal * 100).toFixed(1)}%</span>
            </div>)}
          </div>
        </div>
      ) : renderTable ? (
        <div className="overflow-x-auto">
          {tableData.length ? <table className="w-full text-left text-xs"><thead className="bg-slate-900 text-slate-400"><tr>{Object.keys(tableData[0] || {}).map((k) => <th key={k} className="p-2 font-mono">{k}</th>)}</tr></thead><tbody className="divide-y divide-slate-800 text-slate-300">{tableData.slice(0, 100).map((row: any, idx: number) => <tr key={idx}>{Object.values(row).map((v: any, cell) => <td key={cell} className="p-2 font-mono">{String(v ?? "")}</td>)}</tr>)}</tbody></table> : <p className="text-xs text-slate-400">The chart payload contained no usable observed points.</p>}
        </div>
      ) : (
        <div className="w-full overflow-x-auto">
          <svg role="img" aria-label={`${title}. Horizontal axis: ${xLabel}. Vertical axis: ${yLabel}.`} viewBox={`0 0 ${width} ${svgHeight}`} className="w-full min-w-[520px]" style={{ height: svgHeight }}>
            {Array.from({ length: 5 }, (_, i) => {
              const val = minY + (rangeY * i) / 4;
              const y = yPos(val);
              return <g key={i}><line x1={margin.left} y1={y} x2={width - margin.right} y2={y} stroke="#334155" strokeDasharray="3 4" /><text x={margin.left - 8} y={y + 4} textAnchor="end" fontSize="11" fill="#94a3b8">{fmt(val)}</text></g>;
            })}
            <line x1={margin.left} y1={margin.top} x2={margin.left} y2={svgHeight - margin.bottom} stroke="#64748b" />
            <line x1={margin.left} y1={svgHeight - margin.bottom} x2={width - margin.right} y2={svgHeight - margin.bottom} stroke="#64748b" />
            {isBoxplot && boxes.length ? (() => {
              const band = plotW / Math.max(1, boxes.length);
              const boxWidth = Math.min(54, band * 0.45);
              return boxes.map((box, index) => {
                const x = xPositions.get(box.label) ?? margin.left + band * (index + 0.5);
                const y10 = yPos(box.p10), y25 = yPos(box.p25), y50 = yPos(box.p50), y75 = yPos(box.p75), y90 = yPos(box.p90);
                return <g key={box.label} stroke={colors[index % colors.length]}>
                  <line x1={x} y1={y90} x2={x} y2={y10} strokeWidth="2" />
                  <line x1={x - 8} y1={y90} x2={x + 8} y2={y90} strokeWidth="2" />
                  <line x1={x - 8} y1={y10} x2={x + 8} y2={y10} strokeWidth="2" />
                  <rect x={x - boxWidth / 2} y={y75} width={boxWidth} height={Math.max(1, y25 - y75)} fill={colors[index % colors.length]} fillOpacity="0.25" strokeWidth="2" />
                  <line x1={x - boxWidth / 2} y1={y50} x2={x + boxWidth / 2} y2={y50} strokeWidth="3" />
                  <title>{`${box.label}: p10=${fmt(box.p10)}, p25=${fmt(box.p25)}, median=${fmt(box.p50)}, p75=${fmt(box.p75)}, p90=${fmt(box.p90)}`}</title>
                </g>;
              });
            })() : isWaterfall ? (() => {
              const categories = xValues;
              const band = plotW / Math.max(1, categories.length);
              const barW = Math.max(4, band * 0.58);
              return points.map((point, index) => {
                const x = xPositions.get(String(point.x)) ?? margin.left + band * (index + 0.5);
                const start = point.startValue ?? 0;
                const end = point.endValue ?? point.y;
                const y1 = yPos(start), y2 = yPos(end);
                const color = point.y >= 0 ? colors[2] : colors[3];
                return <g key={`${String(point.x)}-${index}`}>
                  <rect x={x - barW / 2} y={Math.min(y1, y2)} width={barW} height={Math.max(1, Math.abs(y2 - y1))} rx="2" fill={color}>
                    <title>{`${String(point.x)}: delta ${fmt(point.y)}; subtotal ${fmt(end)}`}</title>
                  </rect>
                </g>;
              });
            })() : isBar ? (() => {
              const categories = xValues;
              const band = plotW / Math.max(1, categories.length);
              const barW = isStackedBar ? Math.max(3, band * 0.68) : isHistogram ? Math.max(3, band * 0.94) : Math.max(3, (band * 0.72) / Math.max(1, seriesNames.length));
              if (isStackedBar) {
                return categories.flatMap((x, ci) => {
                  let positiveBase = 0;
                  let negativeBase = 0;
                  const xCenter = xPositions.get(x) ?? margin.left + ci * band + band / 2;
                  return seriesNames.map((series, si) => {
                    const p = (grouped.get(series) || []).find((row) => String(row.x) === x);
                    if (!p) return null;
                    const start = p.y >= 0 ? positiveBase : negativeBase;
                    const end = start + p.y;
                    if (p.y >= 0) positiveBase = end;
                    else negativeBase = end;
                    const yStart = yPos(start);
                    const yEnd = yPos(end);
                    const bx = xCenter - barW / 2;
                    return <g key={`${x}-${series}`}>
                      <rect x={bx} y={Math.min(yStart, yEnd)} width={barW - 1} height={Math.max(1, Math.abs(yStart - yEnd))} rx="1" fill={colors[si % colors.length]}>
                        <title>{`${series} · ${x}: ${fmt(p.y)}; stacked subtotal ${fmt(end)}`}</title>
                      </rect>
                    </g>;
                  });
                });
              }
              return categories.flatMap((x, ci) => seriesNames.map((series, si) => {
                const p = (grouped.get(series) || []).find((row) => String(row.x) === x);
                if (!p) return null;
                const xCenter = xPositions.get(x) ?? margin.left + ci * band + band / 2;
                const yZero = yPos(0);
                const yValue = yPos(p.y);
                const bx = xCenter - (barW * seriesNames.length) / 2 + si * barW;
                return <g key={`${x}-${series}`}>
                  <rect x={bx} y={Math.min(yValue, yZero)} width={barW - 1} height={Math.max(1, Math.abs(yZero - yValue))} rx="2" fill={colors[si % colors.length]}>
                    <title>{`${series} · ${x}: ${fmt(p.y)}${p.ciLow !== null && p.ciLow !== undefined && p.ciHigh !== null && p.ciHigh !== undefined ? ` (95% CI ${fmt(p.ciLow)} to ${fmt(p.ciHigh)})` : ""}`}</title>
                  </rect>
                  {showCI && p.ciLow !== null && p.ciLow !== undefined && p.ciHigh !== null && p.ciHigh !== undefined && (() => {
                    const xErr = bx + (barW - 1) / 2;
                    const yLow = yPos(p.ciLow);
                    const yHigh = yPos(p.ciHigh);
                    return <g stroke={colors[si % colors.length]} strokeWidth="1.6">
                      <line x1={xErr} y1={yHigh} x2={xErr} y2={yLow} />
                      <line x1={xErr - 4} y1={yHigh} x2={xErr + 4} y2={yHigh} />
                      <line x1={xErr - 4} y1={yLow} x2={xErr + 4} y2={yLow} />
                    </g>;
                  })()}
                </g>;
              }));
            })() : grouped.size > 0 ? seriesNames.flatMap((series, si) => {
              const sorted = [...(grouped.get(series) || [])].sort((a, b) => {
                const xa = xPositions.get(String(a.x)) || 0;
                const xb = xPositions.get(String(b.x)) || 0;
                return xa - xb;
              });
              const path = sorted.map((p, i) => `${i ? "L" : "M"}${xPositions.get(String(p.x)) ?? margin.left},${yPos(p.y)}`).join(" ");
              return [
                ...(isLine ? [<path key={`line-${series}`} d={path} fill="none" stroke={colors[si % colors.length]} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />] : []),
                ...sorted.map((p, idx) => {
                  const cx = xPositions.get(String(p.x)) ?? margin.left;
                  const interval = showCI && p.ciLow !== null && p.ciLow !== undefined && p.ciHigh !== null && p.ciHigh !== undefined;
                  return <g key={`${series}-${idx}`}>
                    {interval && <g stroke={colors[si % colors.length]} strokeWidth="1.5">
                      <line x1={cx} y1={yPos(p.ciHigh!)} x2={cx} y2={yPos(p.ciLow!)} />
                      <line x1={cx - 4} y1={yPos(p.ciHigh!)} x2={cx + 4} y2={yPos(p.ciHigh!)} />
                      <line x1={cx - 4} y1={yPos(p.ciLow!)} x2={cx + 4} y2={yPos(p.ciLow!)} />
                    </g>}
                    <circle cx={cx} cy={yPos(p.y)} r={isScatter ? 4 : 2.8} fill={colors[si % colors.length]} stroke="#0f172a" strokeWidth="1"><title>{`${series} · ${String(p.x)}: ${formatNumber(p.y)}${interval ? ` (95% CI ${formatNumber(p.ciLow!)} to ${formatNumber(p.ciHigh!)})` : ""}`}</title></circle>
                  </g>;
                }),
              ];
            }) : null}
            {(isBoxplot ? boxes.map((b) => b.label) : xValues).filter((x, i, all) => all.indexOf(x) === i && (all.length <= 10 || i % Math.ceil(all.length / 10) === 0 || i === all.length - 1)).map((x) => <text key={x} x={xPositions.get(x) ?? margin.left} y={svgHeight - margin.bottom + 18} textAnchor="middle" fontSize="10" fill="#cbd5e1">{x.length > 14 ? `${x.slice(0, 11)}…` : x}</text>)}
            <text x={margin.left + plotW / 2} y={svgHeight - 8} textAnchor="middle" fontSize="11" fill="#94a3b8">{xLabel}</text>
            <text transform={`translate(15 ${margin.top + plotH / 2}) rotate(-90)`} textAnchor="middle" fontSize="11" fill="#94a3b8">{yLabel}</text>
          </svg>
          {seriesNames.length > 1 && <div className="mt-2 flex flex-wrap gap-3">{seriesNames.map((name, i) => <span key={name} className="inline-flex items-center gap-1.5 text-[11px] text-slate-300"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: colors[i % colors.length] }} />{name}</span>)}</div>}
        </div>
      )}
    </div>
  );
}
