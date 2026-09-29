"use client";

import React from "react";
import { BarChart3, LineChart, TrendingUp } from "lucide-react";

interface Props {
  spec?: any;
  height?: number;
}

export default function PlotlyChart({ spec, height = 300 }: Props) {
  if (!spec) {
    return (
      <div className="flex items-center justify-center h-48 bg-slate-950/40 rounded-xl border border-slate-800 text-slate-400 text-xs">
        No visualization specification available.
      </div>
    );
  }

  const chartType = spec.chart_type || "bar";
  const dataRows = spec.data_rows || spec.plotly_figure_json?.data || [];
  const title = spec.title || "Analytical Breakdown";

  // Check if we have Plotly figure json data
  const plotData = spec.plotly_figure_json?.data?.[0];
  const xVals = plotData?.x || [];
  const yVals = plotData?.y || [];

  const maxVal = Math.max(...(yVals.length > 0 ? yVals.map((v: any) => Number(v) || 0) : [1]));

  return (
    <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-sm font-semibold text-white flex items-center space-x-2">
            {chartType === "line" ? (
              <LineChart className="w-4 h-4 text-sky-400" />
            ) : (
              <BarChart3 className="w-4 h-4 text-indigo-400" />
            )}
            <span>{title}</span>
          </h4>
          {spec.explanation && (
            <p className="text-xs text-slate-400 mt-0.5">{spec.explanation}</p>
          )}
        </div>
        <span className="text-[11px] font-mono uppercase bg-slate-800 px-2 py-0.5 rounded text-slate-300 border border-slate-700">
          {chartType}
        </span>
      </div>

      {/* SVG Bar / Line Chart Visualization */}
      {xVals.length > 0 ? (
        <div className="space-y-3 pt-2">
          {xVals.slice(0, 8).map((label: any, idx: number) => {
            const rawVal = yVals[idx] !== undefined ? Number(yVals[idx]) : 0;
            const pct = maxVal > 0 ? Math.max(8, (rawVal / maxVal) * 100) : 10;
            return (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-300 font-sans">{String(label)}</span>
                  <span className="text-white font-semibold">${rawVal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                </div>
                <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${
                      chartType === "line"
                        ? "bg-gradient-to-r from-sky-500 to-indigo-500"
                        : "bg-gradient-to-r from-indigo-500 to-purple-500"
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900 text-slate-400">
              <tr>
                {Object.keys(dataRows[0] || {}).map((k) => (
                  <th key={k} className="p-2 font-mono">
                    {k}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-300">
              {dataRows.slice(0, 5).map((row: any, idx: number) => (
                <tr key={idx}>
                  {Object.values(row).map((v: any, cIdx) => (
                    <td key={cIdx} className="p-2 font-mono">
                      {String(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
