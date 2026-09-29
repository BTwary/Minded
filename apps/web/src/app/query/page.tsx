"use client";

import React, { useEffect, useState } from "react";
import {
  BarChart2,
  CheckCircle2,
  Clock,
  Code2,
  Database,
  Download,
  Play,
  RefreshCw,
  Sparkles,
  Table,
} from "lucide-react";
import { fetchDatasets } from "../../lib/api";
import { Dataset } from "../../types";
import PlotlyChart from "../../components/PlotlyChart";

const SAMPLE_QUERIES = [
  {
    name: "Regional Revenue & Margin",
    sql: "SELECT region, COUNT(*) as tx_count, ROUND(SUM(revenue), 2) as total_revenue, ROUND(AVG(discount_rate)*100, 2) as avg_discount_pct FROM sales GROUP BY region ORDER BY total_revenue DESC",
  },
  {
    name: "Product Unit Economics",
    sql: "SELECT product_id, category, unit_cost, list_price, ROUND((list_price - unit_cost)/list_price * 100, 1) as margin_pct FROM products ORDER BY margin_pct DESC",
  },
  {
    name: "Monthly Revenue Trajectory",
    sql: "SELECT strftime(order_date, '%Y-%m') as month, region, ROUND(SUM(revenue), 2) as monthly_revenue FROM sales GROUP BY 1, 2 ORDER BY 1, 2",
  },
  {
    name: "Customer Segments",
    sql: "SELECT segment, state, COUNT(*) as customer_count FROM customers GROUP BY segment, state ORDER BY customer_count DESC LIMIT 15",
  },
];

export default function QueryWorkbenchPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [sql, setSql] = useState(SAMPLE_QUERIES[0].sql);
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<{
    columns: string[];
    rows: any[];
    execution_time_ms: number;
    row_count: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"table" | "chart">("table");
  const [activeProjectId, setActiveProjectId] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const ds = await fetchDatasets();
        setDatasets(ds);
        const projectRes = await fetch("http://127.0.0.1:8000/api/v1/projects", { credentials: "include" }).catch(() => null);
        if (projectRes?.ok) {
          const projects = await projectRes.json();
          setActiveProjectId(projects?.[0]?.id || "");
        }
      } catch (e) {
        console.error(e);
      }
    }
    load();
  }, []);

  const handleRunQuery = async () => {
    if (!sql.trim()) return;
    if (!activeProjectId) {
      setError("No analytical workspace is available.");
      return;
    }
    setIsRunning(true);
    setError(null);
    const start = performance.now();

    try {
      const res = await fetch("http://127.0.0.1:8000/api/v1/query/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sql, project_id: activeProjectId }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Query execution failed.");
      }

      const duration = Math.round(performance.now() - start);
      setResults({
        columns: data.columns || (data.rows && data.rows.length > 0 ? Object.keys(data.rows[0]) : []),
        rows: data.rows || [],
        execution_time_ms: data.execution_time_ms || duration,
        row_count: data.row_count || (data.rows ? data.rows.length : 0),
      });
    } catch (err: any) {
      setError(err.message || "Execution error in DuckDB engine.");
      setResults(null);
    } finally {
      setIsRunning(false);
    }
  };

  const handleExportCSV = () => {
    if (!results || results.rows.length === 0) return;
    const header = results.columns.join(",");
    const csvRows = results.rows.map((r) =>
      results.columns.map((col) => JSON.stringify(r[col] ?? "")).join(",")
    );
    const blob = new Blob([[header, ...csvRows].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `query_result_${Date.now()}.csv`;
    a.click();
  };

  // Generate chart data if numeric columns exist
  const getChartSpec = () => {
    if (!results || results.rows.length === 0) return null;
    const xCol = results.columns[0];
    const yCol = results.columns.find((c, idx) => idx > 0 && typeof results.rows[0][c] === "number") || results.columns[1];

    return {
      title: "Query Result Visualization",
      chart_type: "bar",
      explanation: `Plot of ${yCol} by ${xCol}`,
      plotly_figure_json: {
        data: [
          {
            type: "bar",
            x: results.rows.map((r) => r[xCol]),
            y: results.rows.map((r) => r[yCol]),
            marker: { color: "#06b6d4" },
          },
        ],
      },
    };
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold text-cyan-400 mb-1">
            <Code2 className="w-4 h-4" />
            <span>INTERACTIVE SQL WORKBENCH (DUCKDB COLUMNAR OLAP)</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">SQL Query Workbench</h1>
          <p className="text-xs text-slate-400 mt-1">
            Execute sub-50ms analytical SQL queries directly on active Parquet tables with AST security &amp; instant visualizer.
          </p>
        </div>

        {/* Active Tables Pill */}
        <div className="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-xl border border-slate-800 text-xs">
          <Database className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-400">Available Tables:</span>
          <span className="font-mono text-cyan-300 font-semibold">
            {datasets.map((d) => d.name).join(", ") || "Loading..."}
          </span>
        </div>
      </div>

      {/* Query Editor Box */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-xl">
        {/* Sample query shortcuts */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-bold uppercase text-slate-400 font-mono">Presets:</span>
          {SAMPLE_QUERIES.map((sq) => (
            <button
              key={sq.name}
              onClick={() => setSql(sq.sql)}
              className="text-xs bg-slate-950 hover:bg-slate-800 text-slate-300 hover:text-white px-2.5 py-1 rounded-lg border border-slate-800 transition-all font-mono"
            >
              {sq.name}
            </button>
          ))}
        </div>

        {/* SQL Textarea */}
        <div className="relative">
          <textarea
            value={sql}
            onChange={(e) => setSql(e.target.value)}
            rows={4}
            placeholder="SELECT * FROM sales LIMIT 20;"
            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-4 text-xs font-mono text-cyan-300 placeholder-slate-600 focus:outline-none focus:border-cyan-500 leading-relaxed resize-y"
          />
        </div>

        {/* Toolbar */}
        <div className="flex items-center justify-between pt-1">
          <span className="text-[11px] text-slate-400 font-mono flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>Read-Only AST AST Enforcement Active</span>
          </span>

          <button
            onClick={handleRunQuery}
            disabled={isRunning || !sql.trim()}
            className="bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 disabled:opacity-50 text-white text-xs font-semibold px-6 py-2.5 rounded-xl shadow-lg shadow-cyan-500/20 flex items-center space-x-2 transition-all"
          >
            {isRunning ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Executing in DuckDB...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Run Query (Ctrl + Enter)</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-xs text-rose-300 font-mono">
          {error}
        </div>
      )}

      {/* Results Container */}
      {results && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4 shadow-xl">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-xs font-mono text-slate-400">
                <Clock className="w-3.5 h-3.5 text-cyan-400" />
                <span>Latency: <strong className="text-white">{results.execution_time_ms}ms</strong></span>
              </div>
              <div className="flex items-center space-x-2 text-xs font-mono text-slate-400">
                <span>Rows: <strong className="text-white">{results.row_count.toLocaleString()}</strong></span>
              </div>
            </div>

            {/* View switcher & Export */}
            <div className="flex items-center space-x-3">
              <div className="bg-slate-950 p-1 rounded-xl border border-slate-800 flex items-center space-x-1">
                <button
                  onClick={() => setActiveTab("table")}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition-all ${
                    activeTab === "table" ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40" : "text-slate-400 hover:text-white"
                  }`}
                >
                  <Table className="w-3.5 h-3.5" />
                  <span>Table</span>
                </button>
                <button
                  onClick={() => setActiveTab("chart")}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition-all ${
                    activeTab === "chart" ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40" : "text-slate-400 hover:text-white"
                  }`}
                >
                  <BarChart2 className="w-3.5 h-3.5" />
                  <span>Visualizer</span>
                </button>
              </div>

              <button
                onClick={handleExportCSV}
                className="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold px-3 py-1.5 rounded-xl border border-slate-700 flex items-center space-x-1.5 transition-all"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Export CSV</span>
              </button>
            </div>
          </div>

          {/* Table Tab */}
          {activeTab === "table" ? (
            <div className="overflow-x-auto max-h-96 rounded-xl border border-slate-800 bg-slate-950">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-900/90 text-slate-400 uppercase text-[10px] font-mono sticky top-0 border-b border-slate-800">
                  <tr>
                    {results.columns.map((c) => (
                      <th key={c} className="px-4 py-2.5 font-bold tracking-wider">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                  {results.rows.slice(0, 100).map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-900/50 transition-colors">
                      {results.columns.map((col) => (
                        <td key={col} className="px-4 py-2 text-slate-300">
                          {typeof row[col] === "number" ? row[col].toLocaleString() : String(row[col] ?? "")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            /* Chart Tab */
            <div>
              {getChartSpec() ? (
                <PlotlyChart spec={getChartSpec()!} />
              ) : (
                <div className="p-8 text-center text-xs text-slate-400 font-mono">
                  No numeric columns available to plot.
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
