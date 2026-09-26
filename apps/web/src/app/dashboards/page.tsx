"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Database, LayoutDashboard, Plus, RefreshCw, Sparkles, UploadCloud } from "lucide-react";
import { fetchDatasets, fetchDatasetData } from "../../lib/api";
import { Dataset } from "../../types";
import PlotlyChart from "../../components/PlotlyChart";

export default function DashboardsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [widgets, setWidgets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const ds = await fetchDatasets();
        setDatasets(ds);

        if (ds.length > 0) {
          // Dynamically build widgets from the active primary dataset
          const primary = ds[0];
          const rawRows = (await fetchDatasetData(primary.id, primary.current_version, 100)) as any;
          const rows = rawRows?.rows || [];

          // Group by category/region if exists
          const catMap: Record<string, number> = {};
          const dateMap: Record<string, number> = {};

          rows.forEach((r: any) => {
            const catKey = r.region || r.category || r.channel || Object.values(r)[1] || "Default";
            const val = Number(r.revenue || r.amount || r.price || r.quantity || 1);
            catMap[String(catKey)] = (catMap[String(catKey)] || 0) + val;

            if (r.order_date || r.date) {
              const dStr = String(r.order_date || r.date).substring(0, 7);
              dateMap[dStr] = (dateMap[dStr] || 0) + val;
            }
          });

          const generatedWidgets: any[] = [];

          if (Object.keys(catMap).length > 0) {
            generatedWidgets.push({
              title: `${primary.name.toUpperCase()} - Category & Territorial Distribution`,
              chart_type: "bar",
              plotly_figure_json: {
                data: [
                  {
                    type: "bar",
                    x: Object.keys(catMap),
                    y: Object.values(catMap),
                  },
                ],
              },
              explanation: `Computed in real-time from active dataset '${primary.name}'.`,
            });
          }

          if (Object.keys(dateMap).length > 0) {
            generatedWidgets.push({
              title: `${primary.name.toUpperCase()} - Timeline Trajectory`,
              chart_type: "line",
              plotly_figure_json: {
                data: [
                  {
                    type: "scatter",
                    mode: "lines+markers",
                    x: Object.keys(dateMap),
                    y: Object.values(dateMap),
                  },
                ],
              },
              explanation: `Monthly volume progression for '${primary.name}'.`,
            });
          }

          setWidgets(generatedWidgets);
        } else {
          setWidgets([]);
        }
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold text-cyan-400 mb-1">
            <LayoutDashboard className="w-4 h-4" />
            <span>REAL-TIME BUSINESS INTELLIGENCE</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Executive Dashboards</h1>
          <p className="text-xs text-slate-400 mt-1">
            Live interactive analytical widgets computed dynamically from active DuckDB columnar tables.
          </p>
        </div>
      </div>

      {/* Widgets Grid or Empty State */}
      {widgets.length === 0 ? (
        <div className="p-12 rounded-2xl bg-slate-900/60 border border-dashed border-slate-800 text-center space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mx-auto text-cyan-400">
            <LayoutDashboard className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-white">No Active Dashboards Available</h3>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              Upload a dataset or load sample data to automatically generate real-time BI widgets.
            </p>
          </div>
          <div className="pt-2">
            <Link
              href="/datasets"
              className="inline-flex items-center space-x-2 bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white text-xs font-semibold px-4 py-2.5 rounded-xl shadow-lg shadow-cyan-500/20 transition-all"
            >
              <UploadCloud className="w-4 h-4" />
              <span>Go to Datasets</span>
            </Link>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {widgets.map((w, idx) => (
            <PlotlyChart key={idx} spec={w} />
          ))}
        </div>
      )}
    </div>
  );
}
