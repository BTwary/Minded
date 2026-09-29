"use client";

import React, { useEffect, useState } from "react";
import { BookOpen, Code2, Plus, Sparkles, Tag } from "lucide-react";
import { fetchMetrics, fetchGlossary } from "../../lib/api";
import { BusinessMetric, GlossaryTerm } from "../../types";

export default function GlossaryPage() {
  const [metrics, setMetrics] = useState<BusinessMetric[]>([]);
  const [terms, setTerms] = useState<GlossaryTerm[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [m, g] = await Promise.allSettled([fetchMetrics(), fetchGlossary()]);
        if (m.status === "fulfilled") setMetrics(m.value);
        if (g.status === "fulfilled") setTerms(g.value);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const defaultMetrics = [
    {
      name: "Net Revenue",
      display_name: "Net Completed Revenue",
      description: "Total gross transaction amount minus promotional discounts and refunds.",
      sql_formula: "SUM(quantity * unit_price * (1.0 - discount_rate))",
      unit: "USD ($)",
      category: "Financial",
    },
    {
      name: "Gross Margin",
      display_name: "Gross Profit Margin",
      description: "Total revenue minus product manufacturing cost divided by total revenue.",
      sql_formula: "SUM(revenue - cost) / NULLIF(SUM(revenue), 0) * 100.0",
      unit: "Percentage (%)",
      category: "Financial",
    },
  ];

  const displayMetrics = metrics.length > 0 ? metrics : defaultMetrics;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold text-cyan-400 mb-1">
            <BookOpen className="w-4 h-4" />
            <span>SEMANTIC LAYER & BUSINESS METRICS</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Business Glossary</h1>
          <p className="text-xs text-slate-400 mt-1">
            Canonical business formulas, semantic metrics, and glossary definitions for AI query alignment.
          </p>
        </div>
      </div>

      {/* Semantic Metrics List */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {displayMetrics.map((m, idx) => (
          <div key={idx} className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-sm text-white">{m.display_name}</span>
              <span className="text-[10px] font-mono bg-slate-800 text-cyan-300 px-2 py-0.5 rounded border border-slate-700">
                {m.category || "Financial"}
              </span>
            </div>
            <p className="text-xs text-slate-400">{m.description}</p>
            
            <div className="pt-2 border-t border-slate-800 space-y-1">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block flex items-center space-x-1">
                <Code2 className="w-3 h-3 text-cyan-400" />
                <span>Canonical DuckDB SQL Expression</span>
              </span>
              <pre className="p-2 rounded bg-slate-950 border border-slate-800 text-cyan-300 font-mono text-[11px] overflow-x-auto">
                {m.sql_formula}
              </pre>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
