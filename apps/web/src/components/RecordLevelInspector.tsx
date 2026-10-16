"use client";

import React, { useMemo, useState } from "react";
import {
  CheckSquare,
  ChevronLeft,
  ChevronRight,
  Database,
  Filter,
  Layers,
  Search,
  Square,
  Table,
} from "lucide-react";

interface RecordLevelInspectorProps {
  rows: Array<Record<string, any>>;
  columns?: string[];
  totalRowCount?: number;
  onScopeChange?: (selectedRows: Array<Record<string, any>>, scopeDescription: string) => void;
}

export default function RecordLevelInspector({
  rows = [],
  columns,
  totalRowCount,
  onScopeChange,
}: RecordLevelInspectorProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
  const [page, setPage] = useState(0);
  const pageSize = 15;

  const resolvedColumns = useMemo(() => {
    if (columns && columns.length > 0) return columns;
    if (rows.length > 0) return Object.keys(rows[0]);
    return [];
  }, [columns, rows]);

  const filteredRows = useMemo(() => {
    if (!searchTerm.trim()) return rows;
    const term = searchTerm.toLowerCase();
    return rows.filter((r) =>
      Object.values(r).some((val) => String(val).toLowerCase().includes(term))
    );
  }, [rows, searchTerm]);

  const totalPages = Math.ceil(filteredRows.length / pageSize) || 1;
  const pageRows = useMemo(() => {
    const start = page * pageSize;
    return filteredRows.slice(start, start + pageSize);
  }, [filteredRows, page, pageSize]);

  const toggleSelectRow = (index: number) => {
    const next = new Set(selectedIndices);
    if (next.has(index)) {
      next.delete(index);
    } else {
      next.add(index);
    }
    setSelectedIndices(next);
  };

  const toggleSelectPage = () => {
    const pageGlobalIndices = pageRows.map((_, i) => page * pageSize + i);
    const allSelected = pageGlobalIndices.every((i) => selectedIndices.has(i));
    const next = new Set(selectedIndices);
    if (allSelected) {
      pageGlobalIndices.forEach((i) => next.delete(i));
    } else {
      pageGlobalIndices.forEach((i) => next.add(i));
    }
    setSelectedIndices(next);
  };

  const selectAllFiltered = () => {
    setSelectedIndices(new Set(filteredRows.map((_, i) => i)));
  };

  const clearSelection = () => {
    setSelectedIndices(new Set());
  };

  const handleApplyScope = () => {
    if (!onScopeChange) return;
    const selectedData = Array.from(selectedIndices).map((i) => filteredRows[i]).filter(Boolean);
    const desc = `${selectedData.length} records selected from ${totalRowCount || rows.length} available rows`;
    onScopeChange(selectedData, desc);
  };

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
      {/* Title & Scope Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center space-x-2 text-[10px] font-bold uppercase tracking-wider text-cyan-400">
            <Table className="w-3.5 h-3.5" />
            <span>Record-Level Preview & Inspection</span>
          </div>
          <h3 className="text-sm font-bold text-white mt-0.5">
            Sample Row Inspection
          </h3>
          <p className="text-xs text-slate-400">
            Inspect sample observations and calculation traces returned from the investigation. Analysis calculations operate on the full authoritative dataset.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <div className="bg-slate-950 border border-slate-800 px-3 py-1.5 rounded-lg text-slate-300">
            <span>Displaying {filteredRows.length.toLocaleString()} sample rows</span>
            {totalRowCount && totalRowCount > filteredRows.length && (
              <span className="text-slate-500"> ({totalRowCount.toLocaleString()} total dataset observations)</span>
            )}
          </div>
          {selectedIndices.size > 0 && (
            <div className="bg-cyan-500/10 border border-cyan-500/30 px-3 py-1.5 rounded-lg text-cyan-300 font-semibold">
              {selectedIndices.size} inspected
            </div>
          )}
        </div>
      </div>

      {/* Controls: Search, Select */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
        <div className="relative flex-1 min-w-[240px]">
          <Search className="w-3.5 h-3.5 absolute left-3 top-3 text-slate-500" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setPage(0);
            }}
            placeholder="Search sample rows (values, IDs, categories)…"
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-sans"
          />
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={toggleSelectPage}
            className="px-2.5 py-1.5 rounded-lg border border-slate-800 bg-slate-950 text-xs text-slate-300 hover:text-white hover:bg-slate-900"
          >
            Select Page
          </button>
          <button
            type="button"
            onClick={selectAllFiltered}
            className="px-2.5 py-1.5 rounded-lg border border-slate-800 bg-slate-950 text-xs text-slate-300 hover:text-white hover:bg-slate-900"
          >
            Select Displayed Rows
          </button>
          {selectedIndices.size > 0 && (
            <button
              type="button"
              onClick={clearSelection}
              className="px-2.5 py-1.5 rounded-lg border border-slate-800 bg-slate-950 text-xs text-slate-400 hover:text-white"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Paginated Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-950 overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-400 uppercase text-[10px] tracking-wider">
              <th className="p-3 w-10 text-center">
                <input
                  type="checkbox"
                  onChange={toggleSelectPage}
                  checked={pageRows.length > 0 && pageRows.every((_, i) => selectedIndices.has(page * pageSize + i))}
                  className="rounded text-cyan-500 focus:ring-cyan-400 bg-slate-900 border-slate-700"
                />
              </th>
              {resolvedColumns.map((col) => (
                <th key={col} className="p-3 font-semibold whitespace-nowrap">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
            {pageRows.length === 0 ? (
              <tr>
                <td colSpan={resolvedColumns.length + 1} className="p-8 text-center text-slate-500 font-sans text-xs">
                  No records match your filter criteria.
                </td>
              </tr>
            ) : (
              pageRows.map((row, idx) => {
                const globalIdx = page * pageSize + idx;
                const isSelected = selectedIndices.has(globalIdx);
                return (
                  <tr
                    key={globalIdx}
                    onClick={() => toggleSelectRow(globalIdx)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? "bg-cyan-950/40 text-cyan-200" : "hover:bg-slate-900/50 text-slate-300"
                    }`}
                  >
                    <td className="p-3 text-center" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleSelectRow(globalIdx)}
                        className="rounded text-cyan-500 focus:ring-cyan-400 bg-slate-900 border-slate-700"
                      />
                    </td>
                    {resolvedColumns.map((col) => (
                      <td key={col} className="p-3 whitespace-nowrap">
                        {row[col] !== undefined && row[col] !== null ? String(row[col]) : <span className="text-slate-600">NULL</span>}
                      </td>
                    ))}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="flex items-center justify-between text-xs text-slate-400 pt-1">
        <div>
          Page {page + 1} of {totalPages}
        </div>
        <div className="flex items-center space-x-1">
          <button
            type="button"
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
            className="p-1.5 rounded-lg border border-slate-800 bg-slate-950 hover:bg-slate-900 disabled:opacity-40"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            type="button"
            disabled={page >= totalPages - 1}
            onClick={() => setPage(page + 1)}
            className="p-1.5 rounded-lg border border-slate-800 bg-slate-950 hover:bg-slate-900 disabled:opacity-40"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
