"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Bell,
  BookOpen,
  Bot,
  Code2,
  Database,
  FileSpreadsheet,
  FileText,
  Home,
  LayoutDashboard,
  Settings,
  Sparkles,
} from "lucide-react";
import Logo from "./Logo";

const navItems = [
  { label: "Home", href: "/", icon: Home },
  { label: "AI Analyst", href: "/analyst", icon: Bot, badge: "AI" },
  { label: "Datasets", href: "/datasets", icon: Database },
  { label: "SQL Workbench", href: "/query", icon: Code2 },
  { label: "Dashboards", href: "/dashboards", icon: LayoutDashboard },
  { label: "Reports", href: "/reports", icon: FileText },
  { label: "Business Glossary", href: "/glossary", icon: BookOpen },
  { label: "Alerts & Monitoring", href: "/alerts", icon: Bell },
  { label: "AI Settings", href: "/settings", icon: Settings, badge: "Universal" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 bg-slate-950 border-r border-slate-800 flex flex-col justify-between p-4 min-h-screen select-none">
      <div className="space-y-6">
        {/* Brand Header with Interactive Logo */}
        <div className="px-2 py-3 border-b border-slate-800/80">
          <Logo />
        </div>

        {/* Navigation Links */}
        <nav className="space-y-1">
          <span className="px-3 text-[10px] font-bold uppercase tracking-wider text-slate-400">
            Core Platform
          </span>
          <div className="pt-2 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium transition-all ${
                    isActive
                      ? "bg-gradient-to-r from-cyan-950/80 via-indigo-950/60 to-slate-900 text-cyan-300 border border-cyan-500/40 shadow-lg shadow-cyan-950/40"
                      : "text-slate-400 hover:text-white hover:bg-slate-900"
                  }`}
                >
                  <div className="flex items-center space-x-2.5">
                    <Icon className={`w-4 h-4 ${isActive ? "text-cyan-400" : "text-slate-400"}`} />
                    <span>{item.label}</span>
                  </div>
                  {item.badge && (
                    <span className="px-1.5 py-0.5 text-[9px] font-mono bg-cyan-500/20 text-cyan-300 rounded font-semibold border border-cyan-500/30">
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        </nav>
      </div>

      {/* Footer Engine Pill */}
      <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-xl text-[11px] space-y-1">
        <div className="flex items-center space-x-1.5 text-cyan-400 font-semibold">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Local Analytical Engine</span>
        </div>
        <p className="text-[10px] text-slate-400 leading-tight">
          DuckDB in-process OLAP with auditable calculations and verification.
        </p>
      </div>
    </aside>
  );
}
