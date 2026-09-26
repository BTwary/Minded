"use client";

import React, { useState } from "react";
import Link from "next/link";

interface LogoProps {
  size?: "sm" | "md" | "lg";
  collapsed?: boolean;
}

export default function Logo({ size = "md", collapsed = false }: LogoProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <Link
      href="/"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="group flex items-center space-x-3 transition-all duration-300 select-none"
    >
      {/* Interactive Glowing 3D Hexagonal Lattice Prism Icon */}
      <div className="relative flex items-center justify-center">
        {/* Glow halo */}
        <div
          className={`absolute -inset-1 rounded-xl bg-gradient-to-r from-amber-500 via-emerald-500 to-yellow-600 opacity-60 blur-sm transition-all duration-500 ${
            isHovered ? "opacity-100 blur-md scale-110" : "opacity-50"
          }`}
        />

        {/* Core Icon Container */}
        <div
          className={`relative z-10 flex items-center justify-center rounded-xl bg-slate-950 border border-amber-500/40 p-2.5 shadow-lg shadow-amber-500/20 transition-transform duration-300 ${
            isHovered ? "scale-105 border-amber-400 rotate-3" : ""
          }`}
        >
          <svg
            className={`w-6 h-6 text-amber-400 transition-all duration-500 ${
              isHovered ? "text-emerald-300 rotate-12 scale-110" : ""
            }`}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            {/* Mining Hexagon Lattice with Pulsing Energy Core */}
            <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5" className="stroke-amber-400 fill-emerald-950/40" />
            <line x1="12" y1="2" x2="12" y2="22" className="stroke-emerald-400" />
            <line x1="2" y1="8.5" x2="22" y2="15.5" className="stroke-amber-300/60" />
            <line x1="2" y1="15.5" x2="22" y2="8.5" className="stroke-amber-300/60" />
            <circle cx="12" cy="12" r="2" className="fill-amber-400 animate-pulse" />
            <circle cx="12" cy="12" r="1.2" className="fill-emerald-300" />
          </svg>
        </div>
      </div>

      {/* Brand Typography */}
      {!collapsed && (
        <div className="flex flex-col">
          <div className="flex items-center space-x-1.5">
            <span className="text-lg font-bold tracking-tight text-white font-serif italic">
              Mind<span className="bg-gradient-to-r from-amber-300 via-yellow-200 to-emerald-400 bg-clip-text text-transparent not-italic font-sans font-black">Ed</span>
            </span>
            <span className="px-1.5 py-0.5 text-[9px] font-mono font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 rounded">
              AA-OS
            </span>
          </div>
          <span className="text-[10px] text-amber-200/60 font-mono tracking-wider uppercase">
            Autonomous Analytical OS
          </span>
        </div>
      )}
    </Link>
  );
}
