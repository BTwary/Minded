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
      {/* Interactive Glowing 3D Neon Ribbon M Emblem with Ambient Cyan & Magenta Halo */}
      <div className="relative flex items-center justify-center emblem-sheen-container perspective-[800px]">
        {/* Ambient Rim Light Halo: Cyan into Fuchsia */}
        <div
          className={`absolute -inset-1.5 rounded-xl bg-gradient-to-r from-cyan-400/80 via-blue-500/60 to-fuchsia-500/80 opacity-70 blur-md transition-all duration-500 ${
            isHovered ? "opacity-100 blur-lg scale-110" : "opacity-45"
          }`}
        />

        {/* 3D Core Emblem Container with Specular Light Reflection */}
        <div
          className={`relative z-10 flex items-center justify-center rounded-xl bg-gradient-to-br from-slate-950 via-[#070b14] to-black border border-cyan-500/40 p-2 shadow-xl shadow-cyan-500/15 transition-all duration-300 transform-gpu overflow-hidden ${
            isHovered
              ? "scale-105 border-cyan-400 rotate-y-6 rotate-x-3 shadow-fuchsia-500/30"
              : "hover:border-cyan-300/80"
          }`}
          style={{ transformStyle: "preserve-3d" }}
        >
          {/* Specular Light Reflection Sweep */}
          <div className="emblem-sheen-reflection" />

          {/* 3D Depth Top Rim Highlight */}
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-200/50 to-transparent" />

          {/* Glowing Neon Ribbon M SVG */}
          <svg
            className={`w-7 h-7 transition-all duration-500 relative z-10 ${
              isHovered
                ? "scale-110 drop-shadow-[0_0_12px_rgba(0,229,255,0.7)]"
                : "drop-shadow-[0_0_6px_rgba(0,229,255,0.4)]"
            }`}
            viewBox="0 0 512 512"
            fill="none"
          >
            <defs>
              <linearGradient id="logoCyan" x1="0%" y1="100%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#00E5FF" />
                <stop offset="50%" stopColor="#00A2FF" />
                <stop offset="100%" stopColor="#7C3AED" />
              </linearGradient>
              <linearGradient id="logoMagenta" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#8B5CF6" />
                <stop offset="60%" stopColor="#D946EF" />
                <stop offset="100%" stopColor="#EC4899" />
              </linearGradient>
            </defs>

            {/* Background Inner Crossing Strand */}
            <path
              d="M 188 210 Q 235 300 256 312 Q 285 290 336 210"
              fill="none"
              stroke="#00E5FF"
              strokeWidth="28"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity="0.85"
            />

            {/* Left Arch with Cross-Over */}
            <path
              d="M 86 344 C 118 190 152 142 196 150 C 236 158 296 280 336 348"
              fill="none"
              stroke="url(#logoCyan)"
              strokeWidth="36"
              strokeLinecap="round"
            />

            {/* Right Arch with Cross-Over */}
            <path
              d="M 426 344 C 394 190 360 142 316 150 C 276 158 216 280 176 348"
              fill="none"
              stroke="url(#logoMagenta)"
              strokeWidth="36"
              strokeLinecap="round"
            />

            {/* Terminal Glowing Spheres */}
            <circle cx="176" cy="348" r="24" fill="#00E5FF" />
            <circle cx="176" cy="348" r="14" fill="#E0F7FF" />
            <circle cx="336" cy="348" r="24" fill="#D946EF" />
            <circle cx="336" cy="348" r="14" fill="#FCE7FE" />

            {/* Core Specular Line Accent */}
            <path
              d="M 102 320 C 126 200 154 156 190 156"
              fill="none"
              stroke="#FFFFFF"
              strokeWidth="8"
              strokeLinecap="round"
              opacity="0.5"
            />
          </svg>
        </div>
      </div>

      {/* Brand Typography */}
      {!collapsed && (
        <div className="flex flex-col">
          <div className="flex items-center space-x-1.5">
            <span className="text-lg font-bold tracking-tight text-white font-serif italic">
              Mind<span className="bg-gradient-to-r from-cyan-400 via-sky-300 to-fuchsia-400 bg-clip-text text-transparent not-italic font-sans font-black">Ed</span>
            </span>
            <span className="px-1.5 py-0.5 text-[9px] font-mono font-bold uppercase tracking-wider bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 rounded">
              AA-OS
            </span>
          </div>
          <span className="text-[10px] text-cyan-400/80 font-mono tracking-wider">
            Autonomous Data Analyst
          </span>
        </div>
      )}
    </Link>
  );
}
