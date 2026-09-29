"use client";

import React, { useEffect, useRef } from "react";

export default function MiningCursorTrail() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    // Check for coarse pointer or reduced motion
    if (typeof window === "undefined") return;
    const isTouch = window.matchMedia("(pointer: coarse)").matches;
    const isReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (isTouch || isReduced) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    let animationFrameId: number;

    const resize = () => {
      if (!canvas) return;
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    resize();
    window.addEventListener("resize", resize);

    interface Particle {
      x: number;
      y: number;
      vx: number;
      vy: number;
      r: number;
      life: number;
      decay: number;
      hue: number;
      sat: number;
      lig: number;
    }

    const dots: Particle[] = [];
    let lastX = window.innerWidth / 2;
    let lastY = window.innerHeight / 2;
    let isHoveringInteractive = false;

    const spawnParticles = (x: number, y: number, count: number, isBurst = false) => {
      for (let i = 0; i < count; i++) {
        const speed = isBurst ? 2.5 + Math.random() * 3.5 : 0.4 + Math.random() * 0.8;
        const angle = Math.random() * Math.PI * 2;
        dots.push({
          x: x + (Math.random() - 0.5) * (isBurst ? 14 : 8),
          y: y + (Math.random() - 0.5) * (isBurst ? 14 : 8),
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
          r: isBurst ? 1.5 + Math.random() * 3.0 : 0.9 + Math.random() * 2.2,
          life: 1.0,
          decay: isBurst ? 0.02 + Math.random() * 0.03 : 0.015 + Math.random() * 0.025,
          hue: Math.random() > 0.3 ? 42 + Math.random() * 12 : 140 + Math.random() * 25, // Gold / Laser Emerald
          sat: 80 + Math.random() * 20,
          lig: 55 + Math.random() * 20,
        });
      }
      if (dots.length > 800) {
        dots.splice(0, dots.length - 800);
      }
    };

    const onMouseMove = (e: MouseEvent) => {
      lastX = e.clientX;
      lastY = e.clientY;

      const target = e.target as HTMLElement | null;
      isHoveringInteractive = !!(
        target &&
        (target.closest("button") ||
          target.closest("a") ||
          target.closest("input") ||
          target.closest("select") ||
          target.closest("textarea") ||
          target.closest("[role='button']") ||
          target.closest("tr"))
      );

      const count = isHoveringInteractive ? 4 + Math.floor(Math.random() * 3) : 2 + Math.floor(Math.random() * 2);
      spawnParticles(e.clientX, e.clientY, count);
    };

    const onMouseDown = (e: MouseEvent) => {
      // Click mining energy burst
      spawnParticles(e.clientX, e.clientY, 16, true);
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mousedown", onMouseDown);

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Render & update particles
      for (let i = dots.length - 1; i >= 0; i--) {
        const d = dots[i];
        d.x += d.vx;
        d.y += d.vy;
        d.life -= d.decay;

        if (d.life <= 0) {
          dots.splice(i, 1);
          continue;
        }

        const alpha = d.life * 0.75;
        ctx.fillStyle = `hsla(${d.hue}, ${d.sat}%, ${d.lig}%, ${alpha})`;
        ctx.beginPath();
        ctx.arc(d.x, d.y, d.r * Math.max(0.2, d.life), 0, Math.PI * 2);
        ctx.fill();
      }

      // Quantum Laser Probe Target Pinpoint
      ctx.fillStyle = isHoveringInteractive ? "rgba(212, 175, 55, 0.95)" : "rgba(56, 239, 125, 0.95)";
      ctx.beginPath();
      ctx.arc(lastX, lastY, isHoveringInteractive ? 3.5 : 2.5, 0, Math.PI * 2);
      ctx.fill();

      // Outer reticle ring
      ctx.strokeStyle = isHoveringInteractive ? "rgba(212, 175, 55, 0.6)" : "rgba(56, 239, 125, 0.4)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(lastX, lastY, isHoveringInteractive ? 10 : 7, 0, Math.PI * 2);
      ctx.stroke();

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mousedown", onMouseDown);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      style={{
        position: "fixed",
        inset: 0,
        pointerEvents: "none",
        zIndex: 99999,
      }}
    />
  );
}
