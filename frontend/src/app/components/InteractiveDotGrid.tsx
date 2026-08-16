"use client";

import { useEffect, useRef } from "react";

interface Dot {
  originX: number;
  originY: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export function InteractiveDotGrid() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const SPACING = 28;
    const RADIUS = 1.2;
    const MOUSE_RADIUS = 120;
    const REPEL_FORCE = 0.35;
    const SPRING_STIFFNESS = 0.08;
    const DAMPING = 0.85;

    let dots: Dot[] = [];
    const mouse = { x: -9999, y: -9999 };

    const initDots = () => {
      dots = [];
      const cols = Math.ceil(width / SPACING) + 1;
      const rows = Math.ceil(height / SPACING) + 1;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const originX = c * SPACING;
          const originY = r * SPACING;
          dots.push({
            originX,
            originY,
            x: originX,
            y: originY,
            vx: 0,
            vy: 0,
          });
        }
      }
    };

    initDots();

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      initDots();
    };

    const handleMouseMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };

    const handleMouseLeave = () => {
      mouse.x = -9999;
      mouse.y = -9999;
    };

    window.addEventListener("resize", handleResize);
    window.addEventListener("mousemove", handleMouseMove, { passive: true });
    document.body.addEventListener("mouseleave", handleMouseLeave);

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Check current theme
      const isDark = document.documentElement.classList.contains("dark");
      const dotColor = isDark ? "rgba(255, 255, 255, 0.07)" : "rgba(20, 23, 28, 0.06)";
      const activeDotColor = isDark ? "rgba(213, 140, 85, 0.45)" : "rgba(213, 140, 85, 0.55)";

      for (let i = 0; i < dots.length; i++) {
        const dot = dots[i];
        const dx = dot.x - mouse.x;
        const dy = dot.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Repel from mouse
        if (dist < MOUSE_RADIUS && dist > 0) {
          const force = (1 - dist / MOUSE_RADIUS) * MOUSE_RADIUS * REPEL_FORCE;
          const angle = Math.atan2(dy, dx);
          dot.vx += Math.cos(angle) * force;
          dot.vy += Math.sin(angle) * force;
        }

        // Spring back to origin
        const springX = (dot.originX - dot.x) * SPRING_STIFFNESS;
        const springY = (dot.originY - dot.y) * SPRING_STIFFNESS;
        dot.vx = (dot.vx + springX) * DAMPING;
        dot.vy = (dot.vy + springY) * DAMPING;

        dot.x += dot.vx;
        dot.y += dot.vy;

        // Draw dot
        ctx.beginPath();
        const isNearMouse = dist < MOUSE_RADIUS;
        ctx.arc(dot.x, dot.y, isNearMouse ? RADIUS * 1.5 : RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = isNearMouse ? activeDotColor : dotColor;
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("mousemove", handleMouseMove);
      document.body.removeEventListener("mouseleave", handleMouseLeave);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        pointerEvents: "none",
        zIndex: 0,
      }}
    />
  );
}
