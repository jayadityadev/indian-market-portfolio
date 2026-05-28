"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, IChartApi, CandlestickSeries, HistogramSeries, CandlestickData, HistogramData, Time } from "lightweight-charts";

interface OHLCVPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface RegimeSegment {
  regime: string;
  start: string;
  end: string;
  days: number;
}

interface Props {
  data: OHLCVPoint[];
  regimeTimeline?: RegimeSegment[];
  height?: number;
}

const REGIME_BG: Record<string, string> = {
  Bull: "rgba(16, 185, 129, 0.06)",
  Bear: "rgba(239, 68, 68, 0.06)",
  Sideways: "rgba(245, 158, 11, 0.06)",
};

export function CandlestickChart({ data, regimeTimeline, height = 400 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    // Get theme
    const isDark = document.documentElement.classList.contains("dark");
    const bg = isDark ? "#0F1C2E" : "#fffefb";
    const textColor = isDark ? "#e0e0e0" : "#313d44";
    const gridColor = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.06)";

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: bg },
        textColor,
        fontFamily: "Inter, system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: gridColor },
        horzLines: { color: gridColor },
      },
      crosshair: {
        mode: 0, // Normal
      },
      rightPriceScale: {
        borderColor: gridColor,
      },
      timeScale: {
        borderColor: gridColor,
        timeVisible: false,
      },
    });

    chartRef.current = chart;

    // Candlestick series
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#10b981",
      downColor: "#ef4444",
      borderDownColor: "#ef4444",
      borderUpColor: "#10b981",
      wickDownColor: "#ef4444",
      wickUpColor: "#10b981",
    });

    const candleData: CandlestickData[] = data.map((d) => ({
      time: d.date as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));
    candleSeries.setData(candleData);

    // Volume histogram
    const hasVolume = data.some((d) => d.volume !== undefined && d.volume > 0);
    if (hasVolume) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        color: "#71c4ef",
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      });

      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });

      const volData: HistogramData[] = data
        .filter((d) => d.volume !== undefined)
        .map((d) => ({
          time: d.date as Time,
          value: d.volume!,
          color: d.close >= d.open ? "rgba(16, 185, 129, 0.3)" : "rgba(239, 68, 68, 0.3)",
        }));
      volumeSeries.setData(volData);
    }

    // Regime background shading
    if (regimeTimeline && regimeTimeline.length > 0) {
      const bgSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "bg",
        lastValueVisible: false,
        priceLineVisible: false,
      });

      chart.priceScale("bg").applyOptions({
        scaleMargins: { top: 0, bottom: 0 },
        visible: false,
      });

      const bgData: HistogramData[] = data.map((d) => {
        const seg = regimeTimeline.find((s) => d.date >= s.start && d.date <= s.end);
        return {
          time: d.date as Time,
          value: 1,
          color: seg ? REGIME_BG[seg.regime] || "transparent" : "transparent",
        };
      });
      bgSeries.setData(bgData);
    }

    // Fit content
    chart.timeScale().fitContent();

    // Resize handler
    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [data, height]);

  if (data.length === 0) return null;

  return (
    <div className="chart-container">
      <div ref={containerRef} style={{ width: "100%", height }} />
    </div>
  );
}
