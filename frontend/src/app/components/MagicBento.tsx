"use client";

import React, { useRef, useState } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";
import { 
  Activity, 
  Target, 
  Cpu, 
  ShieldAlert, 
  TrendingUp, 
  Database
} from "lucide-react";
import "./MagicBento.css";

interface BentoCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  children?: React.ReactNode;
}

type GlowStyle = React.CSSProperties & {
  "--glow-x": string;
  "--glow-y": string;
  "--glow-intensity": number;
};

const BentoCard = ({ title, description, icon, children }: BentoCardProps) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  const glowX = useSpring(mouseX, { stiffness: 300, damping: 30 });
  const glowY = useSpring(mouseY, { stiffness: 300, damping: 30 });
  const [intensity, setIntensity] = useState(0);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    mouseX.set(e.clientX - rect.left);
    mouseY.set(e.clientY - rect.top);
  };

  return (
    <motion.div
      ref={cardRef}
      className="magic-bento-card"
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIntensity(1)}
      onMouseLeave={() => setIntensity(0)}
      style={
        {
          "--glow-x": glowX.get() + "px",
          "--glow-y": glowY.get() + "px",
          "--glow-intensity": intensity,
        } as GlowStyle
      }
      initial={{ opacity: 0, y: 15 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4 }}
    >
      <div className="magic-bento-card__content">
        <div>
          <div className="magic-bento-card__icon">{icon}</div>
          <h3 className="magic-bento-card__title">{title}</h3>
          <p className="magic-bento-card__description">{description}</p>
        </div>
        
        <div className="magic-bento-card__visual">
          {children}
        </div>
      </div>
    </motion.div>
  );
};

const NeuralNetworkVisual = () => {
  const nodes = [
    { left: 18, top: 24, duration: 2.4, delay: 0.1 },
    { left: 32, top: 68, duration: 3.1, delay: 0.5 },
    { left: 43, top: 38, duration: 2.7, delay: 0.9 },
    { left: 56, top: 72, duration: 3.4, delay: 1.2 },
    { left: 62, top: 22, duration: 2.2, delay: 1.6 },
    { left: 74, top: 50, duration: 3.0, delay: 0.3 },
    { left: 84, top: 30, duration: 2.6, delay: 1.0 },
    { left: 88, top: 78, duration: 3.3, delay: 1.8 },
  ];
  return (
    <div className="neural-net">
      {nodes.map((node, i) => (
        <motion.div
          key={i}
          className="neural-node"
          style={{
             left: `${node.left}%`,
             top: `${node.top}%`
          }}
          animate={{ scale: [1, 1.4, 1], opacity: [0.3, 0.9, 0.3] }}
          transition={{ duration: node.duration, repeat: Infinity, delay: node.delay }}
        />
      ))}
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        <line x1="18%" y1="24%" x2="43%" y2="38%" stroke="var(--bento-accent)" strokeWidth="1" strokeOpacity="0.2" />
        <line x1="43%" y1="38%" x2="62%" y2="22%" stroke="var(--bento-accent)" strokeWidth="1" strokeOpacity="0.2" />
        <line x1="62%" y1="22%" x2="84%" y2="30%" stroke="var(--bento-accent)" strokeWidth="1" strokeOpacity="0.2" />
        <line x1="32%" y1="68%" x2="56%" y2="72%" stroke="var(--bento-accent)" strokeWidth="1" strokeOpacity="0.2" />
        <line x1="56%" y1="72%" x2="74%" y2="50%" stroke="var(--bento-accent)" strokeWidth="1" strokeOpacity="0.2" />
        <line x1="74%" y1="50%" x2="88%" y2="78%" stroke="var(--bento-accent)" strokeWidth="1" strokeOpacity="0.2" />
      </svg>
    </div>
  );
};

const MonteCarloVisual = () => {
  const paths = [
    "M 0 50 Q 25 30, 50 45 T 100 20",
    "M 0 50 Q 25 60, 50 35 T 100 65",
    "M 0 50 Q 25 40, 50 55 T 100 40",
    "M 0 50 Q 25 70, 50 60 T 100 80",
    "M 0 50 Q 25 20, 50 30 T 100 15",
  ];
  return (
    <div className="monte-carlo">
      <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
        {paths.map((d, i) => {
          return (
            <path
              key={i}
              d={d}
              className="monte-path"
              style={{
                animationDelay: `${i * 0.4}s`,
                opacity: (i + 1) * 0.15,
                stroke: i === 4 ? "var(--bull, #10b981)" : i === 3 ? "var(--bear, #ef4444)" : "var(--bento-accent)"
              }}
            />
          );
        })}
      </svg>
    </div>
  );
};

export function MagicBento() {
  const cardData = [
    {
      id: "market",
      title: "Real-time Market Data",
      description: "Streaming Indian market updates with low-latency execution pipelines.",
      icon: <Database size={20} />,
      visual: (
        <div className="visual-market-data">
          {[75, 50, 92].map((w, i) => (
            <div key={i} className="market-bar">
              <motion.div 
                className="market-bar__fill" 
                initial={{ width: 0 }}
                animate={{ width: `${w}%` }}
                transition={{ duration: 1, delay: i * 0.2 }}
              />
            </div>
          ))}
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-200)", marginTop: 4 }}>
            <span>NIFTY 50 • NSE</span>
            <span>99.9% Uptime</span>
          </div>
        </div>
      )
    },
    {
      id: "strategy",
      title: "Quant Strategy Suite",
      description: "Backtest and evaluate 6 algorithmic trading strategies across market regimes.",
      icon: <Target size={20} />,
      visual: (
        <div className="visual-strategy">
          {[42, 68, 35, 82, 56, 74, 48, 91, 63, 39].map((height, i) => (
            <motion.div
              key={i}
              className="strategy-candle"
              style={{ height: `${height}%` }}
              animate={{ height: [`${height}%`, `${Math.max(20, height - 15)}%`, `${height}%`] }}
              transition={{ duration: 2.5, repeat: Infinity, delay: i * 0.1 }}
            />
          ))}
        </div>
      )
    },
    {
      id: "ml",
      title: "ML Prediction Engine",
      description: "Calibrated XGBoost classifier trained on purged walk-forward market features.",
      icon: <Cpu size={20} />,
      visual: (
        <div style={{ position: "relative", width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
          <NeuralNetworkVisual />
          <div className="status-grid">
            <div className="status-item"><div className="status-dot pulse" /> XGBoost v2.0</div>
            <div className="status-item"><div className="status-dot" /> Purged 5-Fold</div>
            <div className="status-item"><div className="status-dot pulse" /> CPU Latency &lt;5ms</div>
            <div className="status-item"><div className="status-dot" /> Gate: 0.30 F1</div>
          </div>
        </div>
      )
    },
    {
      id: "risk",
      title: "Advanced Risk Engine",
      description: "Multi-layered risk protection with automated circuit breakers and Monte Carlo drawdowns.",
      icon: <ShieldAlert size={20} />,
      visual: (
        <div className="risk-visual-container">
          <MonteCarloVisual />
          <div className="risk-gauge">
            <motion.div 
              className="gauge-arc"
              animate={{ rotate: [-45, 15, -45] }}
              transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
            />
          </div>
          <div className="risk-stats">
            <div><strong style={{ color: "var(--bull)" }}>LOW</strong> Volatility</div>
            <div><strong style={{ color: "var(--sideways)" }}>1.0x</strong> Exposure</div>
            <div><strong style={{ color: "var(--bull)" }}>SAFE</strong> Stance</div>
          </div>
        </div>
      )
    },
    {
      id: "sentiment",
      title: "Market Regime Discovery",
      description: "Gaussian HMM 3-state detection separating Bull, Bear, and Sideways periods.",
      icon: <TrendingUp size={20} />,
      visual: (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10, padding: "8px 0" }}>
          <motion.div 
            style={{ width: 32, height: 32, borderRadius: "50%", border: "3px solid rgba(16, 185, 129, 0.2)", borderTopColor: "var(--bull)" }}
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          />
          <div style={{ display: "flex", flexDirection: "column" }}>
            <span style={{ color: "var(--bull)", fontWeight: 700, fontSize: 13 }}>Bull Regime</span>
            <span style={{ color: "var(--text-200)", fontSize: 10 }}>Expansion Phase</span>
          </div>
        </div>
      )
    },
    {
      id: "execution",
      title: "Smart Risk Gating",
      description: "Validation-gated promotion protocol preventing out-of-sample ML overfit.",
      icon: <Activity size={20} />,
      visual: (
        <div style={{ display: "flex", flexDirection: "column", gap: 6, width: "100%" }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-200)" }}>
            <span>Promotion Gate</span>
            <span style={{ color: "var(--bull)", fontWeight: 700 }}>VERIFIED</span>
          </div>
          <div style={{ width: "100%", background: "var(--card-bg)", height: 6, borderRadius: 999, overflow: "hidden" }}>
            <motion.div 
              style={{ height: "100%", background: "var(--bull)", borderRadius: 999 }}
              initial={{ width: "0%" }}
              animate={{ width: "100%" }}
              transition={{ duration: 1.5, repeat: Infinity, repeatDelay: 2 }}
            />
          </div>
        </div>
      )
    }
  ];

  return (
    <section className="magic-bento-section">
      <div className="magic-bento-grid">
        {cardData.map((card) => (
          <BentoCard
            key={card.id}
            title={card.title}
            description={card.description}
            icon={card.icon}
          >
            {card.visual}
          </BentoCard>
        ))}
      </div>
    </section>
  );
}
