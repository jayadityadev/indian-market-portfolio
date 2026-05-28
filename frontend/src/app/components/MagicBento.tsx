"use client";

import React, { useRef, useState, useEffect } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";
import { 
  Activity, 
  Target, 
  Cpu, 
  ShieldAlert, 
  TrendingUp, 
  Database, 
  ArrowRight,
  Code
} from "lucide-react";
import "./MagicBento.css";

interface BentoCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  size: "small" | "medium" | "large";
  children?: React.ReactNode;
}

const BentoCard = ({ title, description, icon, size, children }: BentoCardProps) => {
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
      className={`magic-bento-card magic-bento-card--${size}`}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIntensity(1)}
      onMouseLeave={() => setIntensity(0)}
      style={
        {
          "--glow-x": glowX.get() + "px",
          "--glow-y": glowY.get() + "px",
          "--glow-intensity": intensity,
        } as any
      }
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
    >
      <div className="magic-bento-card__content">
        <div className="magic-bento-card__icon">{icon}</div>
        <h3 className="magic-bento-card__title">{title}</h3>
        <p className="magic-bento-card__description">{description}</p>
        
        <div className="magic-bento-card__visual">
          {children}
        </div>
      </div>
      
      <button className="view-code" title="View Documentation">
        <Code size={16} />
      </button>
    </motion.div>
  );
};

const NeuralNetworkVisual = () => {
  const nodes = Array.from({ length: 8 });
  return (
    <div className="neural-net">
      {nodes.map((_, i) => (
        <motion.div
          key={i}
          className="neural-node"
          style={{
            left: `${Math.random() * 80 + 10}%`,
            top: `${Math.random() * 80 + 10}%`,
          }}
          animate={{
            scale: [1, 1.5, 1],
            opacity: [0.3, 1, 0.3],
          }}
          transition={{
            duration: 2 + Math.random() * 2,
            repeat: Infinity,
            delay: Math.random() * 2,
          }}
        />
      ))}
      <svg className="w-full h-full absolute inset-0 opacity-20">
        <motion.path
          d="M 20 50 Q 50 20 80 50 T 140 50"
          stroke="var(--bento-accent)"
          fill="none"
          strokeWidth="1"
          animate={{ pathLength: [0, 1, 0], pathOffset: [0, 0, 1] }}
          transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
        />
      </svg>
    </div>
  );
};

const MonteCarloVisual = () => {
  const paths = Array.from({ length: 12 });
  return (
    <div className="monte-carlo w-full h-full relative">
      <svg className="w-full h-full" viewBox="0 0 200 100" preserveAspectRatio="none">
        {paths.map((_, i) => {
          const endY = 30 + Math.random() * 40;
          return (
            <motion.path
              key={i}
              d={`M 0 50 C 50 ${50 + (i-6)*5}, 150 ${endY}, 200 ${endY}`}
              className="monte-path"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.4 }}
              transition={{ 
                duration: 2, 
                repeat: Infinity, 
                delay: i * 0.1,
                repeatType: "reverse" 
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
      icon: <Database size={24} />,
      size: "medium" as const,
      visual: (
        <div className="visual-market-data">
          {[70, 45, 90].map((w, i) => (
            <div key={i} className="market-bar">
              <motion.div 
                className="market-bar__fill" 
                initial={{ width: 0 }}
                animate={{ width: `${w}%` }}
                transition={{ duration: 1, delay: i * 0.2 }}
              />
            </div>
          ))}
          <div className="flex justify-between text-[10px] text-zinc-500 mt-2">
            <span>NSE/BSE</span>
            <span>99.9% Uptime</span>
          </div>
        </div>
      )
    },
    {
      id: "strategy",
      title: "Quant Strategy",
      description: "Backtest and deploy complex algorithmic trading strategies.",
      icon: <Target size={24} />,
      size: "small" as const,
      visual: (
        <div className="visual-strategy">
          {Array.from({ length: 10 }).map((_, i) => (
            <motion.div
              key={i}
              className="strategy-candle"
              style={{ height: `${Math.random() * 100}%` }}
              animate={{ height: `${20 + Math.random() * 80}%` }}
              transition={{ duration: 2, repeat: Infinity, repeatType: "mirror", delay: i * 0.1 }}
            />
          ))}
        </div>
      )
    },
    {
      id: "ml",
      title: "ML Prediction Engine",
      description: "Deep learning models trained on historical tick data for alpha generation.",
      icon: <Cpu size={24} />,
      size: "large" as const,
      visual: (
        <div className="w-full h-full relative flex flex-col justify-center items-center">
          <NeuralNetworkVisual />
          <div className="status-grid mt-4">
            <div className="status-item"><div className="status-dot pulse" /> Accuracy: 84%</div>
            <div className="status-item"><div className="status-dot" /> Epoch: 124</div>
            <div className="status-item"><div className="status-dot pulse" /> GPU: Active</div>
            <div className="status-item"><div className="status-dot" /> Loss: 0.023</div>
          </div>
          <svg className="radar-svg mt-6 opacity-80" viewBox="0 0 100 100">
            <polygon points="50,10 90,40 75,90 25,90 10,40" fill="none" stroke="var(--visual-border)" strokeWidth="1" />
            <motion.polygon 
              points="50,25 75,45 65,75 35,75 25,45" 
              className="radar-poly"
              animate={{ 
                points: [
                  "50,25 75,45 65,75 35,75 25,45",
                  "50,20 80,40 70,80 30,80 20,40",
                  "50,25 75,45 65,75 35,75 25,45"
                ] 
              }}
              transition={{ duration: 4, repeat: Infinity }}
            />
          </svg>
        </div>
      )
    },
    {
      id: "risk",
      title: "Advanced Risk Engine",
      description: "Multi-layered risk protection with automated circuit breakers and portfolio rebalancing.",
      icon: <ShieldAlert size={24} />,
      size: "large" as const,
      visual: (
        <div className="w-full h-full relative flex flex-col justify-center">
          <MonteCarloVisual />
          <div className="risk-visual-container mt-4">
            <div className="risk-gauge">
              <motion.div 
                className="gauge-arc"
                animate={{ rotate: [-45, 15, -45] }}
                transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
              />
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-[var(--bento-accent)]">98.2%</div>
              <div className="text-[10px] text-zinc-500 uppercase tracking-widest">Confidence Score</div>
            </div>
            <div className="risk-stats">
              <div className="flex flex-col items-center">
                <span className="font-bold text-emerald-500">LOW</span>
                <span>Volatility</span>
              </div>
              <div className="flex flex-col items-center">
                <span className="font-bold text-amber-500">1.2x</span>
                <span>Leverage</span>
              </div>
              <div className="flex flex-col items-center">
                <span className="font-bold text-sky-500">SAFE</span>
                <span>Exposure</span>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: "sentiment",
      title: "News Sentiment",
      description: "AI-driven analysis of news and social media for Indian stocks.",
      icon: <TrendingUp size={24} />,
      size: "small" as const,
      visual: (
        <div className="flex items-center justify-center gap-2">
          <motion.div 
            className="w-12 h-12 rounded-full border-4 border-emerald-500/20 border-t-emerald-500"
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          />
          <span className="text-emerald-500 font-bold">Bullish</span>
        </div>
      )
    },
    {
      id: "execution",
      title: "Smart Execution",
      description: "Optimized order routing for minimal slippage.",
      icon: <Activity size={24} />,
      size: "small" as const,
      visual: (
        <div className="flex flex-col gap-2 w-full">
          <div className="flex justify-between text-[10px] mb-1">
            <span>Slippage</span>
            <span className="text-emerald-500">0.02%</span>
          </div>
          <div className="w-full bg-zinc-800/50 h-1 rounded-full overflow-hidden">
            <motion.div 
              className="h-full bg-emerald-500"
              animate={{ width: ["0%", "100%", "100%"] }}
              transition={{ duration: 3, repeat: Infinity }}
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
            size={card.size}
          >
            {card.visual}
          </BentoCard>
        ))}
      </div>
    </section>
  );
}
