"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";
import { BarChart3, Activity, Cpu, FileText, BookOpen } from "lucide-react";

export default function NavBar() {
  const pathname = usePathname();

  const links = [
    { href: "/", label: "Dashboard", icon: BarChart3 },
    { href: "/regime", label: "Regime Timeline", icon: Activity },
    { href: "/strategies", label: "Strategy Library", icon: BookOpen },
    { href: "/benchmark", label: "Model Benchmark", icon: Cpu },
    { href: "/report", label: "AI Analyst Report", icon: FileText },
  ];

  return (
    <header className="top-navbar">
      <Link href="/" className="navbar-brand">
        <div className="navbar-logo-icon">
          <BarChart3 size={18} />
        </div>
        <div className="navbar-brand-text">
          <span className="navbar-title">Market Intel</span>
          <span className="navbar-subtitle">Indian Equities • Quant Platform</span>
        </div>
      </Link>

      <nav className="navbar-links" aria-label="Main Navigation">
        {links.map((link) => {
          const Icon = link.icon;
          const isActive = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`navbar-link ${isActive ? "navbar-link--active" : ""}`}
            >
              <Icon size={14} />
              <span>{link.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="navbar-actions">
        <div className="system-status-pill" title="Backend API and Causal Datasets Active">
          <div className="status-dot" />
          <span>NIFTY 50 LIVE</span>
        </div>
        <ThemeToggle />
      </div>
    </header>
  );
}
