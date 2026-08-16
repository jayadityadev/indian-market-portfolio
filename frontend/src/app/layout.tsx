import type { Metadata } from "next";
import "./globals.css";
import NavBar from "./components/NavBar";
import { InteractiveDotGrid } from "./components/InteractiveDotGrid";

export const metadata: Metadata = {
  title: "Indian Market Intelligence — Portfolio Analytics Platform",
  description:
    "Regime-aware NIFTY 50 strategy research with backtesting, risk analytics, news context, and validation-gated ML.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        {/* Keep font loading explicit for demo portability; production can migrate to next/font. */}
        {/* eslint-disable-next-line @next/next/no-page-custom-font */}
        <link
          href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Sora:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">
        <InteractiveDotGrid />
        <NavBar />
        <div style={{ position: "relative", zIndex: 1 }}>
          {children}
        </div>
      </body>
    </html>
  );
}
