# Frontend Redesign Plan

This plan details the steps to overhaul the Indian Market Intelligence frontend to exactly match the premium "Bento Box" dashboard aesthetic shown in the reference images.

## Goal

Apply the provided premium UI aesthetic to the existing quantitative trading application. We will maintain the core functionality (strategy backtesting, regime detection, etc.) while completely transforming the visual presentation to match the reference images.

## User Review Required

> [!IMPORTANT]  
> The reference image features a navigation sidebar (Wallet, Portfolio, Projects). Our app currently uses the sidebar for **inputs** (Ticker, Strategy, Dates). 
> **Decision:** I plan to keep our inputs in the sidebar but style them to look as clean and premium as the navigation items in the reference image. The "Run Analysis" button will be styled similarly to the "Log Out" or primary action buttons in the reference. Please confirm if this adaptation is acceptable.

## Proposed Changes

### 1. Global CSS (`src/app/globals.css`)
- **Backgrounds:** Set main body background to `#F7F8FA` and card backgrounds to `#FFFFFF`.
- **Radii:** Update `--card-radius` to `32px`.
- **Shadows:** Introduce the soft, diffused shadow `0 8px 30px rgba(0,0,0,0.04)` for all cards.
- **Accents:** Use the muted dusty blue-grey (`#BCCCDC`) for primary feature cards (like the Total Balance / Strategy Return card) and soft sky blue (`#D1E9F6`) for active states and hover effects.
- **Typography:** Ensure Inter/Geist-like clean sans-serif with appropriate font weights (bold for big numbers, medium for labels).

### 2. Layout & Sidebar (`src/app/page.tsx`)
- Modify the `.dashboard-grid` to use a wider sidebar or better proportions (e.g., 20% / 80%).
- Restyle the sidebar inputs (selects, dates) to have clean, minimal borders with generous padding, matching the airy feel of the reference UI.
- Update the "Run Analysis" button to be a prominent, premium-looking pill button.
- Add a dummy profile section at the top of the sidebar ("Hue Brew, Designer") to perfectly mirror the requested aesthetic, or adapt it to "Trader Profile".

### 3. Dashboard Cards (`src/app/components/MagicBento.tsx` & Data Views)
- **Feature Card:** Create a large card styled with `bg-[#BCCCDC]` for the primary metric (e.g., Final Portfolio Value).
- **Chart Card:** Update the lightweight-charts container to have the 32px border radius and no border, blending seamlessly into a white card.
- **Summary/Metrics Cards:** Use side-by-side mini cards or flex layouts for Win Rate, Max Drawdown, etc., matching the "Income/Spendings" and "Investments" pill-shaped layouts.
- **Lists:** Style the recent trades or historical data to look like the "Income List" (icon on left, name, time, amount on right).

## Verification Plan
1. Start the development server (`npm run dev`).
2. Visually inspect the layout, sidebar, and cards to ensure they match the colors, typography, border-radii, and shadows of the reference images.
3. Test the inputs and "Run Analysis" button to ensure functionality remains intact.
