"""Walk-forward sanity check for strategy suitability classifier.

Implements walk-forward validation from the risk-management and walk-forward-validation skills.
Splits data 70% train / 30% test chronologically.
Trains classifiers on train set, evaluates on test set to detect overfitting.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from classifier_features import build_feature_matrix


def walk_forward_sanity_check(labeled_path: Path, prices_path: Path, regimes_path: Path):
    print("--- Walk-Forward Sanity Check (70/30 Split) ---")
    
    labels_df = pd.read_parquet(labeled_path)
    prices_df = pd.read_parquet(prices_path)
    prices_df.index = pd.to_datetime(prices_df.index)
    
    regimes_df = None
    if regimes_path.exists():
        regimes_df = pd.read_parquet(regimes_path)
        regimes_df.index = pd.to_datetime(regimes_df.index)
        
    unique_dates = labels_df["window_end_date"].unique()
    features_df = build_feature_matrix(prices_df, unique_dates, regimes_df)
    
    full_df = pd.merge(labels_df, features_df, on="window_end_date", how="inner")
    full_df = full_df.sort_values(by="window_end_date").reset_index(drop=True)
    
    feature_cols = ["avg_return", "volatility", "momentum", "max_drawdown", "rsi_at_end", "sma_ratio"]
    if "regime_label" in full_df.columns:
        feature_cols.extend(["regime_label", "regime_stability"])
    
    strategies = full_df["strategy"].unique()
    
    for strategy in strategies:
        strat_df = full_df[full_df["strategy"] == strategy].dropna(subset=feature_cols + ["label"]).copy()
        
        counts = strat_df["label"].value_counts()
        if len(counts) < 2 or counts.min() < 5:
            print(f"\n[Skipped] {strategy:15s} : Highly skewed labels, cannot evaluate OOS.")
            continue
            
        # 70/30 split chronologically (no lookahead)
        split_idx = int(len(strat_df) * 0.7)
        train_df = strat_df.iloc[:split_idx]
        test_df = strat_df.iloc[split_idx:]
        
        # Must check if train/test have both classes
        if len(train_df["label"].unique()) < 2 or len(test_df["label"].unique()) < 2:
            print(f"\n[Skipped] {strategy:15s} : Missing classes in train/test split.")
            continue
            
        X_train = train_df[feature_cols]
        y_train = train_df["label"].astype(int)
        
        X_test = test_df[feature_cols]
        y_test = test_df["label"].astype(int)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=5, 
            class_weight="balanced", random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Train metrics
        train_preds = model.predict(X_train_scaled)
        train_probs = model.predict_proba(X_train_scaled)[:, 1]
        train_acc = accuracy_score(y_train, train_preds)
        train_auc = roc_auc_score(y_train, train_probs)
        
        # Test metrics
        test_preds = model.predict(X_test_scaled)
        test_probs = model.predict_proba(X_test_scaled)[:, 1]
        test_acc = accuracy_score(y_test, test_preds)
        test_auc = roc_auc_score(y_test, test_probs)
        
        print(f"\n--- Strategy: {strategy} ---")
        print(f"  Train AUC: {train_auc:.2f} | Acc: {train_acc:.2f}")
        print(f"  Test  AUC: {test_auc:.2f} | Acc: {test_acc:.2f}")
        
        if train_auc - test_auc > 0.15:
            print("  ⚠️ OVERFIT WARNING: Test AUC significantly lower than Train AUC.")
        elif test_auc < 0.55:
            print("  ⚠️ WEAK SIGNAL WARNING: Test AUC is barely better than random.")
        else:
            print("  ✅ ROBUST: Model holds up out-of-sample.")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    labeled = base / "data" / "labeled_data.parquet"
    prices = base / "data" / "nifty50.parquet"
    regimes = base / "data" / "nifty50_regimes.parquet"
    
    walk_forward_sanity_check(labeled, prices, regimes)
