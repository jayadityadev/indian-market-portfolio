"""Train Classifier Models.

Implements Phase 3 of the Strategy Suitability Classifier pipeline.
Trains one binary classifier per strategy to predict if it will beat Buy & Hold.
Uses TimeSeriesSplit to prevent lookahead bias during evaluation.
Compares Logistic Regression baseline vs Random Forest, calibrates probabilities,
and saves the final models.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from classifier_features import build_feature_matrix


def load_and_prepare_data(labeled_path: Path, prices_path: Path, regimes_path: Path) -> pd.DataFrame:
    """Load data and build feature matrix."""
    labels_df = pd.read_parquet(labeled_path)
    prices_df = pd.read_parquet(prices_path)
    prices_df.index = pd.to_datetime(prices_df.index)
    
    regimes_df = None
    if regimes_path.exists():
        regimes_df = pd.read_parquet(regimes_path)
        regimes_df.index = pd.to_datetime(regimes_df.index)
        
    unique_dates = labels_df["window_end_date"].unique()
    features_df = build_feature_matrix(prices_df, unique_dates, regimes_df)
    
    # Merge features with labels
    full_df = pd.merge(labels_df, features_df, on="window_end_date", how="inner")
    
    # Sort by date to strictly maintain time series order
    full_df = full_df.sort_values(by="window_end_date").reset_index(drop=True)
    return full_df


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> dict[str, float]:
    """Evaluate a model using TimeSeriesSplit."""
    accuracies = []
    f1s = []
    aucs = []
    
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Only evaluate if test set has both classes (otherwise AUC fails)
        if len(y_test.unique()) < 2:
            continue
            
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        try:
            probs = model.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, probs))
        except AttributeError:
            # If no predict_proba (e.g. some setups), use decision_function or skip AUC
            try:
                probs = model.decision_function(X_test)
                aucs.append(roc_auc_score(y_test, probs))
            except AttributeError:
                pass
                
        accuracies.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, zero_division=0))
        
    return {
        "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "auc": sum(aucs) / len(aucs) if aucs else 0.0
    }


def train_and_evaluate_strategy(strategy: str, strat_df: pd.DataFrame, models_dir: Path) -> None:
    """Train models for a single strategy and save the best one."""
    print(f"\n--- Strategy: {strategy} ---")
    
    # Define feature columns explicitly
    feature_cols = ["avg_return", "volatility", "momentum", "max_drawdown", "rsi_at_end", "sma_ratio"]
    if "regime_label" in strat_df.columns:
        feature_cols.extend(["regime_label", "regime_stability"])
        
    X = strat_df[feature_cols]
    y = strat_df["label"]
    
    # Check class balance to prevent CV crashes
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 5:
        print(f"  Highly imbalanced classes ({counts.to_dict()}). Skipping ML.")
        return
    
    # Scale features
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
    
    # Define models
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    rf_model = RandomForestClassifier(max_depth=4, n_estimators=100, min_samples_leaf=5, random_state=42)
    
    # Evaluate with TimeSeriesSplit
    cv = TimeSeriesSplit(n_splits=5)
    
    lr_metrics = evaluate_model(lr_model, X_scaled, y, cv)
    print(f"  Logistic Regression — AUC: {lr_metrics['auc']:.2f}, F1: {lr_metrics['f1']:.2f}, Acc: {lr_metrics['accuracy']:.2f}")
    
    rf_metrics = evaluate_model(rf_model, X_scaled, y, cv)
    print(f"  Random Forest       — AUC: {rf_metrics['auc']:.2f}, F1: {rf_metrics['f1']:.2f}, Acc: {rf_metrics['accuracy']:.2f}")
    
    # We will use Calibrated Random Forest as requested
    print("  Selected: Random Forest (Calibrated)")
    
    # Train on full dataset
    rf_model.fit(X_scaled, y)
    
    # Calibrate probabilities
    calibrated_rf = CalibratedClassifierCV(rf_model, method='isotonic', cv=3)
    calibrated_rf.fit(X_scaled, y)
    
    # Print feature importances from the uncalibrated base RF
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = ", ".join([f"{idx} ({val:.2f})" for idx, val in importances.head(3).items()])
    print(f"  Top features: {top_features}")
    
    # If all AUCs are < 0.55, inform the user
    if max(lr_metrics['auc'], rf_metrics['auc']) < 0.55:
        print(f"  ⚠️ Warning: Model for {strategy} is barely better than random (AUC < 0.55).")
        print("  This means the strategy's performance may be unpredictable from these features.")
        print("  The dashboard will fall back to historical Sharpe when confidence is low.")
        
    # Save scaler and model
    models_dir.mkdir(parents=True, exist_ok=True)
    
    scaler_path = models_dir / f"{strategy.replace(' ', '_')}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    model_path = models_dir / f"{strategy.replace(' ', '_')}_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(calibrated_rf, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", default="data/labeled_data.parquet", help="Path to labeled data")
    parser.add_argument("--prices", default="data/nifty50.parquet", help="Path to prices data")
    parser.add_argument("--regimes", default="data/nifty50_regimes.parquet", help="Path to regimes data")
    parser.add_argument("--models-dir", default="models", help="Directory to save models")
    args = parser.parse_args()
    
    labeled_path = Path(args.labeled)
    prices_path = Path(args.prices)
    regimes_path = Path(args.regimes)
    models_dir = Path(args.models_dir)
    
    if not labeled_path.exists() or not prices_path.exists():
        print("Error: Required data files not found.")
        return
        
    print("Building feature matrix and merging with labels...")
    full_df = load_and_prepare_data(labeled_path, prices_path, regimes_path)
    print(f"Dataset shape: {full_df.shape} ({full_df['strategy'].nunique()} strategies)")
    
    # Train models per strategy
    strategies = full_df["strategy"].unique()
    for strategy in strategies:
        strat_df = full_df[full_df["strategy"] == strategy].copy()
        train_and_evaluate_strategy(strategy, strat_df, models_dir)
        
    print(f"\nAll models trained and saved to {models_dir}/")


if __name__ == "__main__":
    main()
