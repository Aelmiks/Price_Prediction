# This script aggregates quotes from JSON, engineers features, removes outliers,
# trains ExplainableBoostingRegressor (EBM), evaluates performance, and saves the model.
# EBM is interpretable, no Optuna tuning needed (defaults + interactions).

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor
import pickle

# Import shared functions from xgb_final.py
from xgb_final import (
    load_and_filter_quotes,
    build_dataframe,
    preprocess_dataframe,
    remove_outliers_iqr,
    apply_clustering,
    remove_cluster_outliers
)

# Train and evaluate EBM
def train_ebm(X_train, y_train, X_test, y_test):
    """Train EBM and evaluate performance.
    
    Args:
        X_train, X_test, y_train, y_test: Split data.
    
    Returns:
        ebm_model, comparison_df: Trained model and comparison dataframe.
    """
    ebm = ExplainableBoostingRegressor(interactions=3, outer_bags=14, inner_bags=4)
    ebm.fit(X_train, y_train)
    
    y_pred_log = ebm.predict(X_test)
    y_test_exp = np.expm1(y_test)
    y_pred_exp = np.expm1(y_pred_log)
    
    mae = mean_absolute_error(y_test_exp, y_pred_exp)
    mape = np.mean(np.abs((y_test_exp - y_pred_exp) / y_test_exp)) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred_exp - y_test_exp) / (np.abs(y_test_exp) + np.abs(y_pred_exp)))
    mae_weighted = np.average(np.abs(y_test - y_pred_log), weights=y_test)
    
    print(f"EBM MAE: {mae:.2f} €")
    print(f"EBM MAPE: {mape:.2f} %")
    print(f"EBM SMAPE: {smape:.2f} %")
    print(f"EBM Weighted MAE (log-scale): {mae_weighted:.4f}")

    within_5 = np.mean(np.abs(y_test_exp - y_pred_exp) <= 5) * 100
    print(f"EBM Predictions within 5€: {within_5:.2f}%")

    comparison_df = pd.DataFrame({
        "Actual price (€)": y_test_exp,
        "EBM Predicted price (€)": y_pred_exp
    })
    comparison_df["Within 5€?"] = np.abs(y_test_exp - y_pred_exp) <= 5
    print(comparison_df.head(30))
    
    return ebm, comparison_df

# Main execution
if __name__ == "__main__":
    accepted_quotes = load_and_filter_quotes("data/quoteai_last_50000.json")
    df = build_dataframe(accepted_quotes)
    df = preprocess_dataframe(df)
    df = remove_outliers_iqr(df)
    df = apply_clustering(df)
    df = remove_cluster_outliers(df)
    
    y = np.log1p(df["proposed_price_accepted"])
    X = df.drop(columns=["proposed_price_accepted"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ebm_model, comparison_df = train_ebm(X_train, y_train, X_test, y_test)
    
    # Save model
    with open("ebm_model.pkl", "wb") as f:
        pickle.dump(ebm_model, f)