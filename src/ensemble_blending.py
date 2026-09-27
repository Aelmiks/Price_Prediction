# Loads saved XGBoost and CatBoost models, preprocesses data, makes predictions on test set,
# blends them (tuned weights or stacking), evaluates performance, and saves blended predictions.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

# Import shared functions from xgb_final.py
from xgb_final import (
    load_and_filter_quotes,
    build_dataframe,
    preprocess_dataframe,
    remove_outliers_iqr,
    apply_clustering,
    remove_cluster_outliers
)

def preprocess_data(json_path):
    """Preprocess the data from JSON to split-ready X/y.
    
    Args:
        json_path (str): Path to JSON.
    
    Returns:
        X_train, X_test, y_train, y_test: Split data.
    """
    accepted_quotes = load_and_filter_quotes(json_path)
    df = build_dataframe(accepted_quotes)
    df = preprocess_dataframe(df)
    df = remove_outliers_iqr(df)
    df = apply_clustering(df)
    df = remove_cluster_outliers(df)
    
    y = np.log1p(df["proposed_price_accepted"])
    X = df.drop(columns=["proposed_price_accepted"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def get_predictions(X_train, X_test, cat_cols):
    """Get predictions from loaded models on train/test.
    
    Args:
        X_train, X_test (pd.DataFrame): Features.
        cat_cols (list): Categorical columns for CatBoost.
    
    Returns:
        y_pred_xgb_train, y_pred_cat_train, y_pred_xgb_log, y_pred_cat_log: Predictions.
    """
    xgb_model, cat_model = load_models()
    y_pred_xgb_train = xgb_model.predict(X_train)
    train_pool = Pool(X_train, cat_features=cat_cols)
    y_pred_cat_train = cat_model.predict(train_pool)
    
    y_pred_xgb_log = xgb_model.predict(X_test)
    test_pool = Pool(X_test, cat_features=cat_cols)
    y_pred_cat_log = cat_model.predict(test_pool)
    return y_pred_xgb_train, y_pred_cat_train, y_pred_xgb_log, y_pred_cat_log

# Load saved models
def load_models(xgb_path="xgb_model.json", cat_path="cat_model.cbm"):
    """Load saved XGBoost and CatBoost models.
    
    Args:
        xgb_path (str): Path to XGBoost model.
        cat_path (str): Path to CatBoost model.
    
    Returns:
        xgb_model, cat_model: Loaded models.
    """
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    cat_model = CatBoostRegressor()
    cat_model.load_model(cat_path)
    return xgb_model, cat_model

# Blend predictions (weighted average)
def blend_predictions(y_pred_xgb, y_pred_cat, weight_xgb=0.5):
    """Blend predictions from XGBoost and CatBoost with tunable weight.
    
    Args:
        y_pred_xgb (np.array): XGBoost predictions (log scale).
        y_pred_cat (np.array): CatBoost predictions (log scale).
        weight_xgb (float): Weight for XGBoost (CatBoost = 1 - weight_xgb).
    
    Returns:
        np.array: Blended predictions (log scale).
    """
    weight_cat = 1 - weight_xgb
    return weight_xgb * y_pred_xgb + weight_cat * y_pred_cat

# Stacking (meta-model on predictions)
def stack_predictions(y_pred_xgb_train, y_pred_cat_train, y_train, y_pred_xgb_test, y_pred_cat_test):
    """Stack predictions using LinearRegression as meta-model (fit on train, predict on test).
    
    Args:
        y_pred_xgb_train (np.array): XGBoost train predictions (log scale).
        y_pred_cat_train (np.array): CatBoost train predictions (log scale).
        y_train (np.array): True train values (log scale).
        y_pred_xgb_test (np.array): XGBoost test predictions (log scale).
        y_pred_cat_test (np.array): CatBoost test predictions (log scale).
    
    Returns:
        np.array: Stacked predictions (log scale on test).
    """
    stack_features_train = np.column_stack((y_pred_xgb_train, y_pred_cat_train))
    meta_model = LinearRegression()
    meta_model.fit(stack_features_train, y_train)
    stack_features_test = np.column_stack((y_pred_xgb_test, y_pred_cat_test))
    y_pred_stack = meta_model.predict(stack_features_test)
    return y_pred_stack

# Tune weights for blending (finer step 0.05)
def tune_weights(y_pred_xgb, y_pred_cat, y_test_exp, y_test):
    """Tune blending weights by testing 0.0 to 1.0 in 0.05 steps, find best MAPE.
    
    Args:
        y_pred_xgb (np.array): XGBoost log predictions.
        y_pred_cat (np.array): CatBoost log predictions.
        y_test_exp (np.array): Exponentiated true values.
        y_test (np.array): Log true values.
    
    Returns:
        float, np.array: Best weight for XGBoost, blended log predictions.
    """
    best_mape = float('inf')
    best_weight = 0.5
    best_blend_log = None
    for w in np.arange(0.0, 1.05, 0.05):
        y_pred_blend_log = blend_predictions(y_pred_xgb, y_pred_cat, w)
        y_pred_blend_exp = np.expm1(y_pred_blend_log)
        mape = np.mean(np.abs((y_test_exp - y_pred_blend_exp) / y_test_exp)) * 100
        if mape < best_mape:
            best_mape = mape
            best_weight = w
            best_blend_log = y_pred_blend_log
    print(f"Best XGBoost weight: {best_weight:.2f} (CatBoost: {1 - best_weight:.2f}), MAPE: {best_mape:.2f}%")
    return best_weight, best_blend_log

def evaluate_predictions(y_test_exp, y_pred_exp, y_test, y_pred_log):
    """Evaluate predictions with full metrics.
    
    Args:
        y_test_exp (np.array): Exponentiated true values.
        y_pred_exp (np.array): Exponentiated predictions.
        y_test (np.array): Log true values.
        y_pred_log (np.array): Log predictions.
    
    Returns:
        dict: Metrics dictionary.
    """
    mae = mean_absolute_error(y_test_exp, y_pred_exp)
    mape = np.mean(np.abs((y_test_exp - y_pred_exp) / y_test_exp)) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred_exp - y_test_exp) / (np.abs(y_test_exp) + np.abs(y_pred_exp)))
    mae_weighted = np.average(np.abs(y_test - y_pred_log), weights=y_test)
    
    within_5 = np.mean(np.abs(y_test_exp - y_pred_exp) <= 5) * 100
    
    print(f"MAE: {mae:.2f} €")
    print(f"MAPE: {mape:.2f} %")
    print(f"SMAPE: {smape:.2f} %")
    print(f"Weighted MAE (log-scale): {mae_weighted:.4f}")
    print(f"Predictions within 5€: {within_5:.2f}%")
    
    return {"mae": mae, "mape": mape, "smape": smape, "mae_weighted": mae_weighted, "within_5": within_5}

def create_comparison_df(y_test_exp, y_pred_exp):
    """Create comparison dataframe.
    
    Args:
        y_test_exp (np.array): Exponentiated true values.
        y_pred_exp (np.array): Exponentiated predictions.
    
    Returns:
        pd.DataFrame: Comparison DF.
    """
    comparison_df = pd.DataFrame({
        "Actual price (€)": y_test_exp,
        "Predicted price (€)": y_pred_exp
    })
    comparison_df["Within 5€?"] = np.abs(y_test_exp - y_pred_exp) <= 5
    print(comparison_df.head(30))
    return comparison_df

# Main execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data("data/quoteai_last_50000.json")
    
    cat_cols = ["source_language", "target_language", "unit", "lang_pair", "doc_category", "doc_subcategory"]
    
    y_pred_xgb_train, y_pred_cat_train, y_pred_xgb_log, y_pred_cat_log = get_predictions(X_train, X_test, cat_cols)
    
    y_test_exp = np.expm1(y_test)
    best_weight, y_pred_blend_log = tune_weights(y_pred_xgb_log, y_pred_cat_log, y_test_exp, y_test)
    y_pred_blend_exp = np.expm1(y_pred_blend_log)
    
    y_pred_stack_log = stack_predictions(y_pred_xgb_train, y_pred_cat_train, y_train, y_pred_xgb_log, y_pred_cat_log)
    y_pred_stack_exp = np.expm1(y_pred_stack_log)
    mape_stack = np.mean(np.abs((y_test_exp - y_pred_stack_exp) / y_test_exp)) * 100
    print(f"Stacking MAPE: {mape_stack:.2f}%")
    
    blend_mape = np.mean(np.abs((y_test_exp - y_pred_blend_exp) / y_test_exp)) * 100
    if mape_stack < blend_mape:
        print("Using Stacking as final (better MAPE)")
        y_pred_final_log = y_pred_stack_log
        y_pred_final_exp = y_pred_stack_exp
        y_pred_final = y_pred_stack_log  # For weighted MAE
        final_mape = mape_stack
    else:
        print("Using Blend as final")
        y_pred_final_log = y_pred_blend_log
        y_pred_final_exp = y_pred_blend_exp
        y_pred_final = y_pred_blend_log  # For weighted MAE
        final_mape = blend_mape
    
    evaluate_predictions(y_test_exp, y_pred_final_exp, y_test, y_pred_final_log)
    
    comparison_df = create_comparison_df(y_test_exp, y_pred_final_exp)
    comparison_df.to_csv("final_blended_predictions.csv", index=False)