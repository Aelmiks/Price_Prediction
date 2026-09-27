# This script aggregates quotes from JSON, engineers features, removes outliers,
# tunes CatBoost with Optuna/manual CV (Pool for categories), trains the model,
# evaluates performance, and saves the model.
# For CatBoost, no one-hot encoding needed (uses cat_features natively).
# Added dtypes/NaNs prints for debug.

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
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

# Optuna tuning for CatBoost (manual CV with Pool for cat_features)
def tune_catboost(X_train, y_train, cat_cols, n_trials=100):
    """Tune CatBoost hyperparameters with Optuna and manual CV.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        cat_cols (list): Categorical columns.
        n_trials (int): Number of Optuna trials.
    
    Returns:
        dict: Best parameters.
    """
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "depth": trial.suggest_int("depth", 6, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
            "loss_function": "MAE",
            "verbose": 0,
            "random_seed": 42
        }
        
        maes = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X_train):
            X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
            train_pool = Pool(X_t, y_t, cat_features=cat_cols)
            val_pool = Pool(X_v, y_v, cat_features=cat_cols)
            model = CatBoostRegressor(**params)
            model.fit(train_pool)
            y_p = model.predict(val_pool)
            maes.append(mean_absolute_error(y_v, y_p))
        
        return np.mean(maes)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params

# Final model training and evaluation for CatBoost
def final_cat_model_and_evaluation(X_train, X_test, y_train, y_test, best_params, cat_cols):
    """Train final CatBoost model and evaluate performance.
    
    Args:
        X_train, X_test, y_train, y_test: Split data.
        best_params (dict): Tuned parameters.
        cat_cols (list): Categorical columns.
    
    Returns:
        model, comparison_df: Trained model and comparison dataframe.
    """
    best_params.update({"loss_function": "MAE", "verbose": 100, "random_seed": 42})
    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    model = CatBoostRegressor(**best_params)
    model.fit(train_pool)
    
    test_pool = Pool(X_test, cat_features=cat_cols)
    y_pred_log = model.predict(test_pool)
    y_test_exp = np.expm1(y_test)
    y_pred_exp = np.expm1(y_pred_log)
    
    mae = mean_absolute_error(y_test_exp, y_pred_exp)
    mape = np.mean(np.abs((y_test_exp - y_pred_exp) / y_test_exp)) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred_exp - y_test_exp) / (np.abs(y_test_exp) + np.abs(y_pred_exp)))
    mae_weighted = np.average(np.abs(y_test - y_pred_log), weights=y_test)
    
    print(f"MAE: {mae:.2f} €")
    print(f"MAPE: {mape:.2f} %")
    print(f"SMAPE: {smape:.2f} %")
    print(f"Weighted MAE (log-scale): {mae_weighted:.4f}")

    within_5 = np.mean(np.abs(y_test_exp - y_pred_exp) <= 5) * 100
    print(f"Predictions within 5€: {within_5:.2f}%")

    comparison_df = pd.DataFrame({
        "Actual price (€)": y_test_exp,
        "Predicted price (€)": y_pred_exp
    })
    comparison_df["Within 5€?"] = np.abs(y_test_exp - y_pred_exp) <= 5
    print(comparison_df.head(30))
    
    return model, comparison_df

# Main execution for CatBoost
if __name__ == "__main__":
    accepted_quotes = load_and_filter_quotes("data/quoteai_last_50000.json")
    df = build_dataframe(accepted_quotes)
    df = preprocess_dataframe(df)
    print("DataFrame dtypes after preprocess:\n", df.dtypes)  # Debug dtypes
    print("NaNs after preprocess:\n", df.isna().sum())  # Debug NaNs
    df = remove_outliers_iqr(df)
    df = apply_clustering(df)
    df = remove_cluster_outliers(df)
    
    y = np.log1p(df["proposed_price_accepted"])
    X = df.drop(columns=["proposed_price_accepted"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cat_cols = ["source_language", "target_language", "unit", "lang_pair", "doc_category", "doc_subcategory"]
    
    best_params = tune_catboost(X_train, y_train, cat_cols)
    model, comparison_df = final_cat_model_and_evaluation(X_train, X_test, y_train, y_test, best_params, cat_cols)
    
    # Save model
    model.save_model("cat_model.cbm")