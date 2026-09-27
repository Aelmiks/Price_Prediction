# This script aggregates quotes from JSON, engineers features, removes outliers, tunes XGBoost with Optuna/manual CV,
# trains the model, evaluates performance, and saves the model.

import json
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import zscore
from sklearn.cluster import KMeans
from collections import Counter

def load_and_filter_quotes(json_path):
    """Load raw quotes and filter those with accepted price.
    
    Args:
        json_path (str): Path to JSON file.
    
    Returns:
        list: Filtered quotes.
    """
    with open(json_path, "r") as f:
        raw_quotes = json.load(f)
    return [q for q in raw_quotes if q.get("proposed_price_accepted") is not None]

def build_dataframe(accepted_quotes):
    """Build dataframe from filtered quotes, aggregating subprojects.
    
    Args:
        accepted_quotes (list): Filtered quotes.
    
    Returns:
        pd.DataFrame: Built dataframe.
    """
    rows = []
    for q in accepted_quotes:
        subprojects = q.get("subproject", [])
        if not subprojects:
            continue
        
        # Aggregate subprojects
        source_langs = list(set(sub.get("source_language") for sub in subprojects if sub.get("source_language")))
        target_langs = list(set(sub.get("target_language") for sub in subprojects if sub.get("target_language")))
        units = list(set(sub.get("unit") for sub in subprojects if sub.get("unit")))
        doc_types = list(set(sub.get("document_type") for sub in subprojects if sub.get("document_type") and sub.get("document_type")))
        total_units = sum(sub.get("number_of_unit", 0) for sub in subprojects if isinstance(sub.get("number_of_unit"), (int, float)))
        
        primary_source = Counter(source_langs).most_common(1)[0][0] if source_langs else 'Unknown'
        primary_target = Counter(target_langs).most_common(1)[0][0] if target_langs else 'Unknown'
        primary_unit = Counter(units).most_common(1)[0][0] if units else "page"
        primary_doc = Counter(doc_types).most_common(1)[0][0] if doc_types else 'Unknown'
        
        num_unique_langs = len(set(source_langs + target_langs))
        num_unique_docs = len(doc_types)
        num_subprojects = len(subprojects)
        
        rows.append({
            "source_language": primary_source,
            "target_language": primary_target,
            "unit": primary_unit,
            "number_of_unit": total_units,
            "document_type": primary_doc,
            "num_subprojects": num_subprojects,
            "num_unique_langs": num_unique_langs,
            "num_unique_docs": num_unique_docs,
            "requested_deadline": q.get("proposed_deadlines_accepted"),
            "min_proposed_deadline": min(q.get("proposed_deadlines", [np.inf])) if q.get("proposed_deadlines") else None,
            "proposed_price_accepted": q["proposed_price_accepted"]
        })
    
    df = pd.DataFrame(rows)
    print(f"Aggregated quotes: {df.shape[0]}")
    return df

def preprocess_dataframe(df):
    """Preprocess dataframe with feature engineering.
    
    Args:
        df (pd.DataFrame): Built dataframe.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    df["requested_deadline"] = pd.to_numeric(df["requested_deadline"], errors="coerce").fillna(df["requested_deadline"].median())
    df["is_express"] = (df["requested_deadline"] <= 2).astype(int)
    df["flexibility"] = df["requested_deadline"] - df["min_proposed_deadline"].fillna(df["requested_deadline"])
    df["urgency"] = 1 / (df["requested_deadline"] + 1)
    df["lang_pair"] = df["source_language"].astype(str) + "_to_" + df["target_language"].astype(str)

    # Language rarity
    lang_freq = pd.concat([df["source_language"], df["target_language"]]).value_counts(normalize=True)
    df["lang_rarity"] = (df["source_language"].map(lang_freq) + df["target_language"].map(lang_freq)) / 2

    # Interactions
    df["unit_interaction"] = df["number_of_unit"] * df["num_subprojects"]
    df["delay_interaction"] = df["requested_deadline"] * df["num_unique_langs"]

    df["doc_category"] = df["document_type"].str.split("|").str[0].fillna("Unknown") if "document_type" in df else "Unknown"
    df["doc_subcategory"] = df["document_type"].str.split("|").str[1].fillna("Unknown") if "document_type" in df else "Unknown"
    df.drop(columns=["document_type"], inplace=True, errors="ignore")

    # Convert to category
    cat_cols = ["source_language", "target_language", "unit", "lang_pair", "doc_category", "doc_subcategory"]
    for col in cat_cols:
        if col in df:
            df[col] = df[col].astype("category")

    # Fill NaNs separately
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    cat_cols_present = [col for col in cat_cols if col in df.columns]
    for col in cat_cols_present:
        if "Unknown" not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories("Unknown")
        df[col] = df[col].fillna("Unknown")

    return df

def remove_outliers_iqr(df):
    """Remove outliers using IQR.
    
    Args:
        df (pd.DataFrame): Dataframe.
    
    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    q1 = df["proposed_price_accepted"].quantile(0.25)
    q3 = df["proposed_price_accepted"].quantile(0.75)
    iqr = q3 - q1
    return df[
        (df["proposed_price_accepted"] >= q1 - 1.5 * iqr) &
        (df["proposed_price_accepted"] <= q3 + 1.5 * iqr)
    ]

def apply_clustering(df):
    """Apply clustering after one-hot encoding.
    
    Args:
        df (pd.DataFrame): Dataframe.
    
    Returns:
        pd.DataFrame: Dataframe with cluster.
    """
    features = df.drop(columns=[
        "proposed_price_accepted"
    ]).copy()
    features = pd.get_dummies(features, drop_first=True)
    features.fillna(-999, inplace=True)
    stds = features.std()
    features_std = features[stds[stds > 0].index]
    features_std = (features_std - features_std.mean()) / features_std.std()
    df = df.loc[features_std.index]  # Align index if needed
    kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(features_std)
    return df

def remove_cluster_outliers(df):
    """Remove outliers within clusters using Z-score.
    
    Args:
        df (pd.DataFrame): Dataframe with cluster.
    
    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    df["log_price"] = np.log1p(df["proposed_price_accepted"])
    df["z_score"] = df.groupby("cluster")["log_price"].transform(zscore)
    return df[df["z_score"].abs() < 3].drop(columns=["cluster", "log_price", "z_score"])

def tune_xgb(X_train, y_train):
    """Tune XGBoost with Optuna and manual CV (to support enable_categorical).
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    
    Returns:
        dict: Best parameters.
    """
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
            "enable_categorical": True,
            "tree_method": "hist",
            "random_state": 42
        }
        
        maes = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X_train):
            X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = xgb.XGBRegressor(**params)
            model.fit(X_t, y_t)
            y_p = model.predict(X_v)
            maes.append(mean_absolute_error(y_v, y_p))
        
        return np.mean(maes)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    return study.best_trial.params

# Final model and evaluation (full metrics)
def final_model_and_evaluation(X_train, X_test, y_train, y_test, best_params):
    """Train final XGBoost model and evaluate.
    
    Args:
        X_train, X_test, y_train, y_test: Split data.
        best_params (dict): Tuned params.
    
    Returns:
        model, comparison_df: Trained model and DF.
    """
    best_params["enable_categorical"] = True
    best_params["tree_method"] = "hist"
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    
    y_pred_log = model.predict(X_test)
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
    
    best_params = tune_xgb(X_train, y_train)
    model, comparison_df = final_model_and_evaluation(X_train, X_test, y_train, y_test, best_params)
    
    # Save model
    model.save_model("xgb_model.json")