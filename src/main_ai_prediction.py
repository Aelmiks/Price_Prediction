# Fetches all quotes from MySQL DB, processes into accepted_quotes, builds DF with multiprocessing/tqdm/checkpoint,
# tunes/trains CatBoost on full data (new model), evaluates on test split, saves new model/preds.
# Updated: Set OMP_NUM_THREADS=1 to avoid memory leaks in sklearn, reduced batch_size for MiniBatchKMeans.
# Added sparse one-hot, drop quote_no, TruncatedSVD for sparse reduction to prevent OOM.
# Fix: Use col.sparse.to_dense() for sparse columns.
# Fix: Drop quote_no from X to avoid float conversion error in CatBoost.
# Update: Use isinstance for sparse check, increase n_components to 500 for better variance.
# Improvement: Increased trials to 100, optional skip SVD for full dims.

import os
os.environ["OMP_NUM_THREADS"] = "1"  # Fix for OOM in sklearn parallel on Mac

from collections import Counter
import mysql.connector
from dotenv import load_dotenv
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, KFold
import optuna
from tqdm import tqdm
from multiprocessing import Pool as ProcessPool, cpu_count
import signal
import sys
import pickle
from sklearn.cluster import MiniBatchKMeans  # Imported here for clustering
from sklearn.decomposition import TruncatedSVD  # For sparse dimension reduction
from pandas.api.types import is_sparse  # For check sparse

# Import shared functions from xgb_final.py
from xgb_final import (
    preprocess_dataframe,
    remove_outliers_iqr,
    remove_cluster_outliers
)

# Global var for checkpoint
CHECKPOINT_PATH = 'data/checkpoint_df.pkl'

def signal_handler(sig, frame):
    """Handle Ctrl+C: Save checkpoint if df exists, then exit."""
    global df  # Assume df is global or pass it; for simplicity, if defined
    if 'df' in globals():
        print("Ctrl+C detected. Saving checkpoint...")
        df.to_pickle(CHECKPOINT_PATH)
        print("Checkpoint saved.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def fetch_quotes_from_db(table_name='ai_quotes'):
    """Fetch all quotes from MySQL DB (no LIMIT for full 100000+)."""
    # Load .env from project root (Price_Prediction/.env)
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path)
    
    conn = mysql.connector.connect(
        host=os.getenv('HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('PASSWORD'),
        database=os.getenv('DB_NAME')
    )
    cur = conn.cursor(dictionary=True)  # Dict cursor for easy dicts
    
    # Debug: Check if table exists
    cur.execute(f"SHOW TABLES LIKE '{table_name}'")
    table_exists = cur.fetchone()
    print(f"Table '{table_name}' exists: {bool(table_exists)}")
    
    if not table_exists:
        raise ValueError(f"Table '{table_name}' does not exist in DB.")
    
    cur.execute(f"""
        SELECT * FROM {table_name}
        WHERE proposed_price_accepted IS NOT NULL
        ORDER BY quote_no DESC  # Get newest first
    """)  # No LIMIT for all
    quotes = cur.fetchall()
    cur.close()
    conn.close()
    print(f"Fetched {len(quotes)} quotes from DB.")  # Debug count
    return quotes

def process_quote(q):
    """Process single quote for multiprocessing."""
    subprojects = q.get("subproject", [])
    if isinstance(subprojects, str):
        subprojects = json.loads(subprojects)  # Parse if JSON string from DB
    if not subprojects:
        return None
    
    # Parse proposed_deadlines
    proposed_deadlines = q.get("proposed_deadlines", [])
    if isinstance(proposed_deadlines, str):
        try:
            proposed_deadlines = json.loads(proposed_deadlines)
        except json.JSONDecodeError:
            proposed_deadlines = []
    if not isinstance(proposed_deadlines, list):
        proposed_deadlines = []
    try:
        proposed_deadlines = [float(d) for d in proposed_deadlines]
    except (ValueError, TypeError):
        proposed_deadlines = []
    
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
    
    min_proposed_deadline = min(proposed_deadlines) if proposed_deadlines else None
    
    return {
        "quote_no": q["quote_no"],
        "source_language": primary_source,
        "target_language": primary_target,
        "unit": primary_unit,
        "number_of_unit": total_units,
        "document_type": primary_doc,
        "num_subprojects": num_subprojects,
        "num_unique_langs": num_unique_langs,
        "num_unique_docs": num_unique_docs,
        "requested_deadline": q.get("proposed_deadlines_accepted"),
        "min_proposed_deadline": min_proposed_deadline,
        "proposed_price_accepted": q["proposed_price_accepted"]
    }

def build_dataframe(accepted_quotes):
    """Build dataframe with multiprocessing and tqdm."""
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        return pd.read_pickle(CHECKPOINT_PATH)
    
    with ProcessPool(processes=cpu_count()) as pool:
        rows = list(tqdm(pool.imap(process_quote, accepted_quotes), total=len(accepted_quotes), desc="Processing quotes"))
    
    rows = [r for r in rows if r is not None]  # Filter None
    df = pd.DataFrame(rows)
    print(f"Aggregated quotes: {df.shape[0]}")
    
    # Save checkpoint
    df.to_pickle(CHECKPOINT_PATH)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")
    return df

def apply_clustering(df):
    """Apply clustering after one-hot encoding (memory-efficient version with sparse and TruncatedSVD)."""
    print("Preparing features for clustering...")
    features = df.drop(columns=[
        "proposed_price_accepted", "quote_no"  # Drop unique ID to reduce cols
    ]).copy()
    features = pd.get_dummies(features, drop_first=True, sparse=True)  # Sparse to save memory
    features.fillna(-999, inplace=True)
    print(f"Features shape after sparse one-hot: {features.shape}")  # Debug
    # Convert sparse columns to dense
    for col in features.columns:
        if is_sparse(features[col].dtype):
            features[col] = features[col].sparse.to_dense()
    stds = features.std()
    features_std = features[stds[stds > 0].index]
    features_std = (features_std - features_std.mean()) / features_std.std()
    df = df.loc[features_std.index]  # Align index if needed
    
    # Apply TruncatedSVD for sparse reduction
    print("Applying TruncatedSVD to reduce dimensions...")
    svd = TruncatedSVD(n_components=500, random_state=42)  # Increased for better variance
    features_reduced = svd.fit_transform(features_std)
    print(f"Reduced shape after SVD: {features_reduced.shape}")
    print(f"Explained variance ratio cumulative: {svd.explained_variance_ratio_.cumsum()[-1]:.2f}")  # Debug variance retained
    
    kmeans = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=512, n_init="auto")  # Smaller batch for memory
    df["cluster"] = kmeans.fit_predict(features_reduced)
    print("Clustering completed.")
    return df

def tune_catboost(X_train, y_train, cat_cols, n_trials=100):  # Increased to 100 for better tuning
    """Tune CatBoost (from catboost_final.py)."""
    print("Starting Optuna tuning...")
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
    print("Optuna tuning completed.")
    return study.best_trial.params

def main_ai_prediction(table_name='ai_quotes'):
    """Main: Fetch, process, tune/train CatBoost on full, evaluate."""
    print("Starting fetch...")
    accepted_quotes = fetch_quotes_from_db(table_name)
    print("Fetch completed.")
    global df
    df = build_dataframe(accepted_quotes)
    print("Starting preprocess...")
    df = preprocess_dataframe(df)
    print("Preprocess completed.")
    print("Starting outliers removal...")
    df = remove_outliers_iqr(df)
    print("Outliers removal completed.")
    print("Starting clustering...")
    df = apply_clustering(df)
    print("Clustering completed.")
    print("Starting cluster outliers removal...")
    df = remove_cluster_outliers(df)
    print("Cluster outliers removal completed.")
    
    y = np.log1p(df["proposed_price_accepted"])
    X = df.drop(columns=["proposed_price_accepted", "quote_no"])  # Drop quote_no to avoid float conversion error
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cat_cols = ["source_language", "target_language", "unit", "lang_pair", "doc_category", "doc_subcategory"]
    
    best_params = tune_catboost(X_train, y_train, cat_cols)
    best_params.update({"loss_function": "MAE", "verbose": 100, "random_seed": 42})
    
    print("Starting final model training...")
    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    model = CatBoostRegressor(**best_params)
    model.fit(train_pool)
    print("Final model training completed.")
    
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
    
    comparison_df.to_csv("ai_predictions.csv", index=False)
    model.save_model("new_cat_model.cbm")  # Save new trained model

if __name__ == "__main__":
    main_ai_prediction('ai_quotes')