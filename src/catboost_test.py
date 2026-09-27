
import json
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool
from scipy.stats import zscore
from sklearn.cluster import KMeans

# Load the raw file
with open("../data/quoteai_last_50000.json", "r") as f:
    raw_quotes = json.load(f)

# Parsing quotes
accepted_quotes = [q for q in raw_quotes if q.get("proposed_price_accepted") is not None]
rows = []
for q in accepted_quotes:
    subprojects = q.get("subproject", [])
    for sub in subprojects:
        if all(k in sub for k in ["source_language", "target_language", "unit", "number_of_unit", "document_type"]):
            deadline_accepted = q.get("proposed_deadlines_accepted")
            if deadline_accepted is not None:
                rows.append({
                    "source_language": sub["source_language"],
                    "target_language": sub["target_language"],
                    "unit": sub["unit"],
                    "number_of_unit": sub["number_of_unit"],
                    "document_type": sub["document_type"],
                    "requested_deadline": deadline_accepted,
                    "proposed_price_accepted": q["proposed_price_accepted"]
                })

df = pd.DataFrame(rows)

# Handle missing deadline
df["requested_deadline"] = pd.to_numeric(df["requested_deadline"], errors="coerce")
median_deadline = df["requested_deadline"].median()
df["requested_deadline"] = df["requested_deadline"].fillna(median_deadline)

# Feature engineering
df["is_express"] = (df["requested_deadline"] <= 2).astype(int)
df["flexibility"] = df["requested_deadline"] - 1
df["has_rare_doc"] = df["document_type"].apply(lambda d: isinstance(d, str) and "contract" not in d.lower())
df["lang_pair"] = df["source_language"] + "_to_" + df["target_language"]

df["doc_category"] = df["document_type"].str.split("|").str[0]
df["doc_subcategory"] = df["document_type"].str.split("|").str[1]
df.drop(columns=["document_type"], inplace=True)

# --- Removed outliers using IQR (InterQuantile Range) method ---
q1 = df["proposed_price_accepted"].quantile(0.25)
q3 = df["proposed_price_accepted"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df = df[(df["proposed_price_accepted"] >= lower_bound) & (df["proposed_price_accepted"] <= upper_bound)]

# --- Preparation for clustering ---
features_for_clustering = df.drop(columns=[
    "proposed_price_accepted",
    "source_language", "target_language",
    "doc_category", "doc_subcategory"
]).copy()

for col in features_for_clustering.columns:
    if features_for_clustering[col].dtype == "object":
        features_for_clustering[col] = features_for_clustering[col].astype("category").cat.codes

features_for_clustering = features_for_clustering.fillna(-999) # Replacing remaining NaNs (security)

# Filtering columns with std = 0
stds = features_for_clustering.std()
cols_with_variance = stds[stds > 0].index.tolist()
features_std = features_for_clustering[cols_with_variance]

# Standardization
features_std = (features_std - features_std.mean()) / features_std.std()

# --- Clustering + removal of outlier quotes by cluster ---
kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
df["cluster"] = kmeans.fit_predict(features_std)

df["log_price"] = np.log1p(df["proposed_price_accepted"])
df["z_score"] = df.groupby("cluster")["log_price"].transform(zscore)

df = df[df["z_score"].abs() < 3]
df.drop(columns=["cluster", "log_price", "z_score"], inplace=True) # Cleaning

# Define target and features
y = np.log1p(df["proposed_price_accepted"])
X = df.drop(columns=["proposed_price_accepted"])
X = X.fillna(-999)

# Identify categorical features for CatBoost
cat_cols = ["source_language", "target_language", "unit", "lang_pair", "doc_category", "doc_subcategory"]

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "loss_function": "MAE",
        "verbose": 0,
        "random_seed": 42
    }

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    maes = []

    for train_idx, valid_idx in cv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols)

        model = CatBoostRegressor(**params)
        model.fit(train_pool)
        y_pred = model.predict(valid_pool)
        maes.append(mean_absolute_error(y_valid, y_pred))

    return np.mean(maes)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
print("Best trial:")
print(study.best_trial.params)

best_params = study.best_trial.params
best_params.update({
    "loss_function": "MAE",
    "verbose": 100,
    "random_seed": 42
})

train_pool = Pool(X, y, cat_features=cat_cols)
final_model = CatBoostRegressor(**best_params)
final_model.fit(train_pool)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = CatBoostRegressor(**best_params)
model.fit(Pool(X_train, y_train, cat_features=cat_cols))

y_pred_log = model.predict(Pool(X_test, cat_features=cat_cols))
y_pred = np.expm1(y_pred_log)
y_test_exp = np.expm1(y_test)

mae = mean_absolute_error(y_test_exp, y_pred)
mape = np.mean(np.abs((y_test_exp - y_pred) / y_test_exp)) * 100
smape = 100 * np.mean(2 * np.abs(y_pred - y_test_exp) / (np.abs(y_test_exp) + np.abs(y_pred)))
mae_weighted = np.average(np.abs(y_test - y_pred_log), weights=y_test)

print(f"MAE: {mae:.2f} €")
print(f"MAPE: {mape:.2f} %")
print(f"SMAPE: {smape:.2f} %")
print(f"Weighted MAE (log-scale): {mae_weighted:.4f}")

comparison_df = pd.DataFrame({
    "Actual price (€)": y_test_exp,
    "Predicted price (€)": y_pred
})
comparison_df["Within 5€?"] = (comparison_df["Actual price (€)"] - comparison_df["Predicted price (€)"]).abs() <= 5

print(comparison_df.head(30))