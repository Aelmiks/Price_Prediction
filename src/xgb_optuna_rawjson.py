import json
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import zscore
from sklearn.cluster import KMeans

with open("../data/quoteai_last_50000.json", "r") as f:
    raw_quotes = json.load(f)

count = sum(1 for q in raw_quotes if q.get("proposed_price_accepted") is not None)
print("Quotes with accepted price:", count)
print(json.dumps(raw_quotes[0], indent=2))

accepted_quotes = [q for q in raw_quotes if q.get("proposed_price_accepted") is not None]
rows = []
for q in accepted_quotes:
    subprojects = q.get("subproject", [])
    for sub in subprojects:
        if all(k in sub for k in ["source_language", "target_language", "unit", "number_of_unit", "document_type"]):
            rows.append({
                "source_language": sub["source_language"],
                "target_language": sub["target_language"],
                "unit": sub["unit"],
                "number_of_unit": sub["number_of_unit"],
                "document_type": sub["document_type"],
                "requested_deadline": q.get("proposed_deadlines_accepted"),
                "min_proposed_deadline": min(q.get("proposed_deadlines", [])) if q.get("proposed_deadlines") else None,
                "proposed_price_accepted": q["proposed_price_accepted"]
            })

df = pd.DataFrame(rows)
print("Shape:", df.shape)
print(df.head())

# Convert to numeric (e.g., "7" -> 7), coerce invalid entries to NaN
df["requested_deadline"] = pd.to_numeric(df["requested_deadline"], errors="coerce")

# Replace missing values with median (done in one operation)
median_deadline = df["requested_deadline"].median()
df["requested_deadline"] = df["requested_deadline"].fillna(median_deadline)

# Binary feature: is_express = 1 if the requested time is 2 days or less
df["is_express"] = (df["requested_deadline"] <= 2).astype(int)

df["flexibility"] = df["requested_deadline"] - df["min_proposed_deadline"]
df["urgency"] = 1 / (df["requested_deadline"] + 1) # +1 to avoid division by zero

# Flag for rare or incomplete document types
df["has_rare_doc"] = df["document_type"].apply(lambda d: isinstance(d, str) and "contract" not in d.lower())

# Feature lang-pair
df["lang_pair"] = df["source_language"] + "_to_" + df["target_language"]
df["lang_pair"] = df["lang_pair"].astype("category").cat.codes

# Splitting document_type category
df["doc_category"] = df["document_type"].str.split("|").str[0]
df["doc_subcategory"] = df["document_type"].str.split("|").str[1]
df.drop(columns=["document_type"], inplace=True)

# Remove outliers using IQR (Interquartile Range) method
q1 = df["proposed_price_accepted"].quantile(0.25)
q3 = df["proposed_price_accepted"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df["proposed_price_accepted"] >= lower_bound) & (df["proposed_price_accepted"] <= upper_bound)]

# Step 1 — Selecting features for clustering
features_for_clustering = df.drop(columns=[
    "proposed_price_accepted", "source_language", "target_language",
    "doc_category", "doc_subcategory"
]).copy()

# One-hot encoding for categorical features
categorical_cols = ["lang_pair", "unit", "doc_category", "doc_subcategory"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Replace remaining NaNs (if any)
features_for_clustering = features_for_clustering.fillna(-999)

# Step 2 — Filtering columns with std = 0
stds = features_for_clustering.std()
cols_with_variance = stds[stds > 0].index.tolist()
features_std = features_for_clustering[cols_with_variance]

# Standardization
features_std = (features_std - features_std.mean()) / features_std.std()

# Safety check
assert not features_std.isna().any().any(), "NaNs remain in features_std"

# Step 3 — Clustering
kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
df["cluster"] = kmeans.fit_predict(features_std)

# Step 4 — Removing outliers inside each cluster
df["log_price"] = np.log1p(df["proposed_price_accepted"])
df["z_score"] = df.groupby("cluster")["log_price"].transform(zscore)
df = df[df["z_score"].abs() < 3]

# Cleaning
df.drop(columns=["cluster", "log_price", "z_score"], inplace=True)

# Encoding categories
for col in ["lang_pair", "unit", "doc_category", "doc_subcategory", "doc_category"]:
    df[col] = df[col].astype("category").cat.codes

# Target transformation
y = np.log1p(df["proposed_price_accepted"])
X = df.drop(columns=["proposed_price_accepted", "source_language", "target_language", "doc_category", "doc_subcategory"])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the Optuna Goal
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "random_state": 42
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

# Launch of optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
print("Best trial:", study.best_trial.params)
print(X.dtypes)

# Retraining with best parameters
best_params = study.best_trial.params
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)

# Predictions
y_pred_log = best_model.predict(X_test)
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred_log)

comparison_df = pd.DataFrame({
    "Actual price (€)": y_test_exp,
    "Predicted price (€)": y_pred_exp
})
comparison_df["Within 5€?"] = (comparison_df["Actual price (€)"] - comparison_df["Predicted price (€)"]).abs() <= 5

print(comparison_df.head(30))

# Displaying the percentage of correct predictions at €5
comparison_df["Within 5€?"] = (comparison_df["Actual price (€)"] - comparison_df["Predicted price (€)"]).abs() <= 5
within_5 = comparison_df["Within 5€?"].mean() * 100
print(f"\nPredictions within 5€: {within_5:.2f}%")