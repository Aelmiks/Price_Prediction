import json
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from scipy.stats import zscore
from sklearn.cluster import KMeans


def load_and_filter_quotes(json_path):
    with open(json_path, "r") as f:
        raw_quotes = json.load(f)
    return [q for q in raw_quotes if q.get("proposed_price_accepted") is not None]

def build_dataframe(accepted_quotes):
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
    return pd.DataFrame(rows)

def preprocess_dataframe(df):
    # Convert to numeric (e.g., "7" -> 7), coerce invalid entries to NaN
    df["requested_deadline"] = pd.to_numeric(df["requested_deadline"], errors="coerce")
    # Replace missing values with median (done in one operation)
    df["requested_deadline"].fillna(df["requested_deadline"].median(), inplace=True)
    # Binary feature: is_express = 1 if the requested time is 2 days or less
    df["is_express"] = (df["requested_deadline"] <= 2).astype(int)
    df["flexibility"] = df["requested_deadline"] - df["min_proposed_deadline"]
    df["urgency"] = 1 / (df["requested_deadline"] + 1)
    # Flag for rare or incomplete document types
    df["has_rare_doc"] = df["document_type"].apply(lambda d: isinstance(d, str) and "contract" not in d.lower())
    df["lang_pair"] = (df["source_language"] + "_to_" + df["target_language"]).astype("category").cat.codes
    # Splitting document_type category
    df["doc_category"] = df["document_type"].str.split("|").str[0]
    df["doc_subcategory"] = df["document_type"].str.split("|").str[1]
    df.drop(columns=["document_type"], inplace=True)
    return df


def remove_outliers_iqr(df):
    # Remove outliers using IQR (Interquartile Range) method
    q1 = df["proposed_price_accepted"].quantile(0.25)
    q3 = df["proposed_price_accepted"].quantile(0.75)
    iqr = q3 - q1
    return df[
        (df["proposed_price_accepted"] >= q1 - 1.5 * iqr) &
        (df["proposed_price_accepted"] <= q3 + 1.5 * iqr)
    ]


def apply_clustering(df):
    # Apply KMeans clustering to identify patterns in the data
    features = df.drop(columns=[
        "proposed_price_accepted", "source_language", "target_language",
        "doc_category", "doc_subcategory"
    ]).copy()
    # One-hot encoding for categorical features
    features = pd.get_dummies(features, drop_first=True)
    features.fillna(-999, inplace=True)
    stds = features.std()
    features_std = features[stds[stds > 0].index]
    features_std = (features_std - features_std.mean()) / features_std.std()
    df = df.loc[features_std.index]
    kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(features_std)
    return df


def remove_cluster_outliers(df):
    # Remove outliers within each cluster using Z-score
    df["log_price"] = np.log1p(df["proposed_price_accepted"])
    df["z_score"] = df.groupby("cluster")["log_price"].transform(zscore)
    return df[df["z_score"].abs() < 3].drop(columns=["cluster", "log_price", "z_score"])


def encode_features(df):
    # One-hot encoding for categorical features
    for col in ["lang_pair", "unit", "doc_category", "doc_subcategory"]:
        df[col] = df[col].astype("category").cat.codes
    return df

def train_model_with_optuna(X, y, n_trials=30):
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
        # Add cross-validation to evaluate the model
        scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
        return -scores.mean()

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params


def final_model_and_evaluation(X_train, X_test, y_train, y_test, best_params):
    # Train the final model with the best parameters
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    y_test_exp = np.expm1(y_test)
    y_pred_exp = np.expm1(y_pred_log)

    comparison_df = pd.DataFrame({
        "Actual price (€)": y_test_exp,
        "Predicted price (€)": y_pred_exp
    })
    comparison_df["Within 5€?"] = (comparison_df["Actual price (€)"] - comparison_df["Predicted price (€)"]).abs() <= 5

    return model, comparison_df


if __name__ == "__main__":
    # Load and preprocess data
    accepted_quotes = load_and_filter_quotes("data/quoteai_last_50000.json")
    if not accepted_quotes:
        raise ValueError("No valid quotes found in the provided JSON file.")
    df = build_dataframe(accepted_quotes)
    df = preprocess_dataframe(df)
    df = remove_outliers_iqr(df)
    df = apply_clustering(df)
    df = remove_cluster_outliers(df)
    df = encode_features(df)

    # Prepare features and target variable
    y = np.log1p(df["proposed_price_accepted"])
    X = df.drop(columns=["proposed_price_accepted", "source_language", "target_language", "doc_category", "doc_subcategory"])
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model using Optuna for hyperparameter optimization
    best_params = train_model_with_optuna(X_train, y_train)
    # Final model training and evaluation
    model, comparison_df = final_model_and_evaluation(X_train, X_test, y_train, y_test, best_params)

    print(comparison_df.head(25))