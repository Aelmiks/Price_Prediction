import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder

def load_data(path="../data/quotes_grouped.json"):
    df = pd.read_json(path)
    df = df[df["proposed_price_accepted"].notnull()].copy()
    return df

def prepare_data(df):
    X = df[[
        "source_language", "target_language", "unit",
        "number_of_unit", "document_type", "requested_deadline"
    ]].copy()

    y = df["proposed_price_accepted"]

    encoder = OrdinalEncoder()
    X_encoded = X.copy()
    cat_cols = ["source_language", "target_language", "unit", "document_type"]
    X_encoded[cat_cols] = encoder.fit_transform(X[cat_cols])

    return train_test_split(X_encoded, y, test_size=0.2, random_state=42), encoder

def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def explain(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.plots.bar(shap_values, max_display=10)

if __name__ == "__main__":
    df = load_data()
    (X_train, X_test, y_train, y_test), encoder = prepare_data(df)

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE : {mae:.2f} €")

    explain(model, X_train)