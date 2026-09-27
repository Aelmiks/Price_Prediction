import pandas as pd
import numpy as np
import random
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1 – Generating synthetic data
def generate_synthetic_data(n=1000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    languages = ["French", "English", "Spanish", "German", "Arabic"]
    document_types = ["sales_contract", "passport", "birth_certificate", "academic_transcript"]
    units = ["page", "word"]
    currencies = ["eur", "usd"]

    data = []
    for _ in range(n):
        src = random.choice(languages)
        tgt = random.choice([lang for lang in languages if lang != src])
        unit = random.choice(units)
        nb_unit = random.randint(1, 50)
        doc = random.choice(document_types)
        delay = random.randint(1, 10)
        cur = random.choice(currencies)

        base_price = 20 * nb_unit if unit == "page" else 0.2 * nb_unit
        variation = random.randint(-20, 20)
        proposals = [base_price + variation + offset for offset in [-10, 0, 10]]
        accepted_index = random.randint(0, 2)

        # Simuler l’acceptation ou le refus (80% acceptation, 20% refus)
        accepted = random.random() > 0.2
        proposed_price_accepted = proposals[accepted_index] if accepted else None

        data.append({
            "source_language": src,
            "target_language": tgt,
            "unit": unit,
            "number_of_unit": nb_unit,
            "document_type": doc,
            "deadline_unit": "days",
            "requested_deadline": delay,
            "currency": cur,
            "proposed_price": proposals,
            "proposed_price_accepted": proposed_price_accepted,
            "price_accepted": accepted
        })

    return pd.DataFrame(data)


# Step 2 – Data Preparation
def prepare_data_for_regression(df):

    # Here we filter the refusals because we train a regression model on the accepted price.
    # These lines will be used in other models to predict customer acceptance.
    df = df[df["proposed_price_accepted"].notnull()].copy()

    for col in ["source_language", "target_language", "unit", "document_type"]:
        df[col] = df[col].astype("category")

    df["price"] = df["proposed_price_accepted"]
    X = df[[
        "source_language",
        "target_language",
        "unit",
        "number_of_unit",
        "document_type",
        "requested_deadline"
    ]]
    y = df["price"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Step 3 – Training the model
def train_ebm(X_train, y_train):
    ebm = ExplainableBoostingRegressor(random_state=0)
    ebm.fit(X_train, y_train)
    return ebm


# Step 4 – Visualizing Importance
def plot_feature_importance(ebm):
    features = ebm.feature_names
    raw_scores = ebm.term_scores_
    importances = [np.mean(s) if isinstance(s, (list, np.ndarray)) else s for s in raw_scores]

    sns.barplot(x=importances, y=features)
    plt.title("Average importance of variables according to EBM")
    plt.xlabel("Average contribution")
    plt.tight_layout()
    plt.show()


# Step 5 – Prediction from a JSON TIPS
def predict_from_json(ebm, json_input):
    features_to_use = [
        "source_language", "target_language",
        "unit", "number_of_unit",
        "document_type", "requested_deadline"
    ]
    X_new = pd.DataFrame([json_input])[features_to_use]
    predicted_price = ebm.predict(X_new)[0]
    print(f"Prix estimé : {predicted_price:.2f} €")
    return predicted_price


# Main script
if __name__ == "__main__":
    df = generate_synthetic_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    ebm = train_ebm(X_train, y_train)

    # Evaluation
    y_pred = ebm.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE : {mae:.2f} €")

    # Displaying importance
    plot_feature_importance(ebm)

    # JSON prediction test
    new_request = {
        "source_language": "French",
        "target_language": "English",
        "unit": "page",
        "number_of_unit": 12,
        "document_type": "sales_contract",
        "requested_deadline": 3
    }
    predict_from_json(ebm, new_request)