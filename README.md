# Translation Quote Price Prediction

**Explore interpretable machine learning for estimating translation quote prices.**

Python project covering quote aggregation, exploratory data analysis, feature engineering and regression experiments with **XGBoost, CatBoost and Explainable Boosting Machines (EBM)**. A Streamlit interface demonstrates inference with a compatible locally trained CatBoost model.

## Business problem

A translation quote depends on the language pair, document category, volume and turnaround time. This project turns nested quote records into tabular features and compares model families for estimating accepted prices. Predictions support analysis; they are not validated commercial pricing rules.

## Workflow

```text
Authorized quote export → aggregation → feature engineering → model comparison
                                                       └──→ interpretation → demo interface
```

| Area | Work represented in this repository |
| --- | --- |
| Data preparation | JSON, compressed JSON and ZIP ingestion; aggregation by quote identifier |
| Features | Language pair, document category, units, volume, deadlines and quote composition |
| Exploration | Price distributions, categorical analysis and data-quality investigation |
| Modeling | XGBoost, CatBoost, EBM and ensemble blending experiments |
| Optimization | Optuna searches and cross-validation in selected training scripts |
| Interpretation | SHAP explanations and EBM feature effects |
| Evaluation | MAE, MAPE, SMAPE and share of predictions within a €5 error band |
| Interface | Schema-aligned CatBoost inference, feature contributions and scenario export |

## Repository guide

- `src/parse_quotes.py`: aggregate raw quote records into one row per quote.
- `src/xgb_final.py`, `src/catboost_final.py`, `src/ebm_final.py`: model experiments.
- `src/ensemble_blending.py`: model combination experiment.
- `src/app.py`, `src/choices.json`: Streamlit inference interface and display choices.
- `notebooks/`: EDA and model exploration, including locally developed EBM work.
- Legacy database migration helpers are preserved locally and omitted from this public portfolio export.
- `data/README.md`: expected input fields and data handling.

## Set up

The original local environment used Python 3.9.7. Recreating it with current package releases may require a newer Python version or compatible package pins. The dependency list is not a fully locked, cross-platform environment.

```bash
git clone https://github.com/Aelmiks/Price_Prediction.git
cd Price_Prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Provide your own authorized dataset as `data/quoteai_last_50000.json`. No client data is distributed with this repository. Then, from the repository root:

```bash
python src/parse_quotes.py --inpath data/quoteai_last_50000.json
python src/catboost_final.py
```

The aggregation command writes `data/quotes_grouped.json` and `.csv`. The final model scripts read the raw JSON and apply their own preprocessing. CatBoost training writes `cat_model.cbm` in the repository root; the interface searches that location as well as `src/`.

Training includes a substantial hyperparameter search. Review and reduce `n_trials` for a small local experiment. Choose one model experiment at a time.

With a compatible CatBoost model available:

```bash
python -m streamlit run src/app.py
```

The interface derives feature order and categorical columns from the model, but uses defaults for some engineered features. Align these with the training pipeline before interpreting predictions as reliable estimates.

Explore notebooks from their working directory:

```bash
cd notebooks
jupyter notebook
```

The optional `src/main_ai_prediction.py` database workflow requires a private `.env` based on `.env.example` and an authorized development database. Legacy business migration scripts are not included in this public export.

## Evaluation and limitations

Metric calculations are present in the scripts; this export does **not** claim a verified accuracy figure. Private notebook outputs, plots, quote tables, pretrained artifacts and original datasets are intentionally omitted.

Several experiments perform preprocessing, clustering or target-based outlier filtering before the train/test split. This can introduce leakage or make scores optimistic. A rigorous follow-up should split first, fit transformations on training data only, retain an untouched holdout and evaluate currencies and time periods consistently. Percentage metrics also require care with zero or very small prices.

The scripts are exploratory variants rather than one unified training pipeline. Production deployment, end-to-end training and database connections were not validated during publication.

## Data and repository scope

Secrets, datasets, training logs, notebook outputs, serialized models and environments are excluded through `.gitignore`. `src/choices.json` is retained as useful application configuration. No license file is included in this repository.
