# app.py
from pathlib import Path
import json
import tempfile
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, Pool

# -------------------------------------------------------------------
# Backward-compatible caching for old/new Streamlit
def _cache_resource():
    return getattr(st, "cache_resource", None) or st.experimental_singleton

def _cache_data():
    return getattr(st, "cache_data", None) or st.experimental_memo
# -------------------------------------------------------------------

st.set_page_config(page_title="Translation Quote Pricing", layout="centered")

@_cache_resource()
def load_model() -> CatBoostRegressor:
    """Load CatBoost model (.cbm) from common locations."""
    m = CatBoostRegressor()
    here = Path(__file__).resolve().parent
    candidates = [
        here / "new_cat_model.cbm",
        here / "cat_model.cbm",
        here / "model.cbm",
        here.parent / "new_cat_model.cbm",
        here.parent / "cat_model.cbm",
        here.parent / "model.cbm",
    ]
    found = next((p for p in candidates if p.exists()), None)
    if not found:
        st.error(
            "Aucun fichier modèle .cbm trouvé (new_cat_model.cbm / cat_model.cbm / model.cbm). "
            "Place le fichier dans src/ ou à la racine du projet."
        )
        st.stop()
    m.load_model(str(found))
    return m

@_cache_data()
def load_choices() -> dict:
    """Load select-box choices from choices.json (src/ or project root)."""
    here = Path(__file__).resolve().parent
    candidates = [here / "choices.json", here.parent / "choices.json"]
    found = next((p for p in candidates if p.exists()), None)
    if not found:
        st.error("Fichier choices.json introuvable (attendu dans src/ ou à la racine).")
        st.stop()
    with open(found, "r", encoding="utf-8") as f:
        return json.load(f)

def _extract_schema_from_json(model: CatBoostRegressor) -> Tuple[List[str], List[str]]:
    """Fallback: read CatBoost JSON to derive exact feature order and categorical names."""
    def _feat_name(obj: dict, idx: int) -> str:
        return obj.get("name") or obj.get("feature_id") or obj.get("featureId") or str(idx)

    with tempfile.TemporaryDirectory() as td:
        jpath = Path(td) / "model.json"
        model.save_model(str(jpath), format="json")
        meta = json.loads(jpath.read_text(encoding="utf-8"))

    finfo = meta.get("features_info", {})
    feats = []  # (flat_idx, name, kind)

    for f in finfo.get("float_features", []):
        idx = int(f["flat_feature_index"])
        feats.append((idx, _feat_name(f, idx), "float"))
    for f in finfo.get("cat_features", []):
        idx = int(f["flat_feature_index"])
        feats.append((idx, _feat_name(f, idx), "cat"))

    feats_sorted = sorted(feats, key=lambda t: t[0])
    expected_order = [name for _, name, _ in feats_sorted]
    cat_names = [name for _, name, kind in feats_sorted if kind == "cat"]
    return expected_order, cat_names

def extract_schema_from_model(model: CatBoostRegressor) -> Tuple[List[str], List[str]]:
    """
    Preferred path: use CatBoost API (feature_names_ and get_cat_feature_indices).
    Fallback to JSON parsing if API is unavailable.
    """
    try:
        names = list(getattr(model, "feature_names_", []))
        if names and all(isinstance(n, str) for n in names):
            try:
                cat_idx = list(model.get_cat_feature_indices())
                cat_names = [names[i] for i in cat_idx]
            except Exception:
                cat_names = []
            return names, cat_names
    except Exception:
        pass
    return _extract_schema_from_json(model)

def default_value_for(_: str, is_cat: bool) -> Union[float, str]:
    """Neutral safe defaults for missing features."""
    return "-999" if is_cat else 0.0

# ===================== LOAD =====================
model = load_model()
choices = load_choices()
expected_order, cat_names = extract_schema_from_model(model)

st.title("Prédiction de prix (démo)")

# ===================== UI ============================
c1, c2 = st.columns(2)
source = c1.selectbox("Langue source", choices["source_language"])
target = c2.selectbox("Langue cible", choices["target_language"])

doc_cat = st.selectbox("Catégorie documentaire", choices["doc_category"])
doc_sub = st.selectbox("Sous-catégorie", choices.get("doc_subcategory", []))
unit = st.selectbox("Unité", choices["unit"])

# Unit-specific number input (cleaner labels for the demo)
if unit == "word":
    nb_unit = st.number_input("Nombre de mots", min_value=50, value=250, step=50)
elif unit == "minute":
    nb_unit = st.number_input("Durée (minutes)", min_value=1, value=5, step=1)
else:  # page
    nb_unit = st.number_input("Nombre de pages", min_value=1, value=1, step=1)

deadline = st.number_input("Deadline demandée (jours)", min_value=0, value=2, step=1)

# === Derived features (keep in sync with training) ===
is_express = float(deadline <= 2)
flexibility = float(deadline - 1.0)
has_rare_doc = float("contract" not in (doc_cat or "").lower())
lang_pair = f"{source}_to_{target}"
min_proposed_deadline = float(deadline)
num_subprojects = 1.0
num_unique_langs = float(len({source, target}))
num_unique_docs = 1.0
cluster = 0.0
lang_rarity = 1.8  # harmless if not used by the model

# Optional engineered combos (included if present in the model)
urgency = 1.0 / (float(deadline) + 1.0)
unit_interaction = float(nb_unit) * num_subprojects
delay_interaction = float(deadline) * num_unique_langs

# Human-readable row
row_named = {
    "source_language": source,
    "target_language": target,
    "unit": unit,
    "number_of_unit": float(nb_unit),
    "requested_deadline": float(deadline),
    "is_express": is_express,
    "flexibility": flexibility,
    "urgency": urgency,
    "unit_interaction": unit_interaction,
    "delay_interaction": delay_interaction,
    "has_rare_doc": has_rare_doc,
    "lang_pair": lang_pair,
    "doc_category": doc_cat,
    "doc_subcategory": doc_sub,
    "min_proposed_deadline": min_proposed_deadline,
    "num_subprojects": num_subprojects,
    "num_unique_langs": num_unique_langs,
    "num_unique_docs": num_unique_docs,
    "cluster": cluster,
    "lang_rarity": lang_rarity,
}

# ===================== STRICT ALIGNMENT =====================
# Build exactly the columns the model expects, in the expected order
X_one = pd.DataFrame(columns=expected_order)
for col in expected_order:
    is_cat = col in cat_names
    X_one.at[0, col] = row_named.get(col, default_value_for(col, is_cat))

# Dtypes: by names (robust)
float_cols = [c for c in expected_order if c not in cat_names]
cat_cols = list(cat_names)

if float_cols:
    X_one[float_cols] = (
        X_one[float_cols].apply(pd.to_numeric, errors="coerce").astype(float).fillna(0.0)
    )
for c in cat_cols:
    X_one[c] = X_one[c].astype(str)

# Indices of categorical columns for CatBoost Pool
cat_idx_for_pool = [X_one.columns.get_loc(c) for c in cat_cols]

# ===================== OPTIONAL DEBUG (hidden by default) =====================
show_dbg = st.checkbox("🔎 Debug (schéma & dtypes)")
if show_dbg:
    with st.expander("Debug", expanded=True):
        st.write("expected_order (extrait):", expected_order[:25])
        st.write("cat_names:", cat_cols)
        st.write("Dtypes:", X_one.dtypes.astype(str).to_dict())
        st.write("Colonnes envoyées (extrait):", list(X_one.columns)[:25])
        missing_in_named = [c for c in expected_order if c not in row_named]
        extra_named = [c for c in row_named.keys() if c not in expected_order]
        st.write("Colonnes attendues non trouvées dans row_named:", missing_in_named)
        st.write("Clés de row_named non utilisées par le modèle:", extra_named)

# ===================== PREDICTION =====================
if st.button("Prédire"):
    with st.spinner("Calcul du prix..."):
        pool = Pool(X_one, cat_features=cat_idx_for_pool)
        y_log = model.predict(pool)
        price = float(np.expm1(y_log))
        # Safety floor + neat rounding for the demo
        price = max(5.0, round(price, 2))

    st.metric("Prix recommandé (€)", f"{price:,.2f}".replace(",", " "))

    with st.expander("Détail de l’entrée envoyée au modèle"):
        st.dataframe(X_one)

# ===================== SHAP EXPLANATION (optional) =====================
if st.checkbox("🔍 Expliquer la prédiction"):
    pool1 = Pool(X_one, cat_features=cat_idx_for_pool)
    shap_vals = model.get_feature_importance(pool1, type="ShapValues")
    # shap_vals shape: (n_samples, n_features + 1) -> last column is expected value (base)
    base_value = float(shap_vals[0, -1])
    contrib = pd.Series(shap_vals[0, :-1], index=expected_order)
    top_abs = contrib.abs().sort_values(ascending=False).head(10)
    st.caption(f"Base value (log): {base_value:.4f}")
    st.write("Top 10 absolute contributions (log):")
    st.bar_chart(top_abs.to_frame("abs_contrib"))  # nice, quick visual
    st.write("Top 20 signed contributions (log):")
    st.dataframe(contrib.sort_values(key=np.abs, ascending=False).head(20).to_frame("contrib"))

# ===================== EXPORT CURRENT SCENARIO =====================
if st.button("Exporter le scénario (JSON)"):
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in row_named.items()}
    st.download_button(
        "Télécharger",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name="scenario_pricing.json",
        mime="application/json",
    )

# ===================== SIDEBAR: MODEL INFO =====================
with st.sidebar:
    st.caption("🧠 Modèle")
    try:
        st.write("Features:", len(expected_order))
        st.write("Catégorielles:", len(cat_cols))
    except Exception:
        pass