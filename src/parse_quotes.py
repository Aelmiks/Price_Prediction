# src/parse_quotes.py
# Aggregate raw quotes JSON into one row per quote_no.
import os, json, gzip, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

def _open_json_any(path: Path):
    """Open .json, .json.gz, or .zip (containing a single .json) and return parsed object."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            # pick the first .json file
            json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
            if not json_names:
                raise ValueError("No .json file found inside the ZIP.")
            with zf.open(json_names[0]) as f:
                return json.load(f)
    # plain .json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def most_frequent(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return Counter(values).most_common(1)[0][0] if values else None

def safe_min(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return min(xs) if xs else None

def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def load_and_aggregate_quotes(input_path: str, out_json: str = "data/quotes_grouped.json", out_csv: str = "data/quotes_grouped.csv"):
    raw = _open_json_any(Path(input_path))
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of quote dicts.")

    rows = []
    for q in raw:
        # keep quotes that have at least a quote_no
        qno = q.get("quote_no")
        if not qno:
            continue

        subprojects = q.get("subproject") or []
        # tolerate both dict (single) or list
        if isinstance(subprojects, dict):
            subprojects = [subprojects]

        # collect subproject-level fields
        srcs, tgts, units, docs, units_count = [], [], [], [], 0.0
        for sp in subprojects:
            srcs.append(sp.get("source_language"))
            tgts.append(sp.get("target_language"))
            units.append(sp.get("unit"))
            doc = sp.get("document_type")
            if doc in (None, "", "None"):
                doc = None
            docs.append(doc)
            nu = sp.get("number_of_unit")
            try:
                units_count += float(nu) if nu is not None else 0.0
            except Exception:
                pass

        primary_source = most_frequent(srcs)
        primary_target = most_frequent(tgts)
        primary_unit   = most_frequent(units) or "page"
        primary_doc    = most_frequent([d for d in docs if d is not None]) or "Unknown"

        num_subprojects  = len(subprojects) if subprojects else 0
        num_unique_langs = len(set([x for x in srcs if x] + [y for y in tgts if y]))
        num_unique_docs  = len(set([d for d in docs if d]))

        pdead_list = as_list(q.get("proposed_deadlines"))
        requested_deadline = q.get("proposed_deadlines_accepted")
        min_proposed_deadline = safe_min(pdead_list)

        rows.append({
            "quote_no": qno,
            "source_language": primary_source,
            "target_language": primary_target,
            "unit": primary_unit,
            "number_of_unit": float(units_count),
            "document_type": primary_doc,
            "num_subprojects": int(num_subprojects),
            "num_unique_langs": int(num_unique_langs),
            "num_unique_docs": int(num_unique_docs),
            "deadline_unit": q.get("deadline_unit"),
            "requested_deadline": requested_deadline,
            "min_proposed_deadline": min_proposed_deadline,
            "currency": q.get("currency"),
            "proposed_price_accepted": q.get("proposed_price_accepted"),
            "price_accepted": q.get("price_accepted"),
            # keep for potential time series if present in JSON
            "created_at": q.get("created_at"),
        })

    df = pd.DataFrame(rows)

    # Final tidy-ups
    if "document_type" in df.columns:
        df["document_type"] = df["document_type"].fillna("Unknown")
    df["lang_pair"] = (df.get("source_language", pd.Series(["?"]*len(df))).astype(str)
                       + "→" + df.get("target_language", pd.Series(["?"]*len(df))).astype(str))

    # Save outputs
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_json, orient="records", indent=2, force_ascii=False)
    df.to_csv(out_csv, index=False)

    # Basic report
    n_total = len(df)
    n_price = df["proposed_price_accepted"].notnull().sum() if "proposed_price_accepted" in df.columns else 0
    print(f"Aggregated quotes: {n_total} (with accepted price: {n_price})")
    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV : {out_csv}")
    return df

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--inpath", default="data/quoteai_last_50000.json",
                   help="Path to raw input (.json | .json.gz | .zip)")
    p.add_argument("--out_json", default="data/quotes_grouped.json")
    p.add_argument("--out_csv",  default="data/quotes_grouped.csv")
    args = p.parse_args()

    load_and_aggregate_quotes(args.inpath, args.out_json, args.out_csv)