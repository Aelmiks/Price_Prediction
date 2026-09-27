# Local data contract

Supply an authorized JSON array of quote objects locally; no real quote records are included here.

The scripts use these quote-level fields: `quote_no`, `proposed_price_accepted`, `proposed_deadlines_accepted`, `proposed_deadlines` and `subproject`. Aggregation also retains optional `currency`, `deadline_unit`, `price_accepted` and `created_at` values.

Each subproject may include `source_language`, `target_language`, `unit`, `number_of_unit` and `document_type`. Model experiments generally expect `subproject` to be a list; the aggregation helper also accepts a single object. Some experiments split `document_type` as `category|subcategory`.

Raw input defaults to `data/quoteai_last_50000.json`; the aggregator writes `quotes_grouped.json` and `quotes_grouped.csv`. All data files in this directory are ignored. Do not add customer identifiers, contact information or real quote exports to Git.
