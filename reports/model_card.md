# Model Card – Week 3 Baseline

## Problem
- **Predict:** `is_high_value` (binary classification)
- **Unit of analysis:** one row per user
- **Decision enabled:** identify high-value users for prioritization (e.g., marketing or retention)
- **Constraints:** CPU-only; offline-first; batch inference

## Data (contract)
- **Feature table:** `data/processed/features.csv` (optional: `.parquet`)
- **Unit of analysis:** one row per user
- **Target column:** `is_high_value`
  - Positive class: `1`
- **Optional IDs (passthrough):**
  - `user_id`

### Feature schema
| Column name     | Type     | Description |
|-----------------|----------|-------------|
| user_id         | string   | Unique user identifier |
| country         | string   | User country (US, CA, GB) |
| n_orders        | integer  | Number of orders |
| avg_amount      | float    | Average order amount |
| total_amount    | float    | Total spend |
| is_high_value   | integer  | Target label (1 = high value) |

## Splits (draft)
- **Holdout strategy:** random stratified split
- **Leakage risks:** total_amount is derived from n_orders and avg_amount; no future data leakage assumed

## Metrics (draft)
- **Primary:** ROC-AUC (robust to class imbalance)
- **Baseline:** dummy classifier must be reported

## Shipping
- **Artifacts:** trained model, input schema, metrics, holdout tables, environment snapshot
- **Known limitations:** synthetic data; limited feature diversity
- **Monitoring (sketch):**
  - feature drift (country distribution)
  - prediction rate of positive class
