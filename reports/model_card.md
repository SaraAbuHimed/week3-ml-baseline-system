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
  - **Positive class:** `1`
- **Optional IDs (passthrough):**
  - `user_id`

### Feature schema
| Column name     | Type     | Description |
|-----------------|----------|-------------|
| user_id         | string   | Unique user identifier |
| country         | string   | User country |
| n_orders        | integer  | Number of orders |
| avg_amount      | float    | Average order amount |
| total_amount    | float    | Total spend |
| is_high_value   | integer  | Target label (1 = high value) |

## Splits (evaluation plan)
- **Holdout strategy:** random stratified holdout
- **Test size:** 0.20
- **Random seed:** 42
- **Leakage risks:** `total_amount` is derived from `n_orders` and `avg_amount`; no future information is used

## Metrics
- **Primary metric:** ROC-AUC  
  *Chosen because the dataset is imbalanced and ROC-AUC evaluates ranking quality independent of a fixed threshold.*
- **Baseline:** Dummy classifier (most-frequent strategy) reported on the same holdout set

## Shipping
- **Artifacts:**
  - trained model (`model/model.joblib`)
  - metrics (`metrics/baseline_holdout.json`, `metrics/model_holdout.json`)
  - training config (`config.json`)
- **Known limitations:** synthetic data; simplified feature set
- **Monitoring (sketch):**
  - class balance drift
  - feature distribution drift (e.g., country)
  - positive prediction rate over time

