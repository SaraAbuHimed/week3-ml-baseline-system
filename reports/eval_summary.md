# W3D3 Eval Summary

- Run ID: 2025-12-30T16-57-00Z__clf__session42
- Dataset: data/processed/features.csv
- Target: is_high_value
- Unit of analysis: one row per user
- Primary metric: precision

## Holdout results

Baseline (most_frequent) — from `baseline_holdout.json`:
- accuracy: 0.80
- precision: 0.00
- recall: 0.00
- f1: 0.00

Model (logistic regression) — from `holdout_metrics.json`:
- accuracy: 1.00
- precision: 1.00
- recall: 1.00
- f1: 1.00
- roc_auc: 1.00

## Interpretation
Using the same holdout split, the model’s **precision** is 1.00 compared to the baseline precision of 0.00, an absolute improvement of **+1.00**. This indicates the trained model identifies high-value users much more effectively than a naive baseline that always predicts the majority class.

## Caveats / likely failure modes
- The dataset appears small, which can lead to overly optimistic metrics on a single holdout split.
- Perfect scores may indicate overfitting or label leakage if features strongly encode the target.
- Results are based on one random split; cross-validation would provide a more robust performance estimate.

