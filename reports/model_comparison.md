# Historical model comparison

Values extracted from saved outputs; models were not retrained for this report. Empty CV scores mean no confirmed score for that variant.

| Model | Variant | Validation ROC-AUC | CV ROC-AUC | Source |
| --- | --- | ---: | ---: | --- |
| Logistic Regression | baseline without PCA | 0.7705 | Not recorded | [10_modeling.ipynb](../notebooks/10_modeling.ipynb) |
| Decision Tree | optimized | 0.7042 | 0.7055128021 | [decision_tree.ipynb](../notebooks/decision_tree.ipynb) |
| Random Forest | baseline | 0.7670 | Not recorded | [13_RandomForest.ipynb](../notebooks/13_RandomForest.ipynb) |
| Random Forest | optimized | 0.7664 | 0.7597416132 | [13_RandomForest.ipynb](../notebooks/13_RandomForest.ipynb) |
| XGBoost | baseline | 0.7796 | Not recorded | [14_XGBoost.ipynb](../notebooks/14_XGBoost.ipynb) |
| XGBoost | optimized | 0.7769 | 0.772517025 | [14_XGBoost.ipynb](../notebooks/14_XGBoost.ipynb) |
| LightGBM | baseline | 0.7817 | Not recorded | [15_LightGbm.ipynb](../notebooks/15_LightGbm.ipynb) |
| LightGBM | search optimized | 0.7819 | 0.7741 | [15_LightGbm.ipynb](../notebooks/15_LightGbm.ipynb) |
| LightGBM | final scale_pos_weight | 0.7833 | Not recorded | [model_final_evaluation.csv](../data/processed/model_final_evaluation.csv) |

## Interpretation

- Five model families were explored. An optimized Logistic Regression validation score is not recorded in notebook 12; do not describe all five as completed optimized benchmarks.
- Optimization lowered validation ROC-AUC for Random Forest and XGBoost relative to their recorded baselines.
- The final weighted LightGBM score is distinct from the randomized-search variant and its CV score.
- Recorded reports repeatedly show 61,503 validation observations (56,538 negatives, 4,965 positives). Equal counts do not establish identical row membership or preprocessing across experiments.
- Threshold selection was performed on validation. No independent final test performance is established here.
- The historical preprocessing reconstruction verifies feature parity, not absence of leakage in the original preparation.
