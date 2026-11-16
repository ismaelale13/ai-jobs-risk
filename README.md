# Which Jobs Are at Risk? Predicting Local Labor Market Decline with AI Exposure Data

Practical Data Science, WS 2026/27 – Group 5

Predicts whether employment in a U.S. occupation × metro area fell by more than 5 % between
May 2022 and May 2025 (after the release of ChatGPT), and tests whether AI exposure scores
improve the prediction beyond classic labor-market features.

## Data (download by hand into `data/raw/`)

| File | Source | License |
|---|---|---|
| `oesm22all.zip`, `oesm25all.zip` | BLS OEWS, https://www.bls.gov/oes/tables.htm ("All data") | Public domain |
| `db_30_3_excel.zip` | O\*NET 30.3 Database, https://www.onetcenter.org/database.html | CC BY 4.0 |
| `GPTs-are-GPTs-main.zip` | Eloundou et al. (2024), https://github.com/openai/GPTs-are-GPTs | MIT |

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py          # ~10–15 min: dataset, models, SHAP, dashboard data
streamlit run app/dashboard.py  # dashboard on http://localhost:8501
```

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `src/load.py` | Reads the raw ZIPs (cached as parquet in `data/interim/`) |
| 2 | `src/build_dataset.py` | Merges OEWS 2022/2025 on area + SOC code, keeps metro areas with unchanged definitions and cells with ≥ 100 employees in 2022, adds O\*NET features and AI exposure, builds the target → `data/processed/dataset.parquet` |
| 3 | `src/train.py` | Hold-out test set of 20 % of occupations; GroupKFold tuning; logistic regression vs. LightGBM, each without and with AI features; bootstrap CIs; error analysis |
| 4 | `src/explain.py` | SHAP values, global importance and plots |
| 5 | `src/dashboard_data.py` | Compact table for the dashboard |
| – | `app/dashboard.py` | Streamlit app |

Missing occupation scores (mostly "All Other" and aggregated OEWS codes) are imputed with the mean of
the SOC broad group, then minor group, then major group; the flag `ai_imputed` marks them.

## Outputs (`reports/`)

`dataset_stats.json`, `model_comparison.csv`, `metrics.json`, `error_analysis.csv`,
`shap_importance.csv`, `shap_summary.png`, `shap_ai_gpt4_beta.png`.
