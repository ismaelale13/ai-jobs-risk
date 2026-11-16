# Which Jobs Are at Risk? Predicting Local Labor Market Decline with AI Exposure Data

Practical Data Science, WS 2026/27 – Group 5

Predicts whether employment in a U.S. occupation × metro area fell by more than 5 % between
May 2022 and May 2025 (after the release of ChatGPT), and tests whether AI exposure scores
improve the prediction beyond classic labor-market features.

**Live dashboard (no installation needed):** https://ai-jobs-risk-jqxycr4vdbhrus6brfaqxh.streamlit.app/

## Run the code

Requirements: Python 3.11 or newer. Everything else is installed from `requirements.txt`
(exact versions pinned). The repository already contains the cached data extracts
(`data/interim/`), so the raw downloads are **not** needed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py          # ~10 min: dataset, models, calibration, SHAP, dashboard data
streamlit run app/dashboard.py  # dashboard on http://localhost:8501
```

The dashboard can also be started directly (results are included in the repository) without running
the pipeline first.

## Data sources

| Source | Content | License |
|---|---|---|
| BLS OEWS, May 2019, 2022, 2025 ("All data"), https://www.bls.gov/oes/tables.htm | Employment and wages by occupation and area | Public domain |
| O\*NET 30.3 Database, https://www.onetcenter.org/database.html | Job zones, work activities | CC BY 4.0 |
| Eloundou et al. (2024), *GPTs are GPTs*, https://github.com/openai/GPTs-are-GPTs | LLM exposure scores per occupation | MIT |
| GeoNames (via `geonamescache`) | Coordinates of principal cities for the map | CC BY 4.0 |

`data/interim/` holds reduced extracts of these files. To rebuild them from scratch, put the original
downloads (`oesm19all.zip`, `oesm22all.zip`, `oesm25all.zip`, `db_30_3_excel.zip`,
`GPTs-are-GPTs-main.zip`) into `data/raw/` and delete `data/interim/`.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `src/load.py` | Reads the raw ZIPs, or the cached extracts in `data/interim/` |
| 2 | `src/build_dataset.py` | Merges OEWS 2022/2025 on area + SOC code; keeps metro areas with unchanged definitions, cells with ≥ 100 employees and a reliable 2022 estimate (relative standard error ≤ 20 %); adds 2019→2022 local and national trends, O\*NET features and AI exposure; builds the target |
| 2b | `src/eda.py` | Exploratory analysis: 6 figures, descriptive statistics and summary → `reports/eda/` |
| 3 | `src/train.py` | Hold-out test set of 20 % of occupations; GroupKFold tuning; logistic regression vs. LightGBM, each without and with AI features; bootstrap CIs; error analysis |
| 4 | `src/calibrate.py` | Platt scaling on training out-of-fold predictions; decision threshold; reliability diagram |
| 5 | `src/explain.py` | SHAP values, global importance and plots |
| 6 | `src/dashboard_data.py`, `src/geo.py` | Compact table and metro coordinates for the dashboard |
| – | `app/dashboard.py` | Streamlit app |

Missing occupation scores (mostly "All Other" and aggregated OEWS codes) are imputed with the mean of
the SOC broad group, then minor group, then major group; the flag `ai_imputed` marks them.

## Outputs (`reports/`)

`dataset_stats.json`, `model_comparison.csv`, `metrics.json`, `error_analysis.csv`,
`shap_importance.csv`, `shap_summary.png`, `shap_ai_gpt4_beta.png`, `calibration.png`.
`*_v1` files hold the first model version (without 2019 trends and reliability filter) for comparison.
