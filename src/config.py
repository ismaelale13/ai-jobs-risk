"""Central configuration: paths, filters and model settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Raw files (downloaded by hand, see README)
OEWS_2022_ZIP = RAW / "oesm22all.zip"
OEWS_2025_ZIP = RAW / "oesm25all.zip"
ONET_ZIP = RAW / "db_30_3_excel.zip"
ELOUNDOU_ZIP = RAW / "GPTs-are-GPTs-main.zip"

# Sample definition
MIN_EMP_2022 = 100          # drop tiny occupation-metro cells (noisy % changes)
DECLINE_THRESHOLD = -0.05   # target = 1 if employment fell by more than 5 %
AREA_TYPES = {"4", "6"}     # 4 = metropolitan area, 6 = nonmetropolitan area

# O*NET work activities used as task-content features (importance scale)
WORK_ACTIVITIES = {
    "Working with Computers": "wa_computers",
    "Processing Information": "wa_processing_info",
    "Analyzing Data or Information": "wa_analyzing_data",
    "Documenting/Recording Information": "wa_documenting",
    "Thinking Creatively": "wa_creative",
    "Performing Administrative Activities": "wa_administrative",
    "Performing General Physical Activities": "wa_physical",
    "Handling and Moving Objects": "wa_handling_objects",
    "Operating Vehicles, Mechanized Devices, or Equipment": "wa_operating_vehicles",
    "Assisting and Caring for Others": "wa_caring",
    "Performing for or Working Directly with the Public": "wa_public",
    "Guiding, Directing, and Motivating Subordinates": "wa_managing",
}

# Feature groups
BASE_NUMERIC = [
    "log_emp_2022", "log_median_wage_2022", "loc_quotient_2022", "jobs_per_1000_2022",
    "emp_rse_2022", "log_area_emp_2022", "nonmetro", "job_zone",
] + list(WORK_ACTIVITIES.values())
CATEGORICAL = ["major_group"]
AI_FEATURES = ["ai_gpt4_alpha", "ai_gpt4_beta", "ai_gpt4_gamma", "ai_human_beta", "ai_imputed"]

TARGET = "decline"
GROUP_COL = "OCC_CODE"
TEST_SIZE = 0.2
SEED = 42
N_FOLDS = 5
