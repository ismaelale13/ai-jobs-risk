"""Prepare the compact table the dashboard reads (predictions + SHAP values)."""
import pandas as pd

from src import config as C


def run():
    scored = pd.read_parquet(C.PROCESSED / "scored.parquet")
    shap_df = pd.read_parquet(C.PROCESSED / "shap_values.parquet")
    keep = ["AREA", "AREA_TITLE", "OCC_CODE", "OCC_TITLE", "emp_2022", "emp_2025", "pct_change", C.TARGET,
            "risk", "in_test_set", "log_median_wage_2022", "job_zone", "ai_gpt4_beta", "ai_imputed",
            "wa_computers", "wa_documenting", "wa_creative", "wa_physical", "wa_public"]
    out = pd.concat([scored[keep].reset_index(drop=True), shap_df.astype("float32")], axis=1)
    out.to_parquet(C.PROCESSED / "dashboard.parquet", index=False)
    return out


if __name__ == "__main__":
    print(run().shape)
