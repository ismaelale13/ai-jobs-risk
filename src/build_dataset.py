"""Merge OEWS 2022/2025, O*NET and AI exposure into one modelling table."""
import numpy as np
import pandas as pd

from src import config as C
from src.load import load_eloundou, load_oews, load_onet

TOP_CODED_WAGE = 239200.0  # OEWS marks annual wages >= this value with '#'


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def impute_by_soc_hierarchy(codes, table, cols):
    """Attach occupation-level columns to SOC codes.

    Codes missing in `table` (mostly 'All Other' and aggregated OEWS codes) get the mean of
    their SOC broad group (first 6 characters), then minor group (5), then major group (2).
    Returns the filled frame and a flag marking imputed rows.
    """
    out = pd.DataFrame({"soc": sorted(set(codes))}).merge(table[["soc"] + cols], on="soc", how="left")
    imputed = out[cols].isna().any(axis=1)
    for n in (6, 5, 2):
        prefix_mean = table.assign(p=table["soc"].str[:n]).groupby("p")[cols].mean()
        for col in cols:
            missing = out[col].isna()
            if missing.any():
                out.loc[missing, col] = prefix_mean[col].reindex(out.loc[missing, "soc"].str[:n]).values
    out["imputed"] = imputed.astype(int)
    return out


def build():
    o22, o25 = load_oews(2022), load_oews(2025)

    # metro size = all-occupation total employment of the area in 2022
    area_tot = o22[o22["O_GROUP"] == "total"][["AREA", "TOT_EMP"]].rename(columns={"TOT_EMP": "area_emp"})
    d22 = o22[o22["O_GROUP"] == "detailed"].merge(area_tot, on="AREA", how="left")
    d25 = o25[o25["O_GROUP"] == "detailed"][["AREA", "AREA_TITLE", "OCC_CODE", "TOT_EMP"]]

    df = d22.merge(d25, on=["AREA", "OCC_CODE"], suffixes=("_2022", "_2025"))
    n_matched = len(df)
    # keep only areas whose definition (code + name) did not change between releases
    df = df[df["AREA_TITLE_2022"] == df["AREA_TITLE_2025"]].copy()
    n_stable = len(df)

    df["emp_2022"] = _num(df["TOT_EMP_2022"])
    df["emp_2025"] = _num(df["TOT_EMP_2025"])
    df = df.dropna(subset=["emp_2022", "emp_2025"])
    df = df[df["emp_2022"] >= C.MIN_EMP_2022].copy()

    # target
    df["pct_change"] = df["emp_2025"] / df["emp_2022"] - 1
    df["log_change"] = np.log(df["emp_2025"] / df["emp_2022"])
    df[C.TARGET] = (df["pct_change"] < C.DECLINE_THRESHOLD).astype(int)

    # 2022 labor-market features (all known before the outcome period)
    wage = df["A_MEDIAN"].replace("#", str(TOP_CODED_WAGE))
    df["log_median_wage_2022"] = np.log(_num(wage))
    df["log_emp_2022"] = np.log(df["emp_2022"])
    df["loc_quotient_2022"] = _num(df["LOC_QUOTIENT"])
    df["jobs_per_1000_2022"] = _num(df["JOBS_1000"])
    df["emp_rse_2022"] = _num(df["EMP_PRSE"])
    df["log_area_emp_2022"] = np.log(_num(df["area_emp"]))
    df["nonmetro"] = (df["AREA_TYPE"] == "6").astype(int)
    df["major_group"] = df["OCC_CODE"].str[:2]

    # occupation features (O*NET) and AI exposure (Eloundou et al.), imputed via SOC hierarchy
    codes = df["OCC_CODE"].unique()
    onet_cols = ["job_zone"] + list(C.WORK_ACTIVITIES.values())
    onet = impute_by_soc_hierarchy(codes, load_onet(), onet_cols).drop(columns="imputed")
    ai_cols = ["ai_gpt4_alpha", "ai_gpt4_beta", "ai_gpt4_gamma", "ai_human_beta"]
    ai = impute_by_soc_hierarchy(codes, load_eloundou(), ai_cols).rename(columns={"imputed": "ai_imputed"})
    df = df.merge(onet, left_on="OCC_CODE", right_on="soc", how="left").drop(columns="soc")
    df = df.merge(ai, left_on="OCC_CODE", right_on="soc", how="left").drop(columns="soc")

    keep = ["AREA", "AREA_TITLE_2022", "PRIM_STATE", "OCC_CODE", "OCC_TITLE", "emp_2022", "emp_2025",
            "pct_change", "log_change", C.TARGET] + C.BASE_NUMERIC + C.CATEGORICAL + C.AI_FEATURES
    df = df[keep].rename(columns={"AREA_TITLE_2022": "AREA_TITLE"}).reset_index(drop=True)

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.PROCESSED / "dataset.parquet", index=False)

    stats = {
        "pairs_matched_2022_2025": n_matched,
        "pairs_in_stable_areas": n_stable,
        "rows_final": len(df),
        "occupations": df["OCC_CODE"].nunique(),
        "areas": df["AREA"].nunique(),
        "decline_share": round(df[C.TARGET].mean(), 3),
        "occupations_with_imputed_ai_score": int(df.drop_duplicates("OCC_CODE")["ai_imputed"].sum()),
    }
    return df, stats


if __name__ == "__main__":
    _, s = build()
    print(s)
