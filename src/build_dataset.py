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
    o19, o22, o25 = load_oews(2019), load_oews(2022), load_oews(2025)
    local = lambda d: d[d["AREA_TYPE"].isin(C.AREA_TYPES)]
    national = lambda d: d[(d["AREA_TYPE"] == "1") & (d["O_GROUP"] == "detailed")]

    # metro size = all-occupation total employment of the area in 2022
    area_tot = local(o22)[local(o22)["O_GROUP"] == "total"][["AREA", "TOT_EMP"]].rename(columns={"TOT_EMP": "area_emp"})
    d22 = local(o22)[local(o22)["O_GROUP"] == "detailed"].merge(area_tot, on="AREA", how="left")
    d25 = local(o25)[local(o25)["O_GROUP"] == "detailed"][["AREA", "AREA_TITLE", "OCC_CODE", "TOT_EMP"]]

    df = d22.merge(d25, on=["AREA", "OCC_CODE"], suffixes=("_2022", "_2025"))
    n_matched = len(df)
    # keep only areas whose definition (code + name) did not change between releases
    df = df[df["AREA_TITLE_2022"] == df["AREA_TITLE_2025"]].copy()
    n_stable = len(df)

    df["emp_2022"] = _num(df["TOT_EMP_2022"])
    df["emp_2025"] = _num(df["TOT_EMP_2025"])
    df = df.dropna(subset=["emp_2022", "emp_2025"])
    df = df[df["emp_2022"] >= C.MIN_EMP_2022].copy()
    n_size = len(df)
    df = df[_num(df["EMP_PRSE"]) <= C.MAX_RSE_2022].copy()  # reliable 2022 estimates only
    n_rse = len(df)

    # pre-period trend 2019 -> 2022: local (same area definition) and national
    d19 = local(o19)[local(o19)["O_GROUP"] == "detailed"][["AREA", "AREA_TITLE", "OCC_CODE", "TOT_EMP"]]
    d19 = d19.rename(columns={"AREA_TITLE": "AREA_TITLE_2019", "TOT_EMP": "TOT_EMP_2019"})
    df = df.merge(d19, on=["AREA", "OCC_CODE"], how="left")
    e19 = _num(df["TOT_EMP_2019"]).where(df["AREA_TITLE_2019"] == df["AREA_TITLE_2022"])
    df["log_change_2019_2022"] = np.log(df["emp_2022"] / e19)
    df["has_2019"] = df["log_change_2019_2022"].notna().astype(int)
    n19 = national(o19)[["OCC_CODE", "TOT_EMP"]].assign(e19=lambda x: _num(x["TOT_EMP"]))
    n22 = national(o22)[["OCC_CODE", "TOT_EMP"]].assign(e22=lambda x: _num(x["TOT_EMP"]))
    nat = n22.merge(n19, on="OCC_CODE")[["OCC_CODE", "e19", "e22"]]
    nat["nat_log_change_2019_2022"] = np.log(nat["e22"] / nat["e19"])
    df = df.merge(nat[["OCC_CODE", "nat_log_change_2019_2022"]], on="OCC_CODE", how="left")

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
        "pairs_with_min_employment": n_size,
        "pairs_with_reliable_2022_estimate": n_rse,
        "rows_final": len(df),
        "share_with_local_2019_trend": round(df["has_2019"].mean(), 3),
        "share_with_national_2019_trend": round(df["nat_log_change_2019_2022"].notna().mean(), 3),
        "occupations": df["OCC_CODE"].nunique(),
        "areas": df["AREA"].nunique(),
        "decline_share": round(df[C.TARGET].mean(), 3),
        "occupations_with_imputed_ai_score": int(df.drop_duplicates("OCC_CODE")["ai_imputed"].sum()),
    }
    return df, stats


if __name__ == "__main__":
    _, s = build()
    print(s)
