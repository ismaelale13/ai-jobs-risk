"""Build the modelling table for any period t0 -> t1 with the same feature definitions.

Used by the robustness module for the pre-ChatGPT placebo period (2019 -> 2022) and for the
out-of-time test (train on 2019 -> 2022, predict 2022 -> 2025). Features are measured at t0,
the prior trend runs from t_prev to t0, the target compares t1 with t0.
"""
import numpy as np
import pandas as pd

from src import config as C
from src.build_dataset import impute_by_soc_hierarchy
from src.load import load_eloundou, load_oews, load_onet

TOP_CODED_WAGE = {2016: 208000.0, 2019: 208000.0, 2022: 239200.0, 2025: 239200.0}

NUMERIC = ["log_emp", "log_wage", "lq", "jobs_1000", "rse", "log_area_emp", "nonmetro", "job_zone",
           "prior_local", "has_prior", "prior_national"] + list(C.WORK_ACTIVITIES.values())
CATEGORICAL = ["major_group"]
AI = ["ai_gpt4_alpha", "ai_gpt4_beta", "ai_gpt4_gamma", "ai_human_beta", "ai_imputed"]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _local(d):
    return d[d["AREA_TYPE"].isin(C.AREA_TYPES)]


def build_period(t_prev, t0, t1):
    o_prev, o0, o1 = load_oews(t_prev), load_oews(t0), load_oews(t1)
    l0, l1, lp = _local(o0), _local(o1), _local(o_prev)
    area_tot = l0[l0["O_GROUP"] == "total"][["AREA", "TOT_EMP"]].rename(columns={"TOT_EMP": "area_emp"})
    d0 = l0[l0["O_GROUP"] == "detailed"].merge(area_tot, on="AREA", how="left")
    d1 = l1[l1["O_GROUP"] == "detailed"][["AREA", "AREA_TITLE", "OCC_CODE", "TOT_EMP"]]

    df = d0.merge(d1, on=["AREA", "OCC_CODE"], suffixes=("_0", "_1"))
    df = df[df["AREA_TITLE_0"] == df["AREA_TITLE_1"]].copy()
    df["emp0"], df["emp1"] = _num(df["TOT_EMP_0"]), _num(df["TOT_EMP_1"])
    df = df.dropna(subset=["emp0", "emp1"])
    df = df[(df["emp0"] >= C.MIN_EMP_2022) & (_num(df["EMP_PRSE"]) <= C.MAX_RSE_2022)].copy()

    df["pct_change"] = df["emp1"] / df["emp0"] - 1
    df[C.TARGET] = (df["pct_change"] < C.DECLINE_THRESHOLD).astype(int)
    df["log_emp"] = np.log(df["emp0"])
    df["log_wage"] = np.log(_num(df["A_MEDIAN"].replace("#", str(TOP_CODED_WAGE[t0]))))
    df["lq"] = _num(df["LOC_QUOTIENT"])
    df["jobs_1000"] = _num(df["JOBS_1000"])
    df["rse"] = _num(df["EMP_PRSE"])
    df["log_area_emp"] = np.log(_num(df["area_emp"]))
    df["nonmetro"] = (df["AREA_TYPE"] == "6").astype(int)
    df["major_group"] = df["OCC_CODE"].str[:2]

    dp = lp[lp["O_GROUP"] == "detailed"][["AREA", "AREA_TITLE", "OCC_CODE", "TOT_EMP"]]
    dp = dp.rename(columns={"AREA_TITLE": "AREA_TITLE_P", "TOT_EMP": "TOT_EMP_P"})
    df = df.merge(dp, on=["AREA", "OCC_CODE"], how="left")
    e_prev = _num(df["TOT_EMP_P"]).where(df["AREA_TITLE_P"] == df["AREA_TITLE_0"])
    df["prior_local"] = np.log(df["emp0"] / e_prev)
    df["has_prior"] = df["prior_local"].notna().astype(int)
    nat = lambda d: d[(d["AREA_TYPE"] == "1") & (d["O_GROUP"] == "detailed")][["OCC_CODE", "TOT_EMP"]]
    n = nat(o0).merge(nat(o_prev), on="OCC_CODE", suffixes=("_0", "_p"))
    n["prior_national"] = np.log(_num(n["TOT_EMP_0"]) / _num(n["TOT_EMP_p"]))
    df = df.merge(n[["OCC_CODE", "prior_national"]], on="OCC_CODE", how="left")

    codes = df["OCC_CODE"].unique()
    onet_cols = ["job_zone"] + list(C.WORK_ACTIVITIES.values())
    onet = impute_by_soc_hierarchy(codes, load_onet(), onet_cols).drop(columns="imputed")
    ai = impute_by_soc_hierarchy(codes, load_eloundou(), AI[:-1]).rename(columns={"imputed": "ai_imputed"})
    df = df.merge(onet, left_on="OCC_CODE", right_on="soc", how="left").drop(columns="soc")
    df = df.merge(ai, left_on="OCC_CODE", right_on="soc", how="left").drop(columns="soc")

    keep = ["AREA", "AREA_TITLE_0", "OCC_CODE", "OCC_TITLE", "emp0", "emp1", "pct_change", C.TARGET] \
        + NUMERIC + CATEGORICAL + AI
    out = df[keep].rename(columns={"AREA_TITLE_0": "AREA_TITLE"}).reset_index(drop=True)
    out["major_group"] = pd.Categorical(out["major_group"])
    return out
