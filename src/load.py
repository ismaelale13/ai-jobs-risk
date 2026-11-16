"""Read the raw source files straight from their ZIP archives (cached as parquet)."""
import io
import zipfile

import pandas as pd

from src import config as C


def _read_excel_from_zip(zip_path, name_endswith, **kw):
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.split("/")[-1] == name_endswith)
        return pd.read_excel(io.BytesIO(z.read(name)), **kw)


def _cached(name, builder):
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    path = C.INTERIM / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = builder()
    df.to_parquet(path, index=False)
    return df


def load_oews(year):
    """OEWS all-data file, reduced to metro/nonmetro rows (detailed occupations + area totals)."""
    zip_path = C.OEWS_2022_ZIP if year == 2022 else C.OEWS_2025_ZIP

    def build():
        df = _read_excel_from_zip(zip_path, f"all_data_M_{year}.xlsx", dtype=str)
        df.columns = [c.upper() for c in df.columns]
        df = df[df["AREA_TYPE"].isin(C.AREA_TYPES)]
        df = df[df["O_GROUP"].isin(["detailed", "total"])]
        cols = ["AREA", "AREA_TITLE", "AREA_TYPE", "PRIM_STATE", "OCC_CODE", "OCC_TITLE", "O_GROUP",
                "TOT_EMP", "EMP_PRSE", "JOBS_1000", "LOC_QUOTIENT", "A_MEDIAN"]
        return df[cols].reset_index(drop=True)

    return _cached(f"oews_{year}", build)


def load_onet():
    """O*NET occupation features: job zone + selected work activities (importance)."""
    def build():
        jz = _read_excel_from_zip(C.ONET_ZIP, "Job Zones.xlsx")[["O*NET-SOC Code", "Job Zone"]]
        wa = _read_excel_from_zip(C.ONET_ZIP, "Work Activities.xlsx")
        wa = wa[(wa["Scale ID"] == "IM") & wa["Element Name"].isin(C.WORK_ACTIVITIES)]
        wa = wa.pivot_table(index="O*NET-SOC Code", columns="Element Name", values="Data Value")
        wa = wa.rename(columns=C.WORK_ACTIVITIES).reset_index()
        df = jz.merge(wa, on="O*NET-SOC Code", how="outer").rename(columns={"Job Zone": "job_zone"})
        df["soc"] = df["O*NET-SOC Code"].str[:7]
        # several O*NET codes can map to one SOC code: average them
        return df.drop(columns="O*NET-SOC Code").groupby("soc").mean().reset_index()

    return _cached("onet", build)


def load_eloundou():
    """LLM exposure scores (Eloundou et al. 2024) aggregated to 6-digit SOC."""
    def build():
        with zipfile.ZipFile(C.ELOUNDOU_ZIP) as z:
            name = next(n for n in z.namelist() if n.endswith("data/occ_level.csv"))
            df = pd.read_csv(io.BytesIO(z.read(name)))
        df["soc"] = df["O*NET-SOC Code"].str[:7]
        df = df.rename(columns={"dv_rating_alpha": "ai_gpt4_alpha", "dv_rating_beta": "ai_gpt4_beta",
                                "dv_rating_gamma": "ai_gpt4_gamma", "human_rating_beta": "ai_human_beta"})
        keep = ["ai_gpt4_alpha", "ai_gpt4_beta", "ai_gpt4_gamma", "ai_human_beta"]
        return df.groupby("soc")[keep].mean().reset_index()

    return _cached("eloundou", build)
