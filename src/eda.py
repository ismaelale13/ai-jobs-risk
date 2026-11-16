"""Exploratory data analysis: figures and summary tables for the report (reports/eda/)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C

OUT = C.REPORTS / "eda"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

MAJOR_GROUPS = {
    "11": "Management", "13": "Business & financial", "15": "Computer & math", "17": "Architecture & engineering",
    "19": "Life & social science", "21": "Community & social service", "23": "Legal", "25": "Education",
    "27": "Arts, design & media", "29": "Healthcare practitioners", "31": "Healthcare support",
    "33": "Protective service", "35": "Food preparation", "37": "Building & grounds", "39": "Personal care",
    "41": "Sales", "43": "Office & admin support", "45": "Farming & forestry", "47": "Construction",
    "49": "Installation & repair", "51": "Production", "53": "Transportation",
}


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=10)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def fig_target(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = (df["pct_change"].clip(-0.6, 0.8) * 100)
    ax.hist(x, bins=70, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(C.DECLINE_THRESHOLD * 100, color=ORANGE, linewidth=2)
    ax.text(C.DECLINE_THRESHOLD * 100 - 2, ax.get_ylim()[1] * 0.92,
            f"decline > 5 %\n{df[C.TARGET].mean():.0%} of cells", ha="right", color=TEXT, fontsize=9)
    _style(ax, "Employment change 2022→2025 per occupation × metro cell", "Change (%), clipped at −60 / +80",
           "Number of cells")
    _save(fig, "01_target_distribution.png")


def fig_by_major_group(df):
    g = df.groupby("major_group", observed=True)[C.TARGET].agg(["mean", "size"]).reset_index()
    g["label"] = g["major_group"].astype(str).map(MAJOR_GROUPS)
    g = g.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = [ORANGE if l == "Office & admin support" else BLUE for l in g["label"]]
    ax.barh(g["label"], g["mean"] * 100, color=colors, height=0.7)
    ax.axvline(df[C.TARGET].mean() * 100, color=GRAY, linestyle="--", linewidth=1)
    for y, (v, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(v * 100 + 0.8, y, f"{v:.0%}", va="center", fontsize=8, color=MUTED)
    _style(ax, "Cells declining > 5 %, by occupation group", "Cells declining (%)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _save(fig, "02_decline_by_occupation_group.png")
    return g


def fig_ai_exposure(df):
    occ = df.groupby("OCC_CODE").agg(ai=("ai_gpt4_beta", "first"), e22=("emp_2022", "sum"),
                                     e25=("emp_2025", "sum"), decline=(C.TARGET, "mean"))
    occ["bin"] = pd.qcut(occ["ai"], 5, labels=["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
    b = occ.groupby("bin", observed=True).agg(ai=("ai", "mean"), e22=("e22", "sum"), e25=("e25", "sum"),
                                              n=("ai", "size"))
    cells = df.merge(occ["bin"], left_on="OCC_CODE", right_index=True)
    b["decline"] = cells.groupby("bin", observed=True)[C.TARGET].mean()  # share of cells, not of occupations
    b["growth"] = (b["e25"] / b["e22"] - 1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(b.index.astype(str), b["growth"], color=[ORANGE if i == 4 else BLUE for i in range(5)], width=0.6)
    for i, v in enumerate(b["growth"]):
        axes[0].text(i, v + 0.2, f"{v:+.1f} %", ha="center", fontsize=9, color=TEXT)
    _style(axes[0], "Employment growth 2022→2025", "AI exposure quintile (occupations)", "Growth (%)")
    axes[1].bar(b.index.astype(str), b["decline"] * 100, color=[ORANGE if i == 4 else BLUE for i in range(5)],
                width=0.6)
    for i, v in enumerate(b["decline"]):
        axes[1].text(i, v * 100 + 0.6, f"{v:.0%}", ha="center", fontsize=9, color=TEXT)
    _style(axes[1], "Share of metro cells declining > 5 %", "AI exposure quintile (occupations)", "Cells (%)")
    _save(fig, "03_ai_exposure_quintiles.png")
    return b


def fig_prior_trend(df):
    d = df.dropna(subset=["log_change_2019_2022"]).copy()
    d["prior"] = (np.exp(d["log_change_2019_2022"]) - 1) * 100
    d["bin"] = pd.qcut(d["prior"], 10)
    b = d.groupby("bin", observed=True).agg(x=("prior", "median"), y=(C.TARGET, "mean"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(b["x"], b["y"] * 100, color=BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.axhline(df[C.TARGET].mean() * 100, color=GRAY, linestyle="--", linewidth=1)
    ax.text(b["x"].iloc[0], df[C.TARGET].mean() * 100 + 1, "average", color=MUTED, fontsize=8)
    _style(ax, "Cells that grew fast in 2019–2022 were more likely to decline afterwards",
           "Employment change 2019→2022, decile median (%)", "Cells declining 2022→2025 (%)")
    _save(fig, "04_prior_trend_vs_decline.png")
    return b


def fig_metro_size(df):
    d = df.copy()
    d["size"] = np.where(d["nonmetro"] == 1, "Nonmetro",
                         pd.qcut(d["log_area_emp_2022"].where(d["nonmetro"] == 0), 3,
                                 labels=["Small metro", "Medium metro", "Large metro"]).astype(str))
    order = ["Nonmetro", "Small metro", "Medium metro", "Large metro"]
    b = d.groupby("size")[C.TARGET].agg(["mean", "size"]).reindex(order)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(b.index, b["mean"] * 100, color=BLUE, width=0.6)
    for i, v in enumerate(b["mean"]):
        ax.text(i, v * 100 + 0.6, f"{v:.0%}", ha="center", fontsize=9, color=TEXT)
    _style(ax, "Share of cells declining > 5 %, by area size", None, "Cells (%)")
    _save(fig, "05_decline_by_area_size.png")
    return b


def fig_correlations(df):
    cols = ["log_change", "ai_gpt4_beta", "ai_human_beta", "log_change_2019_2022", "nat_log_change_2019_2022",
            "log_median_wage_2022", "log_emp_2022", "loc_quotient_2022", "emp_rse_2022", "job_zone",
            "wa_computers", "wa_physical", "wa_creative", "wa_administrative"]
    names = ["Change 22→25 (target)", "AI exposure (GPT-4)", "AI exposure (human)", "Local trend 19→22",
             "National trend 19→22", "Median wage", "Employment", "Location quotient", "Survey error",
             "Job zone", "Computers", "Physical", "Creative", "Administrative"]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), names, rotation=60, ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(cols)), names, fontsize=8, color=MUTED)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else TEXT)
    ax.set_title("Spearman correlations (first row/column = target)", loc="left", fontsize=12, color=TEXT)
    fig.colorbar(im, ax=ax, shrink=0.7)
    _save(fig, "06_correlations.png")
    return corr["log_change"].drop("log_change").sort_values()


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(C.PROCESSED / "dataset.parquet")
    fig_target(df)
    groups = fig_by_major_group(df)
    ai = fig_ai_exposure(df)
    prior = fig_prior_trend(df)
    size = fig_metro_size(df)
    corr = fig_correlations(df)

    desc = df[C.BASE_NUMERIC + C.AI_FEATURES + ["pct_change"]].describe().T
    desc["missing_share"] = df[desc.index].isna().mean()
    desc.round(3).to_csv(OUT / "descriptive_statistics.csv")

    summary = {
        "rows": len(df), "occupations": int(df["OCC_CODE"].nunique()), "areas": int(df["AREA"].nunique()),
        "decline_share": round(float(df[C.TARGET].mean()), 3),
        "median_pct_change": round(float(df["pct_change"].median()), 3),
        "decline_share_by_group": groups.set_index("label")["mean"].round(3).to_dict(),
        "ai_quintiles": ai[["ai", "growth", "decline"]].round(3).reset_index().astype(str).to_dict(orient="records"),
        "spearman_with_log_change": corr.round(3).to_dict(),
        "decline_share_by_area_size": size["mean"].round(3).to_dict(),
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    report = text_report(df, groups, ai, prior, size, corr, desc)
    (OUT / "eda_report.txt").write_text(report)
    print(report)
    return summary


def text_report(df, groups, ai, prior, size, corr, desc):
    """Readable EDA results (printed when the script runs, saved as reports/eda/eda_report.txt)."""
    pct = lambda v: f"{v:6.1%}"
    L = []
    L.append("=" * 72)
    L.append("EXPLORATORY DATA ANALYSIS")
    L.append("=" * 72)
    L.append(f"Rows (occupation x area cells): {len(df):,}")
    L.append(f"Occupations: {df['OCC_CODE'].nunique()}   Areas: {df['AREA'].nunique()} "
             f"({(df['nonmetro'] == 1).mean():.0%} of cells in nonmetro areas)")

    L.append("\n1) TARGET: employment change May 2022 -> May 2025")
    q = df["pct_change"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    L.append("   Percentiles of change: " + "  ".join(f"p{int(k * 100)}={v:+.1%}" for k, v in q.items()))
    L.append(f"   Cells with decline > 5 % (target = 1): {df[C.TARGET].mean():.1%}  -> classes reasonably balanced")
    L.append(f"   Cells with exactly 0 % change: {(df['pct_change'] == 0).mean():.1%}")
    L.append(f"   Total employment in sample: {df['emp_2022'].sum():,.0f} -> {df['emp_2025'].sum():,.0f} "
             f"({df['emp_2025'].sum() / df['emp_2022'].sum() - 1:+.1%})")

    L.append("\n2) DECLINE SHARE BY OCCUPATION GROUP (highest / lowest 5)")
    g = groups.sort_values("mean", ascending=False)
    for _, r in pd.concat([g.head(5), g.tail(5)]).iterrows():
        L.append(f"   {r['label']:<30s} {pct(r['mean'])}   ({int(r['size']):>6,} cells)")

    L.append("\n3) AI EXPOSURE (GPT-4 beta) QUINTILES OF OCCUPATIONS")
    L.append(f"   {'Quintile':<12s}{'mean score':>11s}{'occupations':>13s}{'emp. growth':>13s}{'cells declining':>17s}")
    for idx, r in ai.iterrows():
        L.append(f"   {str(idx).replace(chr(10), ' '):<12s}{r['ai']:>11.2f}{int(r['n']):>13d}"
                 f"{r['growth'] / 100:>+13.1%}{r['decline']:>17.1%}")
    L.append("   -> only the most exposed quintile shrank; the relation is not linear")

    L.append("\n4) PRIOR TREND 2019 -> 2022 (deciles of local change)")
    for _, r in prior.iterrows():
        L.append(f"   prior change {r['x'] / 100:+7.1%}   -> declining afterwards {r['y']:.1%}")
    L.append(f"   Cells with a 2019 value: {df['has_2019'].mean():.1%}")

    L.append("\n5) DECLINE SHARE BY AREA SIZE")
    for k, r in size.iterrows():
        L.append(f"   {k:<14s} {pct(r['mean'])}   ({int(r['size']):>6,} cells)")

    L.append("\n6) SPEARMAN CORRELATION WITH LOG EMPLOYMENT CHANGE (target, continuous)")
    for k, v in corr.items():
        L.append(f"   {k:<26s} {v:+.3f}")
    ai_comp = df[["ai_gpt4_beta", "wa_computers"]].corr(method="spearman").iloc[0, 1]
    L.append(f"   Note: AI exposure vs. 'working with computers' = {ai_comp:+.2f} (strong overlap)")

    L.append("\n7) MISSING VALUES")
    miss = desc["missing_share"][desc["missing_share"] > 0]
    for k, v in miss.items():
        L.append(f"   {k:<26s} {v:.1%} missing (handled: imputation / native NaN support)")
    if miss.empty:
        L.append("   none")
    L.append(f"   Occupations with imputed AI score: {df.drop_duplicates('OCC_CODE')['ai_imputed'].sum()}")
    L.append("\nFigures: reports/eda/*.png   Tables: descriptive_statistics.csv, eda_summary.json")
    L.append("=" * 72)
    return "\n".join(L)


if __name__ == "__main__":
    run()
