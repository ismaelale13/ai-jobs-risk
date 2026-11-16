"""Streamlit dashboard: pick an occupation and a metro area, see the predicted decline risk and why.

Run:  streamlit run app/dashboard.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

LABELS = {
    "log_emp_2022": "Employment size (2022)",
    "log_median_wage_2022": "Median wage (2022)",
    "loc_quotient_2022": "Local concentration of the job",
    "jobs_per_1000_2022": "Jobs per 1,000 in the area",
    "emp_rse_2022": "Survey uncertainty of the estimate",
    "log_area_emp_2022": "Size of the metro area",
    "nonmetro": "Non-metropolitan area",
    "job_zone": "Required preparation (job zone)",
    "wa_computers": "Working with computers",
    "wa_processing_info": "Processing information",
    "wa_analyzing_data": "Analyzing data",
    "wa_documenting": "Documenting / recording information",
    "wa_creative": "Thinking creatively",
    "wa_administrative": "Administrative activities",
    "wa_physical": "Physical activities",
    "wa_handling_objects": "Handling and moving objects",
    "wa_operating_vehicles": "Operating vehicles / equipment",
    "wa_caring": "Assisting and caring for others",
    "wa_public": "Working directly with the public",
    "wa_managing": "Managing subordinates",
    "major_group": "Occupation group",
    "ai_gpt4_alpha": "AI exposure (GPT-4, narrow)",
    "ai_gpt4_beta": "AI exposure (GPT-4, main)",
    "ai_gpt4_gamma": "AI exposure (GPT-4, broad)",
    "ai_human_beta": "AI exposure (human rating)",
    "ai_imputed": "AI score imputed",
}
AI_COLS = ["ai_gpt4_alpha", "ai_gpt4_beta", "ai_gpt4_gamma", "ai_human_beta", "ai_imputed"]

st.set_page_config(page_title="Which Jobs Are at Risk?", layout="wide")


@st.cache_data
def load():
    df = pd.read_parquet(PROCESSED / "dashboard.parquet")
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    coords = pd.read_csv(PROCESSED / "metro_coords.csv", dtype={"AREA": str})
    return df, metrics, coords


df, metrics, coords = load()
THR = metrics.get("calibration", {}).get("threshold_from_train_oof", 0.5)  # decision threshold
shap_cols = [c for c in df.columns if c.startswith("shap_")]


def risk_band(p):
    if p >= THR + 0.1:
        return "High", "#d64545"
    if p >= THR:
        return "Elevated", "#e0a030"
    if p >= THR - 0.1:
        return "Moderate", "#c9b23a"
    return "Low", "#3a9a5b"


st.title("Which Jobs Are at Risk?")
st.caption("Predicted risk that employment in an occupation and U.S. metro area fell by more than 5% "
           "between May 2022 and May 2025 (after the release of ChatGPT). Data: BLS OEWS, O*NET 30.3, "
           "Eloundou et al. (2024).")

tab_pair, tab_map, tab_rank, tab_model = st.tabs(["Occupation × metro", "Map", "Rankings", "Model"])

with tab_pair:
    c1, c2 = st.columns(2)
    occs = df[["OCC_CODE", "OCC_TITLE"]].drop_duplicates().sort_values("OCC_TITLE")
    occ_title = c1.selectbox("Occupation", occs["OCC_TITLE"],
                             index=int((occs["OCC_TITLE"] == "Accountants and Auditors").to_numpy().argmax()))
    sub = df[df["OCC_TITLE"] == occ_title].sort_values("AREA_TITLE")
    default_area = sub["emp_2022"].to_numpy().argmax()  # largest metro for this occupation
    area = c2.selectbox(f"Metro area ({len(sub)} available)", sub["AREA_TITLE"], index=int(default_area))
    row = sub[sub["AREA_TITLE"] == area].iloc[0]

    band, color = risk_band(row["risk"])
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div style='font-size:0.9rem;color:gray'>Predicted decline risk</div>"
                f"<div style='font-size:2.4rem;font-weight:700;color:{color}'>{row['risk']:.0%}</div>"
                f"<div style='color:{color}'>{band}</div>", unsafe_allow_html=True)
    m2.metric("Employment 2022", f"{row['emp_2022']:,.0f}")
    m3.metric("Employment 2025", f"{row['emp_2025']:,.0f}", f"{row['pct_change']:+.1%}")
    actual = "Fell > 5%" if row["decline"] == 1 else "No fall > 5%"
    hit = (row["risk"] >= THR) == (row["decline"] == 1)
    m4.metric("What actually happened", actual, "model correct" if hit else "model wrong",
              delta_color="normal" if hit else "inverse")
    st.caption("Out-of-sample prediction: " + ("this occupation is in the held-out test set."
               if row["in_test_set"] else "out-of-fold prediction (the model did not see this occupation)."))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Why this risk?")
        contrib = row[shap_cols].astype(float)
        contrib.index = [c.replace("shap_", "") for c in shap_cols]
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
        fig = go.Figure(go.Bar(
            x=top.values, y=[LABELS.get(i, i) for i in top.index], orientation="h",
            marker_color=["#d64545" if v > 0 else "#3a9a5b" for v in top.values]))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Contribution to risk (log-odds, SHAP)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Red bars push the risk up, green bars push it down.")
    with right:
        st.subheader("Occupation profile")
        ai = row["ai_gpt4_beta"]
        avg = df.drop_duplicates("OCC_CODE")["ai_gpt4_beta"].mean()
        st.write(f"**AI exposure score:** {ai:.2f} (average occupation: {avg:.2f})")
        st.progress(min(max(float(ai), 0.0), 1.0))
        if row["ai_imputed"]:
            st.caption("AI score imputed from similar occupations (SOC group).")
        wage = np.exp(row["log_median_wage_2022"])
        if pd.notna(wage):
            st.write(f"**Median annual wage 2022 (this metro):** ${wage:,.0f}")
        st.write(f"**Required preparation (job zone 1–5):** {row['job_zone']:.1f}")
        acts = {k: row[k] for k in ["wa_computers", "wa_documenting", "wa_creative", "wa_physical", "wa_public"]}
        st.dataframe(pd.DataFrame({"Work activity": [LABELS[k] for k in acts],
                                   "Importance (1–5)": [round(v, 1) for v in acts.values()]}),
                     hide_index=True, width="stretch")

with tab_map:
    occs_m = df[["OCC_TITLE"]].drop_duplicates().sort_values("OCC_TITLE")["OCC_TITLE"]
    c1, c2 = st.columns([3, 2])
    occ_m = c1.selectbox("Occupation", occs_m, key="map_occ",
                         index=int((occs_m == "Office Clerks, General").to_numpy().argmax()))
    color_by = c2.radio("Color by", ["Predicted risk", "Actual change 2022–2025", "Model vs. reality"],
                        horizontal=True)
    m = df[df["OCC_TITLE"] == occ_m].merge(coords, on="AREA", how="inner")
    hover = (m["AREA_TITLE"] + "<br>Predicted risk: " + (m["risk"] * 100).round(0).astype(int).astype(str) + "%"
             + "<br>Employment: " + m["emp_2022"].map("{:,.0f}".format) + " → " + m["emp_2025"].map("{:,.0f}".format)
             + " (" + (m["pct_change"] * 100).map("{:+.1f}".format) + "%)")
    size = (np.sqrt(m["emp_2022"]) / np.sqrt(m["emp_2022"]).max() * 28 + 5).clip(upper=34)
    fig = go.Figure()
    if color_by == "Model vs. reality":
        pred = m["risk"] >= THR
        cats = [("Correct: declined", pred & (m["decline"] == 1), "#d64545"),
                ("Correct: no decline", ~pred & (m["decline"] == 0), "#3a9a5b"),
                ("False alarm", pred & (m["decline"] == 0), "#f0a33a"),
                ("Missed decline", ~pred & (m["decline"] == 1), "#7b5cc4")]
        for name, mask, col in cats:
            fig.add_trace(go.Scattergeo(lat=m.loc[mask, "lat"], lon=m.loc[mask, "lon"], text=hover[mask],
                                        hoverinfo="text", mode="markers", name=f"{name} ({mask.sum()})",
                                        marker=dict(size=size[mask], color=col, opacity=0.85,
                                                    line=dict(width=0.5, color="#444"))))
        fig.update_layout(legend=dict(orientation="h", y=-0.02))
    else:
        if color_by == "Predicted risk":
            mid = THR / 0.8  # yellow at the decision threshold on a 0–80 % scale
            color, scale, rng, cbar = (m["risk"] * 100, [[0, "#3a9a5b"], [mid, "#f2d16b"], [1, "#d64545"]],
                                       (0, 80), "Risk (%)")
        else:
            color, scale, rng, cbar = (m["pct_change"].clip(-0.5, 0.5) * 100,
                                       [[0, "#d64545"], [0.5, "#f2f2f2"], [1, "#3a9a5b"]], (-50, 50), "Change (%)")
        fig.add_trace(go.Scattergeo(
            lat=m["lat"], lon=m["lon"], text=hover, hoverinfo="text", mode="markers",
            marker=dict(size=size, color=color, colorscale=scale, cmin=rng[0], cmax=rng[1], opacity=0.85,
                        line=dict(width=0.5, color="#444"), colorbar=dict(title=cbar))))
    fig.update_layout(geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#f7f7f5",
                               subunitcolor="#cfcfcf"), height=560, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Metro areas shown", f"{len(m)}")
    k2.metric("Average predicted risk", f"{m['risk'].mean():.0%}")
    k3.metric("Actually fell > 5%", f"{m['decline'].mean():.0%}")
    k4.metric("Model correct", f"{((m['risk'] >= THR) == (m['decline'] == 1)).mean():.0%}")
    st.caption(f"Risk is calibrated; a metro is flagged as 'decline' when risk ≥ {THR:.0%}. "
               "Bubble size = employment in 2022. Metro areas are placed at their principal city; "
               "nonmetropolitan areas and Puerto Rico are not shown on the map.")

with tab_rank:
    st.subheader("Occupations by average predicted risk")
    occ_rank = (df.groupby(["OCC_TITLE"])
                .agg(metros=("AREA", "size"), avg_risk=("risk", "mean"),
                     ai_exposure=("ai_gpt4_beta", "first"), actual_decline_share=("decline", "mean"))
                .query("metros >= 20").sort_values("avg_risk", ascending=False))
    occ_rank.index.name = "Occupation"
    occ_rank = occ_rank.rename(columns={"metros": "Metros", "avg_risk": "Avg. risk",
                                        "ai_exposure": "AI exposure", "actual_decline_share": "Actually fell"})
    fmt = {"Avg. risk": "{:.0%}", "Actually fell": "{:.0%}", "AI exposure": "{:.2f}"}
    st.markdown("**Highest risk**")
    st.dataframe(occ_rank.head(15).style.format(fmt), width="stretch")
    st.markdown("**Lowest risk**")
    st.dataframe(occ_rank.tail(15).iloc[::-1].style.format(fmt), width="stretch")
    st.caption("Only occupations observed in at least 20 metro areas.")

with tab_model:
    st.subheader("Model performance on the held-out test set")
    res = pd.DataFrame(metrics["results"])[["model", "features", "cv_roc_auc", "roc_auc", "pr_auc", "f1", "accuracy"]]
    st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns[2:]}), hide_index=True,
                 width="stretch")
    g, lo, hi = metrics["lgbm_auc_gain_from_ai"]
    st.write(f"Adding AI exposure to LightGBM changes test ROC-AUC by **{g:+.3f}** "
             f"(95% bootstrap CI {lo:+.3f} to {hi:+.3f}).")
    g2, lo2, hi2 = metrics["lgbm_vs_logreg_auc_gain"]
    st.write(f"LightGBM vs. logistic regression: **{g2:+.3f}** ROC-AUC (95% CI {lo2:+.3f} to {hi2:+.3f}).")
    cal = metrics.get("calibration")
    if cal:
        st.write(f"Calibration (Platt scaling on training out-of-fold predictions): mean predicted risk on the "
                 f"test set {cal['test_mean_predicted_raw']:.0%} → {cal['test_mean_predicted_calibrated']:.0%} "
                 f"(actual {cal['test_decline_share']:.0%}); Brier score {cal['test_brier_raw']:.3f} → "
                 f"{cal['test_brier_calibrated']:.3f}. Decision threshold {cal['threshold_from_train_oof']:.0%}: "
                 f"accuracy {cal['test_accuracy_at_threshold']:.0%}, F1 {cal['test_f1_at_threshold']:.3f}.")
        if (REPORTS / "calibration.png").exists():
            st.image(str(REPORTS / "calibration.png"), width=420)
    st.caption(f"Test set: {metrics['n_test_occupations']} occupations never seen in training "
               f"({metrics['n_test_rows']:,} occupation × metro rows).")
