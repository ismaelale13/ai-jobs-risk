"""Additional validation: placebo test, robustness checks and out-of-time test.

1. Placebo (pre-ChatGPT): does AI exposure add predictive value for 2019 -> 2022, before ChatGPT?
   If it helps equally before and after, the AI score captures older trends, not a ChatGPT effect.
2. Robustness (2022 -> 2025): other decline thresholds (0 %, 10 %), single AI measures,
   employment-weighted evaluation.
3. Out-of-time: train on 2019 -> 2022 only, predict 2022 -> 2025 - the realistic early-warning setup.

All comparisons use the tuned LightGBM / logistic-regression settings of the main model and
5-fold cross-validation grouped by occupation (no occupation in training and validation folds at once).
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C
from src.periods import AI, CATEGORICAL, NUMERIC, build_period

OUT = C.REPORTS / "robustness"
BASE = NUMERIC + CATEGORICAL
WITH_AI = BASE + AI


def lgbm():
    params = joblib.load(C.ROOT / "models" / "lightgbm_with_ai.joblib").get_params()
    return LGBMClassifier(**params)


def logreg(features):
    c = joblib.load(C.ROOT / "models" / "logreg_with_ai.joblib").get_params()["clf__C"]
    num = [f for f in features if f not in CATEGORICAL]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=3000))])


def oof_predictions(df, features, model, target=C.TARGET):
    """Out-of-fold probabilities with occupation-grouped 5-fold CV."""
    p = np.zeros(len(df))
    for tr, va in GroupKFold(n_splits=C.N_FOLDS).split(df, groups=df["OCC_CODE"]):
        m = clone(model).fit(df.iloc[tr][features], df.iloc[tr][target])
        p[va] = m.predict_proba(df.iloc[va][features])[:, 1]
    return p


def auc_gain_ci(df, p_a, p_b, target=C.TARGET, weights=None, n=300):
    """AUC(b) - AUC(a) with a 95 % bootstrap CI over occupations."""
    rng = np.random.default_rng(C.SEED)
    idx_by_occ = df.groupby("OCC_CODE").indices
    occs = np.array(list(idx_by_occ))
    y = df[target].values
    w = None if weights is None else np.asarray(weights)

    def auc(i, p):
        return roc_auc_score(y[i], p[i], sample_weight=None if w is None else w[i])

    full = np.arange(len(df))
    point = auc(full, p_b) - auc(full, p_a)
    diffs = []
    for _ in range(n):
        i = np.concatenate([idx_by_occ[o] for o in rng.choice(occs, len(occs), replace=True)])
        diffs.append(auc(i, p_b) - auc(i, p_a))
    return {"auc_without_ai": auc(full, p_a), "auc_with_ai": auc(full, p_b), "gain": point,
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5))}


def placebo(p1, p2):
    rows = []
    for name, d in [("2019 -> 2022 (before ChatGPT)", p1), ("2022 -> 2025 (after ChatGPT)", p2)]:
        pa, pb = oof_predictions(d, BASE, lgbm()), oof_predictions(d, WITH_AI, lgbm())
        rows.append({"period": name, "rows": len(d), "decline_share": d[C.TARGET].mean(), **auc_gain_ci(d, pa, pb)})
    res = pd.DataFrame(rows)

    # descriptive view: decline share by AI exposure quintile in both periods
    q = []
    for name, d in [("2019-2022", p1), ("2022-2025", p2)]:
        occ = d.groupby("OCC_CODE")["ai_gpt4_beta"].first()
        bins = pd.qcut(occ, 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
        t = d.assign(q=d["OCC_CODE"].map(bins))
        g = t.groupby("q", observed=True).agg(decline_share=(C.TARGET, "mean"), e0=("emp0", "sum"), e1=("emp1", "sum"))
        g["growth"] = g["e1"] / g["e0"] - 1
        q.append(g[["decline_share", "growth"]].add_prefix(f"{name} "))
    quint = pd.concat(q, axis=1)
    return res, quint


def robustness(p2):
    rows = []
    # (a) other decline thresholds
    for thr in [0.0, -0.05, -0.10]:
        d = p2.assign(y=(p2["pct_change"] < thr).astype(int))
        pa = oof_predictions(d, BASE, lgbm(), "y")
        pb = oof_predictions(d, WITH_AI, lgbm(), "y")
        rows.append({"check": f"decline threshold {thr:+.0%}", "positive_share": d["y"].mean(),
                     **auc_gain_ci(d, pa, pb, target="y")})
    base_p = oof_predictions(p2, BASE, lgbm())
    # (b) single AI measures
    for label, cols in [("only GPT-4 exposure (beta)", ["ai_gpt4_beta"]),
                        ("only human-rated exposure (beta)", ["ai_human_beta"])]:
        pb = oof_predictions(p2, BASE + cols, lgbm())
        rows.append({"check": label, "positive_share": p2[C.TARGET].mean(), **auc_gain_ci(p2, base_p, pb)})
    # (c) employment-weighted evaluation (large cells count more)
    full_p = oof_predictions(p2, WITH_AI, lgbm())
    rows.append({"check": "employment-weighted AUC", "positive_share": p2[C.TARGET].mean(),
                 **auc_gain_ci(p2, base_p, full_p, weights=p2["emp0"])})
    # (d) AI exposure as the only occupation information (no O*NET task features)
    no_onet = [f for f in BASE if not f.startswith("wa_") and f != "job_zone"]
    pa = oof_predictions(p2, no_onet, lgbm())
    pb = oof_predictions(p2, no_onet + AI, lgbm())
    rows.append({"check": "without O*NET task features", "positive_share": p2[C.TARGET].mean(),
                 **auc_gain_ci(p2, pa, pb)})
    return pd.DataFrame(rows)


def out_of_time(p1, p2):
    """Train on 2019 -> 2022 (features at 2019), predict 2022 -> 2025 (features at 2022)."""
    test_occ = set(pd.read_parquet(C.PROCESSED / "scored.parquet").query("in_test_set == 1")["OCC_CODE"]) \
        if (C.PROCESSED / "scored.parquet").exists() else set()
    rows = []
    for name, model, feats in [("logistic regression", logreg(WITH_AI), WITH_AI),
                               ("LightGBM without AI", lgbm(), BASE),
                               ("LightGBM with AI", lgbm(), WITH_AI)]:
        p2c = p2.copy()
        p2c["major_group"] = pd.Categorical(p2c["major_group"], categories=p1["major_group"].cat.categories)
        m = clone(model).fit(p1[feats], p1[C.TARGET])
        p = m.predict_proba(p2c[feats])[:, 1]
        in_period = oof_predictions(p2, feats, model)
        mask = p2["OCC_CODE"].isin(test_occ).values
        rows.append({"model": name,
                     "auc_out_of_time": roc_auc_score(p2[C.TARGET], p),
                     "auc_in_period_cv": roc_auc_score(p2[C.TARGET], in_period),
                     "auc_out_of_time_test_occupations": roc_auc_score(p2.loc[mask, C.TARGET], p[mask])
                     if mask.any() else np.nan,
                     "mean_predicted": p.mean(), "actual_share": p2[C.TARGET].mean()})
    return pd.DataFrame(rows)


def plot(plac, oot):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    blue, orange, grid = "#2a78d6", "#eb6834", "#e6e5e1"
    ax = axes[0]
    x = np.arange(len(plac))
    ax.bar(x, plac["gain"], color=[blue, orange], width=0.5)
    ax.errorbar(x, plac["gain"], yerr=[plac["gain"] - plac["ci_low"], plac["ci_high"] - plac["gain"]],
                fmt="none", ecolor="#333", capsize=6, linewidth=1.2)
    ax.axhline(0, color="#888", linewidth=1)
    ax.set_xticks(x, ["2019→2022\nbefore ChatGPT", "2022→2025\nafter ChatGPT"])
    ax.set_title("Placebo: AUC gain from adding AI exposure (95 % CI)", loc="left", fontsize=11)
    ax.set_ylabel("Δ ROC-AUC (LightGBM)")
    ax = axes[1]
    x = np.arange(len(oot))
    ax.bar(x - 0.18, oot["auc_in_period_cv"], width=0.36, color=blue, label="trained in same period (CV)")
    ax.bar(x + 0.18, oot["auc_out_of_time"], width=0.36, color=orange, label="trained on 2019→2022")
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1)
    ax.set_xticks(x, oot["model"].str.replace(" ", "\n", n=1))
    ax.set_ylim(0.45, max(0.75, oot[["auc_in_period_cv", "auc_out_of_time"]].values.max() + 0.03))
    ax.set_title("Out-of-time test on 2022→2025", loc="left", fontsize=11)
    ax.set_ylabel("ROC-AUC")
    ax.legend(fontsize=8, frameon=False)
    for a in axes:
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
        a.grid(axis="y", color=grid)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "robustness_summary.png", dpi=160)
    plt.close(fig)


def report(plac, quint, rob, oot):
    L = ["=" * 72, "VALIDATION: PLACEBO, ROBUSTNESS, OUT-OF-TIME", "=" * 72]
    L.append("\n1) PLACEBO - does AI exposure help predict decline before ChatGPT as well?")
    for _, r in plac.iterrows():
        L.append(f"   {r['period']:<32s} AUC without AI {r['auc_without_ai']:.3f} | with AI {r['auc_with_ai']:.3f}"
                 f" | gain {r['gain']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
    L.append("   Decline share and employment growth by AI exposure quintile:")
    L.append("   " + quint.round(3).to_string().replace("\n", "\n   "))
    L.append("\n2) ROBUSTNESS (2022 -> 2025) - AUC gain from AI features under other choices")
    for _, r in rob.iterrows():
        L.append(f"   {r['check']:<36s} AUC {r['auc_without_ai']:.3f} -> {r['auc_with_ai']:.3f}"
                 f"  gain {r['gain']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
    L.append("\n3) OUT-OF-TIME - train on 2019 -> 2022, predict 2022 -> 2025")
    for _, r in oot.iterrows():
        L.append(f"   {r['model']:<22s} out-of-time AUC {r['auc_out_of_time']:.3f}"
                 f" (unseen test occupations {r['auc_out_of_time_test_occupations']:.3f})"
                 f" vs. same-period CV {r['auc_in_period_cv']:.3f};"
                 f" mean predicted {r['mean_predicted']:.0%} vs actual {r['actual_share']:.0%}")
    L.append("\nFiles: reports/robustness/*.csv, robustness_summary.png")
    L.append("=" * 72)
    return "\n".join(L)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = build_period(2016, 2019, 2022)
    p2 = build_period(2019, 2022, 2025)
    plac, quint = placebo(p1, p2)
    rob = robustness(p2)
    oot = out_of_time(p1, p2)
    plac.round(4).to_csv(OUT / "placebo.csv", index=False)
    quint.round(4).to_csv(OUT / "placebo_quintiles.csv")
    rob.round(4).to_csv(OUT / "robustness_checks.csv", index=False)
    oot.round(4).to_csv(OUT / "out_of_time.csv", index=False)
    plot(plac, oot)
    text = report(plac, quint, rob, oot)
    (OUT / "robustness_report.txt").write_text(text)
    (OUT / "robustness_summary.json").write_text(json.dumps(
        {"placebo": plac.to_dict(orient="records"), "robustness": rob.to_dict(orient="records"),
         "out_of_time": oot.to_dict(orient="records")}, indent=2, default=float))
    print(text)
    return plac, rob, oot


if __name__ == "__main__":
    run()
