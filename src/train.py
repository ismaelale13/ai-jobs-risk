"""Train and evaluate the baseline (logistic regression) and the advanced model (LightGBM).

Design
- Hold-out test set: 20 % of OCCUPATIONS (all their metro areas) never seen in training.
- Tuning: GroupKFold over occupations on the training part only.
- Each model is trained twice: without and with the AI exposure features.
"""
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C

FEATURE_SETS = {
    "without_ai": C.BASE_NUMERIC + C.CATEGORICAL,
    "with_ai": C.BASE_NUMERIC + C.CATEGORICAL + C.AI_FEATURES,
}


def load_data():
    df = pd.read_parquet(C.PROCESSED / "dataset.parquet")
    df["major_group"] = pd.Categorical(df["major_group"])
    return df


def split(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=C.SEED)
    tr, te = next(gss.split(df, groups=df[C.GROUP_COL]))
    assert set(df.iloc[tr][C.GROUP_COL]).isdisjoint(df.iloc[te][C.GROUP_COL])  # no occupation leaks
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def logreg(features):
    num = [f for f in features if f not in C.CATEGORICAL]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), C.CATEGORICAL),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))])
    grid = {"clf__C": [0.01, 0.1, 1.0]}
    return model, grid, GridSearchCV


def lgbm(features):
    model = LGBMClassifier(class_weight="balanced", random_state=C.SEED, verbose=-1, n_jobs=-1)
    grid = {
        "n_estimators": [200, 400, 700],
        "learning_rate": [0.02, 0.05],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [50, 200, 500],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "subsample": [0.8, 1.0],
        "subsample_freq": [1],
        "reg_lambda": [0.0, 1.0, 5.0],
    }
    return model, grid, RandomizedSearchCV


def metrics(y, p):
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "f1": f1_score(y, p >= 0.5),
        "accuracy": accuracy_score(y, p >= 0.5),
    }


def bootstrap_auc_diff(test, p_a, p_b, n=500):
    """95 % CI of AUC(b) - AUC(a), resampling occupations (the unit of independence)."""
    rng = np.random.default_rng(C.SEED)
    occs = test[C.GROUP_COL].unique()
    idx_by_occ = test.groupby(C.GROUP_COL).indices
    y = test[C.TARGET].values
    diffs = []
    for _ in range(n):
        idx = np.concatenate([idx_by_occ[o] for o in rng.choice(occs, len(occs), replace=True)])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p_b[idx]) - roc_auc_score(y[idx], p_a[idx]))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def error_analysis(test, p):
    t = test.assign(pred=p, wrong=((p >= 0.5).astype(int) != test[C.TARGET]).astype(int))
    t["metro_size"] = pd.qcut(t["log_area_emp_2022"], 3, labels=["small", "medium", "large"])

    def agg(g):
        auc = roc_auc_score(g[C.TARGET], g["pred"]) if g[C.TARGET].nunique() > 1 else np.nan
        return pd.Series({"rows": len(g), "occupations": g[C.GROUP_COL].nunique(),
                          "decline_rate": g[C.TARGET].mean(), "error_rate": g["wrong"].mean(), "roc_auc": auc})

    out = []
    for col in ["major_group", "metro_size", "nonmetro", "ai_imputed"]:
        r = t.groupby(col, observed=True).apply(agg).reset_index().rename(columns={col: "value"})
        r.insert(0, "dimension", col)
        out.append(r)
    return pd.concat(out).round(3)


def run():
    C.REPORTS.mkdir(exist_ok=True)
    (C.ROOT / "models").mkdir(exist_ok=True)
    df = load_data()
    train, test = split(df)
    cv = GroupKFold(n_splits=C.N_FOLDS)
    groups = train[C.GROUP_COL]

    results, preds, best = [], {}, {}
    for model_name, factory in [("logreg", logreg), ("lightgbm", lgbm)]:
        for fs_name, feats in FEATURE_SETS.items():
            model, grid, Search = factory(feats)
            kw = {"n_iter": 20, "random_state": C.SEED} if Search is RandomizedSearchCV else {}
            search = Search(model, grid, scoring="roc_auc", cv=cv, n_jobs=1, **kw)
            search.fit(train[feats], train[C.TARGET], groups=groups)
            p = search.best_estimator_.predict_proba(test[feats])[:, 1]
            key = f"{model_name}_{fs_name}"
            preds[key], best[key] = p, search
            results.append({"model": model_name, "features": fs_name, "cv_roc_auc": search.best_score_,
                            **metrics(test[C.TARGET], p), "best_params": search.best_params_})
            joblib.dump(search.best_estimator_, C.ROOT / "models" / f"{key}.joblib")
            print(f"{key:24s} cv_auc={search.best_score_:.3f} test_auc={results[-1]['roc_auc']:.3f}")

    res = pd.DataFrame(results)
    res.drop(columns="best_params").round(4).to_csv(C.REPORTS / "model_comparison.csv", index=False)

    summary = {
        "n_train_rows": len(train), "n_test_rows": len(test),
        "n_train_occupations": int(train[C.GROUP_COL].nunique()),
        "n_test_occupations": int(test[C.GROUP_COL].nunique()),
        "test_decline_share": float(test[C.TARGET].mean()),
        "results": res.to_dict(orient="records"),
        "lgbm_auc_gain_from_ai": bootstrap_auc_diff(test, preds["lightgbm_without_ai"], preds["lightgbm_with_ai"]),
        "lgbm_vs_logreg_auc_gain": bootstrap_auc_diff(test, preds["logreg_with_ai"], preds["lightgbm_with_ai"]),
    }
    (C.REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    error_analysis(test, preds["lightgbm_with_ai"]).to_csv(C.REPORTS / "error_analysis.csv", index=False)

    # Out-of-sample predictions for every row (for the dashboard):
    # training occupations -> out-of-fold predictions, test occupations -> hold-out predictions.
    final = best["lightgbm_with_ai"].best_estimator_
    feats = FEATURE_SETS["with_ai"]
    oof = np.zeros(len(train))
    for tr_i, va_i in cv.split(train, groups=groups):
        m = LGBMClassifier(**final.get_params()).fit(train.iloc[tr_i][feats], train.iloc[tr_i][C.TARGET])
        oof[va_i] = m.predict_proba(train.iloc[va_i][feats])[:, 1]
    scored = pd.concat([train.assign(risk=oof, in_test_set=0),
                        test.assign(risk=preds["lightgbm_with_ai"], in_test_set=1)], ignore_index=True)
    scored.to_parquet(C.PROCESSED / "scored.parquet", index=False)
    return summary, final, scored


if __name__ == "__main__":
    run()
