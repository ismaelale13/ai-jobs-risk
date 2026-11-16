"""SHAP explanations for the final LightGBM model (with AI features)."""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config as C
from src.train import FEATURE_SETS


def run(scored=None):
    feats = FEATURE_SETS["with_ai"]
    model = joblib.load(C.ROOT / "models" / "lightgbm_with_ai.joblib")
    if scored is None:
        scored = pd.read_parquet(C.PROCESSED / "scored.parquet")
    X = scored[feats].copy()
    X["major_group"] = pd.Categorical(X["major_group"], categories=model.booster_.pandas_categorical[0])

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv  # log-odds contributions for class 1 (decline)

    shap_df = pd.DataFrame(sv, columns=[f"shap_{f}" for f in feats])
    shap_df.to_parquet(C.PROCESSED / "shap_values.parquet", index=False)

    # global importance (mean |SHAP|), AI features summed as one block too
    imp = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(C.REPORTS / "shap_importance.csv")
    ai_share = imp[C.AI_FEATURES].sum() / imp.sum()

    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(C.REPORTS / "shap_summary.png", dpi=150)
    plt.close()

    # dependence of risk on the main AI exposure score
    shap.dependence_plot("ai_gpt4_beta", sv, X, interaction_index=None, show=False)
    plt.tight_layout()
    plt.savefig(C.REPORTS / "shap_ai_gpt4_beta.png", dpi=150)
    plt.close()
    return imp, float(ai_share)


if __name__ == "__main__":
    imp, share = run()
    print(imp.head(10))
    print("AI features share of total |SHAP|:", round(share, 3))
