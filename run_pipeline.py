"""Run the full pipeline: raw ZIPs -> dataset -> models -> SHAP -> dashboard data.

Usage:  python run_pipeline.py                     # everything (~18 min)
        python run_pipeline.py --skip-robustness   # main model only (~10 min)
"""
import json
import sys
import time

from src import build_dataset, calibrate, dashboard_data, eda, explain, robustness, train
from src import config as C


def main():
    t0 = time.time()
    print("1/5 Building dataset ...")
    _, stats = build_dataset.build()
    print("   ", stats)
    (C.REPORTS).mkdir(exist_ok=True)
    (C.REPORTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    eda.run()
    print("    EDA figures in reports/eda/")

    print("2/5 Training and evaluating models (grouped CV, this takes a few minutes) ...")
    train.run()
    cal = calibrate.run()
    print(f"    calibrated: mean risk {cal['test_mean_predicted_calibrated']:.2f} vs actual {cal['test_decline_share']:.2f}, "
          f"threshold {cal['threshold_from_train_oof']:.2f}")

    print("3/5 SHAP explanations ...")
    _, ai_share = explain.run()
    print(f"    AI features share of total |SHAP|: {ai_share:.1%}")

    print("4/5 Dashboard data ...")
    dashboard_data.run()

    if "--skip-robustness" not in sys.argv:
        print("5/5 Placebo, robustness and out-of-time tests (~8 min) ...")
        robustness.run()
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Results in reports/, dashboard: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
