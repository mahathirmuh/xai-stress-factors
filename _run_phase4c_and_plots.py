"""Run Phase 4c (cross-model) + generate RF SHAP plots + update summary.json.
Dijalankan SETELAH _shap_data_rf.pkl tersedia.
"""
import sys, pickle, os
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

OUT_DIR = "outputs"

# Load all caches
print("[Loading caches...]", flush=True)
with open(f'{OUT_DIR}/_split_data.pkl', 'rb') as f:
    d = pickle.load(f)
with open(f'{OUT_DIR}/_models.pkl', 'rb') as f:
    m = pickle.load(f)
with open(f'{OUT_DIR}/_shap_data.pkl', 'rb') as f:
    s = pickle.load(f)
with open(f'{OUT_DIR}/_shap_data_rf.pkl', 'rb') as f:
    s_rf = pickle.load(f)

shap_values_rf = s_rf['shap_values_rf']
expected_value_rf = s_rf['expected_value_rf']
X_shap_rf = s_rf['X_shap_rf']
feat_imp_rf = s_rf['feat_imp_rf']
feat_imp_cat = s['feat_imp']
print(f"  RF SHAP loaded: {shap_values_rf.shape}", flush=True)

# ===== Generate RF SHAP plots =====
print("[Generating RF SHAP plots...]", flush=True)
import seaborn as sns
sns.set_style("whitegrid")

plt.figure()
shap.summary_plot(shap_values_rf, X_shap_rf, show=False, max_display=20)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_summary_beeswarm_rf.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: shap_summary_beeswarm_rf.png", flush=True)

plt.figure()
shap.summary_plot(shap_values_rf, X_shap_rf, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_bar_importance_rf.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: shap_bar_importance_rf.png", flush=True)

# ===== Phase 4c: Cross-Model Stability =====
print("\n[Phase 4c: Cross-Model Stability]", flush=True)
from xai_stress_analysis import phase4c_cross_model_stability
cross_model = phase4c_cross_model_stability(feat_imp_cat, feat_imp_rf)
print(f"\nCross-model result: {cross_model}", flush=True)

# ===== Update summary.json =====
print("\n[Updating summary.json with cross-model + RF data]", flush=True)
import json

results = m['results']
best_name = m['best_name']
mean_tau = s['mean_tau']

summary = {
    "best_model": best_name,
    "test_metrics": results[best_name]["test"],
    "cv_r2_mean": results[best_name]["cv_r2_mean"],
    "cv_r2_std": results[best_name]["cv_r2_std"],
    "shap_stability_within_catboost_kendall_tau": mean_tau,
    "cross_model_stability": cross_model,
    "top10_features_catboost": feat_imp_cat.head(10).round(4).to_dict(),
    "top10_features_rf": feat_imp_rf.head(10).round(4).to_dict(),
    "individual_cases": ["high_stress", "low_stress", "borderline"],
}
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\n=== FINAL SUMMARY ===")
print(json.dumps(summary, indent=2, default=str))

print("\n=== Output files ===")
for f in sorted(os.listdir(OUT_DIR)):
    if not f.startswith("_"):
        size = os.path.getsize(f"{OUT_DIR}/{f}") / 1024
        print(f"  {f} ({size:.1f} KB)")
