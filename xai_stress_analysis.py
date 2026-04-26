"""
Explainable AI for Stress Factors
Deciphering Sleep and Lifestyle Impacts on Stress Score

Pipeline lengkap Phase 1-6:
  Phase 1: Setup & EDA
  Phase 2: Feature Engineering & Preprocessing
  Phase 3: Model Training (CatBoost + Random Forest)
  Phase 4: Global SHAP
  Phase 5: Individual SHAP
  Phase 6: Final Report
"""

import os
import json
import warnings
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import kendalltau

from catboost import CatBoostRegressor, Pool
import shap

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ============================================================
# Konfigurasi
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "sleep_health_dataset.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NUMERIC_FEATURES = [
    "age", "bmi", "sleep_duration_hrs", "sleep_quality_score",
    "rem_percentage", "deep_sleep_percentage", "sleep_latency_mins",
    "wake_episodes_per_night", "caffeine_mg_before_bed",
    "alcohol_units_before_bed", "screen_time_before_bed_mins",
    "steps_that_day", "nap_duration_mins", "work_hours_that_day",
    "heart_rate_resting_bpm", "room_temperature_celsius",
    "weekend_sleep_diff_hrs",
    "exercise_day", "sleep_aid_used", "shift_work",
]
CATEGORICAL_FEATURES = [
    "gender", "occupation", "country", "chronotype",
    "mental_health_condition", "season", "day_type",
]
DROP_COLS = [
    "person_id", "cognitive_performance_score",
    "sleep_disorder_risk", "felt_rested",
]
TARGET = "stress_score"
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def banner(title):
    line = "=" * 70
    print(f"\n{line}\n {title}\n{line}")


# ============================================================
# Phase 1 — Setup & EDA
# ============================================================
def phase1_eda():
    banner("PHASE 1 — Setup & EDA")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"Missing values total: {df.isna().sum().sum()}")
    print(f"\nstress_score stats:\n{df[TARGET].describe()}")

    # Distribusi target
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(df[TARGET], bins=40, color="#3a76d8", edgecolor="white")
    axes[0].set_title("Distribusi stress_score")
    axes[0].set_xlabel("stress_score")
    axes[0].set_ylabel("Jumlah")
    sns.boxplot(x=df[TARGET], ax=axes[1], color="#3a76d8")
    axes[1].set_title("Boxplot stress_score (deteksi outlier)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_target_distribution.png"), dpi=120)
    plt.close()

    # Korelasi numerik vs stress_score
    num_corr = df[NUMERIC_FEATURES + [TARGET]].corr()[[TARGET]].drop(TARGET).sort_values(by=TARGET)
    plt.figure(figsize=(7, 9))
    sns.heatmap(num_corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, cbar_kws={"label": "Korelasi vs stress_score"})
    plt.title("Korelasi Fitur Numerik vs stress_score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda_correlation_heatmap.png"), dpi=120)
    plt.close()

    print("\n[OK] Phase 1 plots saved: eda_target_distribution.png, eda_correlation_heatmap.png")
    return df


# ============================================================
# Phase 2 — Preprocessing
# ============================================================
def phase2_preprocess(df):
    banner("PHASE 2 — Preprocessing & Split 70/15/15")

    # Sanity: pastikan kolom drop ada
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    # Stratified split via stress_score bins
    bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_temp, X_test, y_temp, y_test, bins_temp, _ = train_test_split(
        X, y, bins, test_size=0.15, random_state=RANDOM_SEED, stratify=bins
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=RANDOM_SEED, stratify=bins_temp
    )  # 0.15 / 0.85 ≈ 0.1765

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # Cast kategorikal ke str (CatBoost butuh string)
    for c in CATEGORICAL_FEATURES:
        X_train[c] = X_train[c].astype(str)
        X_val[c] = X_val[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    # Sanity check leakage
    assert "sleep_disorder_risk" not in X_train.columns
    assert "cognitive_performance_score" not in X_train.columns
    assert "felt_rested" not in X_train.columns
    print("[OK] Data leakage guard verified — co-outcome columns dropped.")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# Phase 3 — Model Training & Comparison
# ============================================================
def phase3_train(X_train, X_val, X_test, y_train, y_val, y_test):
    banner("PHASE 3 — Train CatBoost + Random Forest")

    cat_idx = [X_train.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    # ---------- CatBoost ----------
    print("\n[Training CatBoost...]")
    t0 = time.time()
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="RMSE",
        early_stopping_rounds=50,
        random_seed=RANDOM_SEED,
        verbose=100,
    )
    cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print(f"CatBoost trained in {time.time()-t0:.1f}s")

    # ---------- Random Forest ----------
    print("\n[Encoding categorical for RF...]")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_rf = X_train.copy()
    X_val_rf = X_val.copy()
    X_test_rf = X_test.copy()
    X_train_rf[CATEGORICAL_FEATURES] = encoder.fit_transform(X_train_rf[CATEGORICAL_FEATURES])
    X_val_rf[CATEGORICAL_FEATURES] = encoder.transform(X_val_rf[CATEGORICAL_FEATURES])
    X_test_rf[CATEGORICAL_FEATURES] = encoder.transform(X_test_rf[CATEGORICAL_FEATURES])

    print("[Training Random Forest...]")
    t0 = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rf_model.fit(X_train_rf, y_train)
    print(f"Random Forest trained in {time.time()-t0:.1f}s")

    # ---------- Evaluate both ----------
    def metrics(y_true, y_pred):
        return {
            "R2": r2_score(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": mean_absolute_error(y_true, y_pred),
        }

    cat_test_pred = cat_model.predict(X_test)
    cat_val_pred = cat_model.predict(X_val)
    rf_test_pred = rf_model.predict(X_test_rf)
    rf_val_pred = rf_model.predict(X_val_rf)

    results = {
        "CatBoost": {"val": metrics(y_val, cat_val_pred), "test": metrics(y_test, cat_test_pred)},
        "RandomForest": {"val": metrics(y_val, rf_val_pred), "test": metrics(y_test, rf_test_pred)},
    }

    # 5-fold CV R² (subsample for speed: 30K rows)
    print("\n[5-Fold CV R² (subsampled 30K rows for speed)...]")
    rng = np.random.RandomState(RANDOM_SEED)
    sub_idx = rng.choice(len(X_train), size=min(30000, len(X_train)), replace=False)
    X_sub = X_train.iloc[sub_idx].reset_index(drop=True)
    y_sub = y_train.iloc[sub_idx].reset_index(drop=True)
    X_sub_rf = X_train_rf.iloc[sub_idx].reset_index(drop=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cat_cv, rf_cv = [], []
    for fold, (tr, te) in enumerate(kf.split(X_sub), 1):
        # CatBoost
        cm = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                               random_seed=RANDOM_SEED, verbose=0)
        cm.fit(Pool(X_sub.iloc[tr], y_sub.iloc[tr], cat_features=cat_idx))
        cat_cv.append(r2_score(y_sub.iloc[te], cm.predict(X_sub.iloc[te])))
        # RF
        rm = RandomForestRegressor(n_estimators=150, min_samples_leaf=5,
                                   n_jobs=-1, random_state=RANDOM_SEED)
        rm.fit(X_sub_rf.iloc[tr], y_sub.iloc[tr])
        rf_cv.append(r2_score(y_sub.iloc[te], rm.predict(X_sub_rf.iloc[te])))
        print(f"  Fold {fold}: CatBoost R²={cat_cv[-1]:.4f}  RF R²={rf_cv[-1]:.4f}")

    results["CatBoost"]["cv_r2_mean"] = float(np.mean(cat_cv))
    results["CatBoost"]["cv_r2_std"] = float(np.std(cat_cv))
    results["RandomForest"]["cv_r2_mean"] = float(np.mean(rf_cv))
    results["RandomForest"]["cv_r2_std"] = float(np.std(rf_cv))

    print("\n=== Model Performance Summary ===")
    print(json.dumps(results, indent=2))

    # Pilih best model berdasarkan val R²
    best_name = "CatBoost" if results["CatBoost"]["val"]["R2"] >= results["RandomForest"]["val"]["R2"] else "RandomForest"
    print(f"\n[BEST MODEL] {best_name} (akan dipakai untuk SHAP)")

    # ---------- Plot model comparison ----------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    metrics_names = ["R2", "RMSE", "MAE"]
    for i, m in enumerate(metrics_names):
        vals = [results["CatBoost"]["test"][m], results["RandomForest"]["test"][m]]
        bars = ax[i].bar(["CatBoost", "RandomForest"], vals,
                         color=["#3a76d8", "#d8763a"])
        ax[i].set_title(f"Test {m}")
        for b, v in zip(bars, vals):
            ax[i].text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                       ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=120)
    plt.close()

    # Save results
    with open(os.path.join(OUT_DIR, "model_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return cat_model, rf_model, X_test_rf, results, best_name


# ============================================================
# Phase 4 — Global SHAP
# ============================================================
def phase4_global_shap(cat_model, X_train, X_test, y_train, y_test):
    banner("PHASE 4 — Global SHAP Analysis (CatBoost)")

    cat_idx = [X_test.columns.get_loc(c) for c in CATEGORICAL_FEATURES]

    # Sample test untuk speed (15K rows fine, but SHAP dependence cukup 5K)
    n_sample = min(5000, len(X_test))
    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = rng.choice(len(X_test), size=n_sample, replace=False)
    X_shap = X_test.iloc[sample_idx].reset_index(drop=True)
    y_shap = y_test.iloc[sample_idx].reset_index(drop=True)

    print(f"Computing SHAP on {n_sample} test samples...")
    explainer = shap.TreeExplainer(cat_model)
    shap_pool = Pool(X_shap, cat_features=cat_idx)
    shap_values = explainer.shap_values(shap_pool)
    expected_value = float(explainer.expected_value)
    print(f"SHAP shape: {shap_values.shape}  expected_value: {expected_value:.4f}")

    # Sanity check: SHAP sum + base = predict
    preds = cat_model.predict(X_shap)
    diff = np.abs(shap_values.sum(axis=1) + expected_value - preds).max()
    print(f"[Sanity] max |SHAP_sum + base - pred| = {diff:.6f}  (target < 0.01)")
    assert diff < 0.01, "SHAP sanity check failed"

    # 4.2 Summary Plot (Beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary_beeswarm.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # 4.3 Bar Plot
    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_bar_importance.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # 4.4 Dependence Plots — top-5 by mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.Series(mean_abs_shap, index=X_shap.columns).sort_values(ascending=False)
    top5 = feat_imp.head(5).index.tolist()
    print(f"Top-5 fitur (mean |SHAP|): {top5}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, feat in enumerate(top5):
        # Skip categorical for dependence plot
        if feat in CATEGORICAL_FEATURES:
            X_dep = X_shap.copy()
            X_dep[feat] = pd.Categorical(X_dep[feat]).codes
            shap.dependence_plot(feat, shap_values, X_dep, ax=axes[i], show=False)
        else:
            shap.dependence_plot(feat, shap_values, X_shap, ax=axes[i], show=False)
        axes[i].set_title(f"Dependence: {feat}")
    if len(top5) < 6:
        axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_dependence_top5.png"), dpi=120)
    plt.close()

    # 4.5 Feature Importance Stability — 5-fold CV SHAP
    print("\n[5-Fold CV SHAP Stability — subsampled 20K rows]")
    rng = np.random.RandomState(RANDOM_SEED)
    sub_idx = rng.choice(len(X_train), size=min(20000, len(X_train)), replace=False)
    X_sub = X_train.iloc[sub_idx].reset_index(drop=True)
    y_sub = y_train.iloc[sub_idx].reset_index(drop=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_rankings = []
    fold_importances = []
    for fold, (tr, te) in enumerate(kf.split(X_sub), 1):
        cm = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                               random_seed=RANDOM_SEED, verbose=0)
        cm.fit(Pool(X_sub.iloc[tr], y_sub.iloc[tr], cat_features=cat_idx))
        sv = shap.TreeExplainer(cm).shap_values(
            Pool(X_sub.iloc[te], cat_features=cat_idx)
        )
        imp = np.abs(sv).mean(axis=0)
        fold_importances.append(imp)
        ranking = pd.Series(imp, index=X_sub.columns).rank(ascending=False).astype(int)
        fold_rankings.append(ranking)
        print(f"  Fold {fold} top-3: {ranking.sort_values().head(3).index.tolist()}")

    # Kendall's Tau pairwise antar fold
    taus = []
    for i in range(5):
        for j in range(i + 1, 5):
            tau, _ = kendalltau(fold_rankings[i], fold_rankings[j])
            taus.append(tau)
    mean_tau = float(np.mean(taus))
    print(f"\nMean pairwise Kendall's Tau (5 folds): {mean_tau:.4f}")

    # Heatmap stability
    rank_df = pd.concat(fold_rankings, axis=1)
    rank_df.columns = [f"Fold{i+1}" for i in range(5)]
    rank_df["mean_rank"] = rank_df.mean(axis=1)
    rank_df = rank_df.sort_values("mean_rank")
    plt.figure(figsize=(8, max(8, len(rank_df) * 0.3)))
    sns.heatmap(rank_df[[f"Fold{i+1}" for i in range(5)]],
                annot=True, fmt=".0f", cmap="YlGnBu_r",
                cbar_kws={"label": "Rank (1 = paling penting)"})
    plt.title(f"Feature Importance Stability — 5-Fold CV\nMean Kendall's Tau = {mean_tau:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_stability_heatmap.png"), dpi=120)
    plt.close()

    return shap_values, expected_value, X_shap, y_shap, top5, mean_tau, feat_imp


# ============================================================
# Phase 5 — Individual SHAP Analysis
# ============================================================
def phase5_individual(cat_model, shap_values, expected_value, X_shap, y_shap):
    banner("PHASE 5 — Individual SHAP Analysis")

    preds = cat_model.predict(X_shap)

    # Pilih 3 individu
    high_idx = np.argmax(preds)  # paling tinggi diprediksi
    low_idx = np.argmin(preds)   # paling rendah diprediksi
    border_idx = int(np.argmin(np.abs(preds - 5.0)))

    cases = {
        "high_stress": int(high_idx),
        "low_stress": int(low_idx),
        "borderline": int(border_idx),
    }

    feature_names = X_shap.columns.tolist()
    narratives = {}

    for label, idx in cases.items():
        pred = float(preds[idx])
        actual = float(y_shap.iloc[idx])
        sv = shap_values[idx]

        print(f"\n[{label.upper()}] idx={idx}  predicted={pred:.2f}  actual={actual:.2f}")

        # Waterfall plot
        explanation = shap.Explanation(
            values=sv,
            base_values=expected_value,
            data=X_shap.iloc[idx].values,
            feature_names=feature_names,
        )
        plt.figure()
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_waterfall_{label}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close()

        # Narasi
        contrib = pd.Series(sv, index=feature_names).sort_values(key=abs, ascending=False)
        top_pos = contrib[contrib > 0].head(3)
        top_neg = contrib[contrib < 0].head(3)
        feat_vals = X_shap.iloc[idx]

        narrative = (
            f"Individu #{idx} memiliki stress_score aktual {actual:.2f} "
            f"dan diprediksi {pred:.2f} (baseline: {expected_value:.2f}).\n"
            f"  Faktor PENINGKAT stres terbesar:\n"
        )
        for f, v in top_pos.items():
            narrative += f"    - {f} = {feat_vals[f]} (SHAP +{v:.3f})\n"
        narrative += "  Faktor PENEKAN stres terbesar:\n"
        for f, v in top_neg.items():
            narrative += f"    - {f} = {feat_vals[f]} (SHAP {v:.3f})\n"
        narratives[label] = narrative
        print(narrative)

    # Force plot interaktif HTML — gunakan high_stress sebagai contoh
    try:
        force = shap.force_plot(
            expected_value,
            shap_values[high_idx],
            X_shap.iloc[high_idx],
            feature_names=feature_names,
        )
        shap.save_html(os.path.join(OUT_DIR, "force_plot.html"), force)
        print("[OK] force_plot.html saved (interactive)")
    except Exception as e:
        print(f"[WARN] Force plot HTML failed: {e}")

    # Save narratives
    with open(os.path.join(OUT_DIR, "individual_narratives.txt"), "w", encoding="utf-8") as f:
        for label, narr in narratives.items():
            f.write(f"=== {label.upper()} ===\n{narr}\n")

    return cases, narratives


# ============================================================
# Phase 6 — Final Report
# ============================================================
def phase6_report(results, best_name, mean_tau, feat_imp, narratives):
    banner("PHASE 6 — Final Summary")

    summary = {
        "best_model": best_name,
        "test_metrics": results[best_name]["test"],
        "cv_r2_mean": results[best_name]["cv_r2_mean"],
        "cv_r2_std": results[best_name]["cv_r2_std"],
        "shap_stability_kendall_tau": mean_tau,
        "top10_features": feat_imp.head(10).round(4).to_dict(),
        "individual_cases": list(narratives.keys()),
    }

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print final report
    print(json.dumps(summary, indent=2))

    # List output files
    print("\n=== Output Files in outputs/ ===")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  - {f}")


# ============================================================
# Main
# ============================================================
def main():
    t_start = time.time()
    df = phase1_eda()
    X_train, X_val, X_test, y_train, y_val, y_test = phase2_preprocess(df)
    cat_model, rf_model, X_test_rf, results, best_name = phase3_train(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    shap_values, expected_value, X_shap, y_shap, top5, mean_tau, feat_imp = phase4_global_shap(
        cat_model, X_train, X_test, y_train, y_test
    )
    cases, narratives = phase5_individual(
        cat_model, shap_values, expected_value, X_shap, y_shap
    )
    phase6_report(results, best_name, mean_tau, feat_imp, narratives)

    print(f"\n[DONE] Total runtime: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
