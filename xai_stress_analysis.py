"""
Explainable AI for Stress Factors
Deciphering Sleep and Lifestyle Impacts on Stress Score

Pipeline lengkap Phase 1-6:
  Phase 1: Setup & EDA
  Phase 2: Feature Engineering & Preprocessing
  Phase 3: Model Training (CatBoost + Random Forest) — modular train/test
  Phase 4a: Global SHAP (CatBoost)
  Phase 4b: Global SHAP (Random Forest)
  Phase 4c: Cross-Model Stability (CatBoost vs RF)
  Phase 5: Individual SHAP
  Phase 6: Final Report

Modular structure:
  encode_for_rf, train_catboost, train_random_forest,
  evaluate_model, test_catboost, test_random_forest
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
from sklearn.preprocessing import OrdinalEncoder
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
    )

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # Cast kategorikal ke str (CatBoost butuh string)
    for c in CATEGORICAL_FEATURES:
        X_train[c] = X_train[c].astype(str)
        X_val[c] = X_val[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    # Sanity check leakage
    for col in ["sleep_disorder_risk", "cognitive_performance_score", "felt_rested"]:
        assert col not in X_train.columns
    print("[OK] Data leakage guard verified — co-outcome columns dropped.")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# Phase 3 — MODULAR Training & Testing Functions
# ============================================================

def encode_for_rf(X_train, X_val, X_test):
    """Encode kategorikal ke ordinal untuk Random Forest.
    Returns: encoder, X_train_rf, X_val_rf, X_test_rf
    """
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_rf = X_train.copy()
    X_val_rf = X_val.copy()
    X_test_rf = X_test.copy()
    X_train_rf[CATEGORICAL_FEATURES] = encoder.fit_transform(X_train_rf[CATEGORICAL_FEATURES])
    X_val_rf[CATEGORICAL_FEATURES] = encoder.transform(X_val_rf[CATEGORICAL_FEATURES])
    X_test_rf[CATEGORICAL_FEATURES] = encoder.transform(X_test_rf[CATEGORICAL_FEATURES])
    return encoder, X_train_rf, X_val_rf, X_test_rf


def train_catboost(X_train, y_train, X_val, y_val,
                   iterations=1000, depth=6, learning_rate=0.05,
                   early_stopping_rounds=50, verbose=100):
    """Train CatBoost Regressor dengan early stopping pada validation set.
    Returns: trained CatBoostRegressor
    """
    print("\n[Training CatBoost...]")
    cat_idx = [X_train.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    t0 = time.time()
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        eval_metric="RMSE",
        early_stopping_rounds=early_stopping_rounds,
        random_seed=RANDOM_SEED,
        verbose=verbose,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print(f"CatBoost trained in {time.time()-t0:.1f}s")
    return model


def train_random_forest(X_train_rf, y_train,
                        n_estimators=300, min_samples_leaf=5, max_depth=None):
    """Train Random Forest Regressor pada data yang sudah di-encode.
    Returns: trained RandomForestRegressor
    """
    print("\n[Training Random Forest...]")
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train_rf, y_train)
    print(f"Random Forest trained in {time.time()-t0:.1f}s")
    return model


def evaluate_model(y_true, y_pred):
    """Hitung R², RMSE, MAE.
    Returns: dict dengan 3 metric
    """
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def test_catboost(cat_model, X_test, y_test, X_val=None, y_val=None):
    """Test CatBoost — hitung metric pada test (dan val kalau di-pass).
    Returns: dict {"val": {...}, "test": {...}}
    """
    print("\n[Testing CatBoost...]")
    out = {}
    if X_val is not None and y_val is not None:
        out["val"] = evaluate_model(y_val, cat_model.predict(X_val))
    out["test"] = evaluate_model(y_test, cat_model.predict(X_test))
    print(f"  Test R²={out['test']['R2']:.4f}  RMSE={out['test']['RMSE']:.4f}  MAE={out['test']['MAE']:.4f}")
    return out


def test_random_forest(rf_model, X_test_rf, y_test, X_val_rf=None, y_val=None):
    """Test Random Forest — hitung metric pada test (dan val kalau di-pass).
    Returns: dict {"val": {...}, "test": {...}}
    """
    print("\n[Testing Random Forest...]")
    out = {}
    if X_val_rf is not None and y_val is not None:
        out["val"] = evaluate_model(y_val, rf_model.predict(X_val_rf))
    out["test"] = evaluate_model(y_test, rf_model.predict(X_test_rf))
    print(f"  Test R²={out['test']['R2']:.4f}  RMSE={out['test']['RMSE']:.4f}  MAE={out['test']['MAE']:.4f}")
    return out


def cross_validate_models(X_train, y_train, X_train_rf, n_splits=5, sample_size=30000):
    """5-Fold CV R² untuk kedua model (subsampled untuk speed)."""
    print(f"\n[{n_splits}-Fold CV R² (subsampled {sample_size} rows)...]")
    rng = np.random.RandomState(RANDOM_SEED)
    sub_idx = rng.choice(len(X_train), size=min(sample_size, len(X_train)), replace=False)
    X_sub = X_train.iloc[sub_idx].reset_index(drop=True)
    y_sub = y_train.iloc[sub_idx].reset_index(drop=True)
    X_sub_rf = X_train_rf.iloc[sub_idx].reset_index(drop=True)
    cat_idx = [X_sub.columns.get_loc(c) for c in CATEGORICAL_FEATURES]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    cat_cv, rf_cv = [], []
    for fold, (tr, te) in enumerate(kf.split(X_sub), 1):
        cm = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                               random_seed=RANDOM_SEED, verbose=0)
        cm.fit(Pool(X_sub.iloc[tr], y_sub.iloc[tr], cat_features=cat_idx))
        cat_cv.append(r2_score(y_sub.iloc[te], cm.predict(X_sub.iloc[te])))
        rm = RandomForestRegressor(n_estimators=150, min_samples_leaf=5,
                                   n_jobs=-1, random_state=RANDOM_SEED)
        rm.fit(X_sub_rf.iloc[tr], y_sub.iloc[tr])
        rf_cv.append(r2_score(y_sub.iloc[te], rm.predict(X_sub_rf.iloc[te])))
        print(f"  Fold {fold}: CatBoost R²={cat_cv[-1]:.4f}  RF R²={rf_cv[-1]:.4f}")

    return {
        "CatBoost": {"cv_r2_mean": float(np.mean(cat_cv)), "cv_r2_std": float(np.std(cat_cv))},
        "RandomForest": {"cv_r2_mean": float(np.mean(rf_cv)), "cv_r2_std": float(np.std(rf_cv))},
    }


def plot_model_comparison(results):
    """Bar chart 3-panel R²/RMSE/MAE CatBoost vs RF."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, m in enumerate(["R2", "RMSE", "MAE"]):
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


def phase3_train(X_train, X_val, X_test, y_train, y_val, y_test):
    """Orchestrator Phase 3 — memanggil fungsi modular train_* dan test_*."""
    banner("PHASE 3 — Train & Test (Modular)")

    # 1. Train CatBoost
    cat_model = train_catboost(X_train, y_train, X_val, y_val)

    # 2. Encode + Train Random Forest
    encoder, X_train_rf, X_val_rf, X_test_rf = encode_for_rf(X_train, X_val, X_test)
    rf_model = train_random_forest(X_train_rf, y_train)

    # 3. Test kedua model
    cat_metrics = test_catboost(cat_model, X_test, y_test, X_val, y_val)
    rf_metrics = test_random_forest(rf_model, X_test_rf, y_test, X_val_rf, y_val)
    results = {"CatBoost": cat_metrics, "RandomForest": rf_metrics}

    # 4. 5-Fold CV
    cv_results = cross_validate_models(X_train, y_train, X_train_rf)
    for name in ["CatBoost", "RandomForest"]:
        results[name].update(cv_results[name])

    # 5. Best model selection
    best_name = "CatBoost" if results["CatBoost"]["val"]["R2"] >= results["RandomForest"]["val"]["R2"] else "RandomForest"
    print(f"\n[BEST MODEL] {best_name}")
    print("\n=== Model Performance Summary ===")
    print(json.dumps(results, indent=2))

    # 6. Plot + save
    plot_model_comparison(results)
    with open(os.path.join(OUT_DIR, "model_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return cat_model, rf_model, X_test_rf, results, best_name


# ============================================================
# Phase 4a — Global SHAP (CatBoost)
# ============================================================
def phase4_global_shap(cat_model, X_train, X_test, y_train, y_test, n_sample=5000):
    banner("PHASE 4a — Global SHAP Analysis (CatBoost)")

    cat_idx = [X_test.columns.get_loc(c) for c in CATEGORICAL_FEATURES]

    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = rng.choice(len(X_test), size=min(n_sample, len(X_test)), replace=False)
    X_shap = X_test.iloc[sample_idx].reset_index(drop=True)
    y_shap = y_test.iloc[sample_idx].reset_index(drop=True)

    print(f"Computing SHAP on {len(X_shap)} test samples...")
    explainer = shap.TreeExplainer(cat_model)
    shap_pool = Pool(X_shap, cat_features=cat_idx)
    shap_values = explainer.shap_values(shap_pool)
    expected_value = float(explainer.expected_value)
    print(f"SHAP shape: {shap_values.shape}  expected_value: {expected_value:.4f}")

    # Sanity check
    preds = cat_model.predict(X_shap)
    diff = np.abs(shap_values.sum(axis=1) + expected_value - preds).max()
    print(f"[Sanity] max |SHAP_sum + base - pred| = {diff:.6f}  (target < 0.01)")
    assert diff < 0.01

    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary_beeswarm.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # Bar Plot
    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_bar_importance.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # Dependence top-5
    feat_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=X_shap.columns).sort_values(ascending=False)
    top5 = feat_imp.head(5).index.tolist()
    print(f"Top-5 fitur (mean |SHAP|): {top5}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, feat in enumerate(top5):
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

    # Stability 5-fold CV SHAP
    print("\n[5-Fold CV SHAP Stability — subsampled 20K rows]")
    rng = np.random.RandomState(RANDOM_SEED)
    sub_idx = rng.choice(len(X_train), size=min(20000, len(X_train)), replace=False)
    X_sub = X_train.iloc[sub_idx].reset_index(drop=True)
    y_sub = y_train.iloc[sub_idx].reset_index(drop=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_rankings = []
    for fold, (tr, te) in enumerate(kf.split(X_sub), 1):
        cm = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                               random_seed=RANDOM_SEED, verbose=0)
        cm.fit(Pool(X_sub.iloc[tr], y_sub.iloc[tr], cat_features=cat_idx))
        sv = shap.TreeExplainer(cm).shap_values(Pool(X_sub.iloc[te], cat_features=cat_idx))
        imp = np.abs(sv).mean(axis=0)
        ranking = pd.Series(imp, index=X_sub.columns).rank(ascending=False).astype(int)
        fold_rankings.append(ranking)
        print(f"  Fold {fold} top-3: {ranking.sort_values().head(3).index.tolist()}")

    taus = []
    for i in range(5):
        for j in range(i + 1, 5):
            tau, _ = kendalltau(fold_rankings[i], fold_rankings[j])
            taus.append(tau)
    mean_tau = float(np.mean(taus))
    print(f"\nMean pairwise Kendall's Tau (5 folds): {mean_tau:.4f}")

    # Heatmap
    rank_df = pd.concat(fold_rankings, axis=1)
    rank_df.columns = [f"Fold{i+1}" for i in range(5)]
    rank_df["mean_rank"] = rank_df.mean(axis=1)
    rank_df = rank_df.sort_values("mean_rank")
    plt.figure(figsize=(8, max(8, len(rank_df) * 0.3)))
    sns.heatmap(rank_df[[f"Fold{i+1}" for i in range(5)]],
                annot=True, fmt=".0f", cmap="YlGnBu_r",
                cbar_kws={"label": "Rank (1 = paling penting)"})
    plt.title(f"Feature Importance Stability — 5-Fold CV (CatBoost)\nMean Kendall's Tau = {mean_tau:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_stability_heatmap.png"), dpi=120)
    plt.close()

    return shap_values, expected_value, X_shap, y_shap, top5, mean_tau, feat_imp


# ============================================================
# Phase 4b — Global SHAP (Random Forest)  [BARU]
# ============================================================
def phase4b_global_shap_rf(rf_model, X_test_rf, y_test, n_sample=500):
    """SHAP analysis untuk Random Forest model.
    RF SHAP jauh lebih lambat dari CatBoost → default sample 500 saja.
    """
    banner("PHASE 4b — Global SHAP Analysis (Random Forest)")

    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = rng.choice(len(X_test_rf), size=min(n_sample, len(X_test_rf)), replace=False)
    X_shap_rf = X_test_rf.iloc[sample_idx].reset_index(drop=True)

    print(f"Computing SHAP on {len(X_shap_rf)} test samples (RF)... (lambat, ~5-15 menit)", flush=True)
    t0 = time.time()
    explainer_rf = shap.TreeExplainer(rf_model)
    shap_values_rf = explainer_rf.shap_values(X_shap_rf)
    # Robust expected_value handling (bisa scalar atau array depending on shap version)
    ev = explainer_rf.expected_value
    expected_value_rf = float(ev) if np.isscalar(ev) else float(np.asarray(ev).item() if np.asarray(ev).size == 1 else np.asarray(ev)[0])
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    print(f"  SHAP shape: {shap_values_rf.shape}  expected_value: {expected_value_rf:.4f}", flush=True)

    # Sanity check
    preds = rf_model.predict(X_shap_rf)
    diff = np.abs(shap_values_rf.sum(axis=1) + expected_value_rf - preds).max()
    print(f"[Sanity RF] max |SHAP_sum + base - pred| = {diff:.6f}")
    assert diff < 0.01, f"RF SHAP sanity check failed: {diff}"

    # Summary Plot RF
    plt.figure()
    shap.summary_plot(shap_values_rf, X_shap_rf, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary_beeswarm_rf.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    # Bar Plot RF
    plt.figure()
    shap.summary_plot(shap_values_rf, X_shap_rf, plot_type="bar",
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_bar_importance_rf.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    # Feature importance RF
    feat_imp_rf = pd.Series(
        np.abs(shap_values_rf).mean(axis=0), index=X_shap_rf.columns
    ).sort_values(ascending=False)
    print(f"\nTop-10 RF SHAP importance:")
    print(feat_imp_rf.head(10).round(4).to_string())

    return shap_values_rf, expected_value_rf, X_shap_rf, feat_imp_rf


# ============================================================
# Phase 4c — Cross-Model Stability (CatBoost vs RF)  [BARU]
# Mengisi Skenario 4 di plan
# ============================================================
def phase4c_cross_model_stability(feat_imp_cat, feat_imp_rf):
    """Bandingkan ranking SHAP importance CatBoost vs Random Forest.
    Skenario 4: Kendall's Tau & Spearman ≥ 0.80
    """
    banner("PHASE 4c — Cross-Model Stability (CatBoost vs RF)")

    # Align features (kedua Series harus punya index yang sama)
    common_features = sorted(set(feat_imp_cat.index) & set(feat_imp_rf.index))
    cat_imp = feat_imp_cat.reindex(common_features)
    rf_imp = feat_imp_rf.reindex(common_features)

    # Compute rankings
    cat_rank = cat_imp.rank(ascending=False).astype(int)
    rf_rank = rf_imp.rank(ascending=False).astype(int)

    # Kendall's Tau (full set)
    tau_full, p_full = kendalltau(cat_rank, rf_rank)
    print(f"Kendall's Tau (semua {len(common_features)} fitur):  {tau_full:.4f}  (p={p_full:.4g})")

    # Top-10 overlap
    top10_cat = set(feat_imp_cat.head(10).index)
    top10_rf = set(feat_imp_rf.head(10).index)
    overlap = top10_cat & top10_rf
    print(f"Top-10 overlap: {len(overlap)}/10 fitur")
    print(f"  CatBoost top-10: {list(feat_imp_cat.head(10).index)}")
    print(f"  RF top-10:       {list(feat_imp_rf.head(10).index)}")
    print(f"  Common:          {sorted(overlap)}")

    # Side-by-side bar chart top-10 dari kedua model
    top10_combined = list(top10_cat | top10_rf)
    df_compare = pd.DataFrame({
        "CatBoost": feat_imp_cat.reindex(top10_combined).fillna(0),
        "RandomForest": feat_imp_rf.reindex(top10_combined).fillna(0),
    }).sort_values("CatBoost", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_compare) * 0.4)))
    df_compare.plot(kind="barh", ax=ax, color=["#3a76d8", "#d8763a"], width=0.8)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Cross-Model Feature Importance Comparison\nKendall's Tau = {tau_full:.3f}  (Skenario 4 target >= 0.80)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_cross_model_comparison.png"), dpi=120)
    plt.close()

    # Scatter rank correlation
    plt.figure(figsize=(8, 8))
    plt.scatter(cat_rank, rf_rank, s=100, alpha=0.6, c="#3a76d8", edgecolor="black")
    for feat in common_features:
        if cat_rank[feat] <= 10 or rf_rank[feat] <= 10:
            plt.annotate(feat, (cat_rank[feat], rf_rank[feat]),
                         fontsize=8, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    lim = max(cat_rank.max(), rf_rank.max()) + 1
    plt.plot([0, lim], [0, lim], "r--", alpha=0.5, label="perfect agreement")
    plt.xlabel("CatBoost Rank (1 = paling penting)")
    plt.ylabel("RandomForest Rank")
    plt.title(f"Rank Correlation CatBoost vs RF\nKendall's Tau = {tau_full:.3f}")
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_cross_model_rank_scatter.png"), dpi=120)
    plt.close()

    status = "PASS" if tau_full >= 0.80 else "MARGINAL"
    print(f"\n[Skenario 4] Kendall's Tau = {tau_full:.4f}  {status}  (target >= 0.80)")

    return {
        "kendall_tau": float(tau_full),
        "p_value": float(p_full),
        "top10_overlap": len(overlap),
        "common_top10": sorted(overlap),
        "status": status,
    }


# ============================================================
# Phase 5 — Individual SHAP Analysis
# ============================================================
def phase5_individual(cat_model, shap_values, expected_value, X_shap, y_shap):
    banner("PHASE 5 — Individual SHAP Analysis")

    preds = cat_model.predict(X_shap)
    high_idx = int(np.argmax(preds))
    low_idx = int(np.argmin(preds))
    border_idx = int(np.argmin(np.abs(preds - 5.0)))

    cases = {"high_stress": high_idx, "low_stress": low_idx, "borderline": border_idx}
    feature_names = X_shap.columns.tolist()
    narratives = {}

    for label, idx in cases.items():
        pred = float(preds[idx])
        actual = float(y_shap.iloc[idx])
        sv = shap_values[idx]
        print(f"\n[{label.upper()}] idx={idx}  predicted={pred:.2f}  actual={actual:.2f}")

        explanation = shap.Explanation(
            values=sv, base_values=expected_value,
            data=X_shap.iloc[idx].values, feature_names=feature_names,
        )
        plt.figure()
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_waterfall_{label}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close()

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

    try:
        force = shap.force_plot(
            expected_value, shap_values[high_idx],
            X_shap.iloc[high_idx], feature_names=feature_names,
        )
        shap.save_html(os.path.join(OUT_DIR, "force_plot.html"), force)
        print("[OK] force_plot.html saved")
    except Exception as e:
        print(f"[WARN] Force plot HTML failed: {e}")

    with open(os.path.join(OUT_DIR, "individual_narratives.txt"), "w", encoding="utf-8") as f:
        for label, narr in narratives.items():
            f.write(f"=== {label.upper()} ===\n{narr}\n")

    return cases, narratives


# ============================================================
# Phase 6 — Final Report
# ============================================================
def phase6_report(results, best_name, mean_tau, feat_imp, narratives,
                  cross_model=None, feat_imp_rf=None):
    banner("PHASE 6 — Final Summary")

    summary = {
        "best_model": best_name,
        "test_metrics": results[best_name]["test"],
        "cv_r2_mean": results[best_name]["cv_r2_mean"],
        "cv_r2_std": results[best_name]["cv_r2_std"],
        "shap_stability_kendall_tau_within_catboost": mean_tau,
        "top10_features_catboost": feat_imp.head(10).round(4).to_dict(),
        "individual_cases": list(narratives.keys()),
    }
    if feat_imp_rf is not None:
        summary["top10_features_rf"] = feat_imp_rf.head(10).round(4).to_dict()
    if cross_model is not None:
        summary["cross_model_stability"] = cross_model

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2, default=str))

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
    shap_values_rf, expected_value_rf, X_shap_rf, feat_imp_rf = phase4b_global_shap_rf(
        rf_model, X_test_rf, y_test
    )
    cross_model = phase4c_cross_model_stability(feat_imp, feat_imp_rf)
    cases, narratives = phase5_individual(
        cat_model, shap_values, expected_value, X_shap, y_shap
    )
    phase6_report(results, best_name, mean_tau, feat_imp, narratives,
                  cross_model=cross_model, feat_imp_rf=feat_imp_rf)

    print(f"\n[DONE] Total runtime: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
