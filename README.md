# Explainable AI for Stress Factors

**Deciphering Sleep and Lifestyle Impacts on Stress Score**

Pipeline Explainable AI (XAI) untuk memprediksi dan menjelaskan faktor-faktor yang berkontribusi terhadap `stress_score` berdasarkan data tidur, gaya hidup, dan demografi.

---

## Status

**100% selesai** — pipeline lengkap dengan modular train/test + dual SHAP analysis.

| Metric | Hasil |
| --- | --- |
| Best Model | **CatBoost Regressor** |
| Test R² | **0.6456** |
| Test RMSE | 0.964 |
| Test MAE | 0.770 |
| 5-Fold CV R² | 0.641 ± 0.006 |
| SHAP Within-CatBoost Stability (Kendall's Tau) | **0.896** ✅ |
| SHAP Cross-Model Stability (CatBoost vs RF) | 0.544 ⚠️ (top-10 overlap 7/10) |

---

## Dataset

| Atribut | Detail |
| --- | --- |
| File | `sleep_health_dataset.csv` |
| Sampel | 100,000 baris |
| Fitur | 31 kolom (27 input + 4 target/drop) |
| Target | `stress_score` (kontinu, 1.0–10.0) |
| Task | Supervised Regression |

**Fitur input (27):**
- 20 numerik: `age`, `bmi`, `sleep_duration_hrs`, `sleep_quality_score`, `rem_percentage`, `deep_sleep_percentage`, `sleep_latency_mins`, `wake_episodes_per_night`, `caffeine_mg_before_bed`, `alcohol_units_before_bed`, `screen_time_before_bed_mins`, `steps_that_day`, `nap_duration_mins`, `work_hours_that_day`, `heart_rate_resting_bpm`, `room_temperature_celsius`, `weekend_sleep_diff_hrs`, `exercise_day`, `sleep_aid_used`, `shift_work`
- 7 kategorikal: `gender`, `occupation`, `country`, `chronotype`, `mental_health_condition`, `season`, `day_type`

**Kolom di-drop (4):** `person_id`, `cognitive_performance_score`, `sleep_disorder_risk`, `felt_rested` (co-outcome → mencegah data leakage)

---

## Metodologi

```text
Phase 1:  EDA                       → distribusi target, korelasi, missing/outlier
Phase 2:  Preprocessing             → drop co-outcomes, split 70/15/15 (stratified)
Phase 3:  Model Training (Modular)  → train_catboost, train_rf, test_catboost, test_rf
Phase 4a: Global SHAP (CatBoost)    → Summary, Bar, Dependence, Within-Stability 5-fold
Phase 4b: Global SHAP (RF)          → Summary, Bar (BARU)
Phase 4c: Cross-Model Stability     → Kendall's Tau CatBoost vs RF (BARU, Skenario 4)
Phase 5:  Individual SHAP           → 3 kasus (High/Low/Borderline) + Waterfall + Force Plot
Phase 6:  Final Report              → automated tests + summary JSON + insights
```

### Modular Functions di Phase 3

| Fungsi | Tugas |
| --- | --- |
| `encode_for_rf(X_train, X_val, X_test)` | OrdinalEncoder kategorikal untuk RF |
| `train_catboost(X_train, y_train, X_val, y_val)` | Train CatBoost + early stopping |
| `train_random_forest(X_train_rf, y_train)` | Train Random Forest |
| `evaluate_model(y_true, y_pred)` | Hitung R² / RMSE / MAE |
| `test_catboost(model, X_test, y_test, X_val, y_val)` | Test CatBoost (Val + Test) |
| `test_random_forest(model, X_test_rf, y_test, X_val_rf, y_val)` | Test RF (Val + Test) |
| `cross_validate_models(...)` | 5-Fold CV R² untuk kedua model |

---

## Setup & Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Pipeline

```bash
python xai_stress_analysis.py
```

Pipeline akan menjalankan seluruh Phase 1-6 secara berurutan dan menghasilkan output di folder `outputs/`. Estimasi waktu: ~5-10 menit (CatBoost + Random Forest + SHAP).

### 3. Lihat Hasil

- **Visualisasi PNG:** `outputs/*.png`
- **Force Plot interaktif:** `outputs/force_plot.html` (buka di browser)
- **Summary JSON:** `outputs/summary.json`
- **Narasi per individu:** `outputs/individual_narratives.txt`

---

## Struktur Folder

```text
kkk1v4/
├── README.md                       (file ini)
├── analisis_dataset.md             (data dictionary)
├── implementation_plan.md          (rencana detail + progress tracker)
├── requirements.txt                (Python dependencies)
├── xai_stress_analysis.py          (script utama, modular Phase 1-6)
├── xai_stress_analysis.ipynb       (notebook self-contained, 45 cells)
├── sleep_health_dataset.csv        (dataset 100K rows)
└── outputs/
    ├── eda_target_distribution.png        (Phase 1)
    ├── eda_correlation_heatmap.png        (Phase 1)
    ├── model_comparison.png               (Phase 3)
    ├── model_metrics.json                 (Phase 3)
    ├── shap_summary_beeswarm.png          (Phase 4a — CatBoost)
    ├── shap_bar_importance.png            (Phase 4a — CatBoost)
    ├── shap_dependence_top5.png           (Phase 4a — CatBoost)
    ├── shap_stability_heatmap.png         (Phase 4a — within-CatBoost stability)
    ├── shap_summary_beeswarm_rf.png       (Phase 4b — RF, BARU)
    ├── shap_bar_importance_rf.png         (Phase 4b — RF, BARU)
    ├── shap_cross_model_comparison.png    (Phase 4c — cross-model bar, BARU)
    ├── shap_cross_model_rank_scatter.png  (Phase 4c — cross-model scatter, BARU)
    ├── shap_waterfall_high_stress.png     (Phase 5)
    ├── shap_waterfall_low_stress.png      (Phase 5)
    ├── shap_waterfall_borderline.png      (Phase 5)
    ├── force_plot.html                    (Phase 5 — interaktif)
    ├── individual_narratives.txt          (Phase 5)
    ├── summary.json                       (Phase 6)
    └── _*.pkl                             (cache intermediate, gitignored)
```

---

## Top-10 Feature Importance (mean |SHAP|) — CatBoost vs Random Forest

| Rank | CatBoost | Mean \|SHAP\| | Random Forest | Mean \|SHAP\| |
| --- | --- | --- | --- | --- |
| 1 | `sleep_quality_score` | 0.6542 | `sleep_quality_score` | 0.6878 |
| 2 | `occupation` | 0.5606 | `occupation` | 0.4393 |
| 3 | `sleep_duration_hrs` | 0.1295 | **`work_hours_that_day`** | 0.1963 |
| 4 | `room_temperature_celsius` | 0.0828 | `sleep_duration_hrs` | 0.0710 |
| 5 | `day_type` | 0.0758 | `room_temperature_celsius` | 0.0327 |
| 6 | `wake_episodes_per_night` | 0.0755 | `deep_sleep_percentage` | 0.0317 |
| 7 | `deep_sleep_percentage` | 0.0548 | `wake_episodes_per_night` | 0.0301 |
| 8 | `alcohol_units_before_bed` | 0.0483 | `heart_rate_resting_bpm` | 0.0281 |
| 9 | `chronotype` | 0.0464 | `bmi` | 0.0220 |
| 10 | `heart_rate_resting_bpm` | 0.0445 | `mental_health_condition` | 0.0206 |

**Top-10 overlap:** **7/10** fitur sama. Top-2 identik (`sleep_quality_score`, `occupation`).

**Perbedaan kunci:** RF memberi `work_hours_that_day` rank 3 (CatBoost di luar top-10) — kemungkinan CatBoost melihat `occupation` sudah menyerap pola jam kerja per profesi.

---

## Insight Utama

1. **Sleep quality is king** — `sleep_quality_score` adalah prediktor #1 di **kedua model** (CatBoost & RF setuju). Intervensi sebaiknya fokus ke **kualitas** tidur, bukan hanya kuantitas.

2. **Occupation matters more than work_hours (di CatBoost)** — Pola pekerjaan (Lawyer +0.99 vs Retired -2.24) memberi efek SHAP ±2.2 di CatBoost. RF justru memberi `work_hours_that_day` peringkat 3 — perbedaan ini menunjukkan CatBoost menangkap interaksi non-linear `occupation` ↔ `work_hours` lebih baik karena native categorical handling.

3. **Lingkungan kamar tidur underrated** — `room_temperature_celsius` masuk top-5 di **kedua model**, sering diabaikan dibanding screen time/kafein. Optimal 16-19°C.

4. **Findings algorithm-agnostic untuk top-tier** — Top-2 fitur (`sleep_quality_score`, `occupation`) identik di CatBoost & RF. Top-10 overlap 7/10. Tier menengah berbeda antar model — perlu kehati-hatian saat klaim ranking di luar top-2.

---

## Verification Tests

| # | Test | Hasil | Status |
| --- | --- | --- | --- |
| 1 | Data leakage guard | 4/4 kolom co-outcome absent | ✅ PASS |
| 2a | SHAP additivity (CatBoost) | max diff = 0.000000 | ✅ PASS |
| 2b | SHAP additivity (RF) | max diff = 0.000000 | ✅ PASS |
| 3 | R² ≥ 0.6 threshold | 0.6456 | ✅ PASS |
| 4 | Cross-Model Stability (Skenario 4) | Kendall's Tau = 0.544, top-10 overlap 7/10 | ⚠️ MARGINAL |
| 5 | 13 output visuals exist | 13/13 | ✅ PASS |

---

## Tech Stack

- **Python 3.12**
- **pandas, numpy, scipy** — data manipulation
- **scikit-learn** — Random Forest, train/test split, metrics
- **CatBoost** — Gradient Boosting Regressor (best model)
- **SHAP 0.51** — TreeExplainer untuk feature attribution
- **matplotlib, seaborn** — visualisasi

---

## Referensi Dokumentasi

- [analisis_dataset.md](analisis_dataset.md) — Detail kolom dataset
- [implementation_plan.md](implementation_plan.md) — Rencana implementasi + progress tracker per phase
- [outputs/summary.json](outputs/summary.json) — Hasil final terstruktur
