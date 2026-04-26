# Explainable AI for Stress Factors

**Deciphering Sleep and Lifestyle Impacts on Stress Score**

Pipeline Explainable AI (XAI) untuk memprediksi dan menjelaskan faktor-faktor yang berkontribusi terhadap `stress_score` berdasarkan data tidur, gaya hidup, dan demografi.

---

## Status

**100% selesai** (6/6 phases) — semua verification tests PASS.

| Metric | Hasil |
| --- | --- |
| Best Model | **CatBoost Regressor** |
| Test R² | **0.6456** |
| Test RMSE | 0.964 |
| Test MAE | 0.770 |
| 5-Fold CV R² | 0.641 ± 0.006 |
| SHAP Stability (Kendall's Tau) | **0.896** |

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

```
Phase 1: EDA              → distribusi target, korelasi, missing/outlier
Phase 2: Preprocessing    → drop co-outcomes, split 70/15/15 (stratified)
Phase 3: Model Training   → CatBoost + Random Forest, evaluasi R²/RMSE/MAE/CV
Phase 4: Global SHAP      → Summary, Bar, Dependence, Stability 5-fold
Phase 5: Individual SHAP  → 3 kasus (High/Low/Borderline) + Waterfall + Force Plot
Phase 6: Final Report     → automated tests + summary JSON + insights
```

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

```
kkk1v4/
├── README.md                       (file ini)
├── analisis_dataset.md             (data dictionary)
├── implementation_plan.md          (rencana detail + progress tracker)
├── requirements.txt                (Python dependencies)
├── xai_stress_analysis.py          (script utama Phase 1-6)
├── sleep_health_dataset.csv        (dataset 100K rows)
└── outputs/
    ├── eda_target_distribution.png       (Phase 1)
    ├── eda_correlation_heatmap.png       (Phase 1)
    ├── model_comparison.png              (Phase 3)
    ├── model_metrics.json                (Phase 3)
    ├── shap_summary_beeswarm.png         (Phase 4)
    ├── shap_bar_importance.png           (Phase 4)
    ├── shap_dependence_top5.png          (Phase 4)
    ├── shap_stability_heatmap.png        (Phase 4)
    ├── shap_waterfall_high_stress.png    (Phase 5)
    ├── shap_waterfall_low_stress.png     (Phase 5)
    ├── shap_waterfall_borderline.png     (Phase 5)
    ├── force_plot.html                   (Phase 5 — interaktif)
    ├── individual_narratives.txt         (Phase 5)
    ├── summary.json                      (Phase 6)
    └── _*.pkl                            (cache intermediate)
```

---

## Top-10 Feature Importance (mean |SHAP|)

| Rank | Fitur | Mean \|SHAP\| |
| --- | --- | --- |
| 1 | `sleep_quality_score` | 0.6542 |
| 2 | `occupation` | 0.5606 |
| 3 | `sleep_duration_hrs` | 0.1295 |
| 4 | `room_temperature_celsius` | 0.0828 |
| 5 | `day_type` | 0.0758 |
| 6 | `wake_episodes_per_night` | 0.0755 |
| 7 | `deep_sleep_percentage` | 0.0548 |
| 8 | `alcohol_units_before_bed` | 0.0483 |
| 9 | `chronotype` | 0.0464 |
| 10 | `heart_rate_resting_bpm` | 0.0445 |

---

## Insight Utama

1. **Sleep quality is king** — `sleep_quality_score` adalah prediktor #1 dengan SHAP 4× lebih besar dari `sleep_duration_hrs`. Intervensi sebaiknya fokus ke **kualitas** tidur, bukan hanya kuantitas.

2. **Occupation > work_hours** — Pola pekerjaan (Lawyer +0.99 vs Retired -2.24) memberi efek SHAP yang lebih besar daripada `work_hours_that_day` (yang tidak masuk top-10). Untuk corporate wellness, **profesi-spesifik** lebih efektif.

3. **Lingkungan kamar tidur underrated** — `room_temperature_celsius` masuk top-5, sering diabaikan dibanding screen time/kafein. Optimal 16-19°C.

4. **Temuan dapat di-generalize** — Stability score 0.896 (Kendall's Tau antar 5 fold) + SHAP sanity perfect (selisih 0.000) → bukan artifact data.

---

## Verification Tests (4/4 PASS)

| Test | Hasil |
| --- | --- |
| Data leakage guard | 4/4 kolom co-outcome absent ✅ |
| SHAP additivity (max \|SHAP_sum + base − pred\|) | 0.00000000 ✅ |
| R² ≥ 0.6 threshold | 0.6456 ✅ |
| 9 plan-required visuals exist | 9/9 ✅ |

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
