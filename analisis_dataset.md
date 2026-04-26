# 📊 Analisis Dataset: Sleep Health & Lifestyle

## Gambaran Umum

| Atribut | Detail |
|---|---|
| **Nama File** | `sleep_health_dataset.csv` |
| **Jumlah Baris** | 100,001 (termasuk header) → **100,000 sampel** |
| **Jumlah Kolom** | **31 fitur** |
| **Tipe Data** | Campuran: numerik kontinu, numerik diskret, kategorikal |

---

## 📋 Deskripsi Kolom

### 🆔 Identifikasi
| Kolom | Tipe | Deskripsi |
|---|---|---|
| `person_id` | Integer | ID unik tiap individu |

---

### 👤 Demografi & Gaya Hidup
| Kolom | Tipe | Rentang/Nilai | Deskripsi |
|---|---|---|---|
| `age` | Integer | 18 – 69 | Usia responden (tahun) |
| `gender` | Kategori | Male, Female, Other | Jenis kelamin |
| `occupation` | Kategori | Driver, Software Engineer, Nurse, Doctor, Student, Lawyer, Manager, Teacher, Retired, Homemaker, Freelancer, Sales | Pekerjaan |
| `bmi` | Float | 16.0 – 41.8 | Body Mass Index |
| `country` | Kategori | Japan, USA, India, Spain, Brazil, Netherlands, UK, Germany, France, Australia, Canada, South Korea, Italy, Sweden, Mexico | Negara asal |
| `chronotype` | Kategori | Morning, Neutral, Evening | Tipe circadian (lark/owl/netral) |
| `mental_health_condition` | Kategori | Healthy, Anxiety, Depression, Both | Kondisi kesehatan mental |

---

### 💤 Metrik Tidur Utama
| Kolom | Tipe | Rentang | Deskripsi |
|---|---|---|---|
| `sleep_duration_hrs` | Float | 3.0 – 10.5 | Durasi tidur malam (jam) |
| `sleep_quality_score` | Float | 1.0 – 8.9 | Skor kualitas tidur (skala 1–10) |
| `rem_percentage` | Float | 10.0 – 30.0 | Persentase tidur fase REM |
| `deep_sleep_percentage` | Float | 7.7 – 30.0 | Persentase tidur dalam (Deep Sleep) |
| `sleep_latency_mins` | Integer | 1 – 45 | Waktu yang dibutuhkan untuk tertidur (menit) |
| `wake_episodes_per_night` | Integer | 0 – 8 | Jumlah terbangun dalam semalam |

---

### ☕ Faktor Perilaku Sebelum Tidur
| Kolom | Tipe | Rentang | Deskripsi |
|---|---|---|---|
| `caffeine_mg_before_bed` | Integer | 0, 40, 80, 100, 150, 200, 300, 400 | Konsumsi kafein sebelum tidur (mg) |
| `alcohol_units_before_bed` | Float | 0.0 – 6.0 | Konsumsi alkohol sebelum tidur (unit) |
| `screen_time_before_bed_mins` | Integer | 5 – 180 | Waktu layar sebelum tidur (menit) |
| `nap_duration_mins` | Integer | 0 – 89 | Durasi tidur siang (menit, 0 = tidak tidur siang) |
| `sleep_aid_used` | Binary | 0, 1 | Penggunaan alat bantu tidur (1=ya) |

---

### 🏃 Aktivitas Fisik & Kesehatan
| Kolom | Tipe | Rentang | Deskripsi |
|---|---|---|---|
| `exercise_day` | Binary | 0, 1 | Olahraga pada hari tersebut (1=ya) |
| `steps_that_day` | Integer | 500 – 17,951 | Jumlah langkah kaki hari itu |
| `heart_rate_resting_bpm` | Integer | 45 – 87 | Detak jantung istirahat (bpm) |

---

### 💼 Pekerjaan & Stres
| Kolom | Tipe | Rentang | Deskripsi |
|---|---|---|---|
| `stress_score` | Float | 1.0 – 10.0 | Skor stres harian |
| `work_hours_that_day` | Float | 0.0 – 15.7 | Jam kerja hari itu |
| `shift_work` | Binary | 0, 1 | Apakah bekerja shift (1=ya) |

---

### 🌡️ Lingkungan & Konteks
| Kolom | Tipe | Nilai | Deskripsi |
|---|---|---|---|
| `room_temperature_celsius` | Float | 15.0 – 28.0 | Suhu kamar tidur (°C) |
| `weekend_sleep_diff_hrs` | Float | -1.0 – 3.0 | Perbedaan durasi tidur weekend vs weekday |
| `season` | Kategori | Spring, Summer, Autumn, Winter | Musim |
| `day_type` | Kategori | Weekday, Weekend | Jenis hari |

---

### 🎯 Variabel Target / Output
| Kolom | Tipe | Nilai | Deskripsi |
|---|---|---|---|
| `cognitive_performance_score` | Float | 0.0 – 100.0 | Skor performa kognitif |
| `sleep_disorder_risk` | Kategori | **Healthy**, Mild, Moderate, Severe | **Target utama** – risiko gangguan tidur |
| `felt_rested` | Binary | 0, 1 | Apakah merasa segar setelah bangun (1=ya) |

---

## 🏷️ Distribusi Nilai Kategorikal (Estimasi)

### Gender
- Male, Female, Other

### Occupation (15 jenis)
Driver, Software Engineer, Nurse, Doctor, Student, Lawyer, Manager, Teacher, Retired, Homemaker, Freelancer, Sales

### Country (15 negara)
Japan, USA, India, Spain, Brazil, Netherlands, UK, Germany, France, Australia, Canada, South Korea, Italy, Sweden, Mexico

### Mental Health Condition
- Healthy, Anxiety, Depression, Both

### Sleep Disorder Risk (Target Klasifikasi)
- **Healthy** → tidur sehat
- **Mild** → gangguan ringan
- **Moderate** → gangguan sedang
- **Severe** → gangguan berat

---

## 🧠 Saran Penggunaan (Machine Learning)

### 1. Klasifikasi Risiko Gangguan Tidur
- **Target:** `sleep_disorder_risk` (4 kelas)
- **Task:** Multi-class classification
- **Algoritma saran:** Random Forest, XGBoost, LightGBM, Neural Network

### 2. Prediksi Performa Kognitif
- **Target:** `cognitive_performance_score` (kontinu)
- **Task:** Regression
- **Algoritma saran:** Gradient Boosting, Ridge/Lasso, SVR

### 3. Prediksi Felt Rested
- **Target:** `felt_rested` (biner)
- **Task:** Binary classification
- **Algoritma saran:** Logistic Regression, SVM, Random Forest

### 4. Clustering Pola Tidur
- **Task:** Unsupervised learning (K-Means, DBSCAN, Hierarchical)
- **Fitur:** sleep_duration_hrs, rem_percentage, deep_sleep_percentage, dll.

---

## ⚠️ Catatan Penting

> [!NOTE]
> Dataset ini memiliki **100,000 sampel** — cukup besar untuk model deep learning.

> [!TIP]
> Kolom `person_id` harus di-drop sebelum training model karena hanya ID.

> [!WARNING]
> Periksa distribusi kelas `sleep_disorder_risk` — kemungkinan imbalanced (lebih banyak "Healthy"). Gunakan teknik SMOTE atau class weighting jika perlu.

> [!IMPORTANT]
> Fitur kategorikal (`gender`, `occupation`, `country`, `chronotype`, `mental_health_condition`, `season`, `day_type`) perlu di-encode (One-Hot Encoding atau Label Encoding) sebelum digunakan.

---

## 📐 Fitur Numerik — Rentang Nilai

| Fitur | Min | Max |
|---|---|---|
| age | 18 | 69 |
| bmi | 16.0 | 41.8 |
| sleep_duration_hrs | 3.0 | 10.5 |
| sleep_quality_score | 1.0 | 8.9 |
| rem_percentage | 10.0 | 30.0 |
| deep_sleep_percentage | 7.7 | 30.0 |
| sleep_latency_mins | 1 | 45 |
| wake_episodes_per_night | 0 | 8 |
| caffeine_mg_before_bed | 0 | 400 |
| alcohol_units_before_bed | 0.0 | 6.0 |
| screen_time_before_bed_mins | 5 | 180 |
| steps_that_day | 500 | 17,951 |
| nap_duration_mins | 0 | 89 |
| stress_score | 1.0 | 10.0 |
| work_hours_that_day | 0.0 | 15.7 |
| heart_rate_resting_bpm | 45 | 87 |
| room_temperature_celsius | 15.0 | 28.0 |
| weekend_sleep_diff_hrs | -1.0 | 3.0 |
| cognitive_performance_score | 0.0 | 100.0 |
