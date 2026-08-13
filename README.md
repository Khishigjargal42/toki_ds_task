# Toki Data Scientist Task — DPD91 Risk Prediction

## 1. Төслийн тойм

Энэхүү төслийн зорилго нь **DPD31 (31 хоногийн хугацаа хэтэрсэн)** болсон зээлдэгч дараагийн шатанд **DPD91+** буюу 91 болон түүнээс дээш хоногийн хугацаа хэтрүүлэх эсэхийг машин сургалтын аргаар таамаглахад оршино.

Үндсэн зорилго нь зөвхөн binary classification хийхээс гадна хэрэглэгчдийг эрсдэлийн оноогоор эрэмбэлж, өндөр эрсдэлтэй хэрэглэгчдэд цуглуулалт болон урьдчилан сэргийлэх арга хэмжээг түрүүлж чиглүүлэх боломж бүрдүүлэх юм.

### Үндсэн workflow

```text
DPD31 хэрэглэгч
      ↓
Machine Learning Model
      ↓
Risk Score
      ↓
Risk Ranking
      ↓
Risk Segmentation
      ↓
Collection / Intervention Prioritization
```

---

## 2. Бизнесийн асуудал

DPD31 болсон хэрэглэгчдийн дундаас хэн нь DPD91+ болох эрсдэл өндөр байгааг урьдчилан тодорхойлох шаардлагатай.

Ингэснээр бизнес:

- өндөр эрсдэлтэй хэрэглэгчдийг эрт илрүүлэх
- collection-ийн нөөцийг эрэмбэлэн хуваарилах
- төлбөрийн сануулга болон харилцааг эрт эхлүүлэх
- өндөр эрсдэлтэй хэрэглэгчдэд илүү анхаарал хандуулах

боломжтой.

Энэхүү model нь хэрэглэгчийг автоматаар татгалзуулах эсвэл шийтгэх зориулалттай биш бөгөөд **эрсдэлийг эрэмбэлэх decision-support tool** хэлбэрээр ашиглахад чиглэнэ.

---

## 3. Өгөгдлийн тойм

Анхны `user_list` өгөгдөл нийт **61,200 мөртэй**.

| Dataset | Мөрийн тоо |
|---|---:|
| Нийт өгөгдөл | 61,200 |
| Label-тэй өгөгдөл | 52,737 |
| Prediction хийх unlabeled өгөгдөл | 8,463 |

Target хувьсагч:

```text
is_dpd_91
```

- `0` — DPD91+ болохгүй
- `1` — DPD91+ болно

Label-тэй өгөгдлийн target distribution:

| Target | Count |
|---|---:|
| 0 | 36,064 |
| 1 | 16,673 |

Positive class-ийн хувь ойролцоогоор **31.6%** бөгөөд imbalance бүхий binary classification problem гэж үзсэн.

---

## 4. Train / Validation / Test split

Энэ асуудал нь хугацааны дараалалтай тул random split ашиглахын оронд **temporal split** ашигласан.

| Dataset | Хугацаа | Мөр |
|---|---|---:|
| Train | Dec 2025 – Mar 2026 | 34,672 |
| Validation | Apr 2026 | 9,170 |
| Test | May 2026 | 8,895 |
| Prediction | Unlabeled data | 8,463 |

`created_month`-ийг temporal split болон analysis-д ашигласан боловч model feature болгон ашиглаагүй.

Temporal split ашигласнаар model-ийг бодит амьдралын ирээдүйн хугацаанд prediction хийх нөхцөлтэй илүү ойр үнэлэх боломжтой.

---

## 5. Data Quality болон Leakage шалгалт

Model-д оруулахын өмнө feature availability болон potential leakage-ийг шалгасан.

### `credit_id`

Зөвхөн хэрэглэгчийг таних identifier болгон ашигласан бөгөөд model feature болгон ашиглаагүй.

### `date_dpd_31`

DPD31 prediction point-тэй холбоотой тул raw model feature болгон ашиглаагүй.

### `created_month`

Temporal split болон analysis-д ашигласан боловч model feature болгон ашиглаагүй.

### `total_payment_amt`

Prediction хийх үеийн мэдээлэл гэдгийг найдвартай баталгаажуулах боломжгүй тул conservative байдлаар хассан.

### `is_unitel`

Prediction population-д бараг бүрэн missing байсан тул хассан.

### `gender`

Prediction population-д маш өндөр missing rate-тэй байсан тул хассан.

### `has_changed_phone`

Prediction population-д маш өндөр missing rate-тэй байсан тул хассан.

### `ostype`

Prediction population-д ашиглах боломжтой мэдээлэл байсан тул хадгалсан.

---

## 6. Missing Value Analysis

Missing values-ийг зөвхөн техникийн асуудал гэж үзэлгүй, өөрөө predictive signal байж болох эсэхийг шалгасан.

Зарим feature-д missing indicator үүсгэсэн.

Missingness-only XGBoost туршилтаар:

```text
ROC-AUC = 0.5680
PR-AUC  = 0.3625
```

гэсэн үр дүн гарсан.

Энэ нь missingness өөрөө target-тэй тодорхой холбоотой байж болохыг харуулсан боловч missingness alone нь үндсэн model-ийн predictive power-ийг бүрэн тайлбарлахгүй.

---

## 7. Feature Engineering

Model-д DPD31 үеийн financial болон behavioral мэдээллүүдийг ашигласан.

### Financial features

Жишээ:

- `limit_dpd31`
- `balance_dpd31`
- `limit_ondue`
- `balance_ondue`
- `invoiced_unpaid_amt`

### Behavioral features

Жишээ:

- transaction count
- average transaction amount
- session activity
- notification reading activity
- transaction recency
- user tenure
- number-change activity

Temporal windows:

```text
trx_l7d
trx_l30d
trx_l60d
trx_l90d

avg_trx_amount_l7d
avg_trx_amount_l30d
avg_trx_amount_l60d
avg_trx_amount_l90d

sessions_l7d
sessions_l30d
sessions_l60d
sessions_l90d

read_noti_l7d
read_noti_l30d
read_noti_l60d
read_noti_l90d
```

Мөн financial ratio болон missingness-related features үүсгэж үнэлсэн.

Final XGBoost model нийт **33 feature** ашигласан.

---

## 8. Model Development

Дараах model-уудыг туршиж харьцуулсан:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- MLP

Model evaluation-д голчлон:

- ROC-AUC
- PR-AUC

ашигласан.

Мөн business application талаас:

- Lift
- Top-N ranking
- Positive capture rate

зэргийг шинжилсэн.

---

## 9. Model Comparison

Эцсийн model comparison:

| Model | Features | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|
| **XGBoost** | 33 | **0.6175** | **0.3875** |
| LightGBM | 33 | 0.6145 | 0.3744 |
| MLP Top 33 | 33 | 0.6052 | 0.3730 |
| MLP Top 20 | 20 | 0.5980 | 0.3641 |
| MLP Top 10 | 10 | 0.5976 | 0.3647 |

XGBoost нь evaluated final candidates дундаас test set дээр хамгийн сайн overall ranking performance үзүүлсэн тул final model-оор сонгосон.

---

## 10. Final XGBoost Model

Final model нь **33 feature бүхий XGBoost binary classifier**.

Model нь хэрэглэгч бүрт DPD91+ болох эрсдэлийг илэрхийлэх risk score гаргана.

### Validation

```text
ROC-AUC = 0.6288
PR-AUC  = 0.4322
```

### Test

```text
ROC-AUC = 0.6175
PR-AUC  = 0.3875
```

Эдгээр үзүүлэлт нь model төгс classifier биш боловч хэрэглэгчдийг эрсдэлийн түвшнээр эрэмбэлэхэд ашиглаж болохуйц predictive signal байгааг харуулж байна.

---

## 11. Feature Importance

Final XGBoost model-ийн хамгийн өндөр importance бүхий feature-үүд:

| Feature | Importance |
|---|---:|
| `sessions_l7d` | 14.89% |
| `sessions_l30d` | 8.18% |
| `number_change_cnt_l90d` | 5.68% |
| `invoiced_unpaid_amt` | 4.52% |
| `trx_recency_l90d` | 3.67% |
| `app_recency_days_missing` | 3.34% |
| `balance_ondue_to_limit` | 3.32% |
| `balance_dpd31` | 3.27% |
| `avg_trx_amount_l90d` | 3.11% |
| `balance_ondue` | 3.10% |

Ялангуяа богино хугацааны session activity болон transaction recency зэрэг behavioral features model-ийн prediction-д чухал хувь нэмэр оруулсан.

---

## 12. Risk Ranking ба Lift Analysis

Model-ийг зөвхөн classification threshold-ээр бус risk ranking хэлбэрээр ашиглах боломжийг шалгасан.

Test set-ийн нийт positive rate:

```text
28.6%
```

Харин model-ийн хамгийн өндөр risk score-той Top 10% хэрэглэгчдийн:

```text
Actual Positive Rate = 46.0%
Lift = 1.61x
```

байсан.

### Test-set Lift

| Top Risk Group | Actual Positive Rate | Lift |
|---|---:|---:|
| Top 10% | 46.0% | **1.61x** |
| Top 20% | 40.7% | 1.42x |
| Top 30% | 37.7% | 1.32x |
| Top 40% | 36.8% | 1.29x |
| Top 50% | 35.2% | 1.23x |

Энэ нь model-ийн хамгийн гол business value нь **өндөр эрсдэлтэй хэрэглэгчдийг эрэмбэлж, нөөцийг тэдгээр хэрэглэгчдэд төвлөрүүлэх** боломж гэдгийг харуулж байна.

---

## 13. Threshold Analysis

Validation set дээр classification threshold-үүдийг харьцуулсан.

Validation дээр F1 хамгийн өндөр байсан threshold:

```text
Threshold = 0.27
F1        = 0.5039
Precision = 0.3572
Recall    = 0.8553
```

Гэхдээ энэ threshold нь predicted positive rate-ийг маш өндөр болгож байсан.

Тиймээс final business output-д fixed binary threshold-ийг үндсэн шийдэл болгоогүй.

Оронд нь:

```text
Risk Score
     ↓
Risk Ranking
     ↓
Top-N Prioritization
```

гэсэн аргачлалыг илүү тохиромжтой гэж үзсэн.

Бодит production threshold-ийг дараах business information дээр үндэслэн тусад нь сонгох шаардлагатай:

- False positive cost
- False negative cost
- Collection capacity
- Intervention strategy

---

## 14. Credit-Level Overlap ба Robustness

Өгөгдөл нь longitudinal шинжтэй бөгөөд нэг `credit_id` олон хугацаанд давтагдах боломжтой.

Тиймээс split хоорондын credit-level overlap-ийг шалгасан.

```text
Train ∩ Validation = 3,138
Train ∩ Test       = 3,523
Validation ∩ Test  = 213
```

Test set-ийн:

```text
40.9%
```

нь Train эсвэл Validation set-д өмнө нь тухайн credit-level мэдээлэлтэй давхцаж байсан.

Энэ нь тухайн өгөгдөлд repeated borrower/event structure байгаатай холбоотой бөгөөд шууд leakage гэж дүгнээгүй.

Гэхдээ model-ийн generalization performance-ийг тайлбарлахдаа энэ overlap-ийг limitation болгон авч үзсэн.

---

## 15. Final Prediction

Final XGBoost model-ийг unlabeled prediction population дээр ажиллуулсан.

```text
Prediction rows = 8,463
Feature count   = 33
```

Final prediction score:

```text
Min risk score  = 0.1502
Max risk score  = 0.5944
Mean risk score = 0.3455
```

Prediction pipeline:

```text
Prediction source
      ↓
24 Behavioral features
      +
9 Financial features
      ↓
33 Final features
      ↓
StandardScaler
      ↓
Final XGBoost
      ↓
Risk Score
```

Prediction feature check:

```text
8463 rows × 33 features
Feature order match = True
Scaling = Successful
Prediction = Successful
```

---

## 16. Risk Segmentation

8,463 prediction хэрэглэгчийг risk score-оор нь эрэмбэлж 4 сегментэд хуваасан.

| Risk Segment | Customers | Share | Avg Risk Score |
|---|---:|---:|---:|
| Top 10% - High Risk | 846 | 10.0% | 0.4502 |
| 10–20% - High-Medium Risk | 846 | 10.0% | 0.4193 |
| 20–30% - Medium Risk | 846 | 10.0% | 0.3988 |
| Bottom 70% - Lower Risk | 5,925 | 70.0% | 0.3123 |

Risk score-ийн дараалал нь segmentation-тэй нийцэж байна.

---

## 17. Business Recommendation

Model-ийг дараах байдлаар ашиглаж болно.

### Top 10% — High Risk

Хамгийн өндөр priority:

- proactive payment reminder
- collection contact
- personalized communication
- early intervention

### 10–20% — High-Medium Risk

Дунд/өндөр priority intervention болон monitoring.

### 20–30% — Medium Risk

Стандарт collection workflow + нэмэлт monitoring.

### Bottom 70% — Lower Risk

Ердийн collection workflow, харьцангуй бага intervention priority.

Энд model-ийн prediction-ийг хэрэглэгчийн эцсийн шийдвэрийг автоматаар гаргах хэрэгсэл бус **decision-support tool** болгон ашиглах нь зөв.

---

## 18. Final Output

Final prediction файл:

```text
toki_final_risk_predictions.csv
```

Файл нь нийт **8,463 мөр**, дараах 6 баганатай:

```text
credit_id
created_month
date_dpd_31
risk_score
risk_rank_pct
risk_segment
```

Жишээ:

| credit_id | created_month | risk_score | risk_rank_pct | risk_segment |
|---|---:|---:|---:|---|
| 2069593 | 202606 | 0.5944 | 0.000118 | Top 10% - High Risk |
| 2603520 | 202606 | 0.5329 | 0.000236 | Top 10% - High Risk |
| 2063776 | 202606 | 0.5323 | 0.000354 | Top 10% - High Risk |

---

## 19. Хязгаарлалтууд

Энэхүү ажлын гол limitations:

1. Model-ийн predictive performance дунд түвшинд байгаа бөгөөд төгс classifier биш.
2. Өгөгдөлд нэг `credit_id` олон хугацаанд давтагдах боломжтой.
3. Train, Validation, Test хооронд credit-level overlap байгаа.
4. Missingness pattern нь хугацааны явцад өөрчлөгдөж болно.
5. Зарим feature-ийг prediction үеийн availability баталгаатай биш байсан тул хассан.
6. False positive болон false negative-ийн business cost тодорхой өгөгдөөгүй.
7. Иймээс model-ийг binary decision system-ээс илүү risk-ranking model хэлбэрээр ашиглах нь тохиромжтой.
8. Production орчинд model drift болон monthly cohort performance-ийг тогтмол хянах шаардлагатай.

---

## 20. Цаашид сайжруулах боломж

Дараагийн шатанд:

- илүү олон хугацааны temporal validation хийх
- credit-disjoint robustness evaluation хийх
- probability calibration хийх
- business cost дээр суурилсан threshold optimization хийх
- collection capacity-д суурилсан intervention policy боловсруулах
- сар бүр model performance мониторинг хийх
- feature drift болон missingness drift хянах
- шаардлагатай үед model retraining хийх
- prediction-time availability баталгаатай нэмэлт behavioral feature оруулах

боломжтой.

---

## 21. Төслийн бүтэц

```text
.
├── Toki_Data_Scientist_Task.ipynb
├── toki_final_risk_predictions.csv
└── README.md
```

### Notebook

`Toki_Data_Scientist_Task.ipynb` нь:

- Data loading
- Data quality audit
- Target definition
- Temporal split
- Leakage analysis
- Feature engineering
- Missing value analysis
- Baseline models
- Model comparison
- XGBoost development
- Model evaluation
- Threshold analysis
- Lift analysis
- Robustness analysis
- Final prediction
- Risk segmentation

бүх workflow-ийг агуулна.

---

## 22. Дүгнэлт

Энэхүү төсөлд DPD31 болсон хэрэглэгч DPD91+ болох эрсдэлийг таамаглах temporal machine learning framework боловсруулсан.

Үнэлэгдсэн model-уудаас XGBoost хамгийн сайн test ranking performance үзүүлсэн.

### Final Test Performance

```text
ROC-AUC = 0.6175
PR-AUC  = 0.3875
```

Business талаас хамгийн чухал үр дүн:

```text
Overall positive rate = 28.6%

Top 10% risk group
Positive rate = 46.0%

Lift = 1.61x
```

Өөрөөр хэлбэл model нь хамгийн өндөр эрсдэлтэй Top 10% хэрэглэгчдийн дотор positive case-уудыг нийт population-тэй харьцуулахад **1.61 дахин өндөр концентрациар** илрүүлж чадсан.

Иймээс энэхүү model-ийн гол хэрэглээ нь:

> **DPD31 болсон хэрэглэгчдийг эрсдэлийн оноогоор эрэмбэлж, collection болон intervention-ийн нөөцийг өндөр эрсдэлтэй хэрэглэгчдэд түрүүлж чиглүүлэх.**

Final prediction pipeline нь **8,463 unlabeled хэрэглэгчид** зориулсан risk score болон risk segment үүсгэж, бизнесийн дараагийн шатны risk-based prioritization хийх боломжийг бүрдүүлсэн.
