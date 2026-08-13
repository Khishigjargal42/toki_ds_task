# Toki Data Scientist Даалгавар — Credit Risk Prediction

## 1. Даалгаврын зорилго

Энэхүү даалгаврын зорилго нь хэрэглэгчийн өмнөх хэрэглээ, санхүүгийн болон зан төлөвийн мэдээлэлд үндэслэн тухайн хэрэглэгч **ирээдүйд 91+ хоногийн хугацаа хэтрүүлэлт (`is_dpd_91 = 1`) үүсгэх эрсдэлтэй эсэхийг** таамаглах machine learning шийдэл боловсруулахад оршино.

Энд зөвхөн classification model байгуулахаас гадна:

- өгөгдлийн чанар ба missing value-ийг шалгах
- feature engineering хийх
- хэд хэдэн model-ийг харьцуулах
- class imbalance-ийг харгалзах
- threshold болон ranking-based шийдвэрийн логик шалгах
- model-ийн ерөнхийлөх чадварыг шалгах
- risk segmentation хийх
- эцсийн prediction output боловсруулах

гэсэн үе шатуудыг хамруулсан.

---

## 2. Dataset overview

Өгөгдөлд хэрэглэгчийн credit, financial болон behavioral шинжүүд багтсан.

### Гол зорилтот хувьсагч

`is_dpd_91`

- `0` — 91+ хоногийн хугацаа хэтрүүлэлт үүсээгүй
- `1` — 91+ хоногийн хугацаа хэтрүүлэлт үүссэн

### Өгөгдлийн үндсэн хэмжээ

| Dataset | Rows |
|---|---:|
| Full user data | 61,200 |
| Labeled data | 52,737 |
| Train | 34,672 |
| Validation | 9,170 |
| Test | 8,895 |
| Final prediction set | 8,463 |

Final prediction set нь target label байхгүй хэрэглэгчдэд risk score тооцоход ашиглагдсан.

---

# 3. Data understanding & preparation

Өгөгдлийн эхний шатанд:

- dataset-ийн бүтэц
- feature-ийн төрөл
- missing value
- target distribution
- labeled/unlabeled record
- date болон cohort-ийн боломжит ялгаа

зэргийг шалгасан.

Missing value-ийг шууд устгахын оронд feature-ийн утга болон missingness өөрөө prediction-д мэдээлэл өгч болох эсэхийг авч үзсэн.

Ялангуяа behavioral feature-үүд дээр missing indicator feature ашигласан.

---

# 4. Feature engineering

Financial болон behavioral мэдээллийг тусад нь боловсруулж, дараа нь model-д ашиглах feature set болгон нэгтгэсэн.

### Financial features

Үндсэн financial feature-үүдээс:

- `limit_dpd31`
- `balance_dpd31`
- `limit_ondue`
- `balance_ondue`
- `invoiced_unpaid_amt`

зэрэг feature-үүд ашигласан.

### Ratio features

Financial feature-үүдээс дараах харьцаануудыг engineered feature болгон үүсгэсэн:

- `balance_ondue_to_limit`
- `balance_dpd31_to_limit`
- `unpaid_to_balance_ondue`
- `unpaid_to_limit_ondue`

### Behavioral features

Хэрэглэгчийн:

- application recency
- transaction frequency
- transaction amount
- session activity
- notification interaction
- user tenure
- number change activity

зэрэг behavioral мэдээллийг 7, 30, 60, 90 хоногийн цонхоор ашигласан.

Нийт эцсийн model feature:

**33 features**

---

# 5. Train / Validation / Test strategy

Model development болон final evaluation-ийг тусгаарлахын тулд:

- Train: 34,672 rows
- Validation: 9,170 rows
- Test: 8,895 rows

хуваарилалт ашигласан.

Validation set нь:

- threshold selection
- model comparison
- ranking analysis

зэрэг model selection-тэй холбоотой шийдвэрүүдэд ашиглагдсан.

Test set нь эцсийн model performance-ийг үнэлэхэд ашиглагдсан.

---

# 6. Important data overlap analysis

Credit-level overlap-ийг тусгайлан шалгасан.

### Credit-level overlap

| Comparison | Overlapping credits |
|---|---:|
| Train ∩ Validation | 3,138 |
| Train ∩ Test | 3,523 |
| Validation ∩ Test | 213 |

Test set-ийн:

- 3,638 rows буюу **40.9%** нь Train эсвэл Validation-д өмнө нь харагдсан credit-тэй холбоотой байсан.

Иймээс test performance-ийг бүрэн unseen-credit generalization гэж тайлбарлахгүй байх шаардлагатай.

Энэ нь model evaluation-ийн чухал limitation бөгөөд notebook болон README-д ил тод тэмдэглэсэн.

---

# 7. Models evaluated

Хэд хэдэн model болон feature configuration-ийг харьцуулсан.

Үүнд:

- Logistic Regression / baseline
- Random Forest
- XGBoost
- LightGBM
- MLP

зэрэг model-ууд орсон.

MLP дээр feature selection-ийн нөлөөг мөн шалгасан:

- Top 5
- Top 10
- Top 15
- Top 20
- Top 33

---

# 8. Final model comparison

Final evaluation дээр хамгийн сайн model нь XGBoost байсан.

| Model | Features | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|
| **XGBoost** | **33** | **0.617548** | **0.387500** |
| LightGBM | 33 | 0.614522 | 0.374430 |
| MLP Top 33 | 33 | 0.605160 | 0.372955 |
| MLP Top 20 | 20 | 0.598024 | 0.364123 |
| MLP Top 10 | 10 | 0.597607 | 0.364724 |

XGBoost нь test ROC-AUC болон PR-AUC-ийн аль алинд хамгийн өндөр үзүүлэлттэй байсан.

### Final XGBoost

**Validation**

- ROC-AUC: **0.628820**
- PR-AUC: **0.432212**

**Test**

- ROC-AUC: **0.617548**
- PR-AUC: **0.387500**

Validation → Test-ийн бууралт байгаа боловч model ranking ability тодорхой хэмжээнд хадгалагдсан.

---

# 9. Why PR-AUC is important

Target class нь balanced биш тул зөвхөн accuracy ашиглах нь тохиромжгүй.

Test set дээр positive rate:

**28.59%**

байсан.

Ийм нөхцөлд:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- Lift
- Positive Capture Rate

зэрэг metric-үүдийг хамтад нь ашигласан.

PR-AUC нь positive class-ийг хэр сайн эрэмбэлж байгааг үнэлэхэд илүү ач холбогдолтой.

---

# 10. Threshold analysis

Default threshold `0.50`-ийг шууд ашиглахын оронд validation set дээр threshold-ийг шалгасан.

Validation дээр хамгийн өндөр F1 үзүүлэлт:

**Threshold = 0.27**

- Precision: **0.3572**
- Recall: **0.8553**
- F1: **0.5039**

Гэхдээ test set дээр threshold 0.27 ашиглахад:

- Precision: **0.3155**
- Recall: **0.8836**
- F1: **0.4649**

гарсан.

Model-ийн predicted positive rate өндөр болсон тул threshold selection-ийг зөвхөн F1-ээр шийдэх нь бизнесийн хувьд заавал хамгийн зөв сонголт биш байж болохыг тэмдэглэсэн.

Иймээс ranking/lift approach-ийг мөн авч үзсэн.

---

# 11. Ranking and Lift analysis

Risk prediction-ийн практик хэрэглээнд бүх хэрэглэгчийг binary `0/1` болгон ангилахаас илүү **хамгийн өндөр эрсдэлтэй хэрэглэгчдийг эрэмбэлэх** нь илүү ашигтай байж болно.

Test set-ийн baseline positive rate:

**28.59%**

### Top-N performance

| Top % | Actual Positive Rate | Lift | Positive Capture Rate |
|---:|---:|---:|---:|
| Top 10% | 46.01% | **1.609** | 16.08% |
| Top 20% | 40.70% | **1.424** | 28.47% |
| Top 30% | 37.74% | **1.320** | 39.60% |
| Top 40% | 36.76% | **1.286** | 51.44% |
| Top 50% | 35.24% | **1.233** | 61.62% |

Эндээс model нь хамгийн өндөр эрсдэлтэй 10%-ийг сонгоход baseline-ээс ойролцоогоор **1.61x өндөр positive rate**-тай хэрэглэгчдийг төвлөрүүлж чадсан.

---

# 12. Risk segmentation

Final unlabeled prediction set дээр XGBoost ашиглан **8,463 хэрэглэгчид risk score** тооцсон.

Risk score-ийн хүрээ:

- Minimum: **0.1502**
- Maximum: **0.5944**
- Mean: **0.3455**

Хэрэглэгчдийг risk rank-аар дөрвөн segment болгон хуваасан.

| Segment | Customers | Avg Risk Score | Share |
|---|---:|---:|---:|
| Top 10% - High Risk | 846 | 0.4502 | 10.00% |
| 10-20% - High-Medium Risk | 846 | 0.4193 | 10.00% |
| 20-30% - Medium Risk | 846 | 0.3988 | 10.00% |
| Bottom 70% - Lower Risk | 5,925 | 0.3123 | 70.01% |

Энэ segmentation нь binary classification-аас илүү operational байдлаар ашиглах боломжтой.

Жишээлбэл:

- Top 10% — хамгийн түрүүнд анхаарах
- 10–20% — өндөр priority
- 20–30% — дунд priority
- Bottom 70% — бага priority

гэсэн байдлаар intervention priority үүсгэж болно.

---

# 13. Feature importance

Final XGBoost model-ийн хамгийн чухал feature-үүд:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `sessions_l7d` | 14.89% |
| 2 | `sessions_l30d` | 8.18% |
| 3 | `number_change_cnt_l90d` | 5.68% |
| 4 | `invoiced_unpaid_amt` | 4.52% |
| 5 | `trx_recency_l90d` | 3.67% |
| 6 | `app_recency_days_missing` | 3.34% |
| 7 | `balance_ondue_to_limit` | 3.32% |
| 8 | `balance_dpd31` | 3.27% |
| 9 | `avg_trx_amount_l90d` | 3.11% |
| 10 | `balance_ondue` | 3.10% |

Behavioral болон financial feature-үүд хоёулаа model-ийн prediction-д хувь нэмэр оруулж байна.

Feature importance-ийг causal relationship гэж тайлбарлаагүй; model-ийн predictive contribution гэж ойлгосон.

---

# 14. Feature-target analysis

Top feature-үүдийн target rate-ийг feature-ийн утгын интервалуудаар шалгасан.

Жишээ нь `trx_recency_l90d`:

- 1–9 days → target rate **22.65%**
- 9–29 days → **23.86%**
- 29–48 days → **30.64%**
- 48–58 days → **30.82%**
- 58–90 days → **35.18%**

Энэ нь transaction recency нэмэгдэх буюу хэрэглэгчийн сүүлийн transaction-оос хойш хугацаа уртсах үед target rate өсөх хандлагатай байгааг харуулж байна.

`balance_ondue_to_limit` болон `balance_dpd31_to_limit` зэрэг ratio feature-үүдийн хувьд мөн өндөр utilization/ratio түвшинд target rate өсөх хандлага ажиглагдсан.

---

# 15. Calibration analysis

Prediction probability болон actual positive rate-ийг probability bins-ээр харьцуулсан.

Model probability-ууд actual rate-ийг зарим түвшинд системтэйгээр өндөр үнэлж байсан.

Жишээ нь:

| Mean Predicted | Actual |
|---:|---:|
| 0.169 | 0.143 |
| 0.243 | 0.191 |
| 0.290 | 0.229 |
| 0.324 | 0.244 |
| 0.354 | 0.290 |
| 0.381 | 0.291 |
| 0.410 | 0.339 |
| 0.440 | 0.318 |
| 0.479 | 0.353 |
| 0.557 | 0.461 |

Иймээс `risk_score`-ийг шууд бодит default probability гэж тайлбарлахгүй.

Энэ даалгаврын хувьд score-ийг **relative risk ranking** болгон ашиглах нь илүү зохистой.

---

# 16. Seen vs Unseen Credit analysis

Test set дотор өмнөх dataset-үүдэд харагдсан credit болон unseen credit-ийг тусад нь шалгасан.

| Credit status | Rows | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Seen-credit | 3,638 | 0.6054 | 0.3532 |
| Unseen-credit | 5,257 | 0.6249 | 0.4082 |

Unseen-credit дээр model-ийн performance илүү өндөр гарсан.

Энэ нь overlap нь performance-ийг заавал хиймлээр өсгөсөн гэж шууд дүгнэх боломжгүйг харуулж байгаа боловч split strategy болон credit-level dependency нь цаашдын production validation-д анхаарах зүйл хэвээр байна.

---

# 17. Final prediction output

Final prediction dataset дээр дараах багануудыг гаргасан:

```text
credit_id
created_month
date_dpd_31
risk_score
risk_rank_pct
risk_segment
```

Нийт:

**8,463 prediction rows**

CSV файл:

```text
toki_final_risk_predictions.csv
```

Output нь risk score-оор эрэмбэлэгдсэн бөгөөд хэрэглэгч бүрийн risk segment-ийг агуулна.

---

# 18. Business interpretation

Энэ model-ийн гол үнэ цэнэ нь зөвхөн `default / non-default` classification хийхэд бус, **эрсдэлийн өндөр хэрэглэгчдийг эрэмбэлэн priority тогтоох** боломжид оршино.

Жишээ operational use case:

1. Top 10% хэрэглэгчийг хамгийн өндөр priority intervention-д оруулах.
2. Дараагийн 10%-ийг high-medium priority болгон авч үзэх.
3. Top 30%-ийг targeted monitoring-д ашиглах.
4. Bottom 70%-ийг бага priority байдлаар боловсруулах.

Ингэснээр бизнесийн нөөцийг бүх хэрэглэгчид ижил хэмжээгээр зарцуулахын оронд өндөр эрсдэлтэй сегментэд төвлөрүүлэх боломжтой.

---

# 19. Limitations

Энэхүү шийдэлд дараах limitations байна:

### 1. Predictive performance moderate

Final test:

- ROC-AUC = **0.618**
- PR-AUC = **0.388**

Иймээс model нь perfect classifier биш бөгөөд production decision-д дангаар нь ашиглах ёсгүй.

### 2. Credit-level overlap

Train, validation болон test-ийн хооронд credit-level overlap байгаа.

Иймээс test result-ийг бүрэн independent customer-level generalization гэж тайлбарлах боломжгүй.

### 3. Probability calibration

Risk score нь actual probability-тэй бүрэн calibrated биш.

Иймээс:

> `risk_score = 0.45`

гэдгийг яг `45% default probability` гэж тайлбарлахгүй.

### 4. Threshold sensitivity

Threshold өөрчлөгдөхөд precision болон recall ихээхэн өөрчлөгдөж байна.

Иймээс production threshold-ийг бизнесийн intervention cost болон risk appetite-тэй уялдуулан сонгох шаардлагатай.

### 5. Temporal validation

Илүү найдвартай production validation хийхийн тулд future-period holdout буюу time-based validation хийх нь зүйтэй.

---

# 20. Recommended next steps

Production deployment-ээс өмнө дараах ажлуудыг хийхийг зөвлөж байна:

1. **Time-based validation**
   - Ирээдүйн cohort-ийг тусад нь holdout хийх.

2. **Credit/customer-level split**
   - Нэг credit/customer-ийн мэдээлэл train болон test-д давхар орохгүй байх split strategy хэрэгжүүлэх.

3. **Probability calibration**
   - Isotonic Regression эсвэл Platt Scaling ашиглан probability calibration шалгах.

4. **Business cost-based threshold**
   - False positive болон false negative-ийн бизнесийн өртгийг тооцож threshold сонгох.

5. **Model monitoring**
   - Data drift
   - Feature drift
   - Prediction distribution
   - Default rate
   - Model performance

   зэргийг production дээр тогтмол хянах.

6. **Explainability**
   - SHAP зэрэг арга ашиглан individual prediction-ийн гол нөлөөлөгч feature-үүдийг тайлбарлах.

7. **Champion / challenger setup**
   - XGBoost-ийг champion model болгон авч, LightGBM зэрэг model-ийг challenger байдлаар production validation-д харьцуулах.

---

# 21. Conclusion

Энэхүү ажлаар хэрэглэгчийн behavioral болон financial мэдээлэлд үндэслэн 91+ хоногийн хугацаа хэтрүүлэлтийн эрсдэлийг үнэлэх end-to-end machine learning pipeline боловсруулсан.

33 feature ашигласан хэд хэдэн model-ийг харьцуулсны үр дүнд **XGBoost** хамгийн сайн test performance үзүүлсэн:

- **ROC-AUC: 0.6175**
- **PR-AUC: 0.3875**

Model нь ялангуяа ranking-based approach дээр практик ач холбогдолтой үр дүн үзүүлсэн. Test set-ийн хамгийн өндөр эрсдэлтэй 10% хэрэглэгчийн target rate **46.01%** байсан нь нийт baseline **28.59%**-оос **1.61x өндөр** байв.

Final prediction set дээр 8,463 хэрэглэгчид risk score тооцож, Top 10%, 10–20%, 20–30%, Bottom 70% гэсэн operational risk segment-үүдэд хуваасан.

Гэхдээ model-ийн performance moderate бөгөөд credit-level overlap, probability calibration болон temporal generalization зэрэг асуудлууд production deployment-ээс өмнө нэмэлтээр шалгагдах шаардлагатай.

Иймээс энэхүү шийдлийг **final automated credit decision maker** гэхээс илүү **risk prioritization болон targeted intervention-д ашиглах predictive ranking system-ийн prototype** гэж үзэх нь зохистой.

---

# 22. Repository structure

```text
.
├── README.md
├── Toki Data Scientist task.ipynb
├── toki_final_risk_predictions.csv
└── data/
    └── Toki Data Scientist task.xlsx
```

> Dataset-ийн privacy болон repository-ийн шаардлагаас шалтгаалан эх өгөгдлийг repository-д оруулах эсэхийг тусад нь шийдвэрлэнэ.

---

# 23. Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# 24. Final takeaway

Энэхүү даалгаврын үндсэн санаа нь:

**Raw data → Data quality analysis → Feature engineering → Model comparison → Validation → Test evaluation → Threshold & ranking analysis → Risk segmentation → Final customer-level prediction**

гэсэн бүрэн pipeline байгуулах явдал байсан.

Final model нь төгс classification хийхээс илүү **эрсдэлийн өндөр хэрэглэгчдийг зөв дараалалд оруулж, бизнесийн intervention-ийн priority тогтоох** тал дээр ашиглах боломжтойг үр дүн харуулж байна.
