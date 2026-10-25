# 10 · เลือก hyperparameter ด้วย CV และ learning curve

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้ใช้ **GridSearchCV** เลือก hyperparameter และอ่าน learning curve เพื่อดูผลของขนาดข้อมูลฝึก

**ใช้สิ่งที่เรียนแล้ว:** KNN/Pipeline จากบท 03, CV จากบท 08 และ overfitting จากบท 09

## ใจความใหม่

- **Parameter:** ค่าที่เรียนจากข้อมูล เช่น coefficient
- **Hyperparameter:** การตั้งค่าวิธีเรียน เช่น k ของ KNN

**Grid search:** ลองทุกชุดค่าที่กำหนด แล้วเลือกจากคะแนน CV

ตัวอย่าง: k=3, 5, 9 × weights สองแบบ = 6 ชุด หากใช้ 5-fold CV จะฝึก 30 ครั้ง แล้วฝึกชุดที่ชนะบน train ทั้งหมดอีกครั้ง เรียกว่า **refit**

| ชื่อในโค้ด | ความหมาย |
|---|---|
| `uniform` | ให้เพื่อนบ้านลงคะแนนเท่ากัน |
| `distance` | ให้เพื่อนบ้านที่ใกล้มีน้ำหนักมากกว่า |
| `knn__n_neighbors` | ค่า n_neighbors ของ step ชื่อ knn |
| `mean_test_score` | ค่าเฉลี่ยคะแนน validation ใน CV |
| `std_test_score` | ส่วนเบี่ยงเบนมาตรฐานของคะแนนระหว่าง fold |
| `best_params_` | ชุดค่าที่เลือก |
| `best_estimator_` | Pipeline ที่ refit แล้ว จึงเรียก `search.predict` ได้ทันที |

Scaler อยู่ใน Pipeline ตลอดการค้นหา

คะแนน best CV อาจสูงเกินจริง เพราะใช้คะแนนนี้เลือกค่าไปแล้ว จึงต้องประเมิน test หลังเลือกเสร็จ

## อ่าน learning curve

| รูปแบบคะแนนเมื่อเพิ่มข้อมูลฝึก | สิ่งที่ควรตรวจ |
|---|---|
| train และ validation ต่ำใกล้กัน | feature หรือความสามารถโมเดลยังไม่พอ |
| train สูง แต่ validation ต่ำกว่า | อาจ overfit ให้ลองจำกัดความซับซ้อนหรือเพิ่มข้อมูล |
| validation สูงขึ้น | ข้อมูลเพิ่มอาจช่วย แต่ไม่รับประกันว่าจะดีขึ้นเสมอ |

**แกนนอน:** จำนวนข้อมูลฝึกต่อ fold จึงจบที่ 96 แถว จาก train ทั้งหมด 120 แถว

กราฟนี้ใช้ค่าที่เลือกจาก train เดิม จึงเป็นเครื่องมือวินิจฉัย ไม่ใช่ผลประเมินอิสระอีกชุด

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรัน **Cell 1–2 ตามลำดับ** หรือเปิด [Notebook ของบทนี้](lesson.ipynb) ใช้ kernel เดิมได้เลย อ่านคำอธิบายใต้โค้ดแต่ละช่วงประกอบผลที่ได้

<!-- CODE:CELLS -->

<!-- CODE:START -->
### Cell 1 — ค้นค่าที่เหมาะด้วย GridSearchCV

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
pipeline = Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())])
search = GridSearchCV(
    pipeline, {"knn__n_neighbors": [3, 5, 9], "knn__weights": ["uniform", "distance"]},
    scoring="accuracy", cv=cv, n_jobs=1,
)
search.fit(X_train, y_train)
results = pd.DataFrame(search.cv_results_)
print(results[["param_knn__n_neighbors", "param_knn__weights",
               "mean_test_score", "std_test_score", "rank_test_score"]]
      .sort_values("rank_test_score").round(3).to_string(index=False))
print("Best parameters:", search.best_params_)
print(f"Best CV accuracy: {search.best_score_:.3f}")
```

**อ่านโค้ด**

- `load_iris(return_X_y=True)` คืน X และ y โดยตรง แทนการคืนชุดข้อมูลพร้อมรายละเอียดแบบบท 02
- `Pipeline([("scale", ...), ("knn", ...)])` ตั้งชื่อแต่ละขั้นเอง เพื่ออ้างชื่อในการค้นพารามิเตอร์
- `knn__n_neighbors` และ `knn__weights` ใช้รูปแบบ `ชื่อขั้น__ชื่อพารามิเตอร์` ให้ GridSearchCV รู้ว่าต้องปรับค่าของ KNN
- `search.fit(X_train, y_train)` ทดลองหกชุดค่าด้วยห้า fold แล้ว refit ชุดที่ชนะบน train ทั้งหมดตามค่าเริ่มต้น
- `pd.DataFrame(search.cv_results_)` ทำผลค้นหาเป็นตาราง แล้วเรียงด้วย `rank_test_score` เพื่ออ่านลำดับจากคะแนน CV
- `best_params_` บอกชุดค่าที่เลือก และ `best_score_` เป็นคะแนน CV เฉลี่ยของชุดนั้น ไม่ใช่ final test

### Cell 2 — ดูผลของจำนวนข้อมูลฝึก

```python
sizes, train_scores, valid_scores = learning_curve(
    search.best_estimator_, X_train, y_train, cv=cv, scoring="accuracy",
    train_sizes=[0.4, 0.6, 0.8, 1.0], shuffle=True, random_state=42,
)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sizes, train_scores.mean(axis=1), "o-", label="Train")
ax.plot(sizes, valid_scores.mean(axis=1), "o-", label="Validation")
ax.set(xlabel="Training examples per fold", ylabel="Accuracy", ylim=(0, 1.05))
ax.legend()
fig.tight_layout()
out = Path("outputs/10-tuning-and-learning-curves")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "learning-curve.png", dpi=140)
plt.show()
plt.close(fig)
print(f"Final test accuracy: {accuracy_score(y_test, search.predict(X_test)):.3f}")
```

**อ่านโค้ด**

- `learning_curve(search.best_estimator_, ...)` ใช้การตั้งค่าที่เลือกแล้ว ฝึกซ้ำด้วยขนาด train หลายระดับ
- `train_sizes=[0.4, 0.6, 0.8, 1.0]` เป็นสัดส่วนของข้อมูลฝึกที่ใช้ได้ในแต่ละ fold ระดับ 1.0 จึงเท่ากับ 96 แถวในบทนี้
- `sizes` คืนจำนวนแถวจริง ส่วน `train_scores` และ `valid_scores` มีหนึ่งแถวต่อขนาดข้อมูล และหนึ่งคอลัมน์ต่อ fold
- `mean(axis=1)` เฉลี่ยคะแนนข้าม fold เพื่อวาดหนึ่งจุดต่อขนาดข้อมูล
- `search.predict(X_test)` ใช้โมเดลที่ GridSearchCV refit ไว้ประเมิน test หลังเลือกค่าเสร็จแล้ว
<!-- CODE:END -->

## อ่านผล

อ่านผลตามลำดับ: ตารางหกทางเลือก → best parameters → learning curve → test accuracy

ไฟล์ภาพ: `outputs/10-tuning-and-learning-curves/learning-curve.png`

![ผลการทดลอง 10-tuning-and-learning-curves](../../assets/figures/10-tuning-and-learning-curves-learning-curve.png)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองคิด:** k ห้าค่า × weights สองค่า × CV สี่ fold ต้อง fit กี่ครั้ง?

**เฉลย:** ฝึก 40 ครั้งสำหรับ CV และ refit อีก 1 ครั้ง รวม 41 ครั้ง

อย่าใช้ผล test เลือกว่าจะขยาย grid อย่างไรซ้ำ ๆ เพราะจะปรับโมเดลเข้าหาความบังเอิญใน test

</details>

## อ้างอิง

- [Handbook: Hyperparameters and Model Validation](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html)
- [Tuning hyperparameters](https://scikit-learn.org/1.7/modules/grid_search.html)
- [Validation and learning curves](https://scikit-learn.org/1.7/modules/learning_curve.html)
