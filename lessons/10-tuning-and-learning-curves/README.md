# 10 · เลือก hyperparameter ด้วย CV และ learning curve

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้ใช้ **GridSearchCV** เลือก hyperparameter และอ่าน learning curve เพื่อดูผลของขนาดข้อมูลฝึก

**ใช้สิ่งที่เรียนแล้ว:** KNN/Pipeline จากบท 03, CV จากบท 08 และ overfitting จากบท 09

## ใจความใหม่

**Parameter** คือค่าที่เรียนจากข้อมูล เช่น coefficient; **hyperparameter** คือการตั้งค่าวิธีเรียน เช่น k ของ KNN

Grid search ลองทุกชุดที่กำหนด: k=3, 5, 9 × weights สองแบบ = 6 ชุด ใช้ 5-fold CV รวม 30 fits แล้ว refit ผู้ชนะบน train ทั้งหมดอีกครั้ง

- `uniform` ให้เพื่อนบ้านลงคะแนนเท่ากัน; `distance` ให้น้ำหนักตัวที่ใกล้มากกว่า
- `knn__n_neighbors` หมายถึง n_neighbors ของ step ชื่อ knn
- `mean_test_score` คือค่าเฉลี่ย validation ใน CV; `std_test_score` คือการกระจายระหว่าง fold
- `best_params_` คือค่าที่เลือก; `best_estimator_` คือ Pipeline ที่ refit แล้ว จึงเรียก `search.predict` ได้ทันที

Scaler อยู่ใน Pipeline ตลอดการค้นหา คะแนน best CV ถูกใช้เลือกค่าแล้วจึงอาจสูงเกินจริง ต้องประเมิน test หลังเลือกเสร็จ

## อ่าน learning curve

| รูปแบบคะแนนเมื่อเพิ่มข้อมูลฝึก | สิ่งที่ควรตรวจ |
|---|---|
| train และ validation ต่ำใกล้กัน | feature หรือความสามารถโมเดลยังไม่พอ |
| train สูง แต่ validation ต่ำกว่า | overfitting; ลองจำกัดความซับซ้อนหรือเพิ่มข้อมูล |
| validation สูงขึ้น | ข้อมูลเพิ่มอาจช่วย แต่ไม่รับประกันว่าจะดีขึ้นเสมอ |

แกนนอนนับข้อมูลฝึก **ต่อ fold** จึงจบที่ 96 จาก train 120 แถว กราฟนี้ใช้ค่าที่เลือกจาก train เดิม เป็นเครื่องมือวินิจฉัย ไม่ใช่ผลประเมินอิสระอีกชุด

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

<!-- CODE:START -->
```python
"""ปรับ hyperparameter ใน CV และดูว่าข้อมูลเพิ่มช่วยหรือไม่"""
# %% GridSearchCV รับ Pipeline ทั้งเส้น
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

# %% ใช้ learning curve เพื่อวินิจฉัยบน train เท่านั้น
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
<!-- CODE:END -->

## อ่านผล

อ่านตารางหกทางเลือก ตามด้วย best parameters และ learning curve สุดท้ายจึงดู test accuracy ภาพบันทึกที่ `outputs/10-tuning-and-learning-curves/learning-curve.png`

![ผลการทดลอง 10-tuning-and-learning-curves](../../assets/figures/10-tuning-and-learning-curves-learning-curve.png)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองคิด:** k ห้าค่า × weights สองค่า × CV สี่ fold ต้อง fit กี่ครั้ง?

**เฉลย:** 40 ครั้งสำหรับ CV และ refit อีก 1 ครั้ง ไม่ขยาย grid โดยเลือกจาก test ซ้ำ เพราะจะปรับเข้าหาความบังเอิญใน test

</details>

## อ้างอิง

- [Handbook: Hyperparameters and Model Validation](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html)
- [Tuning hyperparameters](https://scikit-learn.org/1.7/modules/grid_search.html)
- [Validation and learning curves](https://scikit-learn.org/1.7/modules/learning_curve.html)
