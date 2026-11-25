# %% Cell 1 — ค้นค่าที่เหมาะด้วย GridSearchCV
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

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 1</summary>
#
# ```text
#  param_knn__n_neighbors param_knn__weights  mean_test_score  std_test_score  rank_test_score
#                       3            uniform            0.967           0.017                1
#                       3           distance            0.967           0.017                1
#                       5            uniform            0.958           0.026                3
#                       5           distance            0.958           0.026                3
#                       9            uniform            0.958           0.026                3
#                       9           distance            0.958           0.026                3
# Best parameters: {'knn__n_neighbors': 3, 'knn__weights': 'uniform'}
# Best CV accuracy: 0.967
# ```
#
# k=3 ได้คะแนน CV สูงสุดเท่ากันทั้งสอง weights; GridSearchCV เลือกชุดแรกในลำดับที่ลอง คือ uniform
#
# คะแนน 0.967 เป็น CV accuracy
#
# </details>
#
# **อ่านโค้ด**
#
# - `load_iris(return_X_y=True)` คืน X และ y โดยตรง แทนการคืนชุดข้อมูลพร้อมรายละเอียดแบบบท 02
# - `Pipeline([("scale", ...), ("knn", ...)])` ตั้งชื่อแต่ละขั้นเอง เพื่อระบุว่า GridSearchCV ต้องปรับการตั้งค่าของขั้นใด
# - `knn__n_neighbors` และ `knn__weights` ใช้รูปแบบ `ชื่อขั้น__ชื่อพารามิเตอร์` ให้ GridSearchCV รู้ว่าต้องปรับค่าของ KNN
# - `search.fit(X_train, y_train)` ลองหกชุดค่า แต่ละชุดฝึกและวัด validation ห้ารอบ แล้วเลือกชุดที่มีคะแนนเฉลี่ยสูงสุด<br>จากนั้นฝึกใหม่ด้วย train ทั้งหมดโดยอัตโนมัติ<br>ขั้นตอนฝึกใหม่นี้เรียกว่า **refit**
# - `pd.DataFrame(search.cv_results_)` ทำผลค้นหาเป็นตาราง แล้วเรียงด้วย `rank_test_score` เพื่ออ่านลำดับจากคะแนน CV
# - `best_params_` บอกชุดค่าที่เลือก และ `best_score_` เป็นคะแนน CV เฉลี่ยของชุดนั้น ไม่ใช่ final test
#
# **ชื่อที่ใช้ในตารางผล:** `mean_test_score` คือ accuracy เฉลี่ยของ validation ใน CV<br>ส่วน `std_test_score` คือส่วนเบี่ยงเบนมาตรฐานระหว่าง fold ไม่ใช่คะแนน final test
#
# `weights="uniform"` ให้เพื่อนบ้านลงคะแนนเท่ากัน ส่วน `"distance"` ให้จุดใกล้มีน้ำหนักมากกว่า
#
# %% Cell 2 — ดูผลของจำนวนข้อมูลฝึก
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

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 2</summary>
#
# ```text
# Final test accuracy: 0.933
# ```
#
# ![กราฟ accuracy บน train และ validation เมื่อเพิ่มจำนวนตัวอย่างฝึก](../../assets/figures/10-tuning-and-learning-curves-learning-curve.png)
#
# กราฟเทียบ train กับ validation ตามจำนวนแถวที่ใช้ฝึกต่อ fold ส่วน 0.933 เป็น accuracy บน test ที่กันไว้
#
# </details>
#
# **อ่านโค้ด**
#
# - `learning_curve(search.best_estimator_, ...)` ใช้ค่า k และ weights ที่ GridSearchCV เลือก แล้วสร้าง Pipeline แยกเพื่อ fit ใหม่สำหรับแต่ละขนาดข้อมูลฝึกและแต่ละ fold<br>กราฟจึงแสดงผลของจำนวนข้อมูลที่ใช้ฝึกจริง
# - `train_sizes=[0.4, 0.6, 0.8, 1.0]` เป็นสัดส่วนของข้อมูลฝึกที่ใช้ได้ในแต่ละ fold ระดับ 1.0 จึงเท่ากับ 96 แถวในบทนี้
# - `sizes` คืนจำนวนแถวจริง ส่วน `train_scores` และ `valid_scores` มีหนึ่งแถวต่อขนาดข้อมูล และหนึ่งคอลัมน์ต่อ fold
# - `mean(axis=1)` เฉลี่ยคะแนนข้าม fold เพื่อวาดหนึ่งจุดต่อขนาดข้อมูล
# - `search.predict(X_test)` ใช้ Pipeline ที่ GridSearchCV ฝึกด้วยค่าที่เลือกบน train ทั้ง 120 แถว เพื่อทำนาย test<br>ส่วน Pipeline ที่ฝึกด้วยข้อมูลขนาดต่าง ๆ สำหรับ learning curve ใช้สร้างกราฟเท่านั้น
#
# **ที่มาของ 96 แถวบนแกนนอน:** train ทั้งหมดมี 120 แถว แต่แต่ละรอบของ 5-fold CV ใช้ฝึกเพียง 4 ส่วน จึงมีสูงสุด `120 × 4/5 = 96` แถว<br>ส่วนอีก 24 แถวใช้ validation
