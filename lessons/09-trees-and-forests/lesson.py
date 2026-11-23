# %% Cell 1 — เปรียบเทียบ Decision Tree กับ Random Forest ด้วย CV
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(
    n_samples=800, n_features=10, n_informative=4, n_redundant=2,
    flip_y=0.08, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
models = {
    "shallow_tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "deep_tree": DecisionTreeClassifier(random_state=42),
    "forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=1),
}
scores = {}
for name, model in models.items():
    result = cross_validate(model, X_train, y_train, cv=cv, scoring="accuracy", return_train_score=True)
    scores[name] = result["test_score"].mean()
    print(f"{name}: train={result['train_score'].mean():.3f}, CV={scores[name]:.3f}")

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 1</summary>
#
# ```text
# shallow_tree: train=0.832, CV=0.775
# deep_tree: train=1.000, CV=0.737
# forest: train=0.989, CV=0.815
# ```
#
# ต้นไม้ไม่จำกัดความลึกได้ train 1.000 แต่ CV 0.737 ส่วน Random Forest ได้ CV สูงสุดในสามตัวเลือก จึงเลือกไปฝึกกับ train ทั้งหมด
#
# </details>
#
# **อ่านโค้ด**
#
# - `make_classification` สร้าง 10 feature: 4 คอลัมน์มีข้อมูลสำหรับแยกคลาส 2 คอลัมน์เป็นค่าผสมจากสี่คอลัมน์นั้น และอีก 4 คอลัมน์เป็นค่าที่สุ่มโดยไม่ใช้กำหนดคลาส ส่วน `flip_y=0.08` สุ่มกำหนดคลาสใหม่ให้บางแถว เพื่อจำลองข้อมูลที่มีการระบุคลาส (class label) คลาดเคลื่อน
# - `models` เก็บ Decision Tree ที่แบ่งได้ไม่เกิน 3 ขั้นจากราก, Decision Tree ที่ไม่กำหนดเพดานความลึก และ Random Forest 120 ต้น
# - `cross_validate(..., return_train_score=True)` คืนทั้งคะแนนบนข้อมูลที่ใช้ฝึกและ validation ของแต่ละ fold
# - `result["train_score"]` ใช้ดูผลบนชุดฝึก ส่วน `result["test_score"]` ในคำสั่งนี้หมายถึง validation fold ไม่ใช่ final test
# - `scores[name]` เก็บ accuracy เฉลี่ยจาก CV แล้วเลือกโมเดลที่ได้ค่าเฉลี่ยสูงที่สุด ส่วน `n_jobs=1` ให้ประมวลผลทีละงาน
#
# **การตั้งค่าใน cell นี้:** `max_depth=3` จำกัดความลึกของต้นแรก ส่วน Random Forest ใช้ `n_estimators=120` เป็นต้นไม้ 120 ต้น และ `min_samples_leaf=2` ให้แต่ละใบมีตัวอย่างอย่างน้อย 2 แถว

# %% Cell 2 — ตรวจ feature ของโมเดลที่เลือก
chosen = max(scores, key=scores.get)
model = models[chosen].fit(X_train, y_train)
print("Selected:", chosen, "test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))
importance = permutation_importance(model, X_test, y_test, scoring="accuracy", n_repeats=8, random_state=42)
order = np.argsort(importance.importances_mean)[::-1][:5]
for i in order:
    print(f"feature_{i}: accuracy drop={importance.importances_mean[i]:.3f} +/- {importance.importances_std[i]:.3f}")

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 2</summary>
#
# ```text
# Selected: forest test accuracy: 0.82
# feature_1: accuracy drop=0.149 +/- 0.019
# feature_9: accuracy drop=0.074 +/- 0.022
# feature_8: accuracy drop=0.058 +/- 0.015
# feature_0: accuracy drop=0.015 +/- 0.010
# feature_7: accuracy drop=0.015 +/- 0.007
# ```
#
# Random Forest ได้ test accuracy 0.82 ส่วน accuracy drop 0.149 ของ feature_1 หมายถึงคะแนนลดลงเฉลี่ย 14.9 จุดเปอร์เซ็นต์เมื่อสุ่มสลับคอลัมน์นี้
#
# </details>
#
# **อ่านโค้ด**
#
# - เลือกโมเดลจาก `scores` แล้ว fit บน train ทั้งหมดก่อนประเมิน test ตามวิธีในบท 08
# - `permutation_importance(..., n_repeats=8)` สุ่มสลับค่าภายในคอลัมน์ทีละคอลัมน์ แล้ววัด accuracy ของโมเดลเดิม ทำแปดรอบต่อคอลัมน์เพื่อลดผลจากการสุ่มครั้งเดียว โดยไม่ฝึกโมเดลใหม่
# - `importances_mean` และ `importances_std` คือค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานจากการสลับ 8 รอบของแต่ละคอลัมน์ โดยแต่ละรอบคำนวณ `accuracy ก่อนสลับ − accuracy หลังสลับ` ค่าติดลบหมายถึงรอบนั้นสลับแล้ว accuracy สูงขึ้น
# - `np.argsort(...)[::-1][:5]` เรียง index ของ feature จาก importance มากไปน้อย แล้วเลือกห้าอันดับแรก
# - ค่า permutation importance บน test ใช้รายงานการพึ่งพาแต่ละ feature ของโมเดลที่เลือกแล้ว หากต้องการใช้ importance เลือก feature ให้คำนวณและเลือกบน validation
