# %% Cell 1 — เทียบต้นไม้และป่าด้วย CV
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
# **อ่านโค้ด**
#
# - `make_classification` สร้าง 10 feature: 4 คอลัมน์มีข้อมูลสำหรับแยกคลาส และ 2 คอลัมน์เป็นค่าผสมจากสี่คอลัมน์นั้น ส่วน `flip_y=0.08` สุ่มกำหนดคลาสใหม่ให้บางแถว เพื่อจำลองข้อมูลที่มีการระบุคลาส (class label) คลาดเคลื่อน
# - `models` เก็บต้นไม้จำกัดความลึก 3 ชั้น ต้นไม้ไม่กำหนดเพดานความลึก และ Random Forest 120 ต้น
# - `cross_validate(..., return_train_score=True)` คืนทั้งคะแนนบนข้อมูลที่ใช้ฝึกและ validation ของแต่ละ fold
# - `result["train_score"]` ใช้ดูผลบนชุดฝึก ส่วน `result["test_score"]` ในคำสั่งนี้หมายถึง validation fold ไม่ใช่ final test
# - `scores[name]` เก็บค่าเฉลี่ย CV เพื่อให้ทุกโมเดลเลือกด้วยเกณฑ์เดียวกัน ส่วน `n_jobs=1` จำกัดการทำงานให้ใช้หนึ่งงานพร้อมกัน
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
# **อ่านโค้ด**
#
# - เลือกโมเดลจาก `scores` แล้ว fit บน train ทั้งหมดก่อนประเมิน test ตามวิธีในบท 08
# - `permutation_importance(..., n_repeats=8)` สุ่มสลับค่าภายในคอลัมน์ทีละคอลัมน์ แล้ววัด accuracy ของโมเดลเดิม ทำแปดรอบต่อคอลัมน์เพื่อลดผลจากการสุ่มครั้งเดียว โดยไม่ฝึกโมเดลใหม่
# - `importances_mean` และ `importances_std` คือค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของคะแนนที่ลดลงจากการสลับ
# - `np.argsort(...)[::-1][:5]` เรียง index ของ feature จาก importance มากไปน้อย แล้วเลือกห้าอันดับแรก
# - ผลนี้ใช้รายงานโมเดลที่เลือกแล้ว หากจะนำไปเลือก feature ต้องทำขั้นตอนนั้นบน validation
