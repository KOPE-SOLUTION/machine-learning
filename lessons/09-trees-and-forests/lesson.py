# %% Cell 1 — เตรียมข้อมูลและกันชุดทดสอบ
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train counts [class 0, class 1]:", np.bincount(y_train))

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 1</summary>
#
# ```text
# Train: (600, 10) Test: (200, 10)
# Train counts [class 0, class 1]: [303 297]
# ```
#
# มีชุดฝึก 600 แถวและชุดทดสอบ 200 แถว แต่ละแถวมี features 10 ค่า
#
# </details>
#
# **อ่านโค้ด**
#
# - `make_classification` สร้างโจทย์จำแนกคลาสจำลอง: 4 คอลัมน์ใช้สร้างความแตกต่างระหว่างคลาส, 2 คอลัมน์เป็นค่าผสมเชิงเส้นจากสี่คอลัมน์นั้น และอีก 4 คอลัมน์เป็นค่าสุ่มที่ไม่ใช้สร้างคลาส
# - คำสั่งสลับลำดับคอลัมน์ให้ด้วย จึงไม่ควรสรุปว่า `feature_0` ถึง `feature_3` เป็นสี่คอลัมน์แรกที่ใช้สร้างคลาส
# - `flip_y=0.08` เลือกบางแถวมาสุ่ม label ใหม่ เพื่อจำลอง labels ที่คลาดเคลื่อน<br>สัดส่วนที่เปลี่ยนคลาสจริงอาจต่ำกว่า 8% เพราะสุ่มแล้วอาจได้คลาสเดิม
# - `test_size=0.25` กัน 200 แถวไว้ทดสอบ; อีก 600 แถวใช้ทำ CV และฝึกโมเดลที่เลือก
#
# %% Cell 2 — เตรียมต้นไม้ตื้น ต้นไม้ลึก และ Random Forest
models = {
    "shallow_tree": DecisionTreeClassifier(
        criterion="gini", max_depth=3, random_state=42
    ),
    "deep_tree": DecisionTreeClassifier(
        criterion="gini", max_depth=None, random_state=42
    ),
    "forest": RandomForestClassifier(
        n_estimators=120, criterion="gini", min_samples_leaf=2,
        max_features="sqrt", bootstrap=True, random_state=42, n_jobs=1,
    ),
}
print("Prepared models:", list(models))

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 2</summary>
#
# ```text
# Prepared models: ['shallow_tree', 'deep_tree', 'forest']
# ```
#
# ขั้นนี้สร้างตัวเลือกสามแบบ ยังไม่ได้ fit จึงยังไม่มีคะแนนประเมิน
#
# </details>
#
# **อ่านโค้ด**
#
# - `criterion="gini"` ให้ต้นไม้เลือกเงื่อนไขด้วย Gini ตามตัวอย่างก่อนเริ่มโค้ด
# - `max_depth=3` ให้เดินจากรากถึงใบได้ไม่เกินสามขั้น<br>ส่วน `None` ไม่กำหนดเพดานความลึก แต่ยังหยุดได้เมื่อแบ่งต่อไม่ได้หรือคลาสในจุดนั้นเหมือนกันหมด
# - `n_estimators=120` คือจำนวนต้นไม้ใน forest; `min_samples_leaf=2` กำหนดจำนวนแถวขั้นต่ำในแต่ละใบ
# - `bootstrap=True` สุ่มแถวแบบคืนกลับก่อนฝึกแต่ละต้น<br>ส่วน `max_features="sqrt"` กำหนดจำนวนคอลัมน์ที่สุ่มมาพิจารณาในแต่ละจุดแบ่ง<br>เมื่อมี 10 features จะเริ่มพิจารณา `int(sqrt(10)) = 3` คอลัมน์ แล้วเลือกเงื่อนไขที่ดีที่สุดจากคอลัมน์เหล่านั้น
# - `random_state=42` ช่วยให้ทดลองซ้ำได้; `n_jobs=1` ให้ forest ประมวลผลทีละงาน
#
# <details>
# <summary>จำนวน features และจำนวนแถวใน Random Forest</summary>
#
# หากคอลัมน์ที่สุ่มมายังหาเงื่อนไขแบ่งที่ใช้ได้ไม่พบ scikit-learn อาจตรวจเพิ่มเกิน `max_features` ได้
#
# `min_samples_leaf` ตรวจจำนวนแถวฝึกที่ต่างกันในใบ<br>ส่วนการนับสัดส่วนคลาสและ Gini คำนึงถึงจำนวนครั้งที่แต่ละแถวถูกสุ่มซ้ำด้วย
#
# ในบทนี้ไม่ได้กำหนด `class_weight` หรือ `sample_weight` เพิ่ม
#
# </details>
#
# %% Cell 3 — เตรียม CV ห้ารอบบนชุดฝึก
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(X_train, y_train))
for fold, (fit_idx, valid_idx) in enumerate(folds, start=1):
    print(f"Fold {fold}: fit={len(fit_idx)} rows, validation={len(valid_idx)} rows")

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 3</summary>
#
# ```text
# Fold 1: fit=480 rows, validation=120 rows
# Fold 2: fit=480 rows, validation=120 rows
# Fold 3: fit=480 rows, validation=120 rows
# Fold 4: fit=480 rows, validation=120 rows
# Fold 5: fit=480 rows, validation=120 rows
# ```
#
# แต่ละรอบใช้ 480 แถวฝึก และอีก 120 แถวตรวจผล ส่วน test 200 แถวยังแยกไว้
#
# </details>
#
# **อ่านโค้ด**
#
# - ใช้ `StratifiedKFold` เหมือน EP8 เพื่อรักษาสัดส่วนคลาสในแต่ละ fold
# - `folds` เก็บตำแหน่งแถวไว้ให้ทั้งสามโมเดลใช้การแบ่งเดียวกัน จึงเปรียบเทียบจากข้อมูลชุดเดียวกันในแต่ละรอบ
#
# %% Cell 4 — เปรียบเทียบคะแนน Train กับ CV
scores = {}
cv_by_fold = {}
rows = []
for name, candidate in models.items():
    result = cross_validate(
        candidate, X_train, y_train, cv=folds,
        scoring="accuracy", return_train_score=True,
    )
    train_mean = result["train_score"].mean()
    cv_mean = result["test_score"].mean()
    scores[name] = cv_mean
    cv_by_fold[name] = result["test_score"]
    rows.append({
        "model": name, "train_accuracy": train_mean,
        "cv_accuracy": cv_mean, "cv_std": result["test_score"].std(),
        "train_cv_gap": train_mean - cv_mean,
    })

fold_table = pd.DataFrame(cv_by_fold, index=range(1, 6))
fold_table.index.name = "Fold"
summary = pd.DataFrame(rows).set_index("model")
print("Validation accuracy by fold:")
print(fold_table.round(3).to_string())
print("\nMean scores and train-CV gap:")
print(summary.round(3).to_string())

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 4</summary>
#
# ```text
# Validation accuracy by fold:
#       shallow_tree  deep_tree  forest
# Fold
# 1            0.792      0.750   0.792
# 2            0.775      0.717   0.875
# 3            0.733      0.742   0.800
# 4            0.817      0.783   0.817
# 5            0.758      0.692   0.792
#
# Mean scores and train-CV gap:
#               train_accuracy  cv_accuracy  cv_std  train_cv_gap
# model
# shallow_tree           0.832        0.775   0.028         0.057
# deep_tree              1.000        0.737   0.031         0.263
# forest                 0.989        0.815   0.031         0.174
# ```
#
# ต้นไม้ลึกได้ train 1.000 แต่ CV ประมาณ 0.737 จึงมีสัญญาณ overfitting ชัดเจน ส่วน forest ได้ CV สูงสุดในสามตัวเลือก
#
# </details>
#
# **อ่านโค้ด**
#
# - EP8 ใช้ `cross_val_score` เพื่อเก็บคะแนน validation; บทนี้ใช้ `cross_validate(..., return_train_score=True)` เพื่อเก็บคะแนน train เพิ่มด้วย
# - แต่ละรอบฝึก **สำเนาโมเดลใหม่** บน 480 แถว แล้ววัดทั้ง train 480 แถวและ validation 120 แถว
# - `result["test_score"]` เป็นชื่อ key ของ scikit-learn; ในคำสั่งนี้หมายถึง **คะแนน validation ของแต่ละ fold** ไม่ใช่คะแนนจาก `X_test`
# - `train_cv_gap` คือค่าเฉลี่ย train ลบค่าเฉลี่ย CV; ช่องว่างมากเป็นสัญญาณให้ตรวจ overfitting แต่การเลือกโมเดลใช้ **CV accuracy สูงสุด** ไม่ใช่ช่องว่างน้อยที่สุด
# - `cv_std` บอกการกระจายของคะแนน validation ห้ารอบ ไม่ใช่ช่วงความเชื่อมั่น
#
# %% Cell 5 — ฝึกโมเดลที่เลือกและประเมินบน Test
chosen = max(scores, key=scores.get)
model = models[chosen]
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Selected:", chosen)
print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"Correct predictions: {(pred == y_test).sum()}/{len(y_test)}")

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 5</summary>
#
# ```text
# Selected: forest
# Test accuracy: 0.820
# Correct predictions: 164/200
# ```
#
# </details>
#
# **อ่านโค้ด**
#
# - `max(scores, key=scores.get)` เลือกชื่อโมเดลจาก CV โดยยังไม่ใช้ผลจาก test
# - CV ฝึกสำเนาสำหรับตรวจคะแนน ตัวโมเดลใน `models` จึงยังไม่ได้ fit; เมื่อเลือกแล้วต้องฝึกด้วย train ครบ 600 แถว
# - `predict(X_test)` ทำนาย 200 แถวที่กันไว้ แล้วใช้ `y_test` ตรวจว่าทายถูกกี่แถว
#
# %% Cell 6 — ตรวจการพึ่งพา Features ด้วย Permutation importance
importance = permutation_importance(
    model, X_test, y_test, scoring="accuracy", n_repeats=8, random_state=42,
)
order = np.argsort(importance.importances_mean)[::-1][:5]
for i in order:
    print(
        f"feature_{i}: accuracy drop={importance.importances_mean[i]:.3f}"
        f" +/- {importance.importances_std[i]:.3f}"
    )

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(
    [f"feature_{i}" for i in order], importance.importances_mean[order],
    xerr=importance.importances_std[order], color="tab:blue", capsize=4,
)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=1)
ax.set(xlabel="Mean decrease in accuracy", title="Permutation importance on test")
fig.tight_layout()
out = Path("outputs/09-trees-and-forests")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "permutation-importance.png", dpi=150)
plt.show()
print("Saved:", (out / "permutation-importance.png").resolve())

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 6</summary>
#
# ```text
# feature_1: accuracy drop=0.149 +/- 0.019
# feature_9: accuracy drop=0.074 +/- 0.022
# feature_8: accuracy drop=0.058 +/- 0.015
# feature_0: accuracy drop=0.015 +/- 0.010
# feature_7: accuracy drop=0.015 +/- 0.007
# Saved: <โฟลเดอร์ที่ Notebook ทำงาน>/outputs/09-trees-and-forests/permutation-importance.png
# ```
#
# ![กราฟจาก Cell 6 แสดงค่าเฉลี่ย accuracy ที่ลดลงเมื่อสลับแต่ละคอลัมน์ พร้อมส่วนเบี่ยงเบนมาตรฐานจากการสลับแปดรอบ](../../assets/figures/09-trees-and-forests-importance-output.png)
#
# `feature_1` มีค่าเฉลี่ย accuracy drop ประมาณ 0.149 หรือ **14.9 จุดเปอร์เซ็นต์** จึงเป็นคอลัมน์ที่โมเดลนี้พึ่งพามากที่สุดจากผลที่ตรวจ
#
# </details>
#
# **อ่านโค้ด**
#
# - `permutation_importance` ใช้โมเดลที่ฝึกแล้ว สลับค่าทีละคอลัมน์แล้วทำนายใหม่ โดยคอลัมน์อื่นและ labels อยู่ตามแถวเดิม **ไม่มีการ fit ซ้ำ**
# - `n_repeats=8` ทำแปดรอบต่อคอลัมน์; `importances_mean` เฉลี่ยผลต่าง `accuracy ก่อนสลับ − accuracy หลังสลับ`
# - `importances_std` คือส่วนเบี่ยงเบนมาตรฐานของผลต่างแปดรอบ แสดงด้วยเส้นแนวนอนสีดำที่ปลายแท่ง<br>ความยาวแท่งบอกค่าเฉลี่ย ส่วนเส้นดำบอกการกระจาย ไม่ใช่ช่วงความเชื่อมั่น
# - `np.argsort(...)[::-1][:5]` เรียงตำแหน่งคอลัมน์ตาม importance จากมากไปน้อย แล้วเลือกห้าคอลัมน์แรก
# - `Saved:` แสดงตำแหน่งเต็มของภาพ<br>หากเปิด Notebook ใน `practics` จะบันทึกที่ `practics/outputs/09-trees-and-forests/permutation-importance.png`
