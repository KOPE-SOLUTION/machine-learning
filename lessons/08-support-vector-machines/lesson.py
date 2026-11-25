# %% Cell 1 — สร้างข้อมูลและแยกชุดทดสอบ
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Training labels:", np.unique(y_train))

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 1</summary>
#
# ```text
# Train: (300, 2) Test: (100, 2)
# Training labels: [0 1]
# ```
#
# มีข้อมูลฝึก 300 แถวและข้อมูลทดสอบ 100 แถว แต่ละแถวมีสอง features ส่วน labels มีสองคลาสคือ 0 และ 1
#
# </details>
#
# **อ่านโค้ด**
#
# **1. สร้างข้อมูล:** `make_moons(n_samples=400, ...)` สร้างข้อมูลรูปพระจันทร์เสี้ยวสองกลุ่ม รวม 400 จุด แล้วคืนผลสองส่วนให้ตัวแปร:
#
# - `X` เก็บ **features**: พิกัดสองค่าต่อจุด จึงมีขนาด `(400, 2)`
# - `y` เก็บ **labels**: คลาสจริง 0 หรือ 1 จำนวน 400 ค่า โดยเรียงตรงกับแถวใน `X`
#
# `noise=0.25` เพิ่มค่ารบกวนสุ่มให้พิกัดของแต่ละจุด โดย 0.25 คือส่วนเบี่ยงเบนมาตรฐานของค่ารบกวน เมื่อเพิ่ม noise จุดจะกระจายออกจากแนวโค้งมากขึ้น
#
# **2. แบ่งข้อมูล:** `train_test_split(X, y, ...)` แบ่งแถวพร้อมจับคู่ features กับ label ให้ตรงกัน:
#
# - `X_train`, `y_train`: ชุดฝึก 300 แถว
# - `X_test`, `y_test`: ชุดทดสอบ 100 แถว
#
# `test_size=0.25` กัน 25% ของ 400 แถวไว้ทดสอบ ส่วน `stratify=y` รักษาสัดส่วนคลาสในการแบ่ง

# %% Cell 2 — สร้าง Pipeline ของ SVM สองแบบ
models = {
    "linear": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
    "rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
}
print("Prepared kernels:", list(models))

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 2</summary>
#
# ```text
# Prepared kernels: ['linear', 'rbf']
# ```
#
# เตรียม Pipeline ไว้สองแบบแล้ว ขั้นนี้ยังไม่ได้ฝึกโมเดล จึงยังไม่มีคะแนน Accuracy
#
# </details>
#
# **อ่านโค้ด**
#
# - `models` เก็บ Pipeline สองตัว แต่ละตัวเตรียมให้ข้อมูลผ่าน `StandardScaler` ก่อน `SVC` โดยตัวหนึ่งใช้ `kernel="linear"` อีกตัวใช้ `kernel="rbf"` ขั้นตอนเหล่านี้จะทำงานเมื่อเรียก `fit` หรือ `predict`
# - `C=1.0` เป็นค่าที่กำหนดไว้ให้ทั้งสองแบบ ฝั่ง RBF เพิ่ม `gamma="scale"` เพื่อให้ SVC คำนวณ gamma เมื่อเริ่ม fit โดยใช้ features **หลังผ่าน StandardScaler**
#
# <details>
# <summary>gamma="scale" คำนวณอย่างไร และต่างจาก StandardScaler อย่างไร?</summary>
#
# - **StandardScaler:** แปลงค่า features
# - **gamma="scale":** ให้ SVC คำนวณค่า gamma จากข้อมูลที่ได้รับหลังผ่าน StandardScaler
#
# `gamma="scale"` เป็นการคำนวณค่า gamma ตามสูตร ไม่ได้ปรับสเกล features ซ้ำหรือค้นหาค่า gamma ด้วย CV
#
# SVC ใช้สูตร `gamma = 1 / (จำนวน features × ความแปรปรวนของข้อมูลที่เข้า SVC)`
#
# ข้อมูลบทนี้มีสองคอลัมน์ที่ไม่ใช่ค่าคงที่ หลัง StandardScaler แปลงชุดฝึก แต่ละคอลัมน์มีค่าเฉลี่ยประมาณ 0 และความแปรปรวนประมาณ 1 จึงได้ `gamma ≈ 1 / (2 × 1) = 0.5` ในแต่ละรอบที่ fit
#
# </details>
#
# %% Cell 3 — ดูการแบ่งข้อมูลสำหรับ 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(X_train, y_train))

for fold_no, (fit_idx, valid_idx) in enumerate(folds, start=1):
    print(f"Fold {fold_no}: fit={len(fit_idx)} rows, validation={len(valid_idx)} rows")

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 3</summary>
#
# ```text
# Fold 1: fit=240 rows, validation=60 rows
# Fold 2: fit=240 rows, validation=60 rows
# Fold 3: fit=240 rows, validation=60 rows
# Fold 4: fit=240 rows, validation=60 rows
# Fold 5: fit=240 rows, validation=60 rows
# ```
#
# แต่ละรอบมีจำนวนแถวเท่ากัน แต่สลับส่วนที่ใช้เป็น validation ตามภาพ 5-fold CV
#
# </details>
#
# **อ่านโค้ด**
#
# - `StratifiedKFold(n_splits=5)` กำหนดให้แบ่งชุดฝึกเป็นห้า fold โดยรักษาสัดส่วนคลาส ส่วน `shuffle=True` สุ่มตำแหน่งแถวก่อนจัด fold และ `random_state=42` ทำให้แบ่งข้อมูลชุดเดิมได้เหมือนเดิมเมื่อรันซ้ำ
# - `cv.split(X_train, y_train)` คืนคู่ **ตำแหน่งแถวสำหรับฝึก** กับ **ตำแหน่งแถวสำหรับ validation** ในแต่ละรอบ ไม่ได้คืนโมเดลหรือคะแนน
# - `list(...)` เก็บคู่ตำแหน่งครบห้ารอบไว้ใน `folds` เพื่อให้ Linear และ RBF ใช้การแบ่งชุดเดียวกัน
# - `fit_idx` และ `valid_idx` เป็นตำแหน่งแถวภายใน `X_train` กับ `y_train` ส่วน test 100 แถวยังไม่ถูกนำมาใช้<br>ขั้นนี้ยังไม่ได้ฝึกโมเดล
#
# %% Cell 4 — ฝึกด้วย CV และอ่านคะแนนทีละ Fold
fold_scores = {}
for name, candidate in models.items():
    fold_scores[name] = cross_val_score(candidate, X_train, y_train, cv=folds, scoring="accuracy")

fold_table = pd.DataFrame(fold_scores, index=range(1, len(folds) + 1))
fold_table.index.name = "Fold"
print("Validation accuracy by fold:")
print(fold_table.round(3).to_string())

scores = {}
for name, values in fold_scores.items():
    scores[name] = values.mean()
    print(f"{name} CV accuracy: {values.mean():.3f} +/- {values.std():.3f}")

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 4</summary>
#
# ```text
# Validation accuracy by fold:
#       linear    rbf
# Fold
# 1      0.717  0.900
# 2      0.800  0.833
# 3      0.850  0.950
# 4      0.833  0.950
# 5      0.817  0.900
# linear CV accuracy: 0.803 +/- 0.046
# rbf CV accuracy: 0.907 +/- 0.043
# ```
#
# RBF ได้ค่าเฉลี่ย CV ประมาณ 0.907 สูงกว่า Linear ที่ประมาณ 0.803 คะแนนเหล่านี้คำนวณจาก validation ในแต่ละ fold
#
# </details>
#
# **อ่านโค้ด**
#
# - `cross_val_score` รับ Pipeline ที่จะเปรียบเทียบ และใช้ `cv=folds` แบ่งตามตำแหน่งที่เตรียมไว้<br>แต่ละรอบฝึก **สำเนา Pipeline ใหม่** บน 240 แถว แล้วทำนาย validation 60 แถวก่อนคำนวณ Accuracy
# - `scoring="accuracy"` กำหนดตัวชี้วัด; ผลที่คืนคือคะแนน validation ห้าค่า เก็บใน `fold_scores[name]` ตามชื่อ kernel
# - `fold_table` วางคะแนนรอบเดียวกันไว้แถวเดียวกัน เช่น แถว Fold 1 เปรียบเทียบ Linear กับ RBF บน validation 60 แถวชุดเดียวกัน
# - `values.mean()` เฉลี่ยคะแนน validation ทั้งห้ารอบและเก็บใน `scores[name]` ส่วน `values.std()` คือส่วนเบี่ยงเบนมาตรฐานของห้าคะแนน ค่ายิ่งมาก คะแนนแต่ละรอบยิ่งต่างกันมากขึ้น ตัวเลขหลัง `+/-` จึงบอกการกระจายของคะแนน ไม่ใช่ช่วงความเชื่อมั่น
#
# **ทำไมยังต้อง fit อีก:** `cross_val_score` ฝึกสำเนา Pipeline เพื่อเก็บคะแนน แต่ไม่ได้ส่ง Pipeline ที่ฝึกแล้วกลับมาแทนตัวเดิมใน `models` ดังนั้น Pipeline ใน `models` จึงยังไม่ได้ fit
#
# เมื่อเลือกการตั้งค่าแล้ว จึงฝึก Pipeline ที่จะใช้งานจริงด้วย train ครบ 300 แถว
#
# **ทำไมใช้ Accuracy:** ข้อมูลนี้มีจำนวนแถวของคลาส 0 และ 1 เท่ากัน จึงเริ่มเปรียบเทียบจากสัดส่วนที่ทายถูกได้ หากคลาสไม่สมดุล หรือการทายผิดแต่ละแบบมีผลต่างกัน ให้พิจารณา Precision, Recall และ F1 ที่เรียนในบท 04 ด้วย
#
# เรากำลังเลือกจากสองชุดการตั้งค่าที่เตรียมไว้ใน `models` ยังไม่ได้ค้นหา C/gamma หลายค่า
#
# %% Cell 5 — ฝึกโมเดลที่เลือกแล้วประเมินบน Test
chosen = max(scores, key=scores.get)
model = models[chosen]
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Selected kernel:", chosen)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Correct predictions: {(y_pred == y_test).sum()}/{len(y_test)}")

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 5</summary>
#
# ```text
# Selected kernel: rbf
# Test accuracy: 0.950
# Correct predictions: 95/100
# ```
#
# เลือก RBF จากค่าเฉลี่ย CV แล้วฝึกใหม่ด้วย train ทั้ง 300 แถว ผลบน test คือทายถูก 95 จาก 100 แถว
#
# </details>
#
# **อ่านโค้ด**
#
# - `max(scores, key=scores.get)` เลือก **ชื่อ kernel** ที่มี CV accuracy เฉลี่ยสูงสุด เช่น `"rbf"`; ไม่ได้เลือกจาก test
# - `model.fit(X_train, y_train)` ฝึก Pipeline ที่เลือกด้วยชุดฝึกครบ 300 แถว โดย scaler หาสถิติและแปลง features จาก 300 แถวนี้ก่อน แล้ว SVC จึงฝึกจาก features ที่แปลงแล้วคู่กับ `y_train`
# - `model.predict(X_test)` ให้ scaler แปลง test ด้วยค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานที่เก็บจาก train 300 แถว แล้ว SVC จึงทำนาย โดยไม่ fit ทั้ง scaler และ SVC ใหม่จาก test
# - `accuracy_score` เปรียบเทียบคำทำนายกับ `y_test` ส่วนบรรทัดสุดท้ายแสดงจำนวนที่ทายถูกจาก test 100 แถว

# %% Cell 6 — อ่านแนวแบ่งคลาสและ Support vectors
colors = ListedColormap(["#1558D6", "#BD5A00"])
fig, ax = plt.subplots(figsize=(7, 4.5))
DecisionBoundaryDisplay.from_estimator(
    model, X_train, response_method="predict", grid_resolution=250,
    cmap=colors, alpha=0.18, ax=ax
)
DecisionBoundaryDisplay.from_estimator(
    model, X_train, response_method="decision_function", grid_resolution=250,
    plot_method="contour", levels=[0], colors="black", linewidths=1.5, ax=ax
)
points = ax.scatter(
    X_train[:, 0], X_train[:, 1], c=y_train, cmap=colors,
    edgecolors="white", linewidths=0.4, s=25
)

support_idx = model.named_steps["svc"].support_
support_points = X_train[support_idx]
ax.scatter(
    support_points[:, 0], support_points[:, 1], s=85,
    facecolors="none", edgecolors="black", linewidths=0.8, label="Support vectors"
)
ax.set(xlabel="Feature 1", ylabel="Feature 2", title=f"Training data: {chosen} SVM")
ax.legend(loc="upper right")
fig.colorbar(points, ax=ax, ticks=[0, 1], label="Class")
fig.tight_layout()

out = Path("outputs/08-support-vector-machines")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "boundary.png", dpi=140)
plt.show()
plt.close(fig)
print("Saved:", (out / "boundary.png").resolve())

# %% [markdown]
# <details>
# <summary>ดูผลลัพธ์ Cell 6</summary>
#
# ![กราฟจาก Cell 6 แสดงแนวแบ่ง RBF จุดฝึก และวงแหวน support vectors](../../assets/figures/08-support-vector-machines-boundary.png)
#
# ```text
# Saved: <โฟลเดอร์ที่ Notebook ทำงาน>/outputs/08-support-vector-machines/boundary.png
# ```
#
# ส่วน `<โฟลเดอร์ที่ Notebook ทำงาน>` จะแสดงเป็นเส้นทางจริงในเครื่องที่รัน
#
# </details>
#
# **อ่านโค้ด**
#
# - `DecisionBoundaryDisplay.from_estimator` ครั้งแรกใช้ `response_method="predict"` ให้ Pipeline ทำนายจุดทั่วพื้นที่กราฟ แล้วระบายพื้นตามคลาสที่ได้
# - การเรียกด้วย `response_method="decision_function"` ใช้คะแนนตัดสินใจวาดแนวแบ่ง สำหรับสองคลาสในบทนี้ คะแนนติดลบทำนายคลาส 0 ส่วนคะแนนบวกทำนายคลาส 1<br>`levels=[0]` วาดเส้นทึบที่คะแนนเท่ากับศูนย์ ซึ่งเป็นรอยแบ่งระหว่างสองคลาส คะแนนตัดสินใจนี้ไม่ใช่ความน่าจะเป็น
# - สีของจุดมาจาก label จริง `y_train` ส่วนสีพื้นมาจากคลาสที่โมเดลทำนาย ไม่ใช่ระดับความมั่นใจ
# - `model.named_steps["svc"]` เข้าถึง SVC ภายใน Pipeline; `support_` คือ **ตำแหน่งแถวของ support vectors ในข้อมูลที่ใช้ fit** จึงนำไปเลือกพิกัดเดิมจาก `X_train` แล้ววาดวงแหวนเพื่อระบุจุดฝึกที่ใช้กำหนดแนวแบ่ง
# - **อ่านผลทำนายจากสี:** เทียบสีจุดกับสีพื้นรอบจุด<br>สีตรงกัน → โมเดลทายถูก / สีต่างกัน → โมเดลทายผิด
# - ไฟล์ `boundary.png` บันทึกใน `outputs/08-support-vector-machines` ใต้โฟลเดอร์ที่ Notebook ทำงาน บรรทัด `Saved:` แสดงตำแหน่งเต็ม หากเปิด Notebook ใน `practics` ไฟล์จะอยู่ที่ `practics/outputs/08-support-vector-machines/boundary.png`
