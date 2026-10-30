# %% Cell 1 — เลือก kernel จากคะแนน CV
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {kernel: make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0))
          for kernel in ("linear", "rbf")}
scores = {}
for name, model in models.items():
    values = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    scores[name] = values.mean()
    print(f"{name} CV accuracy: {values.mean():.3f} +/- {values.std():.3f}")
chosen = max(scores, key=scores.get)
model = models[chosen].fit(X_train, y_train)
print("Selected:", chosen)
print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")

# %% [markdown]
# **อ่านโค้ด**
#
# - `make_moons(n_samples=400, noise=0.25, ...)` สร้างข้อมูลสองคลาสรูปโค้ง แล้วแบ่งเป็น train 300 แถวและ test 100 แถว
# - `StratifiedKFold(n_splits=5, shuffle=True, ...)` กำหนดการแบ่ง train เป็นห้าส่วน โดยสลับแถวและรักษาสัดส่วนคลาส
# - dictionary `models` สร้าง Pipeline สองตัวจากค่า kernel `linear` และ `rbf` โดยใช้ scaler และ C แบบเดียวกัน
# - `cross_val_score(..., cv=cv, scoring="accuracy")` คืน accuracy ของ validation ห้ารอบ ใช้ `mean()` หาคะแนนเฉลี่ย และ `std()` ดูว่าคะแนนต่างกันระหว่างรอบมากเพียงใด
# - `max(scores, key=scores.get)` เลือกชื่อ kernel ที่มีค่าเฉลี่ยสูงสุด แล้ว `fit(X_train, y_train)` ฝึกตัวที่เลือกบน train ทั้งหมดก่อนวัด test
#
# **การตั้งค่าใน cell นี้**
#
# - `kernel="linear"` ใช้แนวแบ่งตรง ส่วน `kernel="rbf"` ใช้ความคล้ายตามระยะทางเพื่อสร้างแนวแบ่งโค้ง
# - ทั้งสองแบบตรึง `C=1.0`; ฝั่ง RBF ใช้ `gamma="scale"` ซึ่งเป็นค่าเริ่มต้น จึงเปรียบเทียบเฉพาะ kernel
# - ค่า `+/-` ใน output คือส่วนเบี่ยงเบนมาตรฐานของ accuracy ระหว่าง fold ไม่ใช่ช่วงความเชื่อมั่น

# %% Cell 2 — วาดขอบเขตของโมเดลที่เลือก
fig, ax = plt.subplots(figsize=(6, 4))
DecisionBoundaryDisplay.from_estimator(model, X_train, response_method="predict", alpha=0.3, ax=ax)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k", s=20)
ax.set(xlabel="Feature 1", ylabel="Feature 2", title=f"Training data: {chosen} SVM")
fig.tight_layout()
out = Path("outputs/08-support-vector-machines")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "boundary.png", dpi=140)
plt.show()
plt.close(fig)

# %% [markdown]
# **อ่านโค้ด**
#
# - `DecisionBoundaryDisplay.from_estimator` สร้างจุดบนพื้นที่กราฟ แล้วให้โมเดลที่ฝึกแล้วทำนายเพื่อระบายบริเวณของแต่ละคลาส
# - `response_method="predict"` ใช้ผลทำนายคลาส และส่ง Pipeline ทั้งชุดเข้าไป จึงใช้ scaler เดิมด้วย
# - `X_train[:, 0]` และ `X_train[:, 1]` เป็นพิกัดจุดฝึก ส่วน `c=y_train` ใส่สีตามคลาสจริงของจุดฝึกแต่ละจุด
# - ภาพใช้ train เพื่ออธิบายรูปร่างขอบเขต ส่วนคะแนน test อ่านจาก Cell ก่อนหน้า
