# %% Cell 1 — สร้างข้อมูลและแยกชุดทดสอบ
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Training labels:", np.unique(y_train))

# %% [markdown]
# **อ่านโค้ด**
#
# - `make_moons` สร้างจุดรูปโค้งสองชุด: `X` เก็บพิกัดสองค่า ส่วน `y` เป็น label 0 หรือ 1 ที่กำกับแต่ละจุด
# - `noise=0.25` เพิ่มความกระจายให้จุดไม่เรียงเป็นเส้นโค้งพอดี; ค่า 0.25 เป็นขนาดความกระจาย ไม่ใช่จำนวนจุดที่ทายผิด 25%
# - `test_size=0.25` กันข้อมูล 100 จาก 400 แถวไว้ทดสอบ เหลือ 300 แถวสำหรับฝึกและทำ CV ส่วน `stratify=y` รักษาสัดส่วนคลาส
#
# **ผลที่ได้:** `Train: (300, 2) Test: (100, 2)` และ labels `[0 1]` หมายถึงแต่ละจุดมีสอง features และมีสองคลาส

# %% Cell 2 — เปรียบเทียบ Linear กับ RBF ด้วย CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "linear": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
    "rbf": make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")
    ),
}

scores = {}
for name, candidate in models.items():
    values = cross_val_score(
        candidate, X_train, y_train, cv=cv, scoring="accuracy"
    )
    scores[name] = values.mean()
    print(f"{name} fold accuracy:", np.round(values, 3))
    print(f"{name} CV accuracy: {values.mean():.3f} +/- {values.std():.3f}")

# %% [markdown]
# **อ่านโค้ด**
#
# - `models` เก็บ Pipeline สองตัว ใช้ `StandardScaler` ก่อน SVC ทั้งคู่ ต่างกันที่ kernel: `linear` หรือ `rbf`
# - `StratifiedKFold` แบ่งชุดฝึกเป็นห้าส่วน ส่วนละ 60 แถว และรักษาสัดส่วนคลาส แต่ละรอบใช้ 240 แถวฝึก อีก 60 แถวตรวจผล
# - `cross_val_score` ฝึก **สำเนา Pipeline ใหม่ในแต่ละรอบ** แล้วคืน accuracy ห้าค่า จึงยังต้อง fit Pipeline ตัวที่จะใช้งานจริงหลังเลือก kernel
# - `scores[name]` เก็บค่าเฉลี่ยของแต่ละ kernel ส่วน `std()` แสดงว่าคะแนนแตกต่างกันระหว่างห้ารอบมากเพียงใด ค่า `+/-` นี้ไม่ใช่ช่วงความเชื่อมั่น
#
# **การตั้งค่า:** ใช้ `C=1.0` ทั้งคู่ ฝั่ง RBF ใช้ `gamma="scale"` ให้คำนวณ gamma จาก features ที่เข้า SVC **หลังปรับสเกลในแต่ละรอบ** สำหรับข้อมูลสองคอลัมน์นี้ได้ประมาณ 0.5 เรากำลังเลือกจากสองชุดการตั้งค่านี้ ยังไม่ได้ค้นหา C/gamma ที่ดีที่สุด

# %% Cell 3 — ฝึกโมเดลที่เลือกแล้วประเมินบน Test
chosen = max(scores, key=scores.get)
model = models[chosen]
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Selected kernel:", chosen)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Correct predictions: {(y_pred == y_test).sum()}/{len(y_test)}")

# %% [markdown]
# **อ่านโค้ด**
#
# - `max(scores, key=scores.get)` เลือก **ชื่อ kernel** ที่มี CV accuracy เฉลี่ยสูงสุด เช่น `"rbf"`; ไม่ได้เลือกจาก test
# - `model.fit(X_train, y_train)` ฝึก Pipeline ใหม่ด้วยชุดฝึกครบ 300 แถว: scaler fit/transform ก่อน แล้ว SVC fit จาก features ที่ปรับสเกลแล้ว
# - `model.predict(X_test)` ให้ scaler transform ด้วยค่าเดิมก่อน SVC ทำนาย จึงไม่มีการ fit จาก test
# - `accuracy_score` เปรียบเทียบคำทำนายกับ `y_test` ส่วนบรรทัดสุดท้ายแสดงจำนวนที่ทายถูกจาก test 100 แถว

# %% Cell 4 — อ่านแนวแบ่งคลาสและ Support vectors
colors = ListedColormap(["#1558D6", "#BD5A00"])
fig, ax = plt.subplots(figsize=(7, 4.5))
DecisionBoundaryDisplay.from_estimator(
    model, X_train, response_method="predict", cmap=colors, alpha=0.18, ax=ax
)
points = ax.scatter(
    X_train[:, 0], X_train[:, 1], c=y_train, cmap=colors,
    edgecolors="white", linewidths=0.4, s=25
)

support_idx = model.named_steps["svc"].support_
support_points = X_train[support_idx]
ax.scatter(
    support_points[:, 0], support_points[:, 1], s=85,
    facecolors="none", edgecolors="black", linewidths=0.8,
    label="Support vectors"
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
# **อ่านโค้ด**
#
# - `DecisionBoundaryDisplay.from_estimator` ให้ Pipeline ทำนายจุดทั่วพื้นที่กราฟ แล้วระบายพื้นตามคลาสที่ได้ จึงเห็นบริเวณที่จะทายเป็น 0 หรือ 1
# - สีของจุดมาจาก label จริง `y_train` ถ้าสีจุดต่างจากสีพื้นรอบจุด แสดงว่าโมเดลทายจุดฝึกนั้นผิด
# - `model.named_steps["svc"]` เข้าถึง SVC ภายใน Pipeline; `support_` คือ **ตำแหน่งแถวของ support vectors ในข้อมูลที่ใช้ fit** จึงนำไปเลือกพิกัดเดิมจาก `X_train` แล้ววงแหวนได้ตรงกับแกนกราฟ
# - พื้นที่สีเป็นผลจำแนกคลาส ไม่ใช่ระดับความมั่นใจ ส่วนวงแหวนบอก support vectors ไม่ได้หมายความว่าจุดนั้นทายผิดทุกจุด
# - ไฟล์ `boundary.png` บันทึกใน `outputs/08-support-vector-machines` ใต้โฟลเดอร์ที่ Notebook ทำงาน บรรทัด `Saved:` แสดงตำแหน่งเต็ม หากเปิด Notebook ใน `practics` ไฟล์จะอยู่ที่ `practics/outputs/08-support-vector-machines/boundary.png`
