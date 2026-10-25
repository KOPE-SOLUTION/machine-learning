# %% Cell 1 — ฝึก Pipeline ที่มี PCA
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=0.9, svd_solver="full")),
    ("classifier", LogisticRegression(max_iter=2000)),
])
model.fit(X_train, y_train)
pca = model.named_steps["pca"]
print("Original features:", X.shape[1], "Retained components:", pca.n_components_)
print("Retained variance:", round(pca.explained_variance_ratio_.sum(), 3))
print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")

# %% [markdown]
# **อ่านโค้ด**
#
# - `load_digits(return_X_y=True)` คืนภาพที่คลี่เป็น 64 feature ต่อแถว พร้อม label ตัวเลข 0–9
# - Pipeline ทำ `scale → pca → classifier` ตามลำดับ โดย `fit` เรียนทุกขั้นจาก train เท่านั้น
# - `PCA(n_components=0.9, svd_solver="full")` เลือกจำนวนแกนตามเกณฑ์ variance 90% ไม่ใช่เลือก 0.9 แกน
# - `model.named_steps["pca"]` ดึง PCA ที่ fit แล้วออกมาอ่านผล ส่วน `pca.n_components_` คือจำนวนแกนที่เลือกได้จริง
# - `explained_variance_ratio_.sum()` รวมสัดส่วน variance ที่เก็บไว้ ขณะที่ accuracy คำนวณจากคำทำนายของ Pipeline ทั้งชุด
# - `max_iter=2000` ตั้งเพดานรอบให้ตัวแก้ปัญหาของ Logistic Regression ไม่ใช่จำนวน component ของ PCA

# %% Cell 2 — อ่านพิกัดใหม่และ variance สะสม
Z_train = model[:-1].transform(X_train)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
points = axes[0].scatter(Z_train[:, 0], Z_train[:, 1], c=y_train, cmap="tab10", s=9, alpha=0.6)
axes[0].set(xlabel="PC 1", ylabel="PC 2", title="Training digits")
fig.colorbar(points, ax=axes[0], ticks=range(10))
axes[1].plot(np.arange(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_), "o-")
axes[1].set(xlabel="Components", ylabel="Cumulative explained variance")
fig.tight_layout()
out = Path("outputs/11-principal-components")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "pca.png", dpi=140)
plt.show()
plt.close(fig)

# %% [markdown]
# **อ่านโค้ด**
#
# - `model[:-1]` เลือกทุกขั้นก่อน classifier จึงเหลือ scaler กับ PCA
# - `transform(X_train)` ใช้สถิติและแกนเดิมแปลงข้อมูลเป็น `Z_train` โดยไม่ fit ใหม่
# - `Z_train[:, 0]` และ `Z_train[:, 1]` เลือกเพียงสอง component แรกมาวาด แต่ classifier ใช้ทุก component ที่เก็บไว้
# - `np.cumsum(pca.explained_variance_ratio_)` หาผลรวมสะสมทีละแกน เพื่อดูว่าเพิ่มแกนแล้วเก็บ variance ได้เท่าไร
# - `np.arange(1, pca.n_components_ + 1)` สร้างแกนนอนให้เริ่มนับ component จาก 1
