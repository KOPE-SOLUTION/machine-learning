# 11 · PCA: ลดมิติโดยเก็บความแปรปรวน

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้ใช้ **PCA** ลดจำนวน feature ก่อนจำแนกภาพตัวเลข 8×8

**ใช้สิ่งที่เรียนแล้ว:** variance จาก NumPy, scaling/Pipeline และการเลือกค่าด้วย CV

## ใจความใหม่

PCA สร้าง **component** หรือแกนใหม่ที่ผสม feature เดิมเพื่อเก็บความแปรปรวนให้มาก เช่น ส่วนสูงกับความยาวแขนที่เพิ่มไปด้วยกันอาจสรุปได้ด้วยแกนหลักเดียว นี่ไม่ใช่การเลือกคอลัมน์เดิมบางคอลัมน์

PCA หาแกนจาก X โดยไม่ใช้ y ดังนั้น **เก็บ variance 90% ไม่ได้แปลว่า accuracy 90%** สัญญาณที่กระจายมากอาจไม่ใช่สิ่งที่ช่วยแยกคลาส

โค้ดทำ `StandardScaler → PCA → LogisticRegression` โดย:
- ภาพหนึ่งภาพมี 64 pixel จึงเริ่มด้วย 64 feature
- `n_components=0.9` เลือกจำนวนแกนให้เก็บ variance สะสมตามเกณฑ์ 90% ของ train
- PCA center ข้อมูลแต่ไม่ scale ให้เอง การใส่ scaler ในบทนี้เป็นทางเลือกของตัวอย่าง ไม่ใช่ข้อบังคับสำหรับทุกภาพ
- ข้อมูลใหม่ใช้ scaler และแกน PCA ที่ fit ไว้แล้ว ห้าม fit แกนใหม่บน test

`explained_variance_ratio_` คือสัดส่วน variance ต่อแกน เช่น 0.5, 0.3, 0.1 รวมเป็น 0.9 การเลือกระดับที่เก็บต้องใช้ CV ถ้าจะเปรียบเทียบหลายค่า

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

<!-- CODE:START -->
```python
"""PCA ลดจำนวน feature โดยเรียนทิศทางจาก train"""
# %% ทำ PCA ใน Pipeline
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

# %% แสดง 2 component แรกเพื่อสำรวจ ไม่ใช่ feature ทั้งหมดของโมเดล
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
```
<!-- CODE:END -->

## อ่านผล

จำนวน component เหลือน้อยกว่า 64 และ variance รวมผ่านเกณฑ์ 0.9 กราฟซ้ายแสดง **แค่สองแกนแรก** จาก `model[:-1].transform(X_train)` แต่ classifier ใช้ทุกแกนที่เก็บไว้ จึงตัดสินทั้งระบบจากภาพสองมิติอย่างเดียวไม่ได้

![ผลการทดลอง 11-principal-components](../../assets/figures/11-principal-components-pca.png)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** เปรียบเทียบไม่มี PCA กับเก็บ variance 0.8, 0.9, 0.95

**แนวคำตอบ:** ใช้ CV บน train เปลี่ยนขั้น PCA เป็น `passthrough` หรือปรับ `pca__n_components` แล้ววัดทั้งคะแนนและเวลา การลดมิติไม่รับประกันว่าจะเร็วหรือแม่นขึ้น

</details>

## อ้างอิง

- [Handbook: PCA](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)
- [scikit-learn: PCA](https://scikit-learn.org/1.7/modules/decomposition.html#pca)
