# 11 · PCA: ลดมิติโดยเก็บความแปรปรวน

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 45–65 นาที  
สิ่งที่จะทำได้: อธิบาย component, explained variance และวาง PCA ใน Pipeline ได้

## ก่อนเริ่มบทนี้

ใช้ความเข้าใจ variance จาก NumPy และ Pipeline จากบท 03–06 บทนี้เพิ่มขั้นแปลงพิกัดก่อน classifier ไม่จำเป็นต้องคำนวณ eigenvector ด้วยมือ

## มิติใหม่เป็นการผสม feature เดิม

ภาพตัวเลขขนาด 8×8 มี 64 feature เมื่อคลี่ภาพเป็นแถว PCA หาทิศทางเชิงเส้นที่เก็บความแปรปรวนของข้อมูลได้มาก เรียงเป็น principal components แต่ละ component เป็นการผสมหลายคอลัมน์เดิม จึงไม่ใช่การเลือก pixel บางตำแหน่งทิ้งไว้เฉย ๆ

PCA ไม่ใช้ y เพื่อหาทิศทาง ความแปรปรวนมากอาจไม่ใช่สัญญาณที่ช่วยแยกคลาสมากที่สุด และการเก็บ 90% ของ variance ไม่ได้แปลว่าเก็บ 90% ของ accuracy

สเกลมีผลต่อ PCA ในบทนี้ใช้ StandardScaler ก่อน เพื่อทดลองบนข้อมูลที่แต่ละ pixel ถูกปรับสเกล โดย PCA ของ scikit-learn จะ center แต่ไม่ scale feature ให้เอง การเลือก scale แบบนี้เป็นการตั้งค่าของตัวอย่าง ไม่ใช่ข้อกำหนดของภาพทุกชนิด

## fit บน train แล้วคงแกนเดิม

Pipeline ทำ scale → PCA → logistic regression ค่า n_components=0.9 กับ full solver เลือกจำนวน component ตามสัดส่วนความแปรปรวนสะสมใน train เมื่อทำนายข้อมูลใหม่ต้องใช้ทิศทางเดิม หาก fit PCA ใหม่บน test จะได้ระบบพิกัดคนละชุด

ภาพ scatter แสดงเพียงสอง component แรก ขณะที่ classifier ใช้ทุก component ที่ผ่านเกณฑ์ 90% จึงไม่ควรใช้ความทับกันในภาพสองมิติตัดสินโมเดลทั้งระบบ

## จากคอลัมน์เดิมสู่แกนใหม่

นึกถึงตารางส่วนสูงกับความยาวแขนซึ่งมักเพิ่มไปด้วยกัน PCA อาจสร้างแกนแรกเป็นการผสมสองค่านี้เพื่ออธิบายความต่างหลักระหว่างคน แกนที่สองตั้งฉากกับแกนแรกและเก็บความต่างที่เหลือ **Variance** ในบทนี้คือการกระจายของค่ารอบค่าเฉลี่ย ไม่ใช่ variance ที่หมายถึงความไวของโมเดลต่อชุดฝึกในบท 09

สำหรับ digits หนึ่งภาพ 8×8 ถูกคลี่เป็นหนึ่งแถว 64 คอลัมน์ แต่ละ component จึงผสมค่าจาก pixel เดิมหลายตำแหน่ง จำนวน component ที่เก็บอาจเปลี่ยนเมื่อ train เปลี่ยน เพราะเลือกจากข้อมูลจริงในรอบนั้น

## ตามรอย Pipeline

1. `scale.fit` เรียนค่าเฉลี่ย/สเกลของ pixel ใน train แล้วแปลงทุกคอลัมน์
2. `pca.fit` หาทิศทางจาก feature ที่ scale แล้ว โดยไม่ใช้ label เพื่อหาแกน
3. `LogisticRegression.fit` ใช้พิกัดใหม่ร่วมกับ y เพื่อเรียนจำแนกเลข 0–9
4. `model[:-1].transform(X_train)` เรียกเฉพาะ scaler กับ PCA จึงได้ `Z_train` เพื่อวาดกราฟ ส่วน `model.predict` เรียกครบทั้งสามขั้น

ถ้า explained variance ratio สามแกนแรกเป็น 0.5, 0.3, 0.1 ผลสะสมคือ 0.5, 0.8, 0.9 แต่คำว่า 90% นี้วัดการกระจายของ feature ไม่ใช่เปอร์เซ็นต์ข้อมูลหรือ accuracy ผล test ในบทใช้ตรวจ Pipeline ที่กำหนดล่วงหน้า ถ้าจะเลือกสัดส่วน variance ให้ใช้ CV ตามบท 10

## ลงมือรัน

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/11-principal-components/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` บล็อกนี้รันได้ทั้งชุดโดยไม่ต้องรันบทก่อนหน้า

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

## อ่านผลที่ได้

จำนวน component ที่เหลือน้อยกว่า 64 พร้อม retained variance อย่างน้อย 0.9, test accuracy และกราฟสอง component แรก ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## ภาพตัวอย่างจากการรัน

![ผลการทดลอง 11-principal-components](../../assets/figures/11-principal-components-pca.png)

ภาพนี้สร้างจากโค้ดและ seed ในบท เพื่อช่วยอ่านผลก่อนทดลองด้วยตัวเอง

## แบบฝึกหัด

ออกแบบการเปรียบเทียบไม่มี PCA กับ PCA เก็บ 0.8, 0.9, 0.95 โดยดูทั้งคะแนนและเวลาฝึก จะใช้ข้อมูลชุดใดตัดสิน?

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

เปรียบเทียบด้วย CV บน train ใช้ Pipeline ที่เปลี่ยนขั้น PCA เป็น passthrough หรือปรับ pca__n_components แล้วค่อยประเมินตัวเลือกสุดท้ายบน test การลดมิติไม่ได้รับประกันว่าจะเร็วหรือแม่นขึ้นในข้อมูลทุกขนาด

</details>

## อ่านต่อ

- [Handbook: PCA](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)
- [scikit-learn: PCA](https://scikit-learn.org/1.7/modules/decomposition.html#pca)

