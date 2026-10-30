# 11 · PCA และ Logistic Regression: ลดมิติก่อนจำแนกภาพตัวเลข

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้รับภาพตัวเลข 8×8 ซึ่งมีค่า pixel 64 ค่า ใช้ **PCA** ลดจำนวน feature แล้วให้ **Logistic Regression ทายเลข 0–9** หนึ่งคำตอบต่อภาพ จึงมีทั้งผลการลดมิติและคะแนนการจำแนกที่ต้องอ่านแยกกัน

**ใช้สิ่งที่เรียนแล้ว:** variance จาก NumPy, scaling/Pipeline และการเลือกค่าด้วย CV

## ใจความใหม่

### PCA คืออะไร

**PCA ย่อมาจาก Principal Component Analysis — การวิเคราะห์องค์ประกอบหลัก** เป็นวิธี **Unsupervised Learning สำหรับลดมิติ** เรียนจากข้อมูลนำเข้า `X` โดยไม่ใช้ประเภทของภาพ (class labels) ที่เก็บใน `y`

PCA นำ feature เดิมมาคูณน้ำหนักแล้วรวมเป็นพิกัดบนแกนใหม่ แต่ละแกนเรียกว่า **principal component หรือองค์ประกอบหลัก**

PCA เลือกแกนที่ทำให้พิกัดของจุดกระจายห่างจากค่าเฉลี่ยมาก **“เก็บความแปรปรวนไว้มาก”** จึงหมายถึงยังเก็บการกระจายของข้อมูลเดิมไว้ได้มากหลังลดจำนวนแกน ไม่ได้หมายถึงเก็บคอลัมน์เดิมหรือคำตอบที่ถูกต้องไว้ตามเปอร์เซ็นต์นั้น

**หลักการทำงาน:**

1. ลบค่าเฉลี่ยของแต่ละคอลัมน์ออกจากค่าทุกแถว แล้วหาแกนแรกที่พิกัดของข้อมูลกระจายมากที่สุด
2. หาแกนถัดไปที่ตั้งฉากกับแกนก่อนหน้า เช่น ทำมุม 90 องศาในภาพสองมิติ เพื่อเก็บการกระจายในทิศทางที่ยังไม่ได้เก็บ
3. เก็บเฉพาะบางแกน แล้วคำนวณพิกัดของแต่ละจุดบนแกนที่เก็บไว้ ตัวอย่างหนึ่งแถวจึงเหลือพิกัดน้อยลง

![PCA เรียนแกน PC1 จากข้อมูลสองมิติ ฉายจุดลงแกนและแทนแต่ละจุดด้วยพิกัดเดียว โดยมีส่วนของข้อมูลที่ถูกตัดออก](../../assets/figures/11-principal-components-projection.png)

**อ่านภาพ:** จุดน้ำเงินมีพิกัดเดิมสองค่า เมื่อฉายลงแกนสีส้มจะเหลือพิกัด PC1 ค่าเดียว เส้นเชื่อมแสดงส่วนที่ตัดออก; ภาพสมมติลด 2 มิติเหลือ 1 ส่วนโค้ดเริ่มจาก 64 feature · [เปิดภาพ SVG](../../assets/figures/11-principal-components-projection.svg)

ตัวอย่าง: ส่วนสูงกับความยาวแขนที่เพิ่มไปด้วยกันอาจสรุปได้ด้วยแกนหลักเดียว วิธีนี้จึงต่างจากการเลือกคอลัมน์เดิมบางคอลัมน์

- **จุดเด่น:** ลดจำนวน feature ที่ส่งต่อให้โมเดล และรวมข้อมูลที่สัมพันธ์กันไว้ในแกนจำนวนน้อยลงได้
- **ข้อจำกัด:** แกนใหม่เป็นผลรวมของ feature เดิมที่คูณน้ำหนัก จึงไม่สามารถคลี่รูปแบบโค้งได้โดยทั่วไป แกนใหม่อธิบายยากกว่าคอลัมน์เดิม และทิศทางที่ตัดทิ้งอาจมีข้อมูลช่วยแยกคลาส

PCA หาแกนจาก X โดยไม่ใช้ y ดังนั้น **เก็บ variance 90% ไม่ได้แปลว่า accuracy 90%** เพราะสิ่งที่กระจายมากอาจไม่ช่วยแยกคลาส

### PCA ใน Pipeline ของบทนี้

โค้ดใช้ `StandardScaler → PCA → LogisticRegression` โดยมีรายละเอียดดังนี้:

- **ข้อมูลเข้า:** ภาพ 8×8 มี 64 pixel จึงเริ่มด้วย 64 feature
- **การเลือกจำนวนแกน:** กำหนดสัดส่วน variance ที่ต้องการเก็บ แล้วให้ PCA หาจำนวนแกนที่เพียงพอจาก train
- **การปรับสเกล:** PCA ลบค่าเฉลี่ยของแต่ละคอลัมน์ให้เอง แต่ไม่ได้หารด้วยส่วนเบี่ยงเบนมาตรฐาน เราจึงเลือกใส่ StandardScaler ก่อน PCA ในตัวอย่างนี้
- **ข้อมูลใหม่:** ใช้ scaler และแกน PCA ที่ fit ไว้แล้ว ห้าม fit แกนใหม่บน test

**ตัวอย่างสมมติ:** หากสามแกนเก็บ variance ได้สัดส่วน 0.5, 0.3 และ 0.1 จะรวมเป็น 0.9 หรือ 90% นี่คือสัดส่วนความแปรปรวนที่เก็บได้ ไม่ใช่จำนวนภาพที่ทายถูก

หากจะเปรียบเทียบหลายระดับของ variance ที่เก็บ ให้เลือกด้วย CV

## Logistic Regression: ผู้ทำนายหลังลดมิติ

**PCA** เรียนแกนใหม่จากข้อมูลภาพใน `X` แล้วคืนพิกัดที่มีมิติน้อยลง ส่วน **Logistic Regression** ใช้พิกัดเหล่านั้นเป็น features และใช้เลข 0–9 ที่กำกับแต่ละภาพใน `y` เป็น class labels เพื่อฝึกจำแนกภาพ ทบทวนแนวคิดโมเดลใน [บท 04](../04-validation-and-metrics/README.md)

ต่างจากกรณีสองคลาสในบท 04 ตัวแก้ปัญหาเริ่มต้นที่ใช้กับสิบคลาสนี้เรียนแบบ **multinomial logistic regression**: คำนวณคะแนนให้แต่ละคลาส แล้วใช้ **softmax** แปลงเป็น probability สิบค่าที่รวมกันเป็น 1 และเลือกคลาสที่ได้ค่าสูงสุด จึงไม่ใช้เกณฑ์ 0.5 ตัดสินทุกคลาส

**ตัวอย่างสมมติ:** ถ้าความน่าจะเป็นสูงสุดคือเลข 7 ที่ 0.30 ก็เลือกเลข 7 แม้ค่านั้นยังไม่ถึง 0.5

![ข้อมูล 64 pixel ผ่าน StandardScaler และ PCA ก่อน Logistic Regression ให้ probability สิบคลาส ตัวอย่างเลข 7 มีค่าสูงสุด 0.30 จึงถูกเลือก](../../assets/figures/11-principal-components-classifier.png)

**อ่านภาพ:** PCA คืนพิกัดใหม่ แล้ว Logistic Regression แปลงคะแนนเป็น probability สิบค่าด้วย softmax แท่งสูงสุดคือคำตอบที่เลือก; ตัวเลขในกราฟเป็นตัวอย่างสมมติ ไม่ใช่ผล test · [เปิดภาพ SVG](../../assets/figures/11-principal-components-classifier.svg)

PCA อาจลดข้อมูลรบกวนและจำนวน feature แต่ก็อาจตัดข้อมูลที่ช่วยแยกเลขออกไป ต้องอ่าน accuracy จาก Logistic Regression หลังแปลงข้อมูลด้วย ค่า variance ที่ PCA เก็บไว้ไม่ใช่คะแนนจำแนก

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรัน **Cell 1–2 ตามลำดับ** หรือเปิด [Notebook ของบทนี้](lesson.ipynb) ใช้ kernel เดิมได้เลย อ่านคำอธิบายใต้โค้ดแต่ละช่วงประกอบผลที่ได้

<!-- CODE:CELLS -->

<!-- CODE:START -->
### Cell 1 — ฝึก Pipeline ที่มี PCA

```python
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
```

**อ่านโค้ด**

- `load_digits(return_X_y=True)` คืน `X` ซึ่งมีค่า pixel 64 ค่าต่อภาพ (features) และ `y` ซึ่งระบุว่าแต่ละภาพเป็นเลขใดตั้งแต่ 0–9 (class labels) เลขใน y ใช้ระบุประเภทของภาพ ไม่ใช่ค่าต่อเนื่องสำหรับงาน Regression
- Pipeline ทำ `scale → pca → classifier` ตามลำดับ โดย `fit` เรียนทุกขั้นจาก train เท่านั้น
- `PCA(n_components=0.9, svd_solver="full")` เพิ่มแกนตามลำดับจนสัดส่วนความแปรปรวนสะสมเกิน 90% ของข้อมูลฝึกที่ผ่าน scaler แล้ว ค่า 0.9 เป็นเกณฑ์สัดส่วน ไม่ใช่จำนวนแกน
- `model.named_steps["pca"]` ดึง PCA ที่ fit แล้วออกมาอ่านผล ส่วน `pca.n_components_` คือจำนวนแกนที่เลือกได้จริง
- `explained_variance_ratio_` บอกว่าแต่ละแกนเก็บความแปรปรวนไว้เป็นสัดส่วนเท่าไร และ `.sum()` รวมสัดส่วนของทุกแกนที่เก็บไว้ ส่วน accuracy ต้องเทียบเลขที่ Pipeline ทายกับเลขที่กำกับภาพไว้
- `max_iter=2000` ตั้งเพดานรอบให้ตัวแก้ปัญหาของ Logistic Regression ไม่ใช่จำนวน component ของ PCA

**อ่านผลตามหน้าที่:** `Original features` กับ `Retained components` บอกจำนวนคอลัมน์ก่อนและหลัง PCA; `Retained variance` บอกสัดส่วนความแปรปรวนที่เก็บ ส่วน `Test accuracy` เทียบเลขที่ Logistic Regression ทายกับเลขที่กำกับภาพไว้ ไม่ใช้ variance แทน accuracy

### Cell 2 — อ่านพิกัดใหม่และ variance สะสม

```python
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

**อ่านโค้ด**

- `model[:-1]` เลือกทุกขั้นก่อน classifier จึงเหลือ scaler กับ PCA
- `transform(X_train)` ใช้สถิติและแกนเดิมแปลงข้อมูลเป็น `Z_train` โดยไม่ fit ใหม่
- `Z_train[:, 0]` และ `Z_train[:, 1]` เลือกเพียงสอง component แรกมาวาด แต่ classifier ใช้ทุก component ที่เก็บไว้
- `np.cumsum(pca.explained_variance_ratio_)` หาผลรวมสะสมทีละแกน เพื่อดูว่าเพิ่มแกนแล้วเก็บ variance ได้เท่าไร
- `np.arange(1, pca.n_components_ + 1)` สร้างแกนนอนให้เริ่มนับ component จาก 1
<!-- CODE:END -->

## อ่านผล

จำนวน component เหลือน้อยกว่า 64 และ variance รวมผ่านเกณฑ์ 0.9

- **กราฟซ้าย:** แสดงเพียงสองแกนแรกจาก `model[:-1].transform(X_train)`
- **Classifier:** ใช้ทุกแกนที่เก็บไว้

จึงตัดสินความสามารถของทั้งระบบจากภาพสองมิติเพียงอย่างเดียวไม่ได้

![ผลการทดลอง 11-principal-components](../../assets/figures/11-principal-components-pca.png)

## แบบฝึกท้ายบท

**ลองทำ:** เปรียบเทียบไม่มี PCA กับเก็บ variance 0.8, 0.9, 0.95

<details>
<summary>ดูแนวคำตอบ</summary>

**แนวคำตอบ:** ใช้ CV บน train เพื่อเปรียบเทียบสองกรณี:

- ไม่มี PCA: เปลี่ยนขั้น PCA เป็น `passthrough`
- มี PCA: ปรับ `pca__n_components` เป็นค่าที่ต้องการลอง

วัดทั้งคะแนนและเวลา เพราะการลดมิติไม่รับประกันว่าจะเร็วหรือแม่นขึ้น

</details>

## อ้างอิง

- [Logistic Regression แบบหลายคลาส](https://scikit-learn.org/1.7/modules/linear_model.html#multinomial-case)
- [Handbook: PCA](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)
- [scikit-learn: PCA](https://scikit-learn.org/1.7/modules/decomposition.html#pca)
