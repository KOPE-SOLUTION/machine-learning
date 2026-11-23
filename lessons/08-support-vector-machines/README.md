# 08 · Support Vector Machines และขอบเขตไม่เป็นเส้นตรง

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้ใช้ **SVM จำแนกจุดเป็นคลาส 0 หรือ 1 จากพิกัดสองค่า** เปรียบเทียบแนวแบ่งแบบเส้นตรงกับแบบโค้ง แล้วเลือกด้วยคะแนน Cross-validation ก่อนประเมินบนชุดทดสอบ

**ใช้สิ่งที่เรียนแล้ว:** train/test, StandardScaler, Pipeline และ Accuracy จากบท 03–06 ส่วนใหม่คือหลักการ SVM, kernel และการใช้ CV เปรียบเทียบโมเดล

## SVM คืออะไร

**SVM — Support Vector Machine** อ่านว่า **ซัพพอร์ต เวกเตอร์ แมชชีน** เป็นวิธีเรียนรู้แบบมีผู้สอน (Supervised Learning) โดยรับ features คู่กับ labels เพื่อฝึกโมเดล

| รุ่น | งานที่ใช้ |
|---|---|
| **SVC — Support Vector Classification** | จำแนกคลาส เช่น 0 หรือ 1 และรองรับมากกว่าสองคลาสได้ |
| **SVR — Support Vector Regression** | ทำนายตัวเลขต่อเนื่อง เช่น ราคา |

บทนี้ใช้ **SVC** กับจุดรูปพระจันทร์เสี้ยวสองชุด แต่ละแถวมีสอง features เป็นพิกัดบนกราฟ ส่วน label บอกว่าจุดนั้นอยู่คลาส 0 หรือ 1

## Margin และ Support vectors: SVM เลือกแนวแบ่งอย่างไร

**Decision boundary — แนวแบ่งคลาส** คือ **เส้นทึบสีดำ** ที่แบ่งพื้นที่ว่าโมเดลจะทายเป็นคลาสใด เช่น พื้นที่ฝั่งซ้ายทาย 0 และฝั่งขวาทาย 1

SVM พยายามหาแนวแบ่งที่มี **Margin — แถบรอบแนวแบ่ง** กว้าง ขณะเดียวกันก็ลดการที่จุดฝึกอยู่ใกล้แนวแบ่งเกินไปหรือถูกแบ่งผิดฝั่ง

![แนวแบ่งอยู่ตรงกลาง เส้นประเป็นขอบ margin สองฝั่ง และวงแหวนแสดง support vectors ที่ใช้กำหนดแนวแบ่ง](../../assets/figures/08-support-vector-machines-margin.png)

[เปิดภาพ SVG](../../assets/figures/08-support-vector-machines-margin.svg)

**เส้นประสองเส้นเป็นขอบ Margin** ส่วนแถบระหว่างเส้นประยังแบ่งเป็นสองคลาสตามเส้นทึบตรงกลาง จุดใหม่ที่อยู่ในแถบนี้จึงยังทำนายเป็น 0 หรือ 1 ได้ ไม่ได้มีคลาสที่สามเพิ่มขึ้น

**Support vectors** คือจุดฝึกที่โมเดลใช้กำหนดแนวแบ่ง ในตัวอย่างที่แยกได้ชัด จุดเหล่านี้อยู่บนขอบ margin เมื่อข้อมูลปะปนกัน support vectors อาจอยู่ในแถบ margin หรือผิดฝั่งด้วย จึงไม่ใช่เฉพาะจุดที่ทายผิด

**ทำไมต้องยอมให้บางจุดอยู่ในแถบหรือผิดฝั่ง?** ข้อมูลจริงอาจซ้อนกันหรือมี noise การบังคับให้ตามทุกจุดฝึกอาจทำให้ทายข้อมูลใหม่ได้ไม่ดี เราใช้ค่า **C** ควบคุมความสำคัญของการลดการละเมิด margin เหล่านี้

## Kernel: จากแนวแบ่งตรงเป็นแนวแบ่งโค้ง

**Kernel อ่านว่า เคอร์เนล** เป็นวิธีคำนวณความสัมพันธ์ระหว่างจุดที่ SVM ใช้สร้างแนวแบ่ง

**SVM ใช้ได้ทั้ง Linear และ RBF kernel** บทนี้จึงเปรียบเทียบ SVM สองการตั้งค่า โดยใช้ `SVC` เหมือนกันทั้งคู่

- **Linear kernel:** ได้แนวแบ่งเส้นตรงเมื่อแสดงบนกราฟสอง features
- **RBF — Radial Basis Function:** ใช้ความคล้ายที่ลดลงตามระยะห่าง จุดใกล้กันมีความคล้ายมากกว่า จึงสร้างแนวแบ่งโค้งได้บนกราฟเดิม

RBF อ่านว่า **อาร์ บี เอฟ** ส่วน Radial Basis Function อ่านว่า **เรเดียล เบซิส ฟังก์ชัน**

![Linear ใช้แนวแบ่งตรง ส่วน RBF ใช้แนวแบ่งโค้งบนข้อมูลชุดเดียวกัน วงแหวนระบุ support vectors](../../assets/figures/08-support-vector-machines-model.png)

**อ่านภาพ:** เส้นทึบคือแนวแบ่ง เส้นประฝั่ง Linear คือขอบ margin ส่วนสีพื้นบอกคลาสที่โมเดลจะทำนาย · [เปิดภาพ SVG](../../assets/figures/08-support-vector-machines-model.svg)

RBF รวมผลความคล้ายกับ support vectors เพื่อทำนาย ไม่ใช่การหาเพื่อนบ้านแล้วโหวตแบบ KNN และไม่ได้รับประกันว่าจะดีกว่า Linear ทุกชุดข้อมูล

**จุดเด่น:** เลือก kernel ให้รองรับรูปแบบข้อมูลที่ต่างกันได้

**ข้อจำกัด:** ต้องปรับสเกลและเลือก C ให้เหมาะ ส่วน RBF ต้องเลือก gamma ด้วย การฝึก SVC แบบ kernel อาจช้าเมื่อมีข้อมูลจำนวนมาก

## C และ gamma เปลี่ยนแนวแบ่งอย่างไร

**C และ gamma เป็น Hyperparameters — ไฮเปอร์พารามิเตอร์** คือค่าที่เรากำหนดก่อน `fit` เพื่อควบคุมวิธีที่โมเดลเรียนรู้ จากนั้นโมเดลจึงฝึกหาแนวแบ่งภายใต้ค่าที่กำหนด เราสามารถลองหลายค่าแล้วใช้ CV ช่วยเลือกได้

C ใช้ได้ทั้ง Linear และ RBF ส่วน gamma ใช้ควบคุมความคล้ายของ RBF ในบทนี้

### C: ให้ความสำคัญกับจุดที่อยู่ใน Margin หรือผิดฝั่งมากแค่ไหน

**C อ่านว่า ซี** ใช้ได้ทั้ง Linear และ RBF จุดฝึกอาจทายถูกแต่อยู่ใกล้แนวแบ่งจนเข้าไปในแถบ margin หรืออาจอยู่ผิดฝั่งจนทายผิดก็ได้ SVM พยายามลดทั้งสองกรณี พร้อมรักษา margin ให้กว้าง

- **C น้อย:** ยอมให้จุดฝึกบางจุดอยู่ใน margin หรือผิดฝั่งได้มากขึ้น เพื่อให้ margin กว้าง
- **C มาก:** ให้ความสำคัญกับการลดปัญหาของจุดฝึกเหล่านี้มากขึ้น จึงอาจยอมให้ margin แคบลง

![เปรียบเทียบ Linear SVM บนจุดชุดเดียวกัน ลูกศรชี้จุดสีน้ำเงินที่ C น้อยทายผิด และ C มากปรับแนวแบ่งให้ทายถูก พร้อมลูกศรแสดงความกว้าง Margin](../../assets/figures/08-support-vector-machines-c-effect.png)

**อ่านภาพจากจุดที่ลูกศรชี้:** เมื่อ C น้อย จุดสีน้ำเงินนั้นอยู่ฝั่งที่ทาย 1 เมื่อเพิ่ม C แนวแบ่งเอียงจนจุดเดิมอยู่ฝั่งที่ทาย 0 ได้ถูกต้อง แต่ margin แคบลง ตัวอย่างนี้แสดงสิ่งที่โมเดลยอมแลก ไม่ได้แปลว่า C มากจะดีกับข้อมูลใหม่เสมอ · [เปิดภาพ SVG](../../assets/figures/08-support-vector-machines-c-effect.svg)

### gamma: ความคล้ายลดลงเร็วแค่ไหนเมื่อจุดห่างกัน

**gamma อ่านว่า แกมมา** เป็นค่าที่ใช้กับ RBF ในบทนี้ ลองยึดจุดหนึ่งไว้ แล้วดูว่าจุดอื่นที่ห่างออกไปยังคล้ายจุดนั้นมากเพียงใด

- **gamma น้อย:** ความคล้ายลดลงช้า บริเวณที่ยังคล้ายจุดกลางจึงแผ่กว้าง
- **gamma มาก:** ความคล้ายลดลงเร็ว บริเวณที่ยังคล้ายมากจึงอยู่ใกล้จุดกลาง

![RBF เปรียบเทียบความคล้ายกับจุดกลางเดียวกัน gamma น้อยให้บริเวณสีเข้มกว้าง ส่วน gamma มากให้บริเวณสีเข้มแคบ ลูกศรชี้จุดเปรียบเทียบที่ระยะเท่ากัน](../../assets/figures/08-support-vector-machines-gamma-effect.png)

**สีเข้มหมายถึงคล้ายจุดกลางมาก** SVM นำความคล้ายกับ support vectors หลายจุดมารวมกันเพื่อทำนาย เมื่อ gamma มาก แต่ละจุดมีผลในบริเวณแคบ แนวแบ่งจึงอาจโค้งตามรายละเอียดเฉพาะจุดมากขึ้น ภาพนี้แสดงความคล้าย ส่วนกราฟแนวแบ่งคลาสอยู่ในส่วนขยายด้านล่าง · [เปิดภาพ SVG](../../assets/figures/08-support-vector-machines-gamma-effect.svg)

**จำหน้าที่แยกกัน:** C ควบคุมว่าโมเดลจะพยายามให้จุดฝึกอยู่ถูกฝั่งและพ้น margin มากแค่ไหน ส่วน gamma ควบคุมความคล้ายตามระยะห่างของ RBF ทั้งสองค่ากำหนดก่อน `fit`

<details>
<summary>ดูผลต่อแนวแบ่ง RBF เมื่อเปลี่ยน C และ gamma ทีละค่า</summary>

แถวบนเปลี่ยนเฉพาะ C โดยคง gamma ไว้ ส่วนแถวล่างเปลี่ยนเฉพาะ gamma โดยคง C ไว้ จึงเห็นผลของแต่ละค่าแยกกัน

ลูกศรแถวบนชี้จุดสีน้ำเงินจุดเดียวกันที่ทายผิดเมื่อ C = 0.1 และทายถูกเมื่อ C = 100 ส่วนลูกศรขวาล่างชี้แนวแบ่งที่ล้อมจุดเพียงจุดเดียวเมื่อ gamma = 20 เป็นตัวอย่างของการตามรายละเอียดเฉพาะจุดมากขึ้น

![แถวบนเปลี่ยน C โดยคง gamma ส่วนแถวล่างเปลี่ยน gamma โดยคง C ให้เห็นผลต่อแนวแบ่ง RBF บนจุดฝึกชุดเดียวกัน](../../assets/figures/08-support-vector-machines-parameters.png)

[เปิดภาพ SVG](../../assets/figures/08-support-vector-machines-parameters.svg)

</details>

**ดูความเปลี่ยนแปลงของแนวแบ่ง ไม่เลือกจากกราฟที่ตามจุดฝึกได้ดีที่สุดเพียงอย่างเดียว** ถ้าตามรายละเอียดของ train มากเกินไป อาจเกิด overfitting จึงต้องเปรียบเทียบด้วยข้อมูลที่ไม่ได้ใช้ fit ในรอบนั้น

ใช้ StandardScaler ก่อน SVM เพราะขนาดของ features มีผลต่อระยะและความคล้าย โดยให้ scaler อยู่ใน Pipeline เช่นเดียวกับบทก่อน

## Cross-validation (CV): เลือก kernel ก่อนใช้ Test

แบ่ง 400 แถวเป็น **train 300 แถว** กับ **test 100 แถว** จากนั้นใช้เฉพาะ train เปรียบเทียบ Linear กับ RBF

**Fold อ่านว่า โฟลด์** คือหนึ่งส่วนของข้อมูลที่แบ่งไว้สำหรับ CV บทนี้แบ่ง train เป็นห้าส่วน ส่วนละ 60 แถว แล้วสลับใช้สี่ส่วนฝึกและหนึ่งส่วนตรวจผล:

![แบ่ง train เป็นห้า fold แต่ละรอบฝึก 240 แถวและตรวจ validation 60 แถว เลือก kernel จาก CV ก่อนฝึก train ทั้งหมดแล้วประเมิน test](../../assets/figures/08-support-vector-machines-cross-validation.png)

[เปิดภาพ SVG](../../assets/figures/08-support-vector-machines-cross-validation.svg)

แต่ละ kernel จึงได้ **Accuracy ห้าค่า** แล้วนำมาเฉลี่ยเพื่อเปรียบเทียบกัน โดยใช้ fold ชุดเดียวกันทั้งสอง kernel

**ในแต่ละรอบ:** scaler fit/transform เฉพาะ 240 แถวก่อนส่งให้ SVC fit ส่วน validation 60 แถวใช้ transform และ predict ด้วยค่าที่ฝึกไว้ เมื่อเลือก kernel แล้วจึงฝึก Pipeline ใหม่ด้วย train ทั้ง 300 แถว และใช้ test ประเมินครั้งสุดท้าย

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรัน **Cell 1–5 ตามลำดับ** หรือเปิด [Notebook ของบทนี้](lesson.ipynb)

ลำดับการทดลอง: **แบ่งข้อมูล → สร้าง Pipeline สองแบบ → เปรียบเทียบด้วย CV → ฝึกแบบที่เลือกและทดสอบ → วาดแนวแบ่ง**

<!-- CODE:CELLS -->

<!-- CODE:START -->
### Cell 1 — สร้างข้อมูลและแยกชุดทดสอบ

```python
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
```

**อ่านโค้ด**

- `make_moons` สร้างจุดรูปโค้งสองชุด: `X` เก็บพิกัดสองค่า ส่วน `y` เป็น label 0 หรือ 1 ที่กำกับแต่ละจุด
- `noise=0.25` เพิ่มความกระจายให้จุดไม่เรียงเป็นเส้นโค้งพอดี; ค่า 0.25 เป็นขนาดความกระจาย ไม่ใช่จำนวนจุดที่ทายผิด 25%
- `test_size=0.25` กันข้อมูล 100 จาก 400 แถวไว้ทดสอบ เหลือ 300 แถวสำหรับฝึกและทำ CV ส่วน `stratify=y` รักษาสัดส่วนคลาส

**ผลที่ได้:** `Train: (300, 2) Test: (100, 2)` และ labels `[0 1]` หมายถึงแต่ละจุดมีสอง features และมีสองคลาส

### Cell 2 — สร้าง Pipeline ของ SVM สองแบบ

```python
models = {
    "linear": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
    "rbf": make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")
    ),
}
print("Prepared kernels:", list(models))
```

**อ่านโค้ด**

- `models` เก็บ Pipeline สองตัว ทั้งคู่เป็น SVM สำหรับจำแนกคลาส โดยใช้ `StandardScaler` ก่อน `SVC`; ตัวหนึ่งใช้ `kernel="linear"` อีกตัวใช้ `kernel="rbf"`
- `C=1.0` เป็นค่าที่กำหนดไว้ให้ทั้งสองแบบ ฝั่ง RBF เพิ่ม `gamma="scale"` เพื่อให้ SVC คำนวณ gamma จาก features **หลังผ่าน StandardScaler** เมื่อเริ่ม fit
- ขั้นนี้เพียงสร้าง Pipeline และกำหนดค่า ยังไม่ได้ฝึกโมเดล; `Prepared kernels: ['linear', 'rbf']` แสดงสองแบบที่เตรียมไว้เปรียบเทียบ

<details>
<summary>gamma="scale" คำนวณอย่างไร และต่างจาก StandardScaler อย่างไร?</summary>

**StandardScaler แปลงค่า features** ส่วน **gamma="scale" กำหนดความเร็วที่ความคล้ายของ RBF ลดลงตามระยะ** SVC ไม่ได้ปรับสเกลข้อมูลให้อีกครั้ง

SVC ใช้สูตร `gamma = 1 / (จำนวน features × ความแปรปรวนของข้อมูลที่เข้า SVC)`

ข้อมูลบทนี้มีสองคอลัมน์ที่ไม่ใช่ค่าคงที่ หลัง StandardScaler แปลงชุดฝึก แต่ละคอลัมน์มีค่าเฉลี่ยประมาณ 0 และความแปรปรวนประมาณ 1 จึงได้ `gamma ≈ 1 / (2 × 1) = 0.5` ในแต่ละรอบที่ fit

</details>

### Cell 3 — เปรียบเทียบ Linear กับ RBF ด้วย CV

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = {}
for name, candidate in models.items():
    values = cross_val_score(
        candidate, X_train, y_train, cv=cv, scoring="accuracy"
    )
    scores[name] = values.mean()
    print(f"{name} fold accuracy:", np.round(values, 3))
    print(f"{name} CV accuracy: {values.mean():.3f} +/- {values.std():.3f}")
```

**อ่านโค้ด**

- `StratifiedKFold` แบ่งชุดฝึกเป็นห้าส่วน ส่วนละ 60 แถว และรักษาสัดส่วนคลาส แต่ละรอบใช้ 240 แถวฝึก อีก 60 แถวตรวจผล
- `cross_val_score` ฝึก **สำเนา Pipeline ใหม่ในแต่ละรอบ** แล้วคืน accuracy ห้าค่า จึงยังต้อง fit Pipeline ตัวที่จะใช้งานจริงหลังเลือก kernel
- `scores[name]` เก็บค่าเฉลี่ยของแต่ละ kernel ส่วน `std()` แสดงว่าคะแนนแตกต่างกันระหว่างห้ารอบมากเพียงใด ค่า `+/-` นี้ไม่ใช่ช่วงความเชื่อมั่น

**ทำไมใช้ Accuracy:** ข้อมูลนี้มีจำนวนคลาส 0 และ 1 เท่ากัน จึงเริ่มเปรียบเทียบจากสัดส่วนที่ทายถูกได้ หากคลาสไม่สมดุลหรือการทายผิดแต่ละแบบมีผลต่างกัน ให้พิจารณา Precision, Recall และ F1 ที่เรียนในบท 04 ด้วย

เรากำลังเลือกจากสองชุดการตั้งค่าที่เตรียมไว้ใน `models` ยังไม่ได้ค้นหา C/gamma หลายค่า

### Cell 4 — ฝึกโมเดลที่เลือกแล้วประเมินบน Test

```python
chosen = max(scores, key=scores.get)
model = models[chosen]
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Selected kernel:", chosen)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Correct predictions: {(y_pred == y_test).sum()}/{len(y_test)}")
```

**อ่านโค้ด**

- `max(scores, key=scores.get)` เลือก **ชื่อ kernel** ที่มี CV accuracy เฉลี่ยสูงสุด เช่น `"rbf"`; ไม่ได้เลือกจาก test
- `model.fit(X_train, y_train)` ฝึก Pipeline ใหม่ด้วยชุดฝึกครบ 300 แถว: scaler fit/transform ก่อน แล้ว SVC fit จาก features ที่ปรับสเกลแล้ว
- `model.predict(X_test)` ให้ scaler transform ด้วยค่าเดิมก่อน SVC ทำนาย จึงไม่มีการ fit จาก test
- `accuracy_score` เปรียบเทียบคำทำนายกับ `y_test` ส่วนบรรทัดสุดท้ายแสดงจำนวนที่ทายถูกจาก test 100 แถว

### Cell 5 — อ่านแนวแบ่งคลาสและ Support vectors

```python
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
```

**อ่านโค้ด**

- `DecisionBoundaryDisplay.from_estimator` ครั้งแรกใช้ `response_method="predict"` ให้ Pipeline ทำนายจุดทั่วพื้นที่กราฟ แล้วระบายพื้นตามคลาสที่ได้
- ครั้งที่สองใช้ `decision_function` ซึ่งคืนคะแนนตัดสินใจ สำหรับสองคลาสในบทนี้ ค่าติดลบอยู่ฝั่งคลาส 0 ค่าบวกอยู่ฝั่งคลาส 1; `levels=[0]` จึงวาดเส้นทึบตรงรอยแบ่งที่คะแนนเป็นศูนย์ คะแนนนี้ไม่ใช่ความน่าจะเป็น
- สีของจุดมาจาก label จริง `y_train` ถ้าสีจุดต่างจากสีพื้นรอบจุด แสดงว่าโมเดลทายจุดฝึกนั้นผิด
- `model.named_steps["svc"]` เข้าถึง SVC ภายใน Pipeline; `support_` คือ **ตำแหน่งแถวของ support vectors ในข้อมูลที่ใช้ fit** จึงนำไปเลือกพิกัดเดิมจาก `X_train` แล้ววงแหวนได้ตรงกับแกนกราฟ
- พื้นที่สีเป็นผลจำแนกคลาส ไม่ใช่ระดับความมั่นใจ ส่วนวงแหวนบอก support vectors ไม่ได้หมายความว่าจุดนั้นทายผิดทุกจุด
- ไฟล์ `boundary.png` บันทึกใน `outputs/08-support-vector-machines` ใต้โฟลเดอร์ที่ Notebook ทำงาน บรรทัด `Saved:` แสดงตำแหน่งเต็ม หากเปิด Notebook ใน `practics` ไฟล์จะอยู่ที่ `practics/outputs/08-support-vector-machines/boundary.png`
<!-- CODE:END -->

## อ่านผล

ผลเมื่อรันด้วยค่าที่กำหนดในบท:

```text
linear CV accuracy: 0.803 +/- 0.046
rbf CV accuracy: 0.907 +/- 0.043
Selected kernel: rbf
Test accuracy: 0.950
Correct predictions: 95/100
```

อ่าน CV accuracy เพื่อเปรียบเทียบสอง kernel ก่อน แล้วอ่าน Test accuracy เพื่อประเมิน Pipeline ที่เลือกกับข้อมูล 100 แถวที่กันไว้

กราฟแสดง **จุดฝึกและพื้นที่ที่โมเดลทำนายเป็นแต่ละคลาส** พร้อมวงแหวน support vectors จึงใช้ดูรูปร่างแนวแบ่ง ส่วนผลกับข้อมูลที่ไม่ได้ใช้เลือกโมเดลให้ดูคะแนน test

![แนวแบ่ง RBF พร้อมวงแหวน support vectors จากการรันบทเรียน](../../assets/figures/08-support-vector-machines-boundary.png)

## แบบฝึกท้ายบท

**1. ทำไมไม่เลือก kernel จาก Test accuracy ของทั้งสองแบบ?**

<details>
<summary>ดูเฉลย</summary>

เพราะ test จะเข้ามามีส่วนในการเลือกโมเดล ทำให้ไม่ใช่ชุดที่เก็บไว้ประเมินหลังตัดสินใจเสร็จ ใช้ CV ภายใน train เลือก kernel แล้วจึงประเมิน test ของวิธีที่เลือก

</details>

**2. Scaler ควร fit จาก train 300 แถวก่อนเริ่ม CV หรือไม่?**

<details>
<summary>ดูเฉลย</summary>

ไม่ควร เพราะ scaler จะเห็น validation fold ล่วงหน้า ให้ scaler อยู่ใน Pipeline เพื่อ fit เฉพาะส่วนฝึก 240 แถวของแต่ละรอบ หลังเลือก kernel แล้วจึง fit Pipeline ใหม่จาก train ทั้ง 300 แถวได้

</details>

**3. จุดที่มีวงแหวนต้องเป็นจุดที่โมเดลทายผิดใช่หรือไม่?**

<details>
<summary>ดูเฉลย</summary>

ไม่ใช่ วงแหวนหมายถึง support vectors ที่ใช้กำหนดแนวแบ่ง ซึ่งอาจทายถูกได้ เช่น จุดบนขอบ margin หากต้องการดูว่าทายผิด ให้เทียบสีของจุดกับสีพื้นรอบจุด

</details>

การค้นหลายค่าด้วย GridSearchCV จะเรียนใน [บท 10](../10-tuning-and-learning-curves/README.md) และนำมาเลือก C กับ gamma ของ SVM ใน [บท 16](../16-digits-project/README.md)

## อ้างอิง

- [scikit-learn: Support Vector Machines](https://scikit-learn.org/1.7/modules/svm.html)
- [scikit-learn: ผลของ C และ gamma สำหรับ RBF](https://scikit-learn.org/1.7/auto_examples/svm/plot_rbf_parameters.html)
- [scikit-learn: cross_val_score](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.cross_val_score.html)
- [scikit-learn: SVC](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.SVC.html)
- [scikit-learn: DecisionBoundaryDisplay](https://scikit-learn.org/1.7/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html)
