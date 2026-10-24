# 08 · Support Vector Machines และขอบเขตไม่เป็นเส้นตรง

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เปรียบเทียบ **SVM แบบเส้นตรงกับ RBF** และใช้ cross-validation เลือกวิธีจากชุดฝึก

**ใช้สิ่งที่เรียนแล้ว:** train/test, scaling และ Pipeline จากบท 03–04

## ใจความใหม่

SVM หาแนวแบ่งคลาสพร้อม **margin** หรือระยะเว้นรอบแนวแบ่ง จุดที่กำหนดขอบเขตเรียกว่า **support vectors**

| การตั้งค่า | ความหมาย |
|---|---|
| `kernel="linear"` | ให้เส้นแบ่งตรงในข้อมูลสองมิติ |
| `kernel="rbf"` | ใช้ความคล้ายตามระยะทาง สร้างขอบเขตโค้งได้ |
| `C` | ยิ่งมาก ยิ่งลงโทษการละเมิด margin; ไม่รับประกันผลกับข้อมูลใหม่ |
| `gamma` ของ RBF | ยิ่งมาก อิทธิพลของแต่ละจุดยิ่งอยู่ใกล้ตัว |

บทนี้ตรึง C=1 และ gamma ค่าเริ่มต้นไว้ แล้วเปรียบเทียบเฉพาะ kernel

## Cross-validation (CV)

กัน test 100 แถวก่อน แล้วแบ่ง train 300 แถวเป็น 5 **fold** แต่ละรอบฝึก 240 แถว ตรวจ 60 แถว และหมุนให้ครบ 5 รอบ `StratifiedKFold` ช่วยรักษาสัดส่วนคลาส

`cross_val_score` ฝึก Pipeline ใหม่ทุกรอบ ดังนั้น scaler เรียนเฉพาะ 240 แถวของรอบนั้น เลือก kernel จากคะแนน CV เฉลี่ย → fit บน train ทั้ง 300 แถว → ประเมิน test

ค่า ± ที่พิมพ์เป็นส่วนเบี่ยงเบนระหว่าง fold ไม่ใช่ confidence interval

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

<!-- CODE:START -->
```python
"""SVM แบบเส้นตรงและ RBF บนข้อมูลวงจันทร์เสี้ยว"""
# %% กัน test แล้วเปรียบเทียบโมเดลด้วย cross-validation
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

# %% ภาพขอบเขตช่วยอธิบายผล ไม่ใช้เลือกโมเดลจาก test
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
```
<!-- CODE:END -->

## อ่านผล

ได้ CV accuracy ของ linear/RBF ชื่อวิธีที่เลือก และ test accuracy กราฟแสดงขอบเขตบน **train** เพื่ออ่านรูปร่างการตัดสินใจ บันทึก `outputs/08-support-vector-machines/boundary.png`

![ผลการทดลอง 08-support-vector-machines](../../assets/figures/08-support-vector-machines-boundary.png)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองคิด:** scaler ควรเรียนจาก train 300 แถวก่อนทำ CV หรือไม่?

**เฉลย:** ไม่ควร เพราะจะเห็น validation fold ล่วงหน้า ต้องอยู่ใน Pipeline ที่ fit ใหม่ด้วย 240 แถวของแต่ละรอบ การค้นหลายค่า C/gamma จะเรียนในบท 10

</details>

## อ้างอิง

- [Handbook: SVM](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html)
- [scikit-learn: SVM](https://scikit-learn.org/1.7/modules/svm.html)
