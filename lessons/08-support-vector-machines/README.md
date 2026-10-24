# 08 · Support Vector Machines และขอบเขตไม่เป็นเส้นตรง

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 45–65 นาที  
สิ่งที่จะทำได้: อธิบาย margin, kernel, C และผลของ scaling ได้

## ก่อนเริ่มบทนี้

ใช้การแบ่งข้อมูลและ Pipeline จากบท 03–04 เพิ่มแนวคิด cross-validation ก่อนลอง SVM สองรูปแบบ ไม่จำเป็นต้องแก้สมการ optimization ของ SVM ด้วยมือ

## แบ่งกลุ่มด้วยขอบเขตและ margin

SVM พยายามสร้างขอบเขตที่มีระยะห่างจากตัวอย่างสำคัญทั้งสองด้าน ตัวอย่างที่กำหนดขอบเขตเรียกว่า support vectors ข้อมูลที่ทับกันทำให้ต้องยอมให้บางจุดอยู่ผิดฝั่งหรืออยู่ใน margin

C ควบคุมการลงโทษความผิดพลาดบนข้อมูลฝึก ค่า C มากจะลงโทษการละเมิดมากขึ้น แต่ไม่รับประกันว่าจะทำนายข้อมูลใหม่ดีขึ้น ส่วน RBF kernel ช่วยสร้างขอบเขตโค้งด้วยความคล้ายกันตามระยะทาง gamma ควบคุมความเป็นท้องถิ่นของอิทธิพลจุด

## เลือกจากข้อมูลฝึก

โค้ดเปรียบเทียบ linear กับ RBF ด้วย 5-fold CV ภายใน train เท่านั้น scaler อยู่ใน Pipeline จึงเรียนสถิติใหม่สำหรับแต่ละ fold จากนั้น fit วิธีที่ชนะบน train ทั้งหมดแล้วประเมิน test

กราฟแสดงขอบเขตบนข้อมูลฝึกเพื่อเชื่อมรูปทรงกับกลไกของโมเดล ความสวยของเส้นแบ่งไม่ใช่เกณฑ์วัดผล และ CV mean ± standard deviation เป็นความแปรปรวนระหว่าง fold ไม่ใช่ confidence interval อัตโนมัติ

## Cross-validation คือการหมุนเวียนชุดตรวจ

กัน test ไว้ก่อน แล้วแบ่ง train เป็นห้าส่วนเรียกว่า **fold** ฝึกด้วยสี่ส่วนและตรวจด้วยส่วนที่เหลือ หมุนให้ครบห้ารอบ ตัวอย่างโค้ดมี train 300 แถว: แต่ละรอบฝึก 240 และตรวจ 60 แถว

| รอบ | ใช้ fit | ใช้ validation |
|---|---|---|
| 1 | B, C, D, E | A |
| 2 | A, C, D, E | B |
| 3–5 | สลับส่วนที่เหลือจนครบ | C, D, E ตามรอบ |

`StratifiedKFold` ช่วยรักษาสัดส่วนคลาส ส่วน `cross_val_score` สร้างโมเดลใหม่ในแต่ละรอบ ไม่ได้ฝึกสะสมทับโมเดลเดิม scaler จึงต้องอยู่ใน Pipeline เพื่อเรียนจาก 240 แถวของรอบนั้นเท่านั้น หลังเทียบค่าเฉลี่ยแล้วจึง fit ตัวที่เลือกบน train 300 แถว และตรวจ test 100 แถวครั้งสุดท้าย

## เชื่อมพารามิเตอร์กับรูปขอบเขต

**Decision boundary** คือแนวที่คำทำนายเปลี่ยนจากคลาสหนึ่งเป็นอีกคลาส **Margin** คือแถบรอบแนวแบ่งที่ SVM พยายามเว้นให้กว้าง ส่วน **kernel** เป็นวิธีคำนวณความคล้ายกันซึ่งช่วยสร้างขอบเขตที่ต่างกันได้

- `linear` ให้เส้นตรงในกราฟสองมิติ ส่วน `rbf` ให้ขอบเขตโค้งได้
- `C` มากทำให้ลงโทษจุดที่ละเมิด margin มากขึ้น จึงอาจตามรายละเอียด train มากขึ้น
- `gamma` มากทำให้อิทธิพลของแต่ละจุดอยู่ใกล้ตัวมากขึ้น ค่าเริ่มต้น `scale` คำนวณจากจำนวน feature และ variance ของ input ที่ส่งเข้า SVC

รอบนี้กำหนด C=1 และ gamma เริ่มต้นไว้ก่อน เปรียบเทียบเฉพาะ kernel การเลือกค่าหลายชุดเป็นเนื้อหาบท 10

## คัดลอกโค้ดไปรัน

คัดลอกบล็อก `python` ด้านล่างทั้งชุดไป Code cell ใน Jupyter หรือ Colab แล้วกด Shift+Enter หากเปิด `lesson.ipynb` ของบทนี้อยู่แล้ว ให้รัน cell ที่มีตามลำดับ ไม่ต้องคัดลอกซ้ำ ดู [วิธีเริ่มเรียน](../../docs/start-here.md) หากยังไม่ได้เตรียมแพ็กเกจ

<details>
<summary>ทางเลือก: รันไฟล์ Python จาก repo ในเครื่อง</summary>

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/08-support-vector-machines/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` คัดลอกบล็อกต่อไปนี้ได้ทั้งชุด แต่ละส่วน `# %%` คือช่วงการทดลองเดียวกันกับ notebook

</details>

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

## อ่านผลที่ได้

เห็น CV accuracy ของ linear/RBF พร้อมชื่อวิธีที่เลือก คะแนน test และภาพ boundary.png ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## ภาพตัวอย่างจากการรัน

![ผลการทดลอง 08-support-vector-machines](../../assets/figures/08-support-vector-machines-boundary.png)

ภาพนี้สร้างจากโค้ดและ seed ในบท เพื่อช่วยอ่านผลก่อนทดลองด้วยตัวเอง

## แบบฝึกหัด

เริ่มจากอธิบายว่า CV แต่ละรอบใช้กี่แถว fit และ scaler ต้องเรียนจากแถวใด แล้วคาดการณ์ว่า linear จะแบ่งข้อมูลโค้งได้ดีเพียงใด

แบบฝึกหัดต่อยอดหลังบท 10: วาง grid ของ C=[0.1,1,10] กับ gamma=['scale',0.1,1] สำหรับ RBF แล้วระบุว่าจะวาง StandardScaler ไว้ส่วนใด

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

แต่ละรอบใช้ 240 แถว fit ทั้ง scaler และ SVC อีก 60 แถวใช้ validation เส้นตรงอาจพลาดโครงสร้างโค้ง แต่ต้องดูคะแนนจริงเพื่อเปรียบเทียบ

ใช้ GridSearchCV ครอบ Pipeline และตั้งชื่อพารามิเตอร์แบบ svc__C กับ svc__gamma scaler ต้องอยู่ใน Pipeline เดียวกันเพื่อให้ fit เฉพาะ training fold ไม่ fit แยกบน train ทั้งชุดก่อน CV

</details>

## อ่านต่อ

- [Handbook: SVM](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html)
- [scikit-learn: SVM](https://scikit-learn.org/1.7/modules/svm.html)

