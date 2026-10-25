# 04 · วัดผลให้ตรงโจทย์: accuracy, precision, recall และ F1

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เลือกคะแนนที่สะท้อนความผิดพลาด และใช้ **validation** เลือกโมเดลก่อนประเมิน test

**ใช้สิ่งที่เรียนแล้ว:** fit/predict, baseline และ Pipeline จากบท 01–03

## อ่านคะแนนเมื่อคลาสไม่สมดุล

ถ้าข้อมูล 90% เป็นคลาส 0 การทาย 0 ทุกครั้งจะได้ accuracy 90% แต่พลาดคลาส 1 ทั้งหมด

**Confusion matrix** คือตารางนับว่าคลาสจริงแต่ละคลาสถูกทำนายเป็นคลาสใดบ้าง โดย **แถวเป็นค่าจริง และคอลัมน์เป็นคำทำนาย**

บทนี้เรียกคลาส 1 ว่า **positive** และคลาส 0 ว่า **negative** เช่น ตรวจข้อความสแปม โดยให้สแปมเป็นคลาส 1

![Confusion matrix ตัวอย่าง 100 ข้อความ: แถวจริง 0 มี TN 80 และ FP 10; แถวจริง 1 มี FN 5 และ TP 5 พร้อมสูตร Precision และ Recall](../../assets/figures/04-validation-and-metrics-confusion-matrix.png)

ภาพนี้เป็น **ตัวอย่างสมมติสำหรับอธิบาย** ไม่ใช่ผล test ที่จะได้จาก Cell 3 ([เปิดภาพขนาดเต็ม](../../assets/figures/04-validation-and-metrics-confusion-matrix.svg))

อ่านชื่อแต่ละช่องโดยดูค่าจริงและคำทำนายคู่กัน:

| ค่าจริง / คำทำนาย | ทาย 0 | ทาย 1 |
|---|---|---|
| จริง 0 | **TN — True Negative:** ทาย 0 ถูก | **FP — False Positive:** ทายเป็น 1 ทั้งที่จริงเป็น 0 |
| จริง 1 | **FN — False Negative:** ทายเป็น 0 ทั้งที่จริงเป็น 1 | **TP — True Positive:** ทาย 1 ถูก |

| Metric | ใช้ดูอะไร | สูตร |
|---|---|---|
| Accuracy | ทายถูกทั้งหมดเท่าไร | (TP + TN) / N |
| Precision | ที่ทายว่าเป็น 1 ถูกกี่ส่วน | TP / (TP + FP) |
| Recall | คลาส 1 จริง ตรวจพบกี่ส่วน | TP / (TP + FN) |
| F1 | รวม precision และ recall | 2PR / (P + R) |

**อ่านตัวอย่างในภาพ:** accuracy = 85/100 = 0.85 แต่เมื่อดูคลาส 1 จะเห็นว่า:

- **Precision:** ทายว่าเป็นสแปม 15 ข้อความ ถูกจริง 5 ข้อความ จึงได้ 5/15 ≈ 0.333
- **Recall:** มีสแปมจริง 10 ข้อความ ตรวจพบ 5 ข้อความ จึงได้ 5/10 = 0.5

F1 ของตัวอย่างนี้เท่ากับ 0.4 จึงไม่ควรใช้ accuracy สูงเพียงค่าเดียวตัดสินว่าตรวจพบสแปมได้ดี

โค้ดใช้ `zero_division=0` เพื่อรายงาน 0 เมื่อสูตรของ metric นั้นหารด้วยศูนย์

## เลือกด้วย validation

ข้อมูล 1,500 แถวแบ่งเป็น train 900 แถว, validation 300 แถว และ test 300 แถว แล้วทำตามลำดับนี้:

1. ฝึกโมเดลบน train
2. เลือกโมเดลจาก F1 บน validation
3. ฝึกตัวเลือกนั้นใหม่บน `X_dev` ซึ่งรวม train กับ validation เป็น 1,200 แถว
4. ประเมินด้วย test

### Logistic Regression คืออะไร

**Logistic Regression — การถดถอยโลจิสติก** เป็นโมเดล **Supervised Learning สำหรับ Classification** แม้ชื่อมีคำว่า Regression แต่เป้าหมายในบทนี้คือทายคลาส 0 หรือ 1

**หลักการทำงานในกรณีสองคลาส:**

1. รวม feature ด้วยค่าน้ำหนักที่เรียนจาก train เป็นคะแนนเชิงเส้น
2. แปลงคะแนนผ่านฟังก์ชัน **sigmoid** ให้เป็นค่าความน่าจะเป็นของคลาส 1 ระหว่าง 0 กับ 1
3. ใช้ **threshold** หรือเกณฑ์ตัดสิน เพื่อเปลี่ยนความน่าจะเป็นเป็นคลาส โดยค่าเริ่มต้นคือ 0.5

**ตัวอย่าง:** ความน่าจะเป็นคลาส 1 เท่ากับ 0.8 จะทายเป็น 1 เมื่อใช้ threshold 0.5 แต่ถ้าเพิ่ม threshold เป็น 0.9 ตัวอย่างเดิมจะถูกทายเป็น 0

- **จุดเด่น:** เป็นโมเดลเริ่มต้นที่ฝึกได้รวดเร็วในหลายงาน และให้ probability สำหรับพิจารณา threshold ตามต้นทุนความผิดพลาด
- **ข้อจำกัด:** ขอบเขตการตัดสินใจเป็นเชิงเส้นใน feature ที่ป้อนเข้าไป หากรูปแบบซับซ้อนอาจต้องสร้าง feature เพิ่มหรือใช้โมเดลอื่น

บทนี้ใช้โมเดลนี้เป็นตัวอย่างสำหรับฝึกอ่านคะแนนและเลือกด้วย validation

หากจะปรับ threshold ให้เลือกบน validation โดยพิจารณาต้นทุนของ FP และ FN

**Average precision (AP):** คะแนนที่สรุป precision และ recall จากการเรียงคะแนนทำนายผ่านหลาย threshold

การคำนวณ AP ใช้คะแนนคลาส 1 จาก `predict_proba` ไม่ใช้ label ที่ทายแล้ว และ AP มีความหมายต่างจาก accuracy

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรัน **Cell 1–3 ตามลำดับ** หรือเปิด [Notebook ของบทนี้](lesson.ipynb) ใช้ kernel เดิมได้เลย อ่านคำอธิบายใต้โค้ดแต่ละช่วงประกอบผลที่ได้

<!-- CODE:CELLS -->

<!-- CODE:START -->
### Cell 1 — สร้างข้อมูลและแยก train/validation/test

```python
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=1500, n_features=8, n_informative=5,
    weights=[0.9, 0.1], flip_y=0.02, random_state=42,
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
)
```

**อ่านโค้ด**

- `make_classification` สร้างข้อมูลจำลอง 1,500 แถว 8 feature โดย `n_informative=5` กำหนดจำนวน feature ที่ใช้สร้างสัญญาณแยกคลาส
- `weights=[0.9, 0.1]` ตั้งสัดส่วนคลาสก่อนเติมความผิดพลาดของ label ด้วย `flip_y=0.02` สัดส่วนสุดท้ายจึงอาจไม่ใช่ 90/10 พอดี
- แบ่งรอบแรกกัน test 20% เหลือ `X_dev` 1,200 แถวสำหรับพัฒนาโมเดล
- แบ่งรอบสองกัน validation 25% **ของ X_dev** คือ 300 แถว จึงเหลือ train 900 แถว

### Cell 2 — เปรียบเทียบโมเดลบน validation

```python
models = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
}
valid_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    valid_scores[name] = f1_score(y_valid, pred, zero_division=0)
    print(name, {
        "accuracy": round(accuracy_score(y_valid, pred), 3),
        "precision": round(precision_score(y_valid, pred, zero_division=0), 3),
        "recall": round(recall_score(y_valid, pred), 3),
        "f1": round(valid_scores[name], 3),
    })
```

**อ่านโค้ด**

- `models` เก็บชื่อกับโมเดลแต่ละตัว เพื่อวนฝึกและประเมินด้วยข้อมูลชุดเดียวกัน
- `LogisticRegression(max_iter=1000)` เพิ่มเพดานจำนวนรอบของตัวแก้ปัญหา ไม่ได้หมายความว่าต้องฝึกครบ 1,000 รอบ
- `fit(X_train, y_train)` ใช้ชุดฝึก ส่วน `predict(X_valid)` ใช้ทำนายชุด validation
- `valid_scores[name]` เก็บ F1 ของแต่ละโมเดล เพื่อใช้เลือกใน Cell ถัดไป
- `zero_division=0` ให้รายงาน 0 เมื่อสูตรของ metric นั้นหารด้วยศูนย์ ไม่ได้เปลี่ยนคำทำนายของโมเดล

### Cell 3 — ฝึกตัวเลือกที่ชนะและประเมิน test

```python
chosen = max(valid_scores, key=valid_scores.get)
final = models[chosen].fit(X_dev, y_dev)
test_pred = final.predict(X_test)
positive_column = list(final.classes_).index(1)
test_probability = final.predict_proba(X_test)[:, positive_column]
print("Selected:", chosen)
print("Test confusion matrix (rows=true, columns=predicted):\n",
      confusion_matrix(y_test, test_pred, labels=[0, 1]))
print(f"Test F1: {f1_score(y_test, test_pred):.3f}")
print(f"Test average precision: {average_precision_score(y_test, test_probability):.3f}")
```

**อ่านโค้ด**

- `max(valid_scores, key=valid_scores.get)` เลือก **ชื่อโมเดล** ที่มี F1 บน validation สูงสุด
- `models[chosen].fit(X_dev, y_dev)` ฝึกโมเดลที่เลือกใหม่ด้วย train รวม validation ก่อนประเมิน test
- `list(final.classes_).index(1)` หาว่าคลาส 1 อยู่คอลัมน์ใดของผล `predict_proba`
- `[:, positive_column]` เลือก probability ของคลาส 1 ทุกแถว เพื่อใช้คำนวณ average precision
- `confusion_matrix(..., labels=[0, 1])` กำหนดลำดับคลาสของแถวและคอลัมน์ ส่วน F1 ใช้ label จาก `predict` ต่างจาก AP ที่ใช้ probability
<!-- CODE:END -->

## อ่านผล

baseline มี accuracy สูง แต่ F1 เป็นศูนย์ ให้เทียบ F1 กับ Logistic Regression บน validation ก่อนดูผล test

**Confusion matrix:** แถวเป็นค่าจริง ส่วนคอลัมน์เป็นคำทำนาย

รายงานหลายคลาสจากบท 03 มีค่าเฉลี่ยสองแบบ:

- **Macro average:** ให้น้ำหนักทุกคลาสเท่ากัน
- **Weighted average:** ถ่วงน้ำหนักตามจำนวนตัวอย่างของคลาสนั้น (`support`)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองคิด:** ถ้าต้องลด FN ควรดูอะไรและเลือก threshold จากชุดใด?

**เฉลย:** ดู recall พร้อมติดตาม precision

ใช้ validation เปรียบเทียบ threshold เช่น 0.3, 0.5 และ 0.7 แล้วตรึงค่าที่เลือกก่อนประเมิน test

</details>

## อ้างอิง

- [Logistic Regression](https://scikit-learn.org/1.7/modules/linear_model.html#logistic-regression)
- [Metrics and scoring](https://scikit-learn.org/1.7/modules/model_evaluation.html)
- [Cross-validation](https://scikit-learn.org/1.7/modules/cross_validation.html)
- [Google MLCC: Classification](https://developers.google.com/machine-learning/crash-course/classification)
