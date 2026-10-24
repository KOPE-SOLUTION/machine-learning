# 04 · วัดผลให้ตรงโจทย์: accuracy, precision, recall และ F1

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เลือกคะแนนที่สะท้อนความผิดพลาด และใช้ **validation** เลือกโมเดลก่อนประเมิน test

**ใช้สิ่งที่เรียนแล้ว:** fit/predict, baseline และ Pipeline จากบท 01–03

## อ่านคะแนนเมื่อคลาสไม่สมดุล

ถ้า 90% เป็นคลาส 0 ทาย 0 ทุกครั้งก็ได้ accuracy 90% แต่พลาดคลาส 1 ทั้งหมด บทนี้เรียกคลาส 1 ว่า **positive**

| ค่าจริง / คำทำนาย | ทาย 0 | ทาย 1 |
|---|---|---|
| จริง 0 | TN: ทาย 0 ถูก | FP: แจ้ง 1 ผิด |
| จริง 1 | FN: พลาดคลาส 1 | TP: ทาย 1 ถูก |

| Metric | ใช้ดูอะไร | สูตร |
|---|---|---|
| Accuracy | ทายถูกทั้งหมดเท่าไร | (TP + TN) / N |
| Precision | ที่ทายว่าเป็น 1 ถูกกี่ส่วน | TP / (TP + FP) |
| Recall | คลาส 1 จริง ตรวจพบกี่ส่วน | TP / (TP + FN) |
| F1 | รวม precision และ recall | 2PR / (P + R) |

เช่น TN=80, FP=10, FN=5, TP=5 จะได้ accuracy=0.85 แต่ precision≈0.333 และ recall=0.5 โค้ดใช้ `zero_division=0` ให้รายงาน 0 เมื่อสูตรนั้นหารด้วยศูนย์

## เลือกด้วย validation

ข้อมูล 1,500 แถวแบ่งเป็น train 900 / validation 300 / test 300: ฝึกบน train → เลือกจาก F1 บน validation → ฝึกตัวเลือกนั้นใหม่บน `X_dev` (train รวม validation 1,200 แถว) → ประเมิน test

ตัวเลือกใหม่คือ **Logistic Regression** ซึ่งแปลงคะแนนเชิงเส้นเป็นความน่าจะเป็น ใช้ threshold เริ่มต้น 0.5 แบ่งสองคลาส หากจะปรับ threshold ให้เลือกบน validation ตามต้นทุน FP/FN

ผลยังมี **average precision (AP)** ซึ่งสรุป precision/recall จากการเรียงคะแนนหลาย threshold ต้องส่งคะแนนของคลาส 1 จาก `predict_proba` ไม่ใช้ label ที่ทายแล้ว และ AP ไม่ใช่ accuracy

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

<!-- CODE:START -->
```python
"""Accuracy อาจดูดีเมื่อคลาสไม่สมดุล"""
# %% แบ่ง train / validation / test เป็น 60 / 20 / 20
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

# %% เลือกวิธีจาก F1 บน validation
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

# %% ตัดสินใจแล้วจึง refit และประเมิน test
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
<!-- CODE:END -->

## อ่านผล

baseline มี accuracy สูงแต่ F1 เป็นศูนย์ เปรียบเทียบกับ logistic บน validation ก่อนอ่าน test confusion matrix: **แถวเป็นค่าจริง คอลัมน์เป็นคำทำนาย**

สำหรับรายงานหลายคลาสจากบท 03: macro average เฉลี่ยทุกคลาสเท่ากัน ส่วน weighted average ถ่วงตามจำนวนตัวอย่าง (`support`)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองคิด:** ถ้าต้องลด FN ควรดูอะไรและเลือก threshold จากชุดใด?

**เฉลย:** ดู recall พร้อมติดตาม precision และใช้ validation เลือก threshold เช่น 0.3, 0.5, 0.7 ตรึงทางเลือกก่อนประเมิน test

</details>

## อ้างอิง

- [Metrics and scoring](https://scikit-learn.org/1.7/modules/model_evaluation.html)
- [Cross-validation](https://scikit-learn.org/1.7/modules/cross_validation.html)
- [Google MLCC: Classification](https://developers.google.com/machine-learning/crash-course/classification)
