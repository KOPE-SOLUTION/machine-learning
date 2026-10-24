# 04 · วัดผลให้ตรงโจทย์: accuracy, precision, recall และ F1

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 50–70 นาที  
สิ่งที่จะทำได้: แบ่ง train/validation/test และอ่าน confusion matrix ของปัญหาคลาสไม่สมดุลได้

## ก่อนเริ่มบทนี้

ต่อจากบท 03 ที่เรียก fit/predict ได้แล้ว บทนี้ฝึกเลือกเกณฑ์วัดผล อย่าเพิ่งปรับโมเดลเพื่อไล่คะแนนสูงโดยยังไม่รู้ว่าความผิดพลาดชนิดใดสำคัญ

## คะแนนสูงอาจตอบโจทย์ผิด

ถ้าตัวอย่าง 90% เป็นคลาส 0 การทาย 0 ทุกครั้งจะมี accuracy สูง แต่ตรวจคลาส 1 ไม่พบเลย ในบทนี้กำหนดคลาส 1 เป็น positive และสร้างข้อมูลจำลองเพื่อแยกผลของความไม่สมดุลออกจากบริบทจริง

| Metric | ความหมาย | สูตร |
|---|---|---|
| Accuracy | สัดส่วนที่ทายถูกทั้งหมด | (TP + TN) / N |
| Precision | ในสิ่งที่แจ้งว่า positive ถูกกี่ส่วน | TP / (TP + FP) |
| Recall | ใน positive จริง ตรวจพบกี่ส่วน | TP / (TP + FN) |
| F1 | ค่าเฉลี่ยฮาร์มอนิกของ precision กับ recall | 2PR / (P + R) |

TP คือทาย positive ถูก, FP คือแจ้ง positive ผิด, FN คือพลาด positive, TN คือทาย negative ถูก เมื่อ denominator เป็นศูนย์ ตัวอย่างกำหนด `zero_division=0` เพื่อรายงานกรณีที่โมเดลไม่เคยทาย positive อย่างชัดเจน

## สามชุดมีหน้าที่ต่างกัน

Train ใช้เรียนพารามิเตอร์; validation ใช้เลือกโมเดลหรือ threshold; test ใช้ประเมินหลังตัดสินใจเสร็จ โค้ดกัน test 20% แล้วกัน 25% ของส่วนที่เหลือเป็น validation จึงได้ 60/20/20

Logistic regression ใช้คะแนนเชิงเส้นแปลงเป็นความน่าจะเป็น สำหรับสองคลาส p = 1 / (1 + exp(−z)) แม้ชื่อมี regression แต่มักใช้จำแนกประเภท ในตัวอย่างยังใช้ threshold เริ่มต้น 0.5

Average precision สรุปประสิทธิภาพจากการเรียงคะแนน positive หลาย threshold; ใช้ probability/score ไม่ใช้ hard label และไม่ใช่ accuracy หรือพื้นที่ใต้โค้งที่คำนวณด้วย trapezoid โดยตรง การเลือก metric ในงานจริงต้องสัมพันธ์กับต้นทุน FP/FN

## อ่าน confusion matrix ด้วยตัวเลขเล็ก

สำหรับ `labels=[0, 1]` แถวเป็นค่าจริงและคอลัมน์เป็นคำทำนาย:

| ค่าจริง / ค่าทำนาย | ทาย 0 | ทาย 1 |
|---|---|---|
| จริง 0 | TN | FP |
| จริง 1 | FN | TP |

สมมติมี TN=80, FP=10, FN=5, TP=5 จาก 100 รายการ จะได้ accuracy=0.85 แต่ precision=5/15≈0.333 และ recall=5/10=0.5 จึงเห็นว่าคะแนนรวมสูงยังซ่อนการพลาด positive ได้ คำว่า positive คือคลาสที่เราสนใจ ไม่ได้แปลว่าเป็นผลดีเสมอ

**Threshold** คือจุดตัด เช่น p≥0.5 ให้ทาย 1 ส่วน p<0.5 ให้ทาย 0 Logistic regression เรียนน้ำหนักของ feature แล้วคืนคะแนนความน่าจะเป็น ตัวอย่างนี้ใช้เพื่อฝึกอ่าน metric ยังไม่ต้องคำนวณวิธีหาน้ำหนักด้วยมือ

## ตามรอยโค้ดและการเลือกโมเดล

ข้อมูลจำลอง 1,500 แถวถูกแบ่งเป็น train 900, validation 300, test 300 แถว `X_dev/y_dev` หมายถึง train รวม validation ก่อนแยกย่อย เราใช้ F1 บน validation เลือกระหว่าง baseline กับ logistic จากนั้น `fit(X_dev, y_dev)` ฝึกวิธีที่เลือกใหม่ด้วยข้อมูล 1,200 แถว แล้วจึงประเมิน test

`classes_` บอกว่าคอลัมน์ใดใน `predict_proba` เป็นคลาส 1 ส่วน average precision (AP) มองการเรียงคะแนนหลายจุดตัดและเป็นหัวข้อต่อยอด รอบแรกอ่าน confusion matrix และ F1 ให้ได้ก่อน ค่า AP ไม่ต้องเท่ากับ F1 เพราะวัดคนละมุม

สำหรับหลายคลาส เช่น Iris: **macro average** เฉลี่ย metric ของทุกคลาสเท่ากัน ส่วน **weighted average** ให้น้ำหนักตามจำนวนตัวอย่าง (`support`) จึงอาจมองไม่เห็นปัญหาของคลาสเล็กได้ง่ายกว่า

## คัดลอกโค้ดไปรัน

คัดลอกบล็อก `python` ด้านล่างทั้งชุดไป Code cell ใน Jupyter หรือ Colab แล้วกด Shift+Enter หากเปิด `lesson.ipynb` ของบทนี้อยู่แล้ว ให้รัน cell ที่มีตามลำดับ ไม่ต้องคัดลอกซ้ำ ดู [วิธีเริ่มเรียน](../../docs/start-here.md) หากยังไม่ได้เตรียมแพ็กเกจ

<details>
<summary>ทางเลือก: รันไฟล์ Python จาก repo ในเครื่อง</summary>

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/04-validation-and-metrics/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` คัดลอกบล็อกต่อไปนี้ได้ทั้งชุด แต่ละส่วน `# %%` คือช่วงการทดลองเดียวกันกับ notebook

</details>

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

## อ่านผลที่ได้

baseline มี accuracy สูงแต่ F1 ต่ำหรือเป็นศูนย์ ได้ confusion matrix ที่แถวเป็นค่าจริง คอลัมน์เป็นค่าทำนาย พร้อม F1/AP บน test ของโมเดลที่เลือก ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## แบบฝึกหัด

หากโจทย์ต้องการลด FN จะสนใจ metric ใด และควรใช้ข้อมูลชุดใดเลือก threshold? ลองวางแผนเทียบ threshold 0.3, 0.5, 0.7 โดยยังไม่แตะ test

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

เริ่มดู recall และติดตาม precision ควบคู่ ใช้ validation เลือก threshold ตามข้อจำกัด ตรึงทางเลือกแล้วจึงประเมิน test ครั้งสุดท้าย การลด threshold มักเพิ่มจำนวน positive ที่ทาย แต่ผลแลกเปลี่ยนต้องวัดจริง

</details>

## อ่านต่อ

- [Metrics and scoring](https://scikit-learn.org/1.7/modules/model_evaluation.html)
- [Cross-validation](https://scikit-learn.org/1.7/modules/cross_validation.html)
- [Google MLCC: Classification](https://developers.google.com/machine-learning/crash-course/classification)

