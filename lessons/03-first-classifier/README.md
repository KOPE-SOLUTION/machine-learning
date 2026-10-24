# 03 · โมเดลแรกด้วย scikit-learn และ KNN

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้ใช้ **KNN** ทำนาย Iris จากขนาดกลีบ พร้อมรวมการปรับสเกลและโมเดลเป็น Pipeline

**ใช้สิ่งที่เรียนแล้ว:** X/y, baseline และการเลือก feature จากบท 01–02

## ใจความใหม่

KNN หาเพื่อนบ้านใกล้ที่สุด `k` ตัวแล้วลงคะแนน เช่น 5 ดอกเป็น versicolor 3 ดอกและ virginica 2 ดอก จะทาย versicolor เมื่อให้น้ำหนักเท่ากัน k เล็กไวต่อ noise ส่วน k ใหญ่อาจกลบรายละเอียด

ระยะทางไวต่อสเกล จึงใช้ `StandardScaler` แปลงด้วย `z = (x − ค่าเฉลี่ย train) / ส่วนเบี่ยงเบน train` ค่าที่เท่าค่าเฉลี่ยจะเป็น 0; การ scale ไม่ได้ทำให้ข้อมูลแจกแจงปกติ

| การเรียก Pipeline | สิ่งที่เกิดขึ้น |
|---|---|
| `fit(X_train, y_train)` | scaler เรียนสถิติและแปลง train → KNN เก็บตัวอย่างพร้อมคำตอบ |
| `predict(X_new)` | scaler ใช้สถิติเดิม → KNN หาเพื่อนบ้านและคืนคำตอบ |

กำหนด `k=5` ไว้ก่อนในบทนี้ หากจะเลือก k ใช้ CV ตามบท 10 ไม่เลือกจากคะแนน test ข้อมูลใหม่ต้องมีสี่คอลัมน์ในลำดับและหน่วยเดิม

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

<!-- CODE:START -->
```python
"""Pipeline แรก: scale -> K-nearest neighbors"""
# %% เตรียมข้อมูล
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, stratify=data.target, random_state=42
)

# %% scaler เรียนค่าเฉลี่ย/ส่วนเบี่ยงเบนจาก train เท่านั้น
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Baseline accuracy: {accuracy_score(y_test, baseline.predict(X_test)):.3f}")
print(f"KNN test accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, target_names=data.target_names))

# %% ข้อมูลใหม่: 4 คอลัมน์ในลำดับเดิม หน่วยเซนติเมตร
new_flower = [[5.1, 3.5, 1.4, 0.2]]
label = model.predict(new_flower)[0]
print("New flower:", data.target_names[label])
```
<!-- CODE:END -->

## อ่านผล

เทียบ accuracy ของ KNN กับ baseline และดูดอกใหม่ที่ทายเป็น **setosa** รายงาน precision/recall/F1 จะอ่านโดยละเอียดใน [บท 04](../04-validation-and-metrics/README.md)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองคิด:** fit scaler ก่อนแบ่ง train/test ได้หรือไม่?

**เฉลย:** ไม่ได้ เพราะสถิติจะปนข้อมูล test ต้องแบ่งก่อน แล้ว fit Pipeline เฉพาะ train; test ใช้สถิติเดิม

</details>

## อ้างอิง

- [Handbook: Introducing Scikit-Learn](https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html)
- [scikit-learn: Nearest Neighbors](https://scikit-learn.org/1.7/modules/neighbors.html)
- [Common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html)
