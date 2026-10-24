# M02 · ใช้ scikit-learn ร่วมกับ Mojo

[กลับเส้นทาง Mojo](../README.md) · [Mojo](main.mojo) · [Python reference](reference.py)

เวลาเรียนประมาณ 45–60 นาที เรียน [บท 03](../../lessons/03-first-classifier/README.md) และ M01 ก่อนเริ่ม

**ทดสอบแล้วด้วย Mojo 1.2.0 บน Ubuntu 24.04 / WSL และ CPython 3.12.3** ใช้ environment ตาม [คู่มือติดตั้ง](../README.md) เพื่อให้รุ่นตรงกัน

## แนวคิด

`Python.import_module` นำ Python module มาใช้ผ่าน PythonObject เช่น sklearn.datasets และ sklearn.pipeline ตัวแปร `split` มีผลลัพธ์สี่รายการในลำดับ X_train, X_test, y_train, y_test จึงต้องจับ index ให้ถูก

`def main() raises` ระบุว่าการเรียก Python อาจโยนข้อผิดพลาดได้ ตัวอย่างสร้าง Pipeline เดียวกับบท 03 และ split ด้วย seed เดียวกัน จึงใช้ Python reference เป็นตัวตรวจเส้นทางการเรียก ไม่ใช่การแข่งขันของสอง algorithm

## ใครเป็นผู้คำนวณในแต่ละขั้น

Mojo เป็นตัวเรียกคำสั่ง ส่วน `sklearn` และ NumPy ยังทำงานผ่าน Python interpreter ที่โหลดอยู่ `PythonObject` คือ object ฝั่ง Python ที่เข้าถึงจาก Mojo ได้ ไม่ได้แปลงโค้ด KNN ทั้งหมดเป็น Mojo

| ตำแหน่งใน `split` | ความหมาย | ใช้เมื่อ |
|---|---|---|
| 0 | X_train | `fit` |
| 1 | X_test | `score` และ `predict` |
| 2 | y_train | `fit` |
| 3 | y_test | ตรวจคะแนนใน `score` |

`model.score` ของ classifier นี้คืน accuracy ส่วน `[:5]` แสดงคำทำนายห้าแถวแรก `raises` ยอมให้ error จาก Python ส่งกลับมาหาก import หรือเรียก API ไม่สำเร็จ ถ้าเจอ `ModuleNotFoundError` ให้ตรวจ environment ตามหน้าติดตั้งก่อนเปลี่ยนโค้ดโมเดล

## รันจาก root ของ repo

หลัง activate environment ตามคู่มือ Mojo:

```bash
mojo mojo/02-python-interop/main.mojo
python mojo/02-python-interop/reference.py
```

## โค้ด

<!-- MOJO:START -->
```mojo
# Verified with Mojo 1.2.0 on Ubuntu 24.04 / WSL; see mojo/README.md.
from std.python import Python

def main() raises:
    var datasets = Python.import_module("sklearn.datasets")
    var selection = Python.import_module("sklearn.model_selection")
    var pipeline = Python.import_module("sklearn.pipeline")
    var preprocessing = Python.import_module("sklearn.preprocessing")
    var neighbors = Python.import_module("sklearn.neighbors")

    var data = datasets.load_iris(return_X_y=True)
    var split = selection.train_test_split(
        data[0], data[1], test_size=0.2, stratify=data[1], random_state=42
    )
    var model = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        neighbors.KNeighborsClassifier(n_neighbors=5)
    )
    _ = model.fit(split[0], split[2])
    print("test accuracy:", model.score(split[1], split[3]))
    print("first five predictions:", model.predict(split[1])[:5])
```
<!-- MOJO:END -->

## Python สำหรับเทียบผล

คัดลอกบล็อกนี้ไปรันด้วย Python ใน environment เดียวกันได้ครบชุด เปรียบเทียบคำตอบก่อนทดลองเปลี่ยนค่าหรือจับเวลา

<!-- PYTHON:START -->
```python
"""Python baseline for checking that the Mojo interop path calls the same model."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)
print("test accuracy:", model.score(X_test, y_test))
print("first five predictions:", model.predict(X_test)[:5])
```
<!-- PYTHON:END -->

## อ่านผลและฝึกต่อ

เปรียบเทียบ test accuracy และ prediction ห้าตัวแรกกับ Python reference ใน environment เดียวกัน ค่าเหมือนกันสนับสนุนว่าเรียก pipeline และข้อมูลถูกต้อง รอบตรวจด้วย environment ข้างต้นได้ accuracy 0.933333 และคำทำนายห้าตัวแรก [0, 2, 1, 1, 0] ตรงกัน

ลองเปลี่ยน n_neighbors เป็น 3 ในทั้งสองไฟล์ แล้วตรวจว่าข้อมูล split และ output ตรงกัน หากจะเลือก k เพื่อทำโมเดลสุดท้าย ให้ใช้ CV ตามบท 10

<details><summary>แนวคำตอบ</summary>

การเรียก library จาก Mojo ไม่ได้ทำให้การเลือก k จาก test ถูกต้องขึ้น วิธีประเมิน ML เหมือนเดิม และเวลาที่วัดรวม startup/interpreter อาจบอกต้นทุนการเรียกมากกว่าความเร็วของ training kernel

</details>

## อ้างอิง

- [Python interoperability](https://mojolang.org/docs/manual/python/)
- [Calling Python from Mojo](https://mojolang.org/docs/manual/python/python-from-mojo/)
- [scikit-learn Pipeline](https://scikit-learn.org/1.7/modules/compose.html)

