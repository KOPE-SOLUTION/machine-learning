# M02 · ใช้ KNN จาก scikit-learn ร่วมกับ Mojo

[กลับเส้นทาง Mojo](../README.md) · [Mojo](main.mojo) · [Python reference](reference.py)

เวลาเรียนประมาณ 45–60 นาที เรียน [บท 03](../../lessons/03-first-classifier/README.md) และ M01 ก่อนเริ่ม

**ทดสอบแล้วด้วย Mojo 1.2.0 บน Ubuntu 24.04 / WSL และ CPython 3.12.3** ใช้ environment ตาม [คู่มือติดตั้ง](../README.md) เพื่อให้รุ่นตรงกัน

## โมเดลที่เรียกผ่าน Mojo: KNN

โมเดลยังเป็น **KNeighborsClassifier(n_neighbors=5)** จาก [บท 03](../../lessons/03-first-classifier/README.md) รับขนาดกลีบ Iris สี่ค่า หลังปรับสเกล แล้วทายสายพันธุ์จากเสียงของห้าเพื่อนบ้านใกล้ที่สุด การ fit เก็บตัวอย่าง train พร้อมคำตอบ ส่วน predict หาเพื่อนบ้านของแถวที่ส่งเข้าไป

Mojo เรียก Pipeline นี้ผ่าน Python จึงยังต้องรักษาชื่อหรือลำดับและหน่วยของ feature รวมถึง fit scaler เฉพาะ train เช่นเดิม การเปลี่ยนภาษาเรียกใช้งานไม่ได้เปลี่ยนวิธีเรียนของ KNN

![ทบทวน KNN ใช้ห้าเพื่อนบ้านใกล้สุดและโหวตคลาส เมื่อ versicolor มีสามเสียงและ virginica มีสองเสียงจึงเลือก versicolor](../../assets/figures/03-first-classifier-knn.png)

**ทบทวนภาพจากบท 03:** Mojo เรียก KNN แบบห้าเพื่อนบ้านและให้น้ำหนักเท่ากันเช่นเดียวกัน ภาพเป็นตัวอย่างสมมติสองมิติ ส่วนข้อมูล Iris ในโค้ดมีสี่ feature · [เปิดภาพ SVG](../../assets/figures/03-first-classifier-knn.svg)

## สิ่งที่เพิ่มจากบท 03 และ M01

ใช้ KNN/Pipeline เดิม จึงไม่ทวนการ scale, fit และ predict รอบนี้ให้ Mojo เรียก Python ผ่าน `Python.import_module`

- `PythonObject` คือ object ฝั่ง Python ที่เข้าถึงจาก Mojo ได้ ไม่ได้แปลงอัลกอริทึม scikit-learn เป็น Mojo
- `def main() raises` ยอมให้ข้อผิดพลาดจากการเรียก Python ส่งกลับมา
- ผล `split` มีลำดับ `X_train, X_test, y_train, y_test` จึงใช้ index `0, 2` ตอน fit และ `1, 3` ตอนตรวจ accuracy

หากพบ `ModuleNotFoundError` ให้ตรวจ environment ตามคู่มือติดตั้ง การคำนวณของ scikit-learn/NumPy ยังทำผ่าน Python interpreter ที่โหลดอยู่

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

รัน Python ใน environment เดียวกันเพื่อเทียบคำตอบ

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

## อ่านผล

เทียบ accuracy และคำทำนายห้าตัวแรกกับ Python reference ใน environment เดียวกัน รอบตรวจเดิมได้ **0.933333** และ `[0, 2, 1, 1, 0]` ตรงกัน

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** เปลี่ยน `n_neighbors=3` ทั้งสองไฟล์แล้วเทียบผล

**แนวคำตอบ:** split และ Pipeline ต้องตรงกัน การทดลองนี้ตรวจการเรียกข้ามภาษา; หากจะเลือก k ใช้งานจริงให้ใช้ CV ตามบท 10 การเรียกจาก Mojo ไม่รับประกันว่าจะเร็วกว่า เพราะยังมีต้นทุน startup/interpreter

</details>

## อ้างอิง

- [Python interoperability](https://mojolang.org/docs/manual/python/)
- [Calling Python from Mojo](https://mojolang.org/docs/manual/python/python-from-mojo/)
- [scikit-learn Pipeline](https://scikit-learn.org/1.7/modules/compose.html)
