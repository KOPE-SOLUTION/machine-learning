# M02 · ใช้ scikit-learn ร่วมกับ Mojo

[กลับเส้นทาง Mojo](../README.md) · [Mojo](main.mojo) · [Python reference](reference.py)

เวลาเรียนประมาณ 45–60 นาที เรียน [บท 03](../../lessons/03-first-classifier/README.md) และ M01 ก่อนเริ่ม

**สถานะ: ตัวอย่าง Mojo ยังไม่ผ่านการ compile/run ในการตรวจรอบนี้** อ่าน [environment และข้อจำกัด](../README.md)

## แนวคิด

`Python.import_module` นำ Python module มาใช้ผ่าน PythonObject เช่น sklearn.datasets และ sklearn.pipeline ตัวแปร `split` มีผลลัพธ์สี่รายการในลำดับ X_train, X_test, y_train, y_test จึงต้องจับ index ให้ถูก

`def main() raises` ระบุว่าการเรียก Python อาจโยนข้อผิดพลาดได้ ตัวอย่างสร้าง Pipeline เดียวกับบท 03 และ split ด้วย seed เดียวกัน จึงใช้ Python reference เป็นตัวตรวจเส้นทางการเรียก ไม่ใช่การแข่งขันของสอง algorithm

## รันจาก root ของ repo

หลัง activate environment ตามคู่มือ Mojo:

```bash
mojo mojo/02-python-interop/main.mojo
python mojo/02-python-interop/reference.py
```

## โค้ด

<!-- MOJO:START -->
```mojo
# Target syntax: Mojo documentation 1.2.0; compiler execution not verified here.
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

## อ่านผลและฝึกต่อ

เปรียบเทียบ test accuracy และ prediction ห้าตัวแรกกับ Python reference ใน environment เดียวกัน ค่าเหมือนกันสนับสนุนว่าเรียก pipeline และข้อมูลถูกต้อง แต่ยังต้องมี compiler ที่ทำงานได้ก่อนยืนยันผลของ Mojo

ลองเปลี่ยน n_neighbors เป็น 3 ในทั้งสองไฟล์ แล้วตรวจว่าข้อมูล split และ output ตรงกัน หากจะเลือก k เพื่อทำโมเดลสุดท้าย ให้ใช้ CV ตามบท 10

<details><summary>แนวคำตอบ</summary>

การเรียก library จาก Mojo ไม่ได้ทำให้การเลือก k จาก test ถูกต้องขึ้น วิธีประเมิน ML เหมือนเดิม และเวลาที่วัดรวม startup/interpreter อาจบอกต้นทุนการเรียกมากกว่าความเร็วของ training kernel

</details>

## อ้างอิง

- [Python interoperability](https://mojolang.org/docs/manual/python/)
- [Calling Python from Mojo](https://mojolang.org/docs/manual/python/python-from-mojo/)
- [scikit-learn Pipeline](https://scikit-learn.org/1.7/modules/compose.html)

