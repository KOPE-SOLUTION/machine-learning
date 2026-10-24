# 00 · เปิด notebook และตรวจข้อมูลก่อนเริ่ม ML

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลา: 10–15 นาทีเมื่อติดตั้งแพ็กเกจพร้อมแล้ว

พื้นฐาน: ใช้ Python, notebook, NumPy, pandas และ Matplotlib ได้แล้ว ดู [ทักษะพื้นฐานและวิดีโอทบทวน](../../docs/prerequisites.md)

## ก่อนเริ่มบทนี้

เปิด notebook ที่ใช้เรียน NumPy, pandas และ Matplotlib ได้อยู่แล้ว ขั้นตอนหลักใช้ Jupyter แต่เลือก Colab ได้เช่นกัน หากเปิดไฟล์ notebook ของบทนี้จาก repo อยู่แล้ว ให้รัน cell ที่มีตามลำดับ

บทนี้ตรวจเครื่องมือและอ่านข้อมูลตารางก่อนเริ่ม ML คำใหม่คือ sample, feature และ target แล้วจะไปทดลองโมเดลที่บท 01–03 ผู้ที่ยังไม่มีแพ็กเกจให้ดู [วิธีเริ่มเรียนและ cell ติดตั้ง](../../docs/start-here.md) ส่วนการสร้าง environment ใหม่ด้วย uv/venv อยู่ใน [คู่มือตั้งเครื่อง](../../docs/setup.md)

## ตรวจสิ่งที่หลักสูตรต้องใช้

รันโปรแกรมด้านล่างจาก notebook ที่เริ่มใหม่ บทนี้ตรวจ interpreter และข้อมูลที่ scikit-learn คืนมา โดยไม่ทบทวนวิธีใช้ notebook หรือ indexing ตั้งแต่ต้น

สัญญา input ของ estimator ส่วนใหญ่คือ X มี shape (n_samples, n_features) และ y มีคำตอบหนึ่งค่าต่อแถว NumPy shape ถูกต้องอย่างเดียวยังไม่พอ: ลำดับตัวอย่างใน X/y ต้องตรงกัน และคอลัมน์ตอน predict ต้องมีลำดับ/หน่วยเหมือนตอน fit

`load_iris(as_frame=True)` ทำให้ตรวจชื่อคอลัมน์และ index ได้ก่อนส่งข้อมูลเข้าโมเดล ส่วนรหัส 0,1,2 เป็น label ของสายพันธุ์ ไม่ใช่ค่าที่ควรนำมาเฉลี่ยหรือตีความเป็นลำดับ

## คำใหม่ที่ใช้กับตารางเดิม

ใน ML เราเรียกหนึ่งแถวว่า **sample** หรือตัวอย่าง และคอลัมน์ที่ใช้ทำนายว่า **feature** ส่วนคำตอบที่ต้องการทำนายคือ **target** เช่น หนึ่งแถวเป็นดอกไม้หนึ่งดอก ใช้ขนาดกลีบเป็น feature และสายพันธุ์เป็น target

| ชื่อในโค้ด | ความหมาย | รูปร่างในบทนี้ |
|---|---|---|
| `X` | ตาราง feature ที่ให้โมเดลดู | 150 แถว × 4 คอลัมน์ |
| `y` | คำตอบที่ตรงกับแต่ละแถว | 150 ค่า |
| `data.target_names` | ชื่อที่รหัส 0, 1, 2 แทนอยู่ | setosa, versicolor, virginica |

`sklearn` คือชื่อที่ใช้ import ไลบรารี **scikit-learn** ซึ่งมีเครื่องมือเตรียมข้อมูลและโมเดล ML ส่วน `estimator` คือ object ที่มีวิธี `fit` สำหรับเรียนจากข้อมูล จะได้ลองใช้จริงในบท 01–03

`assert` ตรวจเงื่อนไข: ถ้าถูกโปรแกรมทำต่อ ถ้าผิดจะเกิด `AssertionError` จึงไม่ต้องรอให้โมเดลทำนายผิดก่อนรู้ว่า shape ไม่ตรง อ่านชื่อคอลัมน์จาก `X.head()` จะพบ sepal (กลีบเลี้ยง) และ petal (กลีบดอก) อย่างละความยาว/ความกว้าง หน่วยเซนติเมตร

## คัดลอกโค้ดไปรัน

คัดลอกบล็อก `python` ด้านล่างทั้งชุดไป Code cell ใน Jupyter หรือ Colab แล้วกด Shift+Enter หากเปิด `lesson.ipynb` ของบทนี้อยู่แล้ว ให้รัน cell ที่มีตามลำดับ ไม่ต้องคัดลอกซ้ำ ดู [วิธีเริ่มเรียน](../../docs/start-here.md) หากยังไม่ได้เตรียมแพ็กเกจ

<details>
<summary>ทางเลือก: รันไฟล์ Python จาก repo ในเครื่อง</summary>

```bash
uv run --frozen python lessons/00-setup/lesson.py
```

หากใช้ venv ให้ใช้ Python ของ environment นั้นแทน `uv run --frozen python`

</details>

<!-- CODE:START -->
```python
"""ตรวจสภาพแวดล้อมและรูปร่างข้อมูลก่อนเริ่มเรียน"""
# %% ตรวจรุ่นเครื่องมือ
import importlib.metadata
import platform
import sys

print("Interpreter:", sys.executable)
print("Python:", platform.python_version())
for package in ("numpy", "pandas", "scikit-learn", "matplotlib"):
    print(f"{package}: {importlib.metadata.version(package)}")

# %% อ่านข้อมูลที่ติดมากับ scikit-learn
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
X, y = data.data, data.target
assert X.shape == (150, 4)
assert y.shape == (150,)
print("X shape:", X.shape, "y shape:", y.shape)
print("Classes:", data.target_names)
print(X.head(3).to_string(index=False))
```
<!-- CODE:END -->

## ผลตรวจที่คาด

เห็น X shape (150,4), y shape (150,), ชื่อสามสายพันธุ์และรุ่นแพ็กเกจที่ตรงกับ requirements.txt เมื่อผ่านแล้วให้เริ่ม [บท 01](../01-ml-workflow/README.md)

<details>
<summary>เมื่อ import ไม่ผ่าน: ตรวจ Python ของ notebook</summary>

บรรทัด `Interpreter:` แสดง Python ที่รัน cell นี้ หากติดตั้งแพ็กเกจแล้วแต่ notebook ยังหาไม่พบ อาจเลือก kernel คนละ environment กับที่ติดตั้งไว้ ดู [วิธีแก้ปัญหา](../../docs/start-here.md) และ [คำอธิบาย interpreter/kernel](../../docs/setup.md) เมื่อต้องการตรวจต่อ

</details>

## ตรวจความเข้าใจเรื่องแถวข้อมูล

หากสุ่มลำดับเฉพาะ y แล้วแปลง X และ y เป็น NumPy array ก่อน fit จะเกิดอะไรขึ้น แม้ shape ยังถูกต้อง? อธิบายความต่างระหว่างการจับคู่ด้วย index กับการจับคู่ตามตำแหน่ง

<details>
<summary>แนวคำตอบ</summary>

โมเดลอาจเรียนจากคำตอบของคนละตัวอย่างโดยไม่เกิด shape error เพราะ array จับคู่ตามตำแหน่ง ควรเลือก/แยก X,y จากตารางหรือ index ชุดเดียวกัน และตรวจ alignment ก่อนแปลงเป็น array การสุ่มหรือ reset_index เพียงฝั่งเดียวไม่ได้แก้ความหมายของแถว

</details>

## อ่านต่อ

- [Python: venv](https://docs.python.org/3.12/library/venv.html)
- [uv: working on projects](https://docs.astral.sh/uv/guides/projects/)
- [scikit-learn: load_iris](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_iris.html)
- [ทักษะพื้นฐานและวิดีโอทบทวน](../../docs/prerequisites.md)
- [คู่มือติดตั้งของหลักสูตร](../../docs/setup.md)
