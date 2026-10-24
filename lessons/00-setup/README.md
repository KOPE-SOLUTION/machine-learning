# 00 · ตรวจ environment และสัญญาข้อมูลของ scikit-learn

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลา: 10–15 นาที · เป็นบทตรวจความพร้อมก่อนเนื้อหา ML  
พื้นฐาน: ใช้ Python, notebook, NumPy, pandas และ Matplotlib ได้แล้ว ดู [ทักษะพื้นฐานและวิดีโอทบทวน](../../docs/prerequisites.md)

## ตรวจสิ่งที่หลักสูตรต้องใช้

ใช้ environment แยกตาม [คู่มือติดตั้ง](../../docs/setup.md) เพื่อให้รุ่นแพ็กเกจตรงกัน จากนั้นรันโปรแกรมด้านล่างจาก kernel ใหม่ บทนี้ตรวจ interpreter และข้อมูลที่ scikit-learn คืนมา โดยไม่ทบทวนวิธีใช้ notebook หรือ indexing ตั้งแต่ต้น

สัญญา input ของ estimator ส่วนใหญ่คือ X มี shape (n_samples, n_features) และ y มีคำตอบหนึ่งค่าต่อแถว NumPy shape ถูกต้องอย่างเดียวยังไม่พอ: ลำดับตัวอย่างใน X/y ต้องตรงกัน และคอลัมน์ตอน predict ต้องมีลำดับ/หน่วยเหมือนตอน fit

`load_iris(as_frame=True)` ทำให้ตรวจชื่อคอลัมน์และ index ได้ก่อนส่งข้อมูลเข้าโมเดล ส่วนรหัส 0,1,2 เป็น label ของสายพันธุ์ ไม่ใช่ค่าที่ควรนำมาเฉลี่ยหรือตีความเป็นลำดับ

## รันจาก root ของ repo

```bash
uv run --frozen python lessons/00-setup/lesson.py
```

หากใช้ venv ให้ใช้ Python ของ environment นั้นแทน `uv run --frozen python`

<!-- CODE:START -->
```python
"""ตรวจสภาพแวดล้อมและรูปร่างข้อมูลก่อนเริ่มเรียน"""
# %% ตรวจรุ่นเครื่องมือ
import importlib.metadata
import platform

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

## ตรวจความเข้าใจเรื่องแถวข้อมูล

หากสุ่มลำดับเฉพาะ y แล้วแปลง X และ y เป็น NumPy array ก่อน fit จะเกิดอะไรขึ้น แม้ shape ยังถูกต้อง? อธิบายความต่างระหว่างการจับคู่ด้วย index กับการจับคู่ตามตำแหน่ง

<details>
<summary>แนวคำตอบ</summary>

โมเดลอาจเรียนจากคำตอบของคนละตัวอย่างโดยไม่เกิด shape error เพราะ array จับคู่ตามตำแหน่ง ควรเลือก/แยก X,y จากตารางหรือ index ชุดเดียวกัน และตรวจ alignment ก่อนแปลงเป็น array การสุ่มหรือ reset_index เพียงฝั่งเดียวไม่ได้แก้ความหมายของแถว

</details>

## อ่านต่อ

- [scikit-learn: load_iris](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_iris.html)
- [ทักษะพื้นฐานและวิดีโอทบทวน](../../docs/prerequisites.md)
- [คู่มือติดตั้งของหลักสูตร](../../docs/setup.md)
