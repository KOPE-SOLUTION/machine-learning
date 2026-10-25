# 02 · ตรวจข้อมูลสำหรับ ML: leakage, schema และ EDA

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เลือกข้อมูลที่ใช้ทำนายสายพันธุ์ Iris ได้จริง และตรวจชุดฝึกก่อนสร้างโมเดล

**ใช้สิ่งที่เรียนแล้ว:** NumPy, pandas, Matplotlib และ X/y กับ train/test จาก [บท 01](../01-ml-workflow/README.md) จะไม่ทวนคำสั่งพื้นฐาน

## ใจความใหม่

- **Feature:** ข้อมูลที่มีให้ใช้ตอนทำนาย เช่น ขนาดกลีบของดอกไม้ดอกใหม่ที่ยังไม่รู้สายพันธุ์
- **Target leakage:** การนำคำตอบหรือข้อมูลที่ได้จากคำตอบมาใช้เป็น feature เช่น นำ `species_name` ที่แปลงจาก `target` ไปใส่ใน X แม้ลบคอลัมน์ `target` ออกแล้ว ก็ยังมีคำตอบรั่วเข้าโมเดล
- **EDA (การสำรวจข้อมูล):** ใช้สถิติและกราฟตรวจข้อมูลก่อนพัฒนาโมเดล บทนี้กัน test ไว้ก่อน แล้วสำรวจเฉพาะ **train**

## ลงมือทำ

สร้าง `ml-02.ipynb` แล้วรัน **Cell 1–10 แยกกันตามลำดับ** ใน Jupyter เพื่อให้ตารางท้ายแต่ละ cell แสดงครบ ใช้ kernel เดิมจากบท 01

<!-- CODE:CELLS -->

<!-- CODE:START -->
### Cell 1 — เตรียมเครื่องมือ

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
```

เครื่องมือเดิมจากบท 01: รันจบโดยไม่มีข้อความถือว่าปกติ

### Cell 2 — โหลดข้อมูลและอ่านโครงสร้างตาราง

```python
data = load_iris(as_frame=True)
frame = data.frame.copy()

print("Table shape:", frame.shape)
frame.head(3)
```

- **Shape `(150, 5)`:** 150 แถว และ 5 คอลัมน์ ได้แก่ขนาดกลีบสี่ค่าและ target หนึ่งค่า
- **หนึ่งแถว:** ดอกไม้หนึ่งดอก
- **คำศัพท์:** sepal คือกลีบเลี้ยง ส่วน petal คือกลีบดอก โดยขนาดที่วัดมีหน่วยเซนติเมตร

### Cell 3 — เพิ่มคอลัมน์ตัวอย่างและดูการจับคู่รหัสกับชื่อ

```python
frame["sample_id"] = np.arange(len(frame))
label_names = dict(enumerate(data.target_names))
frame["species_name"] = frame["target"].map(label_names)  # สร้างจากคำตอบ: ใช้ทำนายไม่ได้

frame.loc[[0, 50, 100], ["sample_id", "target", "species_name"]]
```

- **`species_name`:** สร้างจากคำตอบใน `target` หากนำไปใช้เป็น feature จะเกิด target leakage
- **`sample_id`:** เป็นลำดับแถวในตัวอย่างนี้ จึงตัดออกจาก X เพราะไม่ใช่ลักษณะดอกไม้

**บรรทัด `.loc`:** แสดงแถว index 0, 50, 100 เพื่อดูหนึ่งตัวอย่างต่อสายพันธุ์ ได้รหัส 0→setosa, 1→versicolor และ 2→virginica

ตารางเดิมยังมี 150 แถว ขั้นตอนนี้ตรวจการจับคู่รหัสกับชื่อ ยังไม่ได้แบ่ง train/test

### Cell 4 — เลือก feature ที่ใช้ได้และแยก X กับ y

```python
features = list(data.feature_names)
forbidden = {"target", "species_name", "sample_id"}
assert forbidden.isdisjoint(features)

X = frame[features]
y = frame["target"]
assert X.index.equals(y.index)

print("Approved features:", features)
print("Excluded columns:", sorted(forbidden))
print("X shape:", X.shape)
print("y shape:", y.shape)
```

ได้ `X: (150, 4)` และ `y: (150,)` โดย X มีสี่คอลัมน์ที่วัดได้ และแถวของ X กับ y ตรงกัน

**Schema:** ข้อตกลงเรื่องชื่อ ลำดับ ชนิดข้อมูล และหน่วยของแต่ละคอลัมน์ ข้อมูลตอนทำนายต้องตรงกับตอนฝึก

ตัวอย่าง: ส่งค่ามิลลิเมตรแทนเซนติเมตรไม่ได้ แม้ shape จะถูกต้อง

### Cell 5 — กัน test แล้วเลือกแถวสำหรับสำรวจ

```python
train_idx, test_idx = train_test_split(
    frame.index, test_size=0.2, stratify=y, random_state=42
)
assert set(train_idx).isdisjoint(test_idx)

train = frame.loc[train_idx].copy()
print("Train rows:", len(train_idx), "Reserved test rows:", len(test_idx))
print("Training class counts:\n", train["target"].value_counts().sort_index())
```

ได้ train 120 แถว (คลาสละ 40) และ test 30 แถว

**ใช้ train เท่านั้น** สำหรับสถิติและกราฟใน cell ถัดไป

การตรวจ index ไม่ซ้ำช่วยกันแถวเดียวกันข้ามชุด แต่ยังไม่รับรองว่าเป็นคนละตัวอย่างจริง ดูกรณีคนเดิมหลายแถวในบท 17

### Cell 6 — สรุปช่วงค่า ค่าหาย และชนิดข้อมูล

```python
summary = train[features].agg(["min", "max", "mean", "std"]).T
summary["missing"] = train[features].isna().sum()
summary["dtype"] = train[features].dtypes.astype(str)

summary.round(3)
```

ตารางมีหนึ่งแถวต่อ feature เพื่อใช้ตรวจช่วงค่า ค่าหาย และชนิดข้อมูล โดยใช้พื้นฐาน pandas ที่เรียนแล้ว

**ผลของชุดนี้:** `missing=0` ทุกคอลัมน์ และชนิดข้อมูลเป็น `float64` ทั้งหมด

### Cell 7 — ตรวจ NaN และค่าที่เป็นอนันต์

```python
matrix = train[features].to_numpy()

assert np.isfinite(matrix).all(), "Handle non-finite values inside a training pipeline"
assert len(features) == matrix.shape[1]

print("Matrix shape:", matrix.shape)
print("All feature values are finite")
```

**ผลตรวจ:** matrix มี shape `(120, 4)` และไม่มี NaN หรือ infinity

การผ่านเงื่อนไขนี้ยังไม่รับรองว่าค่าที่วัดถูกต้องทุกค่า หากพบค่าหาย จะจัดการใน Pipeline ตามบท 06

### Cell 8 — ตรวจ feature ที่ซ้ำและคอลัมน์ค่าคงที่

```python
duplicate_mask = train[features].duplicated(keep=False)
constant_features = train[features].columns[
    train[features].nunique() <= 1
].tolist()

print("Duplicate feature rows in train:", duplicate_mask.sum())
print("Constant features in train:", constant_features)
train.loc[duplicate_mask, features + ["target"]]
```

**ผลตรวจ:** พบ feature ซ้ำ 2 แถว และไม่มีคอลัมน์ค่าคงที่ (`[]`)

**อย่าเพิ่งลบแถวซ้ำ:** ต้องตรวจว่าเป็นคนละดอกที่วัดได้เท่ากัน หรือเป็นการบันทึกดอกเดิมซ้ำ

### Cell 9 — วาดกราฟจาก train และอ่านความหมาย

```python
fig, ax = plt.subplots(figsize=(6, 4))

for label, name in label_names.items():
    subset = train[train["target"] == label]
    ax.scatter(
        subset[features[2]],
        subset[features[3]],
        label=name,
        alpha=0.75
    )

ax.set(
    xlabel=features[2],
    ylabel=features[3],
    title="Approved features: training data only"
)
ax.legend()
fig.tight_layout()
plt.show()
```

**อ่านกราฟ:** ใช้ความยาวกับความกว้างกลีบดอกของ train โดย setosa แยกค่อนข้างชัด ส่วนอีกสองสายพันธุ์ยังทับกันบางส่วน

- สีมาจากคำตอบเพื่อช่วยอ่านภาพ แต่ไม่ส่งเป็น feature
- การเห็นกลุ่มแยกจากกันยังไม่ใช่คะแนนโมเดล
- กราฟแสดงสอง feature แต่ X ที่เลือกไว้ยังมีสี่ feature

### Cell 10 — บันทึกตารางและรูปกราฟ

```python
import os

out = Path("outputs/02-data-and-plots")
if os.name == "nt":
    out = (
        Path.home()
        / "Documents/L/L04_CoreData/machine-learning"
        / "practics/outputs/02-data-and-plots"
    )

out.mkdir(parents=True, exist_ok=True)

summary.to_csv(out / "feature-audit.csv", index_label="feature")
fig.savefig(out / "iris.png", dpi=140)
plt.close(fig)

print("Saved CSV:", (out / "feature-audit.csv").resolve())
print("Saved figure:", (out / "iris.png").resolve())
```

- **Windows:** บันทึกใน `practics/outputs/02-data-and-plots/` ของโปรเจกต์ `Documents/L/L04_CoreData/machine-learning` ใต้โฟลเดอร์ผู้ใช้ โดย `Path.home()` เลือกผู้ใช้ปัจจุบันให้
- **โปรเจกต์อยู่ที่อื่น:** แก้ส่วน `Documents/L/L04_CoreData/machine-learning` ให้ตรงกับตำแหน่งของตัวเอง
- **WSL/Colab:** บันทึกใน `outputs/02-data-and-plots/` ใต้โฟลเดอร์ที่ notebook ทำงาน

สองบรรทัดสุดท้ายแสดงตำแหน่งเต็มของ CSV และ PNG ที่บันทึกแล้ว
<!-- CODE:END -->

## อ่านผล

ผลที่ได้มีสามส่วน:

- รายการ feature สี่คอลัมน์ที่ใช้ทำนายได้
- ตารางตรวจข้อมูล บันทึกเป็น `feature-audit.csv`
- กราฟของชุดฝึก บันทึกเป็น `iris.png`

เปิดไฟล์ตามตำแหน่งที่ Cell 10 แสดง แล้วต่อ [บท 03: KNN และ Pipeline](../03-first-classifier/README.md)

![ข้อมูลฝึกและ feature ที่ผ่านการตรวจ](../../assets/figures/02-data-and-plots-iris.png)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองคิด:** ลบ target แล้วแต่เก็บ species_name ไว้ ใช้เป็น X ได้หรือไม่?

**เฉลย:** ใช้ไม่ได้ เพราะ `species_name` สร้างจากคำตอบ การนำคอลัมน์นี้เข้า X คือ **target leakage**

ผลตรวจที่กรอกหลังรู้สายพันธุ์ก็ต้องตัดออก หากยังไม่มีข้อมูลนั้นตอนทำนาย

</details>

## อ้างอิง

- [scikit-learn: Data leakage](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage)
- [pandas: ตรวจแถวซ้ำ](https://pandas.pydata.org/pandas-docs/version/2.3/reference/api/pandas.DataFrame.duplicated.html)
- [ทบทวนพื้นฐานและลิงก์บทเรียนในช่อง](../../docs/prerequisites.md)
