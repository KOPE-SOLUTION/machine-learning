# %% Cell 1 — เตรียมเครื่องมือ
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# %% [markdown]
# ใช้เครื่องมือชุดเดียวกับบท 01: NumPy สำหรับ array, pandas สำหรับตาราง และ Matplotlib สำหรับกราฟ ส่วน `load_iris` โหลดข้อมูลตัวอย่าง และ `train_test_split` แบ่งข้อมูลฝึกกับข้อมูลทดสอบ
#
# `Path` ใช้จัดการตำแหน่งไฟล์ใน Cell 10 แม้ไม่ได้เรียก `pd` โดยตรงในตัวอย่างนี้ ตารางที่ได้จาก `load_iris(as_frame=True)` ก็เป็น DataFrame ของ pandas
#
# **ผลที่ควรเห็น:** ไม่มีข้อความหรือกราฟ และ cell รันจบโดยไม่มี error

# %% Cell 2 — โหลดข้อมูลและอ่านโครงสร้างตาราง
data = load_iris(as_frame=True)
frame = data.frame.copy()

print("Table shape:", frame.shape)
frame.head(3)

# %% [markdown]
# `as_frame=True` ให้ข้อมูลเป็นตาราง pandas ส่วน `data.frame` รวม feature สี่คอลัมน์กับ `target` ไว้ด้วยกัน จึงต่างจาก `data.data` ในบท 01 ที่มีเฉพาะ feature การใช้ `.copy()` สร้างตารางสำหรับทดลองแยกจากตารางต้นทาง
#
# `frame.shape` คืนค่า (จำนวนแถว, จำนวนคอลัมน์) จึงได้ `Table shape: (150, 5)` หนึ่งแถวแทนดอกไม้หนึ่งดอก ส่วนห้าคอลัมน์ประกอบด้วยข้อมูลที่วัดได้สี่ค่าและคำตอบหนึ่งค่า:
#
# | คอลัมน์ | ความหมาย |
# |---|---|
# | `sepal length (cm)` | ความยาวกลีบเลี้ยง |
# | `sepal width (cm)` | ความกว้างกลีบเลี้ยง |
# | `petal length (cm)` | ความยาวกลีบดอก |
# | `petal width (cm)` | ความกว้างกลีบดอก |
# | `target` | รหัสสายพันธุ์ 0, 1 หรือ 2 |
#
# `frame.head(3)` เลือกสามแถวแรกเพื่อดูหน้าตาตาราง ไม่ได้ลด `frame` ให้เหลือสามแถว เมื่อวาง expression นี้เป็นบรรทัดสุดท้ายของ Code cell Jupyter จะแสดงตารางให้อัตโนมัติโดยไม่ต้องใช้ `print()` หากรันเป็นไฟล์ `.py` จะไม่มีการแสดง expression สุดท้ายแบบ Notebook

# %% Cell 3 — เพิ่มคอลัมน์ตัวอย่างและดูการจับคู่รหัสกับชื่อ
frame["sample_id"] = np.arange(len(frame))
label_names = dict(enumerate(data.target_names))
frame["species_name"] = frame["target"].map(label_names)  # สร้างจากคำตอบ: ใช้ทำนายไม่ได้

frame.loc[[0, 50, 100], ["sample_id", "target", "species_name"]]

# %% [markdown]
# **อ่านทีละบรรทัด**
#
# 1. `len(frame)` ได้ 150 และ `np.arange(150)` สร้างเลข 0 ถึง 149 จากนั้นเก็บเป็นคอลัมน์ `sample_id` หนึ่งค่าต่อหนึ่งแถว เลขนี้เป็นลำดับที่เราสร้างขึ้น ไม่ใช่ค่าที่วัดจากดอกไม้
# 2. `data.target_names` มีชื่อ setosa, versicolor, virginica ตามลำดับ `enumerate(...)` จับคู่แต่ละชื่อกับเลขเริ่มที่ 0 และ `dict(...)` เปลี่ยนคู่เหล่านั้นเป็นตารางจับคู่ในรูป dictionary: `{0: "setosa", 1: "versicolor", 2: "virginica"}`
# 3. `frame["target"].map(label_names)` อ่านรหัสในแต่ละแถวแล้วแทนด้วยชื่อจาก dictionary ผลถูกเก็บในคอลัมน์ `species_name` จึงเป็นข้อมูลคำตอบเดิมที่เปลี่ยนรูปแบบ ไม่ใช่ผลทำนายของโมเดล
# 4. บรรทัดสุดท้ายเลือกบางแถวและบางคอลัมน์มาแสดง เพื่อให้เห็นการจับคู่ครบทั้งสามสายพันธุ์
#
# **แยกส่วนบรรทัด `.loc`**
#
# รูปแบบในที่นี้คือ `frame.loc[รายการ index ของแถว, รายการชื่อคอลัมน์]`
#
# | ส่วนของคำสั่ง | หน้าที่ |
# |---|---|
# | `frame.loc[...]` | เลือกข้อมูลตาม label ของแถวและคอลัมน์ |
# | `[0, 50, 100]` ก่อนเครื่องหมาย comma | เลือกแถวที่มี index label เป็น 0, 50 และ 100 |
# | `["sample_id", "target", "species_name"]` หลัง comma | เลือกเฉพาะสามคอลัมน์นี้ และเรียงตามรายการ |
# | ไม่มี `=` เพื่อกำหนดค่ากลับ | แสดงผลที่เลือก โดยไม่ได้เปลี่ยน `frame` |
#
# วงเล็บเหลี่ยมชั้นนอกเป็นส่วนของ `.loc[...]` ส่วนวงเล็บด้านในสองชุดเป็น list ของแถวและคอลัมน์
#
# **ทำไมเป็น 0, 50, 100?** Iris ที่โหลดในบทนี้เรียงแถว 0–49 เป็น setosa, 50–99 เป็น versicolor และ 100–149 เป็น virginica จึงเลือกตัวอย่างหนึ่งแถวจากแต่ละสายพันธุ์ให้เห็นว่า `.map()` จับคู่ถูกต้อง เลขเหล่านี้ไม่ใช่กฎสำหรับ dataset อื่น ไม่ใช่การสุ่ม และไม่ใช่การแบ่ง train/test
#
# ผลที่ควรเห็น:
#
# | index | sample_id | target | species_name |
# |---:|---:|---:|---|
# | 0 | 0 | 0 | setosa |
# | 50 | 50 | 1 | versicolor |
# | 100 | 100 | 2 | virginica |
#
# คอลัมน์ซ้ายสุดของตารางที่แสดงคือ **index** ไม่ใช่คอลัมน์ข้อมูลที่เพิ่มมาอีกหนึ่งคอลัมน์ แม้ค่าจะตรงกับ `sample_id` ในตัวอย่างนี้ แต่ `.loc` เลือกจาก index ไม่ได้ค้นจากคอลัมน์ `sample_id`
#
# `.loc` ใช้ **index label** ส่วน `.iloc` ใช้ **ตำแหน่งแถว** ที่เริ่มนับจาก 0 หากเปลี่ยน index เป็นข้อความ `.loc[[0, 50, 100], ...]` จะใช้ไม่ได้เมื่อไม่มี label เหล่านี้ การดูสามแถวนี้ก็ยังทำให้ `frame` มี 150 แถวเหมือนเดิม
#
# **เชื่อมกับ ML:** ตอนทำนายดอกไม้ที่ยังไม่รู้สายพันธุ์ เราไม่มี `species_name` ให้ใช้ หากใส่คอลัมน์นี้ใน X จะเป็นการส่งคำตอบเข้าไปด้วย เรียกว่า **target leakage** การลบเฉพาะ `target` จึงยังไม่พอ ส่วน `sample_id` ในบทนี้เป็นเพียงลำดับแถวและอาจสะท้อนการเรียงข้อมูล จึงตัดออกด้วย ไม่ได้หมายความว่า ID ใช้ไม่ได้ในทุกโจทย์
#
# รายละเอียดการเลือกด้วย label อ่านได้ใน [pandas: DataFrame.loc](https://pandas.pydata.org/pandas-docs/version/2.3/reference/api/pandas.DataFrame.loc.html)

# %% Cell 4 — เลือก feature ที่ใช้ได้และแยก X กับ y
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

# %% [markdown]
# `features` เป็น list ของชื่อคอลัมน์ที่วัดจากดอกไม้ทั้งสี่ค่า เราเลือกจากความหมายและข้อมูลที่มีตอนทำนาย ไม่ได้เลือกจากคะแนนบน test วิธีระบุรายการที่อนุญาตไว้ชัดเจนนี้เรียกว่า allowlist
#
# `forbidden` เป็น set ของคอลัมน์ที่ไม่ใช้เป็น feature ในโจทย์นี้ ส่วน `.isdisjoint(features)` ตรวจว่าสองรายการไม่มีชื่อร่วมกัน `assert` ให้โปรแกรมหยุดหากเงื่อนไขเป็นเท็จ การประกาศ `forbidden` เพียงอย่างเดียวไม่ได้ลบคอลัมน์ออกจากตาราง; บรรทัด `X = frame[features]` เป็นผู้เลือกคอลัมน์จริง
#
# `X` เป็น DataFrame ของ feature ส่วน `y` เป็น Series ของคำตอบ เราเลือกทั้งสองจากตารางเดียวกันแล้วใช้ `X.index.equals(y.index)` ตรวจว่า index และลำดับแถวตรงกัน จำนวนแถวเท่ากันอย่างเดียวไม่พอ และการตรวจ index ก็ไม่ได้พิสูจน์ว่าคำตอบต้นฉบับถูกต้องทุกค่า
#
# **ผลที่ควรเห็น:** Approved features มีสี่คอลัมน์ขนาดกลีบ; Excluded columns เป็น `['sample_id', 'species_name', 'target']` โดย `sorted()` เรียงชื่อเพื่อให้อ่านง่าย รูปร่างคือ `X shape: (150, 4)` และ `y shape: (150,)`
#
# **Schema** คือข้อตกลงของข้อมูล เช่น ชื่อและลำดับคอลัมน์ ชนิดข้อมูล และหน่วย ตอนฝึกและทำนายต้องใช้ความหมายเดียวกัน ตัวเลขหน่วยมิลลิเมตรที่ส่งเข้าตำแหน่งซึ่งคาดว่าเป็นเซนติเมตรยังมี shape ถูกต้อง แต่ความหมายผิด ส่วนการจับคู่แถวของ X/y เป็นอีกเรื่องที่ต้องตรวจควบคู่กัน

# %% Cell 5 — กัน test แล้วเลือกแถวสำหรับสำรวจ
train_idx, test_idx = train_test_split(
    frame.index, test_size=0.2, stratify=y, random_state=42
)
assert set(train_idx).isdisjoint(test_idx)

train = frame.loc[train_idx].copy()
print("Train rows:", len(train_idx), "Reserved test rows:", len(test_idx))
print("Training class counts:\n", train["target"].value_counts().sort_index())

# %% [markdown]
# ก่อนหน้านี้เราอ่านโครงสร้างและแสดงการจับคู่รหัสกับชื่อจากข้อมูลที่ทราบอยู่แล้ว ต่อจากนี้สถิติและกราฟที่อาจนำไปใช้ตัดสินใจพัฒนาโมเดลจะมาจาก **train เท่านั้น**
#
# - `frame.index` คือ index ของทุกแถว ครั้งนี้เราส่ง index ไปแบ่งก่อน จึงได้ `train_idx` และ `test_idx` กลับมา
# - `test_size=0.2` กัน 20% ไว้ทดสอบ: 30 แถว และเหลือ train 120 แถว
# - `stratify=y` รักษาสัดส่วนสายพันธุ์ให้ใกล้ข้อมูลเดิม ไม่ได้เพิ่มจำนวนคลาสที่มีน้อยให้เท่าคลาสอื่น
# - `random_state=42` ทำให้แบ่งซ้ำได้เมื่อข้อมูลและเงื่อนไขเดิม เลข 42 ไม่ได้ทำให้โมเดลแม่นขึ้น
# - `set(train_idx).isdisjoint(test_idx)` ตรวจว่าไม่มี index แถวเดียวกันในทั้งสองชุด แต่ไม่ได้ยืนยันว่าคนละแถวเป็นคนละดอกหรือคนละเหตุการณ์จริง
# - `frame.loc[train_idx]` เลือกแถวตามรายการ index โดยใช้ทุกคอลัมน์ เพราะไม่ได้ระบุรายการคอลัมน์หลัง comma จากนั้น `.copy()` สร้างตาราง train สำหรับสำรวจ
# - `value_counts()` นับจำนวนแต่ละรหัสสายพันธุ์ และ `sort_index()` เรียงรหัสเป็น 0, 1, 2 ส่วน `\n` ในข้อความคือการขึ้นบรรทัดใหม่
#
# ผลที่ควรเห็น:
#
# ```text
# Train rows: 120 Reserved test rows: 30
# Training class counts:
# target
# 0    40
# 1    40
# 2    40
# Name: count, dtype: int64
# ```
#
# ทั้งสามคลาสมี train คลาสละ 40 เพราะ Iris ชุดนี้มีคลาสละ 50 ตั้งแต่ต้น เราเก็บ `test_idx` ไว้และไม่ใช้สถิติหรือกราฟจาก test เลือก feature หรือวิธีสร้างโมเดล หลักการนี้ช่วยลด [data leakage](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage)

# %% Cell 6 — สรุปช่วงค่า ค่าหาย และชนิดข้อมูล
summary = train[features].agg(["min", "max", "mean", "std"]).T
summary["missing"] = train[features].isna().sum()
summary["dtype"] = train[features].dtypes.astype(str)

summary.round(3)

# %% [markdown]
# **EDA** (Exploratory Data Analysis) คือการใช้สถิติและกราฟสำรวจข้อมูล ส่วน **data audit** คือการตรวจข้อมูลเป็นขั้นตอน เช่น ตรวจช่วงค่า ค่าหาย และชนิดข้อมูล
#
# `train[features]` เลือกเฉพาะ feature ของชุดฝึก; `.agg([...])` คำนวณหลายสถิติ; `.T` สลับแถวกับคอลัมน์ให้ตารางสรุปมีหนึ่งแถวต่อ feature
#
# | คำสั่ง/คอลัมน์ | อ่านอย่างไร |
# |---|---|
# | `min`, `max` | ค่าน้อยที่สุดและมากที่สุดที่พบ |
# | `mean` | ค่าเฉลี่ย |
# | `std` | ส่วนเบี่ยงเบนมาตรฐาน ช่วยอธิบายการกระจายรอบค่าเฉลี่ย |
# | `.isna().sum()` | ตรวจค่าหายแล้วนับแยกตามคอลัมน์ เก็บใน `missing` |
# | `.dtypes.astype(str)` | อ่านชนิดข้อมูลแล้วแปลงเป็นข้อความ เก็บใน `dtype` |
# | `summary.round(3)` | แสดงตัวเลขทศนิยมสามตำแหน่ง โดยไม่ได้กำหนดกลับให้ `summary` |
#
# pandas จับคู่ผลของ `missing` และ `dtype` กับแถวของ summary ตามชื่อ feature ที่เป็น index
#
# ผลที่ควรเห็น:
#
# | feature | min | max | mean | std | missing | dtype |
# |---|---:|---:|---:|---:|---:|---|
# | sepal length (cm) | 4.3 | 7.9 | 5.842 | 0.841 | 0 | float64 |
# | sepal width (cm) | 2.0 | 4.4 | 3.048 | 0.449 | 0 | float64 |
# | petal length (cm) | 1.1 | 6.9 | 3.770 | 1.769 | 0 | float64 |
# | petal width (cm) | 0.1 | 2.5 | 1.205 | 0.763 | 0 | float64 |
#
# ชุด train นี้ไม่พบค่าหายใน feature และทุกคอลัมน์เป็นตัวเลขทศนิยม สถิตินี้ยังไม่ใช่คะแนนของโมเดล และไม่ได้ทำให้สรุปว่าข้อมูลถูกต้องทุกค่า

# %% Cell 7 — ตรวจ NaN และค่าที่เป็นอนันต์
matrix = train[features].to_numpy()

assert np.isfinite(matrix).all(), "Handle non-finite values inside a training pipeline"
assert len(features) == matrix.shape[1]

print("Matrix shape:", matrix.shape)
print("All feature values are finite")

# %% [markdown]
# `.to_numpy()` แปลงตาราง feature ของ train เป็น array สองมิติ โดยแถวยังแทนดอกไม้และคอลัมน์ยังเรียงตาม `features`
#
# `np.isfinite(matrix)` ตรวจแต่ละตำแหน่งว่าไม่ใช่ NaN, บวกอนันต์ หรือลบอนันต์ แล้ว `.all()` ตรวจว่าทุกตำแหน่งผ่าน ส่วนข้อความหลัง comma ใน `assert` เป็นข้อความที่จะแสดงเมื่อเงื่อนไขไม่ผ่าน
#
# `matrix.shape[1]` คือจำนวนคอลัมน์ จึงเทียบกับจำนวนชื่อใน `features` ได้
#
# ผลที่ควรเห็น:
#
# ```text
# Matrix shape: (120, 4)
# All feature values are finite
# ```
#
# หาก assertion ไม่ผ่าน ให้ตรวจค่าที่ผิดปกติก่อน บท 06 จะอธิบายการจัดการค่าหาย การเติมค่าหายและการปรับสเกลที่ต้องเรียนสถิติ จะเรียนจาก train หรือ training fold ใน Pipeline ไม่ใช้ test มาคำนวณสถิติเหล่านั้น
#
# การตรวจนี้ไม่ตรวจความถูกต้องของหน่วยหรือการวัด ตัวเลขที่วัดผิดก็อาจเป็นค่าจำกัดและผ่านได้

# %% Cell 8 — ตรวจ feature ที่ซ้ำและคอลัมน์ค่าคงที่
duplicate_mask = train[features].duplicated(keep=False)
constant_features = train[features].columns[
    train[features].nunique() <= 1
].tolist()

print("Duplicate feature rows in train:", duplicate_mask.sum())
print("Constant features in train:", constant_features)
train.loc[duplicate_mask, features + ["target"]]

# %% [markdown]
# `duplicated(keep=False)` เปรียบเทียบค่าของ feature ทั้งสี่คอลัมน์และคืน Series ของ True/False โดยทำเครื่องหมาย **ทุกแถว** ที่อยู่ในกลุ่มค่าซ้ำ รวมแถวแรกด้วย ไม่ได้ใช้ index หรือ sample_id มาตัดสินว่าซ้ำ การรวม `duplicate_mask.sum()` จึงนับจำนวนแถวที่ถูกทำเครื่องหมาย
#
# `nunique()` นับจำนวนค่าที่แตกต่างกันของแต่ละคอลัมน์ เงื่อนไข `<= 1` เลือกคอลัมน์ที่มีค่าแตกต่างกันไม่เกินหนึ่งค่า แล้วใช้ `.columns[...].tolist()` ดึงชื่อคอลัมน์เหล่านั้นเป็น list ในขั้นนี้เราได้ตรวจแล้วว่าไม่มีค่าหายหรือค่าอนันต์
#
# **อ่านบรรทัดสุดท้าย:** `train.loc[duplicate_mask, features + ["target"]]` ใช้ boolean mask เลือกเฉพาะแถวที่เป็น True และใช้ `+` ต่อ list ของชื่อ feature สี่ตัวกับ `["target"]` เพื่อแสดงห้าคอลัมน์ `+` ตรงนี้ไม่ได้บวกค่าตัวเลขในตาราง และไม่ได้เพิ่ม target ลงใน X
#
# ผลที่ควรเห็น:
#
# ```text
# Duplicate feature rows in train: 2
# Constant features in train: []
# ```
#
# ตารางแสดง index 101 และ 142 ซึ่งมีค่าตามลำดับเป็น 5.8, 2.7, 5.1, 1.9 และ target เป็น 2 ทั้งคู่ นี่คือ **สองแถวในคู่ซ้ำหนึ่งคู่** ไม่ใช่สองคู่ ส่วน `[]` หมายถึงไม่พบคอลัมน์ค่าคงที่
#
# อย่าลบแถวเพียงเพราะค่าที่วัดเหมือนกัน: อาจเป็นดอกไม้คนละดอกที่มีขนาดเท่ากัน หรือเป็นการบันทึกดอกเดิมซ้ำ ต้องตรวจที่มาและรหัสต้นทาง หากเป็นคนหรือเหตุการณ์เดียวกันหลายแถว ต้องพิจารณาการแบ่งตามกลุ่มในบท 17

# %% Cell 9 — วาดกราฟจาก train และอ่านความหมาย
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

# %% [markdown]
# `plt.subplots()` สร้างรูป `fig` และพื้นที่วาด `ax` ขนาด 6 × 4 นิ้ว จากนั้น `label_names.items()` คืนคู่รหัสกับชื่อให้วนทีละสายพันธุ์ เช่น `label=0`, `name="setosa"`
#
# `train["target"] == label` สร้าง boolean mask ของสายพันธุ์นั้น และ `train[...]` เลือกแถวเป็น `subset`
#
# ชื่อ feature อยู่ใน list ที่เริ่มนับตำแหน่งจาก 0: `features[2]` จึงเป็น petal length และ `features[3]` เป็น petal width เราส่งสองค่านี้เป็นแกนนอนและแกนตั้งของ `scatter()` ส่วน `label=name` ตั้งชื่อชุดจุดสำหรับ legend และ `alpha=0.75` กำหนดความทึบของจุด
#
# `ax.set()` กำหนดชื่อแกนกับชื่อกราฟ; `legend()` แสดงชื่อสายพันธุ์คู่กับสี; `tight_layout()` ช่วยจัดระยะองค์ประกอบ และ `plt.show()` แสดงกราฟ
#
# **อ่านกราฟ:** หนึ่งจุดแทนหนึ่งดอกใน train รวม 120 ดอก แต่บางจุดซ้อนกันได้ setosa อยู่ซ้ายล่างและแยกค่อนข้างชัด ส่วน versicolor กับ virginica มีบริเวณใกล้หรือทับซ้อนกัน
#
# สีมาจาก label ที่ทราบในชุดฝึก ใช้เพื่อสำรวจ ไม่ได้ส่ง label เข้า X กราฟนี้แสดงเพียงสอง feature แต่รายการ feature ที่เลือกไว้ยังมีสี่ตัว การแยกกลุ่มด้วยสายตายังไม่บอกคะแนนทำนายข้อมูลใหม่

# %% Cell 10 — บันทึกตารางและรูปกราฟ
out = Path("outputs/02-data-and-plots")
out.mkdir(parents=True, exist_ok=True)

summary.to_csv(out / "feature-audit.csv", index_label="feature")
fig.savefig(out / "iris.png", dpi=140)
plt.close(fig)

print("Saved CSV:", (out / "feature-audit.csv").resolve())
print("Saved figure:", (out / "iris.png").resolve())

# %% [markdown]
# `Path(...)` ระบุโฟลเดอร์ปลายทางสัมพันธ์กับ working directory ของ Notebook ส่วน `mkdir(parents=True, exist_ok=True)` สร้างโฟลเดอร์แม่ที่ยังไม่มี และยอมให้รันซ้ำเมื่อมีโฟลเดอร์นั้นแล้ว
#
# เครื่องหมาย `/` ระหว่าง Path กับชื่อไฟล์ใช้ต่อเส้นทาง ไม่ใช่หารตัวเลข `to_csv(..., index_label="feature")` บันทึก summary โดยตั้งชื่อคอลัมน์ index เป็น feature; ค่าใน CSV ยังมีความละเอียดเดิม เพราะ Cell 6 ปัดทศนิยมเฉพาะผลที่แสดง
#
# `fig.savefig(..., dpi=140)` บันทึกรูปจากตัวแปร fig ที่ยังเก็บไว้หลังแสดงกราฟใน Cell 9 และ `plt.close(fig)` ปิดรูปที่ใช้งานเสร็จแล้ว `.resolve()` แสดงตำแหน่งเต็มเพื่อให้หาไฟล์ได้ง่าย
#
# **ผลที่ควรได้:** ไฟล์ `feature-audit.csv` และ `iris.png` ในโฟลเดอร์ `outputs/02-data-and-plots/` ตำแหน่งเต็มต่างกันได้ตามเครื่อง การรันซ้ำจะเขียนทับไฟล์สองชื่อนี้ บทนี้บันทึกผลการสำรวจ ยังไม่ได้ฝึกหรือบันทึกโมเดล
