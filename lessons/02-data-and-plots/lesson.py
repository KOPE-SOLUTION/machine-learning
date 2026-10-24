# %% Cell 1 — เตรียมเครื่องมือ
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# %% [markdown]
# เครื่องมือเดิมจากบท 01 รันจบโดยไม่มีข้อความถือว่าปกติ

# %% Cell 2 — โหลดข้อมูลและอ่านโครงสร้างตาราง
data = load_iris(as_frame=True)
frame = data.frame.copy()

print("Table shape:", frame.shape)
frame.head(3)

# %% [markdown]
# ได้ตาราง `(150, 5)`: ขนาดกลีบสี่ค่าและ target หนึ่งค่า หนึ่งแถวคือดอกไม้หนึ่งดอก; sepal คือกลีบเลี้ยง และ petal คือกลีบดอก หน่วยเซนติเมตร

# %% Cell 3 — เพิ่มคอลัมน์ตัวอย่างและดูการจับคู่รหัสกับชื่อ
frame["sample_id"] = np.arange(len(frame))
label_names = dict(enumerate(data.target_names))
frame["species_name"] = frame["target"].map(label_names)  # สร้างจากคำตอบ: ใช้ทำนายไม่ได้

frame.loc[[0, 50, 100], ["sample_id", "target", "species_name"]]

# %% [markdown]
# `species_name` แปลงมาจากคำตอบ จึงห้ามใช้ทำนาย ส่วน `sample_id` เป็นลำดับแถวในตัวอย่างนี้ ไม่ใช่ลักษณะดอกไม้ จึงตัดออกด้วย
#
# บรรทัด `.loc` แสดง index 0, 50, 100 เพื่อดูหนึ่งตัวอย่างต่อสายพันธุ์ ได้รหัส 0→setosa, 1→versicolor, 2→virginica ตารางเดิมยังมี 150 แถว นี่คือการตรวจการจับคู่ ไม่ใช่การแบ่ง train/test

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
# ได้ `X: (150, 4)` และ `y: (150,)` เลือกเฉพาะสี่คอลัมน์ที่วัดได้ และตรวจว่าแถวของ X/y ตรงกัน
#
# **Schema** คือข้อตกลงเรื่องชื่อ ลำดับ ชนิดข้อมูล และหน่วย ตอนทำนายต้องตรงกับตอนฝึก เช่น ส่งมิลลิเมตรแทนเซนติเมตรไม่ได้ แม้ shape จะถูก

# %% Cell 5 — กัน test แล้วเลือกแถวสำหรับสำรวจ
train_idx, test_idx = train_test_split(
    frame.index, test_size=0.2, stratify=y, random_state=42
)
assert set(train_idx).isdisjoint(test_idx)

train = frame.loc[train_idx].copy()
print("Train rows:", len(train_idx), "Reserved test rows:", len(test_idx))
print("Training class counts:\n", train["target"].value_counts().sort_index())

# %% [markdown]
# ได้ train 120 แถว (คลาสละ 40) และ test 30 แถว ใช้ **train เท่านั้น** สำหรับสถิติและกราฟใน cell ถัดไป
#
# การตรวจ index ไม่ซ้ำช่วยกันแถวเดียวกันข้ามชุด แต่ยังไม่รับรองว่าเป็นคนละตัวอย่างจริง; กรณีคนเดิมหลายแถวอยู่ในบท 17

# %% Cell 6 — สรุปช่วงค่า ค่าหาย และชนิดข้อมูล
summary = train[features].agg(["min", "max", "mean", "std"]).T
summary["missing"] = train[features].isna().sum()
summary["dtype"] = train[features].dtypes.astype(str)

summary.round(3)

# %% [markdown]
# ตารางมีหนึ่งแถวต่อ feature ใช้ตรวจช่วงค่า ค่าหาย และชนิดข้อมูล ชุดนี้ `missing=0` ทุกคอลัมน์และเป็น `float64` ทั้งหมด การอ่านสถิติใช้พื้นฐาน pandas เดิม

# %% Cell 7 — ตรวจ NaN และค่าที่เป็นอนันต์
matrix = train[features].to_numpy()

assert np.isfinite(matrix).all(), "Handle non-finite values inside a training pipeline"
assert len(features) == matrix.shape[1]

print("Matrix shape:", matrix.shape)
print("All feature values are finite")

# %% [markdown]
# ได้ matrix `(120, 4)` และผ่านการตรวจว่าไม่มี NaN หรือ infinity การผ่านนี้ยังไม่รับรองว่าค่าที่วัดถูกต้องทุกค่า หากมีค่าหายจะจัดการใน Pipeline ตามบท 06

# %% Cell 8 — ตรวจ feature ที่ซ้ำและคอลัมน์ค่าคงที่
duplicate_mask = train[features].duplicated(keep=False)
constant_features = train[features].columns[
    train[features].nunique() <= 1
].tolist()

print("Duplicate feature rows in train:", duplicate_mask.sum())
print("Constant features in train:", constant_features)
train.loc[duplicate_mask, features + ["target"]]

# %% [markdown]
# พบ feature ซ้ำ 2 แถว และไม่มีคอลัมน์ค่าคงที่ (`[]`) **อย่าเพิ่งลบแถวซ้ำ**: ต้องแยกว่าเป็นคนละดอกที่วัดได้เท่ากัน หรือบันทึกดอกเดิมซ้ำ

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
# กราฟใช้ความยาวกับความกว้างกลีบดอกของ train: setosa แยกค่อนข้างชัด ส่วนอีกสองสายพันธุ์ยังทับกันบางส่วน
#
# สีใช้คำตอบเพื่อช่วยอ่านภาพ ไม่ส่งเป็น feature และภาพแยกกลุ่มได้ยังไม่ใช่คะแนนโมเดล รายการ feature ที่เลือกยังมีสี่ค่า

# %% Cell 10 — บันทึกตารางและรูปกราฟ
import os

out = Path("outputs/02-data-and-plots")
if os.name == "nt":  # Windows: เลี่ยง Documents ที่บางเครื่องป้องกันการเขียน
    out = Path.home() / "Projects/machine-learning/practics/outputs/02-data-and-plots"

out.mkdir(parents=True, exist_ok=True)

summary.to_csv(out / "feature-audit.csv", index_label="feature")
fig.savefig(out / "iris.png", dpi=140)
plt.close(fig)

print("Saved CSV:", (out / "feature-audit.csv").resolve())
print("Saved figure:", (out / "iris.png").resolve())

# %% [markdown]
# Windows บันทึกที่ `Projects/machine-learning/practics/outputs/02-data-and-plots/` ใต้โฟลเดอร์ผู้ใช้ โดย `Path.home()` เลือกผู้ใช้ปัจจุบันให้ ส่วน WSL/Colab บันทึกใน `outputs/02-data-and-plots/` ใต้โฟลเดอร์ที่ Notebook ทำงาน
#
# ผลสองบรรทัดแสดงตำแหน่งเต็มของ CSV และ PNG ที่บันทึกแล้ว ใช้เปิดไฟล์ได้ทันที
