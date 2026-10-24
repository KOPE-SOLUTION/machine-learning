# 02 · ตรวจข้อมูลสำหรับ ML: leakage, schema และ EDA

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลอง: 35–50 นาที  
สิ่งที่จะทำได้: ใช้ทักษะพื้นฐาน pandas/NumPy/Matplotlib ตรวจ feature ที่ใช้ได้จริง รักษา X/y alignment และสำรวจ train โดยไม่ใช้ test เลือกโมเดล

## ก่อนเริ่มบทนี้

ต่อจากบท 01 ที่แยก train/test แล้ว ใช้ pandas และ Matplotlib ที่คุ้นเคยมาตรวจว่าคอลัมน์ใดควรเข้าโมเดล บทนี้ยังไม่ฝึก classifier

## ก่อนคำนวณ ต้องรู้ว่าหนึ่งแถวแทนอะไร

ตารางที่ merge/groupby ได้ถูกต้องอาจยังไม่เหมาะกับการเรียนรู้: หนึ่งคนอาจมีหลายแถว หนึ่งแถวอาจรวมเหตุการณ์หลังเวลาทำนาย หรือ index ของ X กับ y อาจสลับกัน บทนี้หนึ่งแถวคือดอกไม้หนึ่งดอก และเป้าหมายคือทำนายสายพันธุ์จากขนาดกลีบที่วัดได้

ตรวจสามเรื่องก่อนเลือกโมเดล: หน่วยตัวอย่าง, เวลาที่รู้แต่ละคอลัมน์ และที่มาของ target Shape ถูกต้องแต่ label ผิดแถวยังทำให้ fit ผ่านได้ จึงไม่ควรใช้เพียงขนาด array เป็นหลักฐานว่าข้อมูลพร้อม

## แยก feature ออกจากคำตอบและรหัสระบุตัวอย่าง

ตัวอย่างเพิ่มสองคอลัมน์ให้เห็นความเสี่ยง:

| คอลัมน์ | ที่มา | ใช้เป็น feature ในโจทย์นี้หรือไม่ |
|---|---|---|
| ขนาดกลีบ 4 ค่า | วัดจากดอกไม้ก่อนรู้คำตอบ | ใช้ได้ |
| target | สายพันธุ์ที่ต้องทำนาย | เป็น y |
| species_name | แปลง target เป็นชื่อสายพันธุ์ | ใช้ไม่ได้: target leakage |
| sample_id | หมายเลขแถวในชุดข้อมูล | ตัดออก: เป็นรหัสแถว ไม่มีความหมายสำหรับทำนายดอกใหม่ |

การเขียน `frame.drop(columns="target")` อย่างเดียวไม่พอ เพราะ species_name ยังเฉลยคำตอบอยู่ จึงกำหนดรายการ feature ที่อนุญาตจากความหมายของข้อมูล แล้วตรวจว่าไม่มีคอลัมน์ต้องห้าม

ID ไม่ใช่สิ่งต้องห้ามในทุกโจทย์ แต่ต้องมีเหตุผลตามการใช้งาน ตัวอย่างนี้ ID สืบทอดลำดับแถวของ dataset จึงอาจจับรูปแบบการจัดเก็บแทนลักษณะของดอกไม้

## EDA ที่มีผลต่อการสร้างโมเดล

กัน test ก่อนดูสถิติหรือกราฟเพื่อเลือก feature จากนั้นใช้ทักษะเดิมตรวจค่าหาย ช่วงค่า หน่วย คอลัมน์คงที่ แถว feature ซ้ำ และสัดส่วนคลาสบน train สถิตินี้ช่วยกำหนดสมมติฐาน ไม่ใช่คะแนนทำนาย

แถวที่มี feature เท่ากันไม่ได้แปลว่าเป็นตัวอย่างเดียวกันเสมอ อย่าลบทิ้งอัตโนมัติ ต้องตรวจหน่วยข้อมูลและรหัสต้นทาง ถ้าพบคนหรือเหตุการณ์เดียวกันหลายแถว ให้ออกแบบ group split ตามบท 17

การเติมค่าหายหรือ scale ด้วยสถิติทั้ง dataset ก่อน split จะนำข้อมูลจาก test มาใช้ เราจะให้ Pipeline เรียนสถิติจาก training fold ในบท 03 และ 06 สีใน scatter ใช้ label เพื่ออ่านรูปแบบได้ แต่ label ต้องไม่ถูกส่งเข้า X

## แปลคำว่า audit, schema และ leakage

**Data audit** คือการตรวจข้อมูลอย่างเป็นขั้นตอน ส่วน **schema** คือข้อตกลงเรื่องคอลัมน์ ชนิดข้อมูล หน่วยและลำดับแถว ตัวอย่างนี้ใช้ sepal length/width และ petal length/width หน่วยเซนติเมตร หากเปลี่ยนหน่วยเฉพาะตอนทำนาย แม้ shape เดิมก็ผิดข้อตกลง

**Leakage** เกิดเมื่อข้อมูลที่ไม่ควรรู้ในการทำนายมีอิทธิพลต่อการฝึกหรือเลือกโมเดล เช่น ใส่ `species_name` ที่แปลมาจากคำตอบลงใน X จะคล้ายส่งเฉลยให้โมเดล ผลที่สูงจึงไม่บอกว่าจะทายดอกใหม่ได้ดี **EDA** (exploratory data analysis) คือการใช้สถิติและกราฟสำรวจรูปแบบ โดยบทนี้ใช้เฉพาะ train

## ตามรอยโค้ด

1. `features` เลือกสี่คอลัมน์ตามความหมายข้อมูล ส่วน `forbidden` ใช้ตรวจว่าคำตอบและรหัสแถวไม่ปนมา
2. `train_idx/test_idx` เป็น index ของแถว การเลือกด้วย `.loc` จึงรักษาคู่ระหว่างขนาดดอกไม้กับสายพันธุ์ไว้
3. ตาราง `summary` มีหนึ่งแถวต่อ feature: `missing` นับค่าหาย, `std` บอกการกระจาย และ `dtype` บอกชนิดข้อมูล
4. `np.isfinite` ตรวจว่าไม่มี NaN หรือ infinity หากไม่ผ่าน อย่าลบ assertion ทิ้งเพื่อให้รันต่อ ให้ตรวจข้อมูลและเรียนการแทนค่าหายในบท 06
5. กราฟใช้ petal length/width สองค่า แต่รายการ feature ที่อนุญาตยังมีสี่ค่า สีเป็นคำตอบสำหรับอ่านภาพ ไม่ใช่คอลัมน์เพิ่มให้โมเดล

## ลงมือรัน

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/02-data-and-plots/lesson.py
```

หากใช้ venv ให้ใช้ Python ของ environment นั้นแทน `uv run --frozen python` โค้ดรันได้ทั้งชุดโดยไม่ต้องรันบทก่อนหน้า

<!-- CODE:START -->
```python
"""ตรวจ feature schema และความเสี่ยง leakage ก่อนสร้างโมเดล"""
# %% กำหนดหน่วยตัวอย่างและข้อมูลที่ใช้ได้ ณ เวลาทำนาย
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris(as_frame=True)
frame = data.frame.copy()
frame["sample_id"] = np.arange(len(frame))
label_names = dict(enumerate(data.target_names))
frame["species_name"] = frame["target"].map(label_names)  # สร้างจากคำตอบ: ใช้ทำนายไม่ได้

# ระบุ allowlist ตามความหมายข้อมูล ไม่เลือกทุกคอลัมน์นอกจาก target อัตโนมัติ
features = list(data.feature_names)
forbidden = {"target", "species_name", "sample_id"}
assert forbidden.isdisjoint(features)
X, y = frame[features], frame["target"]
assert X.index.equals(y.index)
print("Approved features:", features)
print("Excluded columns:", sorted(forbidden))

# %% กัน test ก่อนใช้กราฟและสถิติเพื่อตัดสินใจเกี่ยวกับโมเดล
train_idx, test_idx = train_test_split(
    frame.index, test_size=0.2, stratify=y, random_state=42
)
assert set(train_idx).isdisjoint(test_idx)
train = frame.loc[train_idx].copy()
print("Train rows:", len(train_idx), "Reserved test rows:", len(test_idx))
print("Training class counts:\n", train["target"].value_counts().sort_index())

# %% ตรวจคุณภาพและยืนยันว่า target ไม่ได้ปะปนใน feature
summary = train[features].agg(["min", "max", "mean", "std"]).T
summary["missing"] = train[features].isna().sum()
summary["dtype"] = train[features].dtypes.astype(str)
print("Training feature audit:\n", summary.round(3))
matrix = train[features].to_numpy()
assert np.isfinite(matrix).all(), "Handle non-finite values inside a training pipeline"
assert len(features) == matrix.shape[1]
print("Duplicate feature rows in train:", train[features].duplicated(keep=False).sum())
print("Constant features in train:", train[features].columns[train[features].nunique() <= 1].tolist())

# %% หนึ่งจุดคือหนึ่งดอก สีใช้ label เพื่อสำรวจ แต่ label ไม่เข้าสู่ X
fig, ax = plt.subplots(figsize=(6, 4))
for label, name in label_names.items():
    subset = train[train["target"] == label]
    ax.scatter(subset[features[2]], subset[features[3]], label=name, alpha=0.75)
ax.set(xlabel=features[2], ylabel=features[3], title="Approved features: training data only")
ax.legend()
fig.tight_layout()
out = Path("outputs/02-data-and-plots")
out.mkdir(parents=True, exist_ok=True)
summary.to_csv(out / "feature-audit.csv", index_label="feature")
fig.savefig(out / "iris.png", dpi=140)
plt.show()
plt.close(fig)
```
<!-- CODE:END -->

## อ่านผลที่ได้

เห็น feature ที่อนุญาตสี่ค่า คอลัมน์ที่ถูกตัดออก train 120 แถว และ test ที่กันไว้ 30 แถว โปรแกรมบันทึก `feature-audit.csv` และ `iris.png` ใน `outputs/02-data-and-plots/` โดยยังไม่ใช้ test เลือก feature

## ภาพตัวอย่างจากการรัน

![ข้อมูลฝึกและ feature ที่ผ่านการตรวจ](../../assets/figures/02-data-and-plots-iris.png)

ขนาดกลีบสองแกนอาจช่วยแยกบางคลาสได้ แต่ภาพนี้ยังไม่ยืนยันประสิทธิภาพกับข้อมูลใหม่

## แบบฝึกหัด

1. เพิ่มคอลัมน์ `predicted_species_from_expert` ซึ่งผู้เชี่ยวชาญกรอกหลังทราบผลตรวจแล้ว ควรใช้ทำนายตอนวัดดอกไม้หรือไม่? อธิบายจากเวลาที่ข้อมูลมีให้ใช้
2. หาก merge ตารางการวัดกับตาราง label แล้วจำนวนแถวเพิ่มเป็นสองเท่า จะตรวจ key และความสัมพันธ์ของตารางอย่างไร?
3. ถ้าพบสัดส่วนคลาสต่างกันมาก ควรเปลี่ยนสิ่งใดในแผนประเมินก่อนเริ่มลอง algorithm?

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

1. ใช้ไม่ได้ในเวลาทำนายที่กำหนด แม้ชื่อคอลัมน์จะไม่ได้มีคำว่า target เพราะเป็นข้อมูลที่รู้ภายหลัง
2. ตรวจ key ซ้ำและความสัมพันธ์ one-to-one/one-to-many ใช้ `merge(validate="one_to_one")` เมื่อสัญญาข้อมูลควรเป็นแบบนั้น ก่อนตรวจว่า X/y ยังแทนตัวอย่างชุดเดียวกัน ไม่แก้ด้วยการสุ่มลบครึ่งหนึ่ง
3. เพิ่ม baseline และเลือก metric ที่สะท้อนคลาสสำคัญ เช่น precision/recall หรือ macro F1 ตามบท 04 พร้อมตรวจความเหมาะสมของ stratification และหน่วยการแบ่งข้อมูล

</details>

## อ่านต่อ

- [scikit-learn: Common pitfalls และ leakage](https://scikit-learn.org/1.7/common_pitfalls.html)
- [ทบทวน pandas — KOPE SOLUTION](https://www.youtube.com/watch?v=Ejgb1q1A8uU&list=PLtB6tPOso1cY5OS1uyQjvrktkA-rHw2pU)
- [ทบทวน Matplotlib — KOPE SOLUTION](https://www.youtube.com/watch?v=tUmJrKmaos8&list=PLea-bfI_jPIA)
