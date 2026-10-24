# 06 · เตรียมตัวเลข หมวดหมู่ และค่าที่หาย

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 50–70 นาที  
สิ่งที่จะทำได้: สร้าง ColumnTransformer, imputation และ one-hot encoding ใน Pipeline ได้

## ก่อนเริ่มบทนี้

ใช้การเลือกคอลัมน์และจัดการ missing จาก pandas ร่วมกับ Pipeline ในบท 03 และ regression ในบท 05 บทนี้เน้นใช้กฎที่เรียนจาก train ซ้ำกับข้อมูลใหม่

## จาก pandas สู่ preprocessing ที่เรียนจาก train

จากพื้นฐาน pandas คุณจัดการค่าหาย รวมตาราง และเลือกคอลัมน์ได้แล้ว จุดใหม่ในบทนี้คือแยกการแปลงแบบมีกฎคงที่ออกจากการแปลงที่ต้องเรียนสถิติ เช่น median, ค่าเฉลี่ย หรือ vocabulary ขั้นที่ต้องเรียนต้อง fit ใหม่ภายใน training fold จึงย้ายมาอยู่ใน Pipeline เพื่อใช้กฎเดิมกับข้อมูลใหม่

## โมเดลต้องเห็น feature ที่มีความหมาย

ตารางส่งสินค้าจำลองมีระยะทาง น้ำหนัก และประเภทบริการ เป้าหมายเป็นค่าบริการจำลองที่มี noise เราแทรกค่าน้ำหนักที่หายเพื่อฝึกการจัดการข้อมูล ไม่ใช่หลักฐานว่าการแทนค่าด้วย median เหมาะกับทุกงาน

Median imputation เรียนค่าทดแทนจาก train แล้วนำค่าเดิมไปใช้กับ test การลบทุกแถวที่มีค่าว่างอาจลดข้อมูลมากหรือเปลี่ยนสัดส่วนกลุ่มผู้ใช้ ควรสำรวจสาเหตุที่หาย และอาจเพิ่ม missing indicator หากมีประโยชน์

## หมวดหมู่ไม่ใช่ลำดับ

การแทน standard=0, express=1, same_day=2 อาจทำให้โมเดลเข้าใจว่าหมวดหมู่มีระยะห่างและลำดับ One-hot encoding สร้างคอลัมน์แยกแทน ในบทนี้ `handle_unknown="ignore"` ทำให้หมวดหมู่ใหม่กลายเป็นศูนย์ทุกคอลัมน์ของ field นั้น โปรแกรมจึงรันต่อได้ แต่ไม่ได้แปลว่าเรียนรู้หมวดหมู่ใหม่แล้ว

`ColumnTransformer` ระบุว่าแต่ละคอลัมน์ผ่านขั้นตอนไหน แล้วต่อกับ Ridge ซึ่งเพิ่มโทษขนาด coefficient เพื่อจำกัดความซับซ้อน พารามิเตอร์ alpha ยิ่งมากยิ่งลงโทษมาก การปรับ alpha ต้องใช้ validation/CV

เมื่อใช้ข้อมูลจริง ให้ตรวจว่า feature มีอยู่ก่อนเวลาทำนาย เช่นค่าบริการสุดท้ายหรือสถานะส่งสำเร็จจะรั่วข้อมูลเข้าสู่โจทย์ที่ต้องทำนายตอนรับคำสั่งซื้อ

## ดูหนึ่งแถวก่อนและหลังแปลง

สมมติ `service` มีแค่ standard กับ express การเข้ารหัสแบบ one-hot จะได้สองคอลัมน์: standard เป็น `[1, 0]` และ express เป็น `[0, 1] ลำดับจริงดูจาก `get_feature_names_out()` ส่วน same_day ที่ไม่เคยอยู่ใน train จะเป็น `[0, 0]` ในตัวอย่างนี้ จึงไม่มีระยะห่างสมมติแบบรหัส 0, 1, 2

**Imputation** คือการแทนค่าที่หาย ค่ากลางอย่าง median ต้องคำนวณจาก train ส่วน **regularization** คือการเพิ่มเงื่อนไขให้โมเดลไม่ใช้ coefficient ใหญ่เกินไป Ridge เป็น linear regression ที่เพิ่มโทษแบบนี้ โดยใช้ `alpha` ควบคุมความแรง

## ตามรอยข้อมูลใน Pipeline

| ข้อมูลเข้า | ขั้นตอน | ผลที่ส่งต่อ |
|---|---|---|
| distance_km, weight_kg | เติม median → scale | ตัวเลขสองคอลัมน์ที่ใช้สถิติ train |
| service | เติมหมวดหมู่ที่พบบ่อย → one-hot | คอลัมน์บอกการเป็นสมาชิกแต่ละบริการ |
| ผลจากทั้งสองทาง | ColumnTransformer ต่อคอลัมน์ → Ridge | คำทำนายค่าบริการหนึ่งค่าต่อแถว |

`ColumnTransformer` ในตัวอย่างทิ้งคอลัมน์ที่ไม่ได้ระบุโดยปริยาย จึงต้องเพิ่มชื่อคอลัมน์ในรายการ numeric ด้วยเมื่อทำแบบฝึกหัด หลัง fit โค้ดพิมพ์ตัวอย่างสามแถวที่แปลงแล้วเพื่อให้ตรวจด้วย pandas ได้

MAE ของ baseline มัธยฐานและ Ridge ใช้หน่วยค่าบริการจำลองเดียวกัน ส่วนตัวอย่าง same_day ใช้สาธิตการรับ input ที่ไม่รู้จัก การรันสำเร็จยังไม่บอกว่าค่าทำนายบริการใหม่เชื่อถือได้

## ลงมือรัน

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/06-feature-engineering/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` คัดลอกบล็อกต่อไปนี้ได้ทั้งชุด แต่ละส่วน `# %%` คือช่วงการทดลองเดียวกันกับ notebook

<!-- CODE:START -->
```python
"""จัดการ missing values และข้อมูลหมวดหมู่ภายใน Pipeline"""
# %% สร้างข้อมูลส่งสินค้าจำลอง ไม่ใช่ข้อมูลธุรกิจจริง
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rng = np.random.default_rng(42)
n = 400
distance = rng.uniform(1, 100, n)
weight = rng.uniform(0.2, 8, n)
service = rng.choice(["standard", "express"], n)
y = 30 + 2 * distance + 5 * weight + 40 * (service == "express") + rng.normal(0, 8, n)
X = pd.DataFrame({"distance_km": distance, "weight_kg": weight, "service": service})
X.loc[rng.choice(n, 30, replace=False), "weight_kg"] = np.nan
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %% แยกวิธีแปลงข้อมูลตามชนิดคอลัมน์
numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
categorical = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)
preprocess = ColumnTransformer([
    ("numeric", numeric, ["distance_km", "weight_kg"]),
    ("categorical", categorical, ["service"]),
])
model = make_pipeline(preprocess, Ridge(alpha=1.0))
model.fit(X_train, y_train)
baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
print(f"Baseline MAE: {mean_absolute_error(y_test, baseline.predict(X_test)):.3f}")
print(f"Test MAE (synthetic cost units): {mean_absolute_error(y_test, model.predict(X_test)):.3f}")
print("Transformed columns:", preprocess.get_feature_names_out())
preview = preprocess.transform(X_train.iloc[:3])
print("First three transformed rows:\n",
      pd.DataFrame(preview, columns=preprocess.get_feature_names_out()).round(3).to_string(index=False))

# %% หมวดหมู่ที่ไม่เคยเห็นจะไม่ทำให้โปรแกรมหยุด แต่ผลอาจไม่น่าเชื่อถือ
new_order = pd.DataFrame({"distance_km": [20.0], "weight_kg": [np.nan], "service": ["same_day"]})
print("Unseen-category prediction:", model.predict(new_order).round(2))
```
<!-- CODE:END -->

## อ่านผลที่ได้

พิมพ์ชื่อ feature หลังแปลงและ MAE หน่วยจำลอง ตัวอย่าง service=same_day รันได้แม้ไม่เคยอยู่ใน train ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## แบบฝึกหัด

เพิ่มคอลัมน์จำนวนชิ้นเข้าในข้อมูลและ ColumnTransformer จากนั้นออกแบบการตรวจว่าเมื่อค่าประเภทบริการไม่รู้จัก ควรแจ้งผู้ใช้หรือ fallback อย่างไร

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

เพิ่มคอลัมน์ตัวเลขในรายชื่อ numeric และพิจารณาว่าสัมพันธ์กับ target หรือไม่ การตั้ง handle_unknown ป้องกัน error ของ encoder เท่านั้น ระบบใช้งานจริงยังควรตรวจรายการหมวดหมู่ รายงานสัดส่วน unknown และกำหนดการตอบสนองเอง

</details>

## อ่านต่อ

- [Handbook: Feature Engineering](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html)
- [Pipelines and composite estimators](https://scikit-learn.org/1.7/modules/compose.html)
- [Imputation](https://scikit-learn.org/1.7/modules/impute.html)

