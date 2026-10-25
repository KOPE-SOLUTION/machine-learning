# 06 · เตรียมตัวเลข หมวดหมู่ และค่าที่หาย

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เตรียมตารางที่มีตัวเลข หมวดหมู่ และค่าหาย ด้วย **ColumnTransformer** แล้วทำนายค่าบริการส่งสินค้าจำลอง

**ใช้สิ่งที่เรียนแล้ว:** pandas, Pipeline/StandardScaler จากบท 03 และ regression/MAE จากบท 05

## ใจความใหม่

**SimpleImputer:** เติมค่าหายโดยเรียนค่าทดแทนจาก train แล้วใช้ค่าเดิมกับข้อมูลใหม่ จึงต้องอยู่ใน Pipeline แทนการเติมทั้งตารางก่อนแบ่งข้อมูล

**One-hot encoding:** แยกคอลัมน์ตามหมวดหมู่ เพื่อไม่สร้างลำดับสมมติแบบรหัส 0, 1, 2

ตัวอย่าง: ถ้าลำดับคอลัมน์คือ standard และ express จะได้ standard→`[1, 0]` และ express→`[0, 1]` ตรวจลำดับจริงด้วย `get_feature_names_out()`

| ข้อมูลเข้า | ขั้นตอนใน ColumnTransformer |
|---|---|
| distance_km, weight_kg | เติม median → scale |
| service | เติมค่าที่พบบ่อย → one-hot |
| ผลจากทั้งสองทาง | ต่อคอลัมน์ → ส่งเข้า Ridge |

**Ridge:** Linear Regression ที่เพิ่มโทษกำลังสองของ coefficient เพื่อลดการใช้ค่าน้ำหนักใหญ่เกินไป

ค่า `alpha` คุมความแรงของโทษ หากจะเลือกค่าให้ใช้ validation หรือ CV

**หมวดหมู่ใหม่:** `handle_unknown="ignore"` แปลง service ที่ไม่เคยเห็นเป็นศูนย์ทุกคอลัมน์ของ field นั้น โปรแกรมจึงรันได้ แต่โมเดลยังไม่เคยเรียนบริการใหม่นี้

การเติม median เป็นวิธีตัวอย่าง ต้องพิจารณาสาเหตุของค่าหายก่อนนำไปใช้จริง

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรัน **Cell 1–3 ตามลำดับ** หรือเปิด [Notebook ของบทนี้](lesson.ipynb) ใช้ kernel เดิมได้เลย อ่านคำอธิบายใต้โค้ดแต่ละช่วงประกอบผลที่ได้

<!-- CODE:CELLS -->

<!-- CODE:START -->
### Cell 1 — สร้างตารางตัวเลข หมวดหมู่ และค่าหาย

```python
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
```

**อ่านโค้ด**

- `y = 30 + ...` สร้างค่าบริการจำลองจากระยะทาง น้ำหนัก ประเภทบริการ และ noise
- `40 * (service == "express")` เพิ่มค่าบริการ 40 เฉพาะแถว express โดยเงื่อนไข True/False ทำหน้าที่เป็น 1/0
- `pd.DataFrame(...)` รวม feature ทั้งสามให้มีชื่อคอลัมน์ ซึ่งจะใช้เลือกวิธีแปลงข้อมูลใน Cell ถัดไป
- `rng.choice(n, 30, replace=False)` สุ่ม index 30 แถวแบบไม่ซ้ำ แล้ว `X.loc[..., "weight_kg"] = np.nan` ทำให้น้ำหนักของแถวเหล่านั้นหายไปเพื่อจำลองข้อมูลไม่ครบ
- สร้าง y ก่อนทำค่าหาย เพื่อจำลองว่าค่าบริการมีคำตอบอยู่แล้ว แต่ข้อมูลน้ำหนักบางแถวไม่ได้ถูกบันทึก

### Cell 2 — เตรียมแต่ละชนิดข้อมูลแล้วฝึก Ridge

```python
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
```

**อ่านโค้ด**

- `numeric` เป็น Pipeline ที่เติมค่าหายด้วย median แล้วปรับสเกล ส่วน `categorical` เติมค่าที่พบบ่อยแล้วทำ one-hot
- แต่ละรายการใน `ColumnTransformer` มีสามส่วน: ชื่อขั้นตอน, ตัวแปลง และรายชื่อคอลัมน์ที่จะส่งเข้าไป
- `make_pipeline(preprocess, Ridge(alpha=1.0))` รวมผลจากตัวแปลงทั้งสองทางก่อนส่งเข้า Ridge โดย `fit` เรียนทุกขั้นจาก train
- `get_feature_names_out()` คืนชื่อคอลัมน์หลังแปลง เช่น คอลัมน์ service ที่แตกออกเป็น one-hot
- `preprocess.transform(X_train.iloc[:3])` ใช้ค่าที่เรียนไว้แปลงสามแถวแรกเป็นตัวอย่าง โดยไม่ได้ fit ใหม่
- `DummyRegressor(strategy="median")` ให้ baseline ทายมัธยฐานของ y_train เพื่อเทียบ MAE กับโมเดล

### Cell 3 — ลองข้อมูลใหม่ที่มีหมวดหมู่ไม่เคยพบ

```python
new_order = pd.DataFrame({"distance_km": [20.0], "weight_kg": [np.nan], "service": ["same_day"]})
print("Unseen-category prediction:", model.predict(new_order).round(2))
```

**อ่านโค้ด**

- `new_order` เป็น DataFrame หนึ่งแถวที่มีชื่อคอลัมน์ตรงกับตอนฝึก แต่ใส่น้ำหนักเป็น `np.nan` และบริการเป็น `same_day`
- เมื่อเรียก `model.predict(new_order)` ตัว imputer จะเติมน้ำหนักด้วย median จาก train
- `handle_unknown="ignore"` ทำให้บริการที่ไม่เคยพบถูกแทนด้วยศูนย์ทุกคอลัมน์ one-hot ของ service
- `round(2)` ปัดเฉพาะผลที่แสดงเป็นสองตำแหน่ง การรันผ่านยังไม่ยืนยันว่าคำทำนายสำหรับบริการใหม่น่าเชื่อถือ
<!-- CODE:END -->

## อ่านผล

- ตรวจชื่อ feature และสามแถวหลังแปลงข้อมูล
- เทียบ MAE ของ Ridge กับ baseline ที่ทายค่ามัธยฐาน
- ตัวอย่าง `same_day` ทำนายได้ แต่การไม่เกิด error ยังไม่ยืนยันความน่าเชื่อถือสำหรับหมวดหมู่ใหม่

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** เพิ่มคอลัมน์จำนวนชิ้น แล้วออกแบบการรับ service ที่ไม่เคยเห็น

**แนวคำตอบ:**

- เพิ่มชื่อคอลัมน์จำนวนชิ้นในรายการ numeric ของ ColumnTransformer
- ตรวจ service ที่ไม่เคยเห็น แล้วกำหนดว่าจะให้ผู้ใช้แก้ข้อมูลหรือใช้วิธีสำรอง การป้องกัน error ด้วย encoder เพียงอย่างเดียวยังไม่พอ

</details>

## อ้างอิง

- [Handbook: Feature Engineering](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html)
- [Pipelines and composite estimators](https://scikit-learn.org/1.7/modules/compose.html)
- [Imputation](https://scikit-learn.org/1.7/modules/impute.html)
