# 06 · เตรียมตัวเลข หมวดหมู่ และค่าที่หาย

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เตรียมตารางที่มีตัวเลข หมวดหมู่ และค่าหาย ด้วย **ColumnTransformer** แล้วทำนายค่าบริการส่งสินค้าจำลอง

**ใช้สิ่งที่เรียนแล้ว:** pandas, Pipeline/StandardScaler จากบท 03 และ regression/MAE จากบท 05

## ใจความใหม่

การแทนค่าหายด้วย median ต้องเรียนค่าจาก **train** แล้วใช้ค่าเดิมกับข้อมูลใหม่ จึงใส่ `SimpleImputer` ใน Pipeline แทนการเติมทั้งตารางก่อนแบ่งข้อมูล

**One-hot encoding** แยกคอลัมน์ตามหมวดหมู่ เช่น standard→`[1, 0]` และ express→`[0, 1]` เพื่อไม่สร้างลำดับสมมติแบบรหัส 0, 1, 2 ดูลำดับคอลัมน์จริงด้วย `get_feature_names_out()`

| ข้อมูลเข้า | ขั้นตอนใน ColumnTransformer |
|---|---|
| distance_km, weight_kg | เติม median → scale |
| service | เติมค่าที่พบบ่อย → one-hot |
| ผลจากทั้งสองทาง | ต่อคอลัมน์ → ส่งเข้า Ridge |

**Ridge** คือ linear regression ที่เพิ่มโทษกำลังสองของ coefficient เพื่อลดการใช้ค่าน้ำหนักใหญ่เกินไป `alpha` คุมความแรง; หากจะเลือกค่าให้ใช้ validation/CV

`handle_unknown="ignore"` ทำให้ service ใหม่เป็นศูนย์ทุกคอลัมน์ของ field นั้น โปรแกรมรันได้ แต่ยังไม่เคยเรียนบริการใหม่นี้ การเติม median ก็เป็นวิธีตัวอย่าง ไม่ได้เหมาะกับสาเหตุค่าหายทุกแบบ

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

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

## อ่านผล

ดูชื่อ feature และสามแถวหลังแปลง แล้วเทียบ MAE ของ Ridge กับ baseline มัธยฐาน ตัวอย่าง `same_day` ทำนายได้ แต่การไม่เกิด error ไม่ได้ยืนยันความน่าเชื่อถือสำหรับหมวดหมู่ใหม่

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** เพิ่มคอลัมน์จำนวนชิ้น แล้วออกแบบการรับ service ที่ไม่เคยเห็น

**แนวคำตอบ:** เพิ่มชื่อคอลัมน์ในรายการ numeric ของ ColumnTransformer ด้วย ส่วน unknown service ควรตรวจและกำหนดว่าจะให้ผู้ใช้แก้ข้อมูลหรือใช้วิธีสำรอง ไม่พึ่ง encoder ป้องกัน error อย่างเดียว

</details>

## อ้างอิง

- [Handbook: Feature Engineering](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html)
- [Pipelines and composite estimators](https://scikit-learn.org/1.7/modules/compose.html)
- [Imputation](https://scikit-learn.org/1.7/modules/impute.html)
