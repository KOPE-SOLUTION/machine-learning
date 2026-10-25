# %% Cell 1 — สร้างตารางตัวเลข หมวดหมู่ และค่าหาย
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

# %% [markdown]
# **อ่านโค้ด**
#
# - `y = 30 + ...` สร้างค่าบริการจำลองจากระยะทาง น้ำหนัก ประเภทบริการ และ noise
# - `40 * (service == "express")` เพิ่มค่าบริการ 40 เฉพาะแถว express โดยเงื่อนไข True/False ทำหน้าที่เป็น 1/0
# - `pd.DataFrame(...)` รวม feature ทั้งสามให้มีชื่อคอลัมน์ ซึ่งจะใช้เลือกวิธีแปลงข้อมูลใน Cell ถัดไป
# - `rng.choice(n, 30, replace=False)` สุ่ม index 30 แถวแบบไม่ซ้ำ แล้ว `X.loc[..., "weight_kg"] = np.nan` ทำให้น้ำหนักของแถวเหล่านั้นหายไปเพื่อจำลองข้อมูลไม่ครบ
# - สร้าง y ก่อนทำค่าหาย เพื่อจำลองว่าค่าบริการมีคำตอบอยู่แล้ว แต่ข้อมูลน้ำหนักบางแถวไม่ได้ถูกบันทึก

# %% Cell 2 — เตรียมแต่ละชนิดข้อมูลแล้วฝึก Ridge
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

# %% [markdown]
# **อ่านโค้ด**
#
# - `numeric` เป็น Pipeline ที่เติมค่าหายด้วย median แล้วปรับสเกล ส่วน `categorical` เติมค่าที่พบบ่อยแล้วทำ one-hot
# - แต่ละรายการใน `ColumnTransformer` มีสามส่วน: ชื่อขั้นตอน, ตัวแปลง และรายชื่อคอลัมน์ที่จะส่งเข้าไป
# - `make_pipeline(preprocess, Ridge(alpha=1.0))` รวมผลจากตัวแปลงทั้งสองทางก่อนส่งเข้า Ridge โดย `fit` เรียนทุกขั้นจาก train
# - `get_feature_names_out()` คืนชื่อคอลัมน์หลังแปลง เช่น คอลัมน์ service ที่แตกออกเป็น one-hot
# - `preprocess.transform(X_train.iloc[:3])` ใช้ค่าที่เรียนไว้แปลงสามแถวแรกเป็นตัวอย่าง โดยไม่ได้ fit ใหม่
# - `DummyRegressor(strategy="median")` ให้ baseline ทายมัธยฐานของ y_train เพื่อเทียบ MAE กับโมเดล

# %% Cell 3 — ลองข้อมูลใหม่ที่มีหมวดหมู่ไม่เคยพบ
new_order = pd.DataFrame({"distance_km": [20.0], "weight_kg": [np.nan], "service": ["same_day"]})
print("Unseen-category prediction:", model.predict(new_order).round(2))

# %% [markdown]
# **อ่านโค้ด**
#
# - `new_order` เป็น DataFrame หนึ่งแถวที่มีชื่อคอลัมน์ตรงกับตอนฝึก แต่ใส่น้ำหนักเป็น `np.nan` และบริการเป็น `same_day`
# - เมื่อเรียก `model.predict(new_order)` ตัว imputer จะเติมน้ำหนักด้วย median จาก train
# - `handle_unknown="ignore"` ทำให้บริการที่ไม่เคยพบถูกแทนด้วยศูนย์ทุกคอลัมน์ one-hot ของ service
# - `round(2)` ปัดเฉพาะผลที่แสดงเป็นสองตำแหน่ง การรันผ่านยังไม่ยืนยันว่าคำทำนายสำหรับบริการใหม่น่าเชื่อถือ
