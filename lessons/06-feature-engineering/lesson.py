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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print("Train/Test rows:", len(X_train), len(X_test))
print("Missing values (train):", X_train.isna().sum().to_dict())
print("First three training rows:\n", X_train.iloc[:3].round(3).to_string(index=False))

# %% [markdown]
# **อ่านโค้ด**
#
# - `y = 30 + ...` สร้างค่าบริการจำลองจากระยะทาง น้ำหนัก ประเภทบริการ และ noise ไม่ใช่อัตราค่าจัดส่งจริง และไม่ได้ส่งสูตรนี้ให้โมเดล
# - `40 * (service == "express")` เพิ่มค่าบริการ 40 เฉพาะ express โดย True/False ทำหน้าที่เป็น 1/0
# - `pd.DataFrame(...)` เก็บระยะทาง น้ำหนัก และประเภทบริการเป็นข้อมูลนำเข้า `X` (features) ส่วนค่าบริการที่สร้างไว้เก็บใน `y` (target) ชื่อคอลัมน์ใน X ช่วยให้ ColumnTransformer เลือกแต่ละกลุ่มไปแปลงได้
# - `rng.choice(n, 30, replace=False)` เลือก 30 แถวไม่ซ้ำ แล้วใส่ `np.nan` ในน้ำหนัก สร้างค่าบริการ `y` ก่อนลบค่าน้ำหนัก เพื่อจำลองกรณีที่ทราบค่าบริการแล้ว แต่บันทึกน้ำหนักพัสดุไว้ไม่ครบ
# - `test_size=0.25` กัน 100 แถวไว้ทดสอบ อีก 300 แถวใช้ฝึก โดยยังไม่คำนวณค่าทดแทนหรือ fit ตัวแปลงใด
# - ผลท้าย cell แสดงจำนวนแถว ค่าหายของ train และข้อมูล 3 แถวก่อนแปลง เพื่อเทียบกับ Cell 2

# %% Cell 2 — เตรียมแต่ละชนิดข้อมูลแล้วฝึก Ridge
numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
categorical = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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
feature_names = preprocess.get_feature_names_out()
print("Transformed columns:", feature_names)
preview = preprocess.transform(X_train.iloc[:3])
print("First three transformed rows:\n",
      pd.DataFrame(preview, columns=feature_names).round(3).to_string(index=False))

# %% [markdown]
# **อ่านโค้ด**
#
# - `numeric` เติม median ก่อนปรับสเกล ส่วน `categorical` เติมค่าที่พบบ่อยก่อนแยกหมวดหมู่เป็น one-hot
# - `sparse_output=False` ให้ encoder คืนอาร์เรย์ปกติ เพื่อแสดงตารางตัวอย่างได้ตรงไปตรงมา เหมาะกับสองหมวดหมู่ในบทนี้; ถ้ามีหมวดหมู่จำนวนมากควรพิจารณารูปแบบ sparse เพื่อประหยัดหน่วยความจำ
# - รายการใน `ColumnTransformer` มีสามส่วน: ชื่อขั้นตอน, ตัวแปลง และรายชื่อคอลัมน์ ชื่อ `numeric` และ `categorical` จะปรากฏเป็นคำนำหน้าในชื่อ feature หลังแปลง
# - `model.fit(X_train, y_train)` คำนวณค่าที่ใช้เติมช่องว่าง ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานสำหรับปรับสเกล และรายการประเภทบริการจาก train แล้วฝึก Ridge ส่วน `model.predict(X_test)` ใช้ค่าที่เก็บไว้เหล่านี้กับ test
# - `get_feature_names_out()` ให้ชื่อ 4 คอลัมน์ตามลำดับจริง: ตัวเลข 2 คอลัมน์ และ service ที่แยกเป็น express/standard อีก 2 คอลัมน์
# - `preprocess.transform(X_train.iloc[:3])` แปลงสามแถวเดิมใน Cell 1 โดยไม่ fit ซ้ำ ตัวเลขอาจติดลบหลัง scale; คอลัมน์ one-hot ยังคงเป็น 0/1 เพราะไม่ผ่าน StandardScaler
# - `DummyRegressor(strategy="median")` ใช้มัธยฐานของค่าบริการใน `y_train` ทายทุกแถว เป็น baseline สำหรับเทียบ MAE ส่วนมัธยฐานของน้ำหนักพัสดุใช้เติมค่าน้ำหนักที่หาย จึงเป็นคนละค่าและคนละหน้าที่
#
# **การตั้งค่าใน cell นี้:** `handle_unknown="ignore"` ทำให้บริการที่ไม่เคยพบมีค่า one-hot เป็น 0 ทุกคอลัมน์เดิม ส่วน `Ridge(alpha=1.0)` กำหนดความเข้มของ L2 ไว้ล่วงหน้า หากจะเลือกค่า alpha ให้ใช้ validation/CV ตาม [บท 04](../04-validation-and-metrics/README.md) และเก็บ test ไว้ประเมินท้ายสุด
#
# **อ่านคะแนน:** คำทำนายเป็นค่าบริการหนึ่งตัวเลขต่อแถว MAE เทียบค่านั้นกับค่าบริการจริงใน test ทั้ง 100 แถว ส่วนตารางสามแถวที่แสดงเป็นข้อมูลนำเข้าหลังแปลง ไม่ใช่ค่าบริการที่โมเดลทำนาย

# %% Cell 3 — ลองข้อมูลใหม่ที่มีหมวดหมู่ไม่เคยพบ
new_order = pd.DataFrame({
    "distance_km": [20.0],
    "weight_kg": [np.nan],
    "service": ["same_day"],
})
new_features = pd.DataFrame(
    preprocess.transform(new_order), columns=feature_names
)
print("New order after preprocessing:\n", new_features.round(3).to_string(index=False))
print("Unseen-category prediction:", model.predict(new_order).round(2))

# %% [markdown]
# **อ่านโค้ด**
#
# - `new_order` มีชื่อคอลัมน์ตรงกับตอนฝึก แต่ใส่น้ำหนักเป็น `np.nan` และบริการเป็น `same_day`
# - `preprocess.transform(new_order)` เติมน้ำหนักที่หายด้วยมัธยฐานจาก train แล้วปรับสเกลด้วยค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานเดิม ส่วน service_express และ service_standard เป็น 0 ทั้งคู่ เพราะ same_day ไม่อยู่ในรายการที่เรียนไว้
# - ค่าศูนย์ทั้งคู่หมายถึงหมวดหมู่ที่ encoder ไม่เคยพบในตัวอย่างนี้ ไม่ได้หมายถึงบริการฟรี และไม่มีการเรียนค่าน้ำหนักของ same_day ขึ้นมาใหม่
# - `model.predict(new_order)` รับตารางดิบ แล้วเตรียมข้อมูลด้วย Pipeline เดิมก่อนทำนาย ไม่ต้องส่ง `new_features` เข้าไปอีก
# - `round(2)` ปัดเฉพาะการแสดงคำทำนาย การรันผ่านยังไม่ยืนยันว่าคำทำนายสำหรับบริการใหม่น่าเชื่อถือ
