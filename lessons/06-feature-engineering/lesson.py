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
# - `y = 30 + ...` กำหนดสูตรสำหรับสร้างข้อมูลตัวอย่าง: เริ่มที่ 30 บวก 2 เท่าของระยะทาง บวก 5 เท่าของน้ำหนัก บวก 40 เมื่อเป็น express และบวก noise ผลที่คำนวณได้เก็บใน `y` เป็นค่าจัดส่งจำลองของแต่ละแถว
# - `40 * (service == "express")` เพิ่มค่าบริการ 40 เฉพาะ express โดย True/False ทำหน้าที่เป็น 1/0
# - `pd.DataFrame(...)` เก็บระยะทาง น้ำหนัก และประเภทบริการเป็นข้อมูลนำเข้า `X` (features) ส่วนค่าบริการที่สร้างไว้เก็บใน `y` (target) ชื่อคอลัมน์ใน X ช่วยให้ ColumnTransformer เลือกแต่ละกลุ่มไปแปลงได้
# - `rng.choice(n, 30, replace=False)` เลือก 30 แถวไม่ซ้ำ แล้วใส่ `np.nan` ในน้ำหนัก สร้างค่าบริการ `y` ก่อนลบค่าน้ำหนัก เพื่อจำลองกรณีที่ทราบค่าบริการแล้ว แต่บันทึกน้ำหนักพัสดุไว้ไม่ครบ
# - `test_size=0.25` กัน 100 แถวไว้ทดสอบ อีก 300 แถวใช้ฝึก `X_train` เก็บ features ส่วน `y_train` เก็บค่าบริการของแถวฝึก ตอนจบ cell นี้ยังไม่ได้ fit ตัวเตรียมข้อมูลหรือ Ridge
# - ผลท้าย cell แสดงจำนวนแถว ค่าหายของ train และข้อมูล 3 แถวก่อนแปลง เพื่อเทียบกับ Cell 2

# %% Cell 2 — เตรียมข้อมูล ฝึก Ridge และดูน้ำหนักที่เรียนได้
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

ridge = model.named_steps["ridge"]
coefficients = pd.DataFrame({"feature": feature_names, "coefficient": ridge.coef_})
print(f"Learned intercept: {ridge.intercept_:.3f}")
print("Learned coefficients:\n", coefficients.round(3).to_string(index=False))

# %% [markdown]
# **อ่านโค้ด**
#
# - `numeric` กำหนดลำดับเติม median ก่อนปรับสเกล ส่วน `categorical` กำหนดลำดับเติมค่าที่พบบ่อยก่อนแปลงหมวดหมู่เป็น one-hot การสร้าง Pipeline ยังไม่ได้คำนวณค่าทดแทนหรือแปลงข้อมูล
# - **ทำไมเติม service ทั้งที่ยังครบ?** ข้อมูลจำลองมีค่าน้ำหนักหาย แต่ service ไม่มีช่องว่าง เราใส่ `SimpleImputer(strategy="most_frequent")` ไว้รองรับ service ที่เป็น `np.nan` ในข้อมูลภายหลัง โดยใช้ประเภทที่พบบ่อยที่สุดใน train มาเติม
# - `sparse_output=False` ให้ encoder คืนอาร์เรย์ปกติ เพื่อแสดงตารางตัวอย่างได้ตรงไปตรงมา เหมาะกับสองหมวดหมู่ในบทนี้; ถ้ามีหมวดหมู่จำนวนมากควรพิจารณารูปแบบ sparse เพื่อประหยัดหน่วยความจำ
# - รายการใน `ColumnTransformer` มีสามส่วน: ชื่อขั้นตอน, ตัวแปลง และรายชื่อคอลัมน์ ชื่อ `numeric` และ `categorical` จะปรากฏเป็นคำนำหน้าในชื่อ feature หลังแปลง
# - `model = make_pipeline(preprocess, Ridge(alpha=1.0))` ต่อ ColumnTransformer เข้ากับ Ridge ตัวแปร `model` จึงเป็น Pipeline ทั้งชุด โดยยังไม่ได้เริ่มฝึก
# - `model.fit(X_train, y_train)` ทำสองขั้นตามลำดับ: **(1)** ให้ `preprocess` fit และ transform ชุดฝึกผ่าน `fit_transform` จนได้ features 4 คอลัมน์ **(2)** ส่ง features ที่แปลงแล้วพร้อม `y_train` ให้ Ridge fit เพื่อหาน้ำหนักกับ intercept เราไม่ต้องเรียกสองขั้นนี้แยกเอง
# - ภายใน `numeric` ตัวเติมค่า fit เพื่อหา median แล้ว transform เพื่อเติมช่องว่างก่อน จากนั้น StandardScaler จึง fit จากตัวเลขที่เติมครบแล้วและ transform เพื่อปรับสเกล ส่วน OneHotEncoder ใน `categorical` fit เพื่อเก็บรายชื่อประเภทบริการ แล้ว transform เป็น 0/1
# - `model.predict(X_test)` ให้ตัวเตรียมข้อมูล transform `X_test` ด้วยค่าที่เก็บไว้ แล้วให้ Ridge predict ด้วยน้ำหนักกับ intercept ที่ฝึกไว้ ทั้งสองส่วนไม่ fit ใหม่จาก test เราส่ง `X_test` ดิบเข้า Pipeline ได้โดยไม่ต้องแปลงเองก่อน
# - `get_feature_names_out()` ให้ชื่อ 4 คอลัมน์ตามลำดับจริง: ตัวเลข 2 คอลัมน์ และ service ที่แยกเป็น express/standard อีก 2 คอลัมน์
# - `preprocess.transform(X_train.iloc[:3])` แปลงสามแถวเดิมใน Cell 1 โดยไม่ fit ซ้ำ ตัวเลขอาจติดลบหลัง scale; คอลัมน์ one-hot ยังคงเป็น 0/1 เพราะไม่ผ่าน StandardScaler
# - `DummyRegressor(strategy="median")` ใช้มัธยฐานของค่าบริการใน `y_train` ทายทุกแถว เป็น baseline สำหรับเทียบ MAE ส่วนมัธยฐานของน้ำหนักพัสดุใช้เติมค่าน้ำหนักที่หาย จึงเป็นคนละค่าและคนละหน้าที่
# - `model.named_steps["ridge"]` เข้าถึง Ridge ที่ฝึกแล้วใน Pipeline ชื่อ `"ridge"` มาจาก `make_pipeline` ที่ตั้งชื่อขั้นตอนตามชื่อคลาส
# - `ridge.coef_` เก็บน้ำหนักทั้ง 4 ค่าตามลำดับเดียวกับ `feature_names`: ตำแหน่ง `[0]` คือ w₁ ของระยะทางหลังปรับสเกล ส่วน `[3]` คือ w₄ ของ service_standard; Python เริ่มนับ index ที่ 0 ส่วน `ridge.intercept_` คือค่าคงที่ b
# - อ่านน้ำหนักตัวเลขเป็นความชันต่อค่า z โดยคง feature อื่นไว้ ส่วนการเปลี่ยน express เป็น standard ใช้ผลต่าง `ridge.coef_[3] - ridge.coef_[2]` เพราะคอลัมน์ express เปลี่ยนจาก 1 เป็น 0 พร้อมกับ standard เปลี่ยนจาก 0 เป็น 1
#
# **การตั้งค่าใน cell นี้:** `handle_unknown="ignore"` ทำให้บริการที่ OneHotEncoder ไม่เคยพบตอน fit มีค่าเป็น 0 เฉพาะคอลัมน์ one-hot ของ service ส่วนคอลัมน์ตัวเลขยังแปลงตามปกติ `Ridge(alpha=1.0)` กำหนดความเข้มของ L2 ไว้ล่วงหน้า หากจะเลือกค่า alpha ให้ใช้ validation/CV ตาม [บท 04](../04-validation-and-metrics/README.md) โดยนำ Pipeline ทั้งชุดเข้า CV และเก็บ test ไว้ประเมินท้ายสุด ค่า alpha เป็นค่าตั้ง ส่วน `coef_` กับ `intercept_` เป็นค่าที่เรียนจาก train
#
# **อ่านคะแนน:** คำทำนายเป็นค่าบริการหนึ่งตัวเลขต่อแถว MAE เปรียบเทียบค่าบริการที่ทำนายกับค่าบริการจริงใน test ทั้ง 100 แถว ส่วนตารางสามแถวที่แสดงเป็นข้อมูลนำเข้าหลังแปลง ไม่ใช่ค่าบริการที่โมเดลทำนาย

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
manual_prediction = ridge.intercept_ + new_features.to_numpy() @ ridge.coef_
print("Prediction from coefficients:", manual_prediction.round(2))

# %% [markdown]
# **อ่านโค้ด**
#
# - `new_order` สร้างรายการใหม่หลังฝึกโมเดล โดยใช้ชื่อคอลัมน์เดิม ใส่น้ำหนักเป็น `np.nan` และบริการเป็น `same_day`
# - **ค่าหายกับหมวดหมู่ใหม่ต่างกัน:** `np.nan` ในน้ำหนักเป็นช่องว่างที่ตัวเติมค่าต้องเติม ส่วนข้อความ `same_day` มีค่าอยู่แล้ว ตัวเติม service จึงปล่อยข้อความนี้ไว้ แล้วส่งต่อให้ OneHotEncoder จัดการ
# - `preprocess.transform(new_order)` เติมน้ำหนักที่หายด้วยมัธยฐานจาก train แล้วปรับสเกลด้วยค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานเดิม ส่วน service_express และ service_standard เป็น 0 ทั้งคู่ เพราะ same_day ไม่อยู่ในรายชื่อประเภทบริการที่ OneHotEncoder เก็บไว้ตอน fit กับ X_train
# - ค่าศูนย์ทั้งคู่หมายถึงหมวดหมู่ที่ encoder ไม่เคยพบในตัวอย่างนี้ ไม่ได้หมายถึงบริการฟรี และไม่มีการเรียนค่าน้ำหนักของ same_day ขึ้นมาใหม่
# - `model.predict(new_order)` รับตารางดิบ แล้วเตรียมข้อมูลด้วย Pipeline เดิมก่อนทำนาย ไม่ต้องส่ง `new_features` เข้าไปอีก
# - `new_features.to_numpy() @ ridge.coef_` นำ features ที่แปลงแล้วมาคูณน้ำหนักทีละคอลัมน์แล้วรวมกัน จากนั้นบวก `ridge.intercept_` จึงเป็นการแทนค่าสมการ Ridge โดยตรง ผลต้องตรงกับ `model.predict(new_order)` เมื่อใช้ตัวเลขเต็มความละเอียด
# - `round(2)` ปัดเฉพาะการแสดงคำทำนาย
