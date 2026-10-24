"""จัดการ missing values และข้อมูลหมวดหมู่ภายใน Pipeline"""
# %% สร้างข้อมูลส่งสินค้าจำลอง ไม่ใช่ข้อมูลธุรกิจจริง
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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
print(f"Test MAE (synthetic cost units): {mean_absolute_error(y_test, model.predict(X_test)):.3f}")
print("Transformed columns:", preprocess.get_feature_names_out())

# %% หมวดหมู่ที่ไม่เคยเห็นจะไม่ทำให้โปรแกรมหยุด แต่ผลอาจไม่น่าเชื่อถือ
new_order = pd.DataFrame({"distance_km": [20.0], "weight_kg": [np.nan], "service": ["same_day"]})
print("Unseen-category prediction:", model.predict(new_order).round(2))
