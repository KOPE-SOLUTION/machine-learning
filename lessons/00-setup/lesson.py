"""ตรวจสภาพแวดล้อมและรูปร่างข้อมูลก่อนเริ่มเรียน"""
# %% ตรวจรุ่นเครื่องมือ
import importlib.metadata
import platform

print("Python:", platform.python_version())
for package in ("numpy", "pandas", "scikit-learn", "matplotlib"):
    print(f"{package}: {importlib.metadata.version(package)}")

# %% อ่านข้อมูลที่ติดมากับ scikit-learn
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
X, y = data.data, data.target
assert X.shape == (150, 4)
assert y.shape == (150,)
print("X shape:", X.shape, "y shape:", y.shape)
print("Classes:", data.target_names)
print(X.head(3).to_string(index=False))
