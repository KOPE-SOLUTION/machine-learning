"""เลือกรูปแบบการแบ่งข้อมูลให้ตรงกับคำถามเรื่องคนใหม่และอนาคต"""
# %% คนเดิมหลายแถว: แยกเป็นกลุ่มก่อนประเมินคนใหม่
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(42)
groups = np.repeat(np.arange(60), 5)
user_signal = rng.normal(size=(60, 3))
X = user_signal[groups] + rng.normal(0, 0.05, (len(groups), 3))
user_label = rng.integers(0, 2, size=60)
y = user_label[groups]
split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(split.split(X, y, groups))
assert set(groups[train_idx]).isdisjoint(groups[test_idx])
model = KNeighborsClassifier(n_neighbors=3).fit(X[train_idx], y[train_idx])
print("Held-out user groups:", len(np.unique(groups[test_idx])))
print(f"New-user accuracy (random user labels): {model.score(X[test_idx], y[test_idx]):.3f}")

# %% เวลาต้องเดินไปข้างหน้า: ตัวอย่างมีหนึ่ง observation ต่อเวลา
t = np.arange(180)
series = 0.1 * t + np.sin(t / 6) + rng.normal(0, 0.2, len(t))
# ทำนายค่าถัดไปจากค่าก่อนหน้า ไม่ใช้ centered rolling ที่เห็นอนาคต
lag = series[:-1].reshape(-1, 1)
target = series[1:]
for fold, (train_idx, valid_idx) in enumerate(TimeSeriesSplit(n_splits=4, gap=1).split(lag), start=1):
    assert train_idx.max() < valid_idx.min()
    model = LinearRegression().fit(lag[train_idx], target[train_idx])
    pred = model.predict(lag[valid_idx])
    persistence = lag[valid_idx, 0]
    print(f"Fold {fold}: train ends={train_idx.max()}, validation starts={valid_idx.min()}, "
          f"model MAE={mean_absolute_error(target[valid_idx], pred):.3f}, "
          f"last-value MAE={mean_absolute_error(target[valid_idx], persistence):.3f}")
