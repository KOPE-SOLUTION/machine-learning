# %% Cell 1 — เปรียบเทียบการสุ่มแถวกับการแยกคน
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, TimeSeriesSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(42)
groups = np.repeat(np.arange(60), 5)
user_signal = rng.normal(size=(60, 3))
X = user_signal[groups] + rng.normal(0, 0.05, (len(groups), 3))
user_label = rng.integers(0, 2, size=60)
y = user_label[groups]

# การสุ่มแถวใช้เทียบให้เห็นการปะปนของคนเดิม ไม่ใช้เลือกวิธีประเมินจากคะแนนสูง
row_train, row_test = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
row_model = KNeighborsClassifier(n_neighbors=3).fit(X[row_train], y[row_train])
shared_users = set(groups[row_train]) & set(groups[row_test])
print("Random row split - shared users:", len(shared_users))
print(f"Random row-split accuracy: {row_model.score(X[row_test], y[row_test]):.3f}")

split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(split.split(X, y, groups))
assert set(groups[train_idx]).isdisjoint(groups[test_idx])
model = KNeighborsClassifier(n_neighbors=3).fit(X[train_idx], y[train_idx])
print("Group split - shared users:", len(set(groups[train_idx]) & set(groups[test_idx])))
print("Held-out user groups:", len(np.unique(groups[test_idx])))
print(f"New-user accuracy (random user labels): {model.score(X[test_idx], y[test_idx]):.3f}")

# %% [markdown]
# **อ่านโค้ด**
#
# - `np.repeat(np.arange(60), 5)` สร้างรหัสคน 0–59 โดยแต่ละคนมีห้าแถว รวมเป็น 300 แถว
# - `user_signal[groups]` ใช้รหัสคนเลือก feature ตั้งต้นของคนนั้น แล้วเติม noise เล็กน้อย ส่วน `user_label[groups]` ทำให้ห้าแถวของคนเดียวกันได้ label เดียวกัน
# - `set(groups[row_train]) & set(groups[row_test])` หารหัสคนที่อยู่ทั้ง train และ test หลังสุ่มแถว เพื่อแสดงปัญหาคนเดิมข้ามชุด
# - `next(split.split(X, y, groups))` รับคู่ index จาก `GroupShuffleSplit` รอบแรก โดยตัวแบ่งใช้ groups กันคนออกทั้งกลุ่ม
# - `isdisjoint` คืน True เมื่อไม่มีรหัสคนร่วมกันระหว่าง train กับ test และ `assert` จะหยุดพร้อมข้อผิดพลาดหากเงื่อนไขนี้ไม่จริง ส่วน `np.unique(groups[test_idx])` หารหัสคนที่ไม่ซ้ำกันใน test เพื่อนำไปนับจำนวนคน
# - สำหรับ KNN classifier ในบทนี้ `model.score(...)` คืน accuracy ของข้อมูลที่ส่งให้ประเมิน จึงต้องอ่านพร้อมจำนวนคนที่ซ้ำ ไม่เลือกวิธีแบ่งจากคะแนนสูงสุด
#
# **การตั้งค่าและจำนวนแถว:** `GroupShuffleSplit(test_size=0.2)` กัน 20% ของ 60 คน จึงเป็น 12 คน × 5 แถว = test 60 แถว ไม่ได้สุ่มกันเพียงบางแถวของคนเดิม ค่า `Group split - shared users` ต้องเป็น 0

# %% Cell 2 — แบ่งข้อมูลตามเวลาและเทียบค่าล่าสุด
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

# %% [markdown]
# **อ่านโค้ด**
#
# - `series[:-1].reshape(-1, 1)` เลือกค่าตั้งแต่ตัวแรกถึงก่อนตัวสุดท้ายเป็น feature หนึ่งคอลัมน์ ส่วน `series[1:]` เลื่อนคำตอบไปข้างหน้าหนึ่งตำแหน่ง
# - แต่ละแถวจึงจับคู่ค่าก่อนหน้ากับค่าถัดไป โดย `lag` และ `target` มีจำนวนแถวเท่ากัน
# - `TimeSeriesSplit(n_splits=4, gap=1).split(lag)` คืน index สี่รอบ ฝึกจากอดีตและเว้นหนึ่งแถวก่อนช่วง validation
# - `train_idx.max() < valid_idx.min()` ตรวจว่าแถวฝึกอยู่ก่อนแถว validation
# - `persistence = lag[valid_idx, 0]` ใช้ค่าล่าสุดเป็นคำทำนาย baseline แล้ววัด MAE กับ target ชุดเดียวกับโมเดล
#
# การใช้ค่าจริงก่อนหน้าของทุกจุดเหมาะกับ one-step-ahead ตามโจทย์นี้ ไม่ใช่การทายหลายก้าวโดยยังไม่เห็นค่าจริงระหว่างทาง
#
# **การตั้งค่าใน cell นี้:** `gap=1` เว้นหนึ่งแถวระหว่าง train กับ validation ส่วน `Fold` ที่พิมพ์ระบุรอบประเมิน และ MAE ของโมเดลกับ last-value baseline ใช้คำตอบจริงของช่วง validation เดียวกัน
