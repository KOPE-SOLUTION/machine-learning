# 17 · ข้อมูลคนเดิม ข้อมูลเวลา และการประเมินให้ตรงการใช้งาน

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เลือกการแบ่งข้อมูลให้ตรงกับงาน **ทำนายคนใหม่** หรือ **ทำนายอนาคต**

**ใช้สิ่งที่เรียนแล้ว:** train/test, CV, KNN และ regression/MAE ไม่เลือกวิธีแบ่งจากคะแนนที่สูงที่สุด

## คนเดิมหลายแถว: แบ่งตามกลุ่ม

ข้อมูลจำลองมี 60 คน คนละ 5 แถว แต่ละคนมี feature คล้ายกันและ label เดียวกันที่สุ่มไว้ การสุ่มแถวอาจทำให้โมเดลจำคนเดิมที่อยู่ทั้ง train/test จึงไม่ได้วัดผลกับคนใหม่

`GroupShuffleSplit` กันคนออกทั้งกลุ่ม `test_size=0.2` คือ 20% ของ **กลุ่ม**: ในบทนี้ได้ 12 คน / 60 แถว เพราะจำนวนแถวต่อคนเท่ากัน `groups` ใช้แบ่งข้อมูล ไม่ส่งเป็น feature เข้า KNN

ให้ดู `Shared users` ว่าต้องเป็น 0 เมื่อประเมินคนใหม่ label ที่สุ่มระหว่างคนไม่มีรูปแบบให้ทายคนใหม่อย่างสม่ำเสมอ และ test เพียง 12 คนทำให้คะแนนแกว่งได้มาก

## ข้อมูลเวลา: ฝึกอดีต ตรวจช่วงถัดไป

`TimeSeriesSplit` ขยายช่วง train ไปข้างหน้าแล้วตรวจช่วงถัดไป `gap=1` เว้นหนึ่งแถวระหว่างสองช่วง ขนาด gap ที่เหมาะต้องดูช่วงเวลาของ feature/label จริง

เช่น series `[10, 12, 11, 15]` สร้างคู่ lag→target เป็น `10→12, 12→11, 11→15` ใช้ค่าก่อนหน้าทำนายค่าถัดไป เทียบกับ **persistence baseline** ที่ทายเท่าค่าล่าสุด

บทนี้ประเมินแบบ **one-step-ahead**: ตอนทำนายแต่ละจุดรู้ค่าจริงก่อนหน้าแล้ว ไม่ใช่การทายหลายก้าวจากวันเริ่มเดียว ซึ่งใช้ค่าจริงระหว่างทางแบบนี้ไม่ได้

คะแนนเวลาเป็น validation ของแต่ละ fold ยังไม่มี final future holdout หากใช้เลือกโมเดล ต้องกันช่วงอนาคตสุดท้ายเพิ่ม

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

<!-- CODE:START -->
```python
"""เลือกรูปแบบการแบ่งข้อมูลให้ตรงกับคำถามเรื่องคนใหม่และอนาคต"""
# %% คนเดิมหลายแถว: แยกเป็นกลุ่มก่อนประเมินคนใหม่
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
```
<!-- CODE:END -->

## อ่านผล

random row split อาจมีคนซ้ำสองฝั่ง แต่ group split ต้องมี Shared users=0 ส่วนทุก time fold ต้องมี train ก่อน validation อ่าน MAE ของโมเดลเทียบกับค่าล่าสุดในแต่ละ fold ยิ่งต่ำยิ่งดี

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองคิด:** ทายเดือนหน้าสำหรับร้านเดิม กับร้านที่ไม่เคยเห็น ใช้การประเมินเดียวกันได้หรือไม่?

**เฉลย:** ต้องแยกโจทย์ ร้านเดิมแบ่งตามเวลา ส่วนร้านใหม่กันทั้งร้านออกจาก train พร้อมคุมเวลาตามการใช้งาน แล้วรายงานแยกกัน

</details>

## อ้างอิง

- [Cross-validation: groups and time series](https://scikit-learn.org/1.7/modules/cross_validation.html)
- [Common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html)
- [Google MLCC: Production ML systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems)
