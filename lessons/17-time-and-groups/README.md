# 17 · ประเมิน KNN และ Linear Regression กับข้อมูลกลุ่มและเวลา

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เลือกการแบ่งข้อมูลให้ตรงกับงาน **ทำนายคนใหม่** หรือ **ทำนายอนาคต**

**ใช้สิ่งที่เรียนแล้ว:** train/test, CV, KNN และ regression/MAE

**Group split** และ **Time-based split** เป็นวิธีแบ่งข้อมูลเพื่อประเมินโมเดล ไม่ใช่อัลกอริทึมทำนาย จุดประสงค์คือให้ข้อมูลที่ใช้ตรวจสอบต่างจากชุดฝึกในลักษณะเดียวกับงานจริง

| งานที่ต้องการประเมิน | วิธีแบ่ง | จุดเด่นและข้อจำกัด |
|---|---|---|
| ทำนายคนที่ไม่เคยอยู่ในชุดฝึก | กันคนออกทั้งกลุ่ม | ลดการวัดผลจากการจำคนเดิม แต่เมื่อกลุ่มมีน้อยคะแนนจะไม่แน่นอนมาก |
| ทำนายช่วงเวลาถัดไป | ฝึกอดีต ตรวจช่วงหลัง | สอดคล้องลำดับเวลาจริง แต่รูปแบบในอนาคตอาจเปลี่ยนจากช่วงที่ตรวจ |

เลือกวิธีแบ่งตามการใช้งานจริง ไม่เลือกจากวิธีที่ให้คะแนนสูงที่สุด

## โมเดลที่ใช้ทดลองเรื่องการแบ่งข้อมูล

**กรณีคนเดิมหลายแถว:** ใช้ **KNN** จาก [บท 03](../03-first-classifier/README.md) ผ่าน `KNeighborsClassifier(n_neighbors=3)` เลือกสามตัวอย่างที่ใกล้ที่สุดแล้วโหวตคลาส 0/1 ถ้าคนเดิมหลุดไปทั้ง train/test ตัวอย่างใกล้ที่สุดอาจเป็นข้อมูลของคนเดิม คะแนนจึงอาจสูงโดยไม่ได้แสดงความสามารถกับคนใหม่

**กรณีข้อมูลเวลา:** ใช้ **Linear Regression** จาก [บท 05](../05-linear-regression/README.md) รับค่าก่อนหน้าหนึ่งค่าเป็น X และเรียนสมการ `ค่าถัดไป ≈ b + w × ค่าก่อนหน้า` จากช่วงอดีต แล้วเทียบกับ **persistence baseline** ที่ทายค่าถัดไปให้เท่าค่าล่าสุด เช่น ล่าสุดเป็น 12 ก็ทาย 12 โดยไม่เรียน w/b

KNN ไวต่อระยะทางและตัวอย่างอ้างอิง ส่วน Linear Regression ใช้ความสัมพันธ์เชิงเส้น การแบ่งกลุ่มหรือเวลาไม่ได้เปลี่ยนข้อจำกัดของโมเดล แต่ช่วยให้การประเมินตรงกับงานที่ต้องการใช้จริง

## คนเดิมหลายแถว: แบ่งตามกลุ่ม

ข้อมูลจำลองมี 60 คน คนละ 5 แถว แต่ละคนมี feature คล้ายกันและใช้ label เดียวกันที่สุ่มไว้

**ปัญหาของการสุ่มแถว:** คนเดิมอาจอยู่ทั้ง train และ test โมเดลจึงอาจจำคนเดิมได้ แทนที่จะวัดความสามารถกับคนใหม่

**GroupShuffleSplit:** กันคนออกทั้งกลุ่ม โดย `test_size=0.2` หมายถึง 20% ของจำนวนกลุ่ม

บทนี้จึงได้ test 12 คน หรือ 60 แถว เพราะแต่ละคนมีจำนวนแถวเท่ากัน

- `groups` ใช้แบ่งข้อมูล ไม่ส่งเป็น feature เข้า KNN
- `Shared users` ต้องเป็น 0 เมื่อประเมินการทำนายคนใหม่

label ที่สุ่มระหว่างคนไม่มีรูปแบบให้ทายคนใหม่อย่างสม่ำเสมอ อีกทั้ง test มีเพียง 12 คน คะแนนจึงแกว่งได้มาก

## ข้อมูลเวลา: ฝึกอดีต ตรวจช่วงถัดไป

**TimeSeriesSplit:** ขยายช่วง train ไปข้างหน้า แล้วใช้ช่วงถัดไปเป็น validation

**`gap=1`:** เว้นหนึ่งแถวระหว่างสองช่วง ขนาด gap ที่เหมาะต้องพิจารณาช่วงเวลาของ feature และ label จริง

**Lag:** ค่าจากเวลาก่อนหน้า เช่น series `[10, 12, 11, 15]` สร้างคู่ lag→target ได้เป็น `10→12, 12→11, 11→15`

**Persistence baseline:** ทายค่าถัดไปให้เท่ากับค่าล่าสุด เพื่อใช้เทียบกับโมเดล

**One-step-ahead:** ทำนายครั้งละหนึ่งก้าว โดยรู้ค่าจริงก่อนหน้าของแต่ละจุดแล้ว นี่คือรูปแบบที่บทนี้ประเมิน

หากทำนายหลายก้าวจากวันเริ่มเดียว จะยังไม่รู้ค่าจริงระหว่างทาง จึงนำค่าจริงเหล่านั้นมาใช้แบบโค้ดนี้ไม่ได้

คะแนนเวลาเป็น validation ของแต่ละ fold และยังไม่มี final future holdout หากจะใช้คะแนนเลือกโมเดล ต้องกันช่วงอนาคตสุดท้ายไว้ประเมินเพิ่ม

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรัน **Cell 1–2 ตามลำดับ** หรือเปิด [Notebook ของบทนี้](lesson.ipynb) ใช้ kernel เดิมได้เลย อ่านคำอธิบายใต้โค้ดแต่ละช่วงประกอบผลที่ได้

<!-- CODE:CELLS -->

<!-- CODE:START -->
### Cell 1 — เปรียบเทียบการสุ่มแถวกับการแยกคน

```python
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
```

**อ่านโค้ด**

- `np.repeat(np.arange(60), 5)` สร้างรหัสคน 0–59 โดยแต่ละคนมีห้าแถว รวมเป็น 300 แถว
- `user_signal[groups]` ใช้รหัสคนเลือก feature ตั้งต้นของคนนั้น แล้วเติม noise เล็กน้อย ส่วน `user_label[groups]` ทำให้ห้าแถวของคนเดียวกันได้ label เดียวกัน
- `set(groups[row_train]) & set(groups[row_test])` หารหัสคนที่อยู่ทั้ง train และ test หลังสุ่มแถว เพื่อแสดงปัญหาคนเดิมข้ามชุด
- `next(split.split(X, y, groups))` รับคู่ index จาก `GroupShuffleSplit` รอบแรก โดยตัวแบ่งใช้ groups กันคนออกทั้งกลุ่ม
- `isdisjoint` ตรวจว่ารหัสคนไม่ซ้ำสองฝั่ง ส่วน `np.unique(groups[test_idx])` ใช้นับจำนวนคนที่กันไว้
- สำหรับ KNN classifier ในบทนี้ `model.score(...)` คืน accuracy ของข้อมูลที่ส่งให้ประเมิน จึงต้องอ่านพร้อมจำนวนคนที่ซ้ำ ไม่เลือกวิธีแบ่งจากคะแนนสูงสุด

### Cell 2 — แบ่งข้อมูลตามเวลาและเทียบค่าล่าสุด

```python
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

**อ่านโค้ด**

- `series[:-1].reshape(-1, 1)` เลือกค่าตั้งแต่ตัวแรกถึงก่อนตัวสุดท้ายเป็น feature หนึ่งคอลัมน์ ส่วน `series[1:]` เลื่อนคำตอบไปข้างหน้าหนึ่งตำแหน่ง
- แต่ละแถวจึงจับคู่ค่าก่อนหน้ากับค่าถัดไป โดย `lag` และ `target` มีจำนวนแถวเท่ากัน
- `TimeSeriesSplit(n_splits=4, gap=1).split(lag)` คืน index สี่รอบ ฝึกจากอดีตและเว้นหนึ่งแถวก่อนช่วง validation
- `train_idx.max() < valid_idx.min()` ตรวจว่าแถวฝึกอยู่ก่อนแถว validation
- `persistence = lag[valid_idx, 0]` ใช้ค่าล่าสุดเป็นคำทำนาย baseline แล้ววัด MAE กับ target ชุดเดียวกับโมเดล

การใช้ค่าจริงก่อนหน้าของทุกจุดเหมาะกับ one-step-ahead ตามโจทย์นี้ ไม่ใช่การทายหลายก้าวโดยยังไม่เห็นค่าจริงระหว่างทาง
<!-- CODE:END -->

## อ่านผล

- **Random row split:** อาจมีคนเดิมซ้ำทั้งสองฝั่ง
- **Group split:** ต้องมี `Shared users=0`
- **Time split:** ทุก fold ต้องมี train อยู่ก่อน validation

อ่าน MAE ของโมเดลเทียบกับ persistence baseline ในแต่ละ time fold โดยค่าต่ำกว่าดีกว่า

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองคิด:** ทายเดือนหน้าสำหรับร้านเดิม กับร้านที่ไม่เคยเห็น ใช้การประเมินเดียวกันได้หรือไม่?

**เฉลย:** ต้องแยกการประเมินให้ตรงกับสองโจทย์:

- **ร้านเดิม:** แบ่งตามเวลา เพื่อวัดการทำนายเดือนถัดไป
- **ร้านใหม่:** กันทั้งร้านออกจาก train พร้อมควบคุมลำดับเวลาให้ตรงการใช้งาน

รายงานผลของสองโจทย์แยกกัน

</details>

## อ้างอิง

- [Cross-validation: groups and time series](https://scikit-learn.org/1.7/modules/cross_validation.html)
- [Common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html)
- [Google MLCC: Production ML systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems)
