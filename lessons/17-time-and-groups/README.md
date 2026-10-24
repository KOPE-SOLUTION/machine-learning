# 17 · ข้อมูลคนเดิม ข้อมูลเวลา และการประเมินให้ตรงการใช้งาน

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 45–65 นาที  
สิ่งที่จะทำได้: เลือก group split/time split และอธิบายข้อจำกัดของ random split ได้

## ก่อนเริ่มบทนี้

เรียน train/test ในบท 01 และ CV ในบท 08–10 ก่อน บทนี้กลับมาถามว่าสิ่งที่ต้องทำนายคือคนใหม่หรือเวลาใหม่ เพราะคำตอบกำหนดวิธีแบ่งข้อมูล

## หนึ่งแถวไม่ได้เป็นอิสระเสมอไป

หากผู้ใช้คนหนึ่งมีหลายแถว โมเดลอาจจำลักษณะเฉพาะบุคคลได้ การสุ่มแถวทำให้คนเดิมอยู่ทั้ง train/test และตอบคำถามต่างจากการใช้งานกับคนใหม่ GroupShuffleSplit แยกกลุ่มคนให้ไม่ซ้ำกันทั้งสองฝั่ง

ตัวอย่างสร้าง signal ประจำตัวผู้ใช้และ label สุ่มต่อคน จึงไม่มีความสัมพันธ์ที่ควรทำนาย label ของคนใหม่ได้สม่ำเสมอ คะแนนที่ได้จากคนใหม่อาจแกว่งมากเมื่อมี test เพียง 12 คน อย่าตีความคะแนนรอบเดียวเป็นหลักฐานเชิงสถิติที่หนักแน่น

## ข้อมูลอนาคตต้องอยู่หลังอดีต

TimeSeriesSplit ใช้ช่วงก่อนหน้าเป็น train และช่วงถัดไปเป็น validation ค่า gap กันระยะระหว่างสองช่วงตามที่กำหนด แต่ขนาด gap ต้องเลือกตามช่วงเวลาของ feature และ label จริง ไม่ได้มีค่าเดียวที่ป้องกัน leakage ทุกกรณี

ตัวอย่างใช้ค่าที่เวลา t−1 ทำนาย t และเป็นการประเมินแบบ one-step-ahead: เมื่อทำนายแต่ละเวลาจะรู้ค่าจริงของเวลาก่อนหน้าแล้ว จึงต่างจากการพยากรณ์หลายก้าวล่วงหน้าจากวันเริ่มต้นเดียว ซึ่งห้ามใช้ค่าจริงระหว่างทางแบบเดียวกัน

แต่ละ fold เทียบกับ baseline ใช้ค่าล่าสุด ไม่มี final future holdout แยกในตัวอย่างนี้ จึงเป็นผล validation สำหรับเรียนวิธี split หากนำไปเลือกโมเดลแล้วต้องกันช่วงอนาคตสุดท้ายไว้อีกชุด

## เชื่อมกับการใช้งานจริง

ตรวจซ้ำเมื่อข้อมูลหรือพฤติกรรมเปลี่ยน บันทึกช่วงเวลาที่ฝึก แหล่งข้อมูล metric รายกลุ่ม และกรณีที่ระบบควรปฏิเสธการทำนาย การประเมินให้ตรงการใช้งานมักสำคัญกว่าการเปลี่ยน algorithm

## การทดลองสองแบบตอบคนละคำถาม

ข้อมูลจำลองมี 60 คน คนละห้าแถว รวม 300 แถว คนเดียวกันมี feature คล้ายกันและใช้ label เดียวกัน แต่ label ถูกสุ่มระหว่างคน จึงไม่มีรูปแบบที่ควรทายคนใหม่ได้อย่างสม่ำเสมอ

โค้ดลอง random split ของแถวก่อนเพื่อให้เห็นคะแนนที่อาจสูงเมื่อคนเดิมอยู่สองฝั่ง แล้วลอง group split ที่กันคน 12 คนออกไปทั้งกลุ่ม ให้ดูจำนวน `Shared users` พร้อมคะแนน ไม่ใช้คะแนนสูงกว่าเลือกวิธี split เพราะโจทย์ “แถวใหม่ของคนเดิม” กับ “คนใหม่” ต่างกัน

`groups` เป็นรหัสสำหรับจัดแบ่ง ไม่ส่งเป็นคอลัมน์เข้า KNN ส่วน `next(split.split(...))` ดึงคู่ index จากการแบ่งหนึ่งครั้ง `test_size=0.2` ของ GroupShuffleSplit หมายถึงสัดส่วน **กลุ่ม** ไม่ใช่จำนวนแถวเสมอไป ที่ได้ 60 แถวในบทนี้เพราะทุกคนมีห้าแถวเท่ากัน

## อ่าน lag และการเดินตามเวลา

ถ้า series เป็น `[10, 12, 11, 15]` การทำ `lag=series[:-1]` และ `target=series[1:]` สร้างคู่ `(10→12), (12→11), (11→15)` หนึ่งแถวจึงใช้ค่าก่อนหน้าทำนายค่าถัดไป ไม่ใช่เอา target ปัจจุบันเป็น feature

TimeSeriesSplit ขยายช่วง train ไปข้างหน้าแล้วตรวจช่วงถัดไป `gap=1` เว้นหนึ่งแถวก่อน validation ในตัวอย่างนี้ไม่เท่ากับกำหนด horizon หนึ่งวันกับทุกชุดข้อมูล การออกแบบ gap ต้องดูว่าแต่ละแถวครอบคลุมช่วงเวลาเท่าไร

**Persistence baseline** ทายว่าค่าถัดไปเท่ากับค่าล่าสุด เช่น 12→12 โมเดลซับซ้อนควรเทียบกับวิธีง่ายนี้ก่อน ค่า MAE ในแต่ละ fold ยิ่งต่ำยิ่งดี และเป็น validation score สำหรับรูปแบบการใช้งาน one-step-ahead ที่อธิบายไว้

## คัดลอกโค้ดไปรัน

คัดลอกบล็อก `python` ด้านล่างทั้งชุดไป Code cell ใน Jupyter หรือ Colab แล้วกด Shift+Enter หากเปิด `lesson.ipynb` ของบทนี้อยู่แล้ว ให้รัน cell ที่มีตามลำดับ ไม่ต้องคัดลอกซ้ำ ดู [วิธีเริ่มเรียน](../../docs/start-here.md) หากยังไม่ได้เตรียมแพ็กเกจ

<details>
<summary>ทางเลือก: รันไฟล์ Python จาก repo ในเครื่อง</summary>

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/17-time-and-groups/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` บล็อกนี้รันได้ทั้งชุดโดยไม่ต้องรันบทก่อนหน้า

</details>

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

## อ่านผลที่ได้

assert ยืนยันกลุ่มคนไม่ซ้ำระหว่าง train/test และทุก time fold มี train ก่อน validation พร้อม MAE เทียบค่าล่าสุด ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## แบบฝึกหัด

ออกแบบ split สำหรับข้อมูลร้านค้า 20 ร้าน บันทึกทุกวัน และต้องทำนายทั้งเดือนหน้าในร้านเดิมกับร้านที่ไม่เคยเห็น ระบุว่าเป็นการทดลองเดียวกันหรือไม่

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

เป็นคนละคำถาม ร้านเดิมเดือนหน้าต้องแยกตามเวลา ส่วนร้านใหม่ต้องกันร้านออกจาก train และยังคุมเวลาให้ตรงการใช้งาน อาจออกแบบการทดสอบสองชุดเพื่อรายงานแยก ไม่เลือก random split เพียงเพราะคะแนนสูงกว่า

</details>

## อ่านต่อ

- [Cross-validation: groups and time series](https://scikit-learn.org/1.7/modules/cross_validation.html)
- [Common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html)
- [Google MLCC: Production ML systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems)

