# EP2 · ตรวจข้อมูลเพิ่มเติม (เลือกอ่าน)

[กลับบท 02](../lessons/02-data-and-plots/README.md)

ใช้หลังรัน Cell 1–5 ของบทหลักใน notebook เดียวกัน ขั้นตอนเหล่านี้ช่วยตรวจปัญหาเพิ่มเติม แต่ไม่จำเป็นต่อการทำความเข้าใจลำดับหลักของ EP2

## 1. ตรวจว่าข้อมูลกับคำตอบตรงแถว และไม่มี index อยู่ทั้งสองชุด

```python
assert X_train.index.equals(y_train.index)
assert X_test.index.equals(y_test.index)
assert set(X_train.index).isdisjoint(X_test.index)

print("Row checks passed")
```

`assert` ทำงานต่อเมื่อเงื่อนไขเป็น True และเกิด AssertionError หยุด cell ตรงบรรทัดนั้นเมื่อเป็น False ส่วน `isdisjoint` ให้ True เมื่อไม่มีสมาชิกที่ซ้ำกันระหว่างสองชุด

ใน Iris บทนี้ index ไม่ซ้ำ และ train_test_split จัดแถวกับคำตอบให้สอดคล้องกันอยู่แล้ว จึงใช้ตรวจยืนยัน ไม่ได้เปลี่ยนวิธีแบ่งข้อมูล การไม่มี index ซ้ำยังไม่รับรองว่าเป็นคนละตัวอย่างจริง เช่น คนเดิมอาจถูกบันทึกหลายแถว ดูกรณีนี้ในบท 17

## 2. ตรวจ NaN และค่าอนันต์

```python
import numpy as np

matrix = X_train.to_numpy()
print("All values are finite:", np.isfinite(matrix).all())
```

`isfinite` ตรวจว่าค่าไม่ใช่ NaN, inf หรือ -inf ส่วน `all()` ตรวจว่าทุกค่าผ่าน ผลของบทนี้เป็น True หากได้ False ต้องตรวจค่าที่มีปัญหาก่อนนำไปใช้; การเติมค่าหายจะเรียนในบท 06 การผ่านเงื่อนไขนี้ยังไม่รับรองว่าค่าที่วัดสมเหตุสมผล

## 3. ตรวจแถวที่มีขนาดกลีบเหมือนกัน

```python
duplicate_mask = X_train.duplicated(keep=False)
duplicates = X_train.loc[duplicate_mask].copy()
duplicates["target"] = y_train.loc[duplicate_mask]

print("Duplicate feature rows:", duplicate_mask.sum())
duplicates
```

`keep=False` ทำเครื่องหมายทุกแถวในกลุ่มที่ซ้ำ รวมทั้งแถวแรก ได้ 2 แถว คือ index 101 และ 142 เป็น **หนึ่งคู่** ที่มีขนาดกลีบเหมือนกัน ไม่ใช่สองคู่

คอลัมน์ target ในตาราง duplicates ใช้ดูคำตอบประกอบ ไม่ได้เพิ่มเข้า X_train อย่าเพิ่งลบแถวซ้ำ เพราะอาจเป็นคนละดอกที่วัดได้เท่ากัน ต้องตรวจที่มาของข้อมูลก่อน

## 4. ตรวจคอลัมน์ที่ไม่มีความแตกต่างในชุดฝึก

```python
constant_features = X_train.columns[X_train.nunique() <= 1].tolist()
print("Constant features:", constant_features)
```

`nunique()` นับจำนวนค่าที่แตกต่างกัน เช่น คอลัมน์ที่ทุกแถวเป็น 1 จะนับได้ 1 คอลัมน์นั้นไม่มีความต่างให้ใช้แยกตัวอย่างในชุดฝึก ผลของบทนี้เป็น `[]` คือไม่พบคอลัมน์ดังกล่าว

## อ้างอิง

- [Python: assert](https://docs.python.org/3.11/reference/simple_stmts.html#the-assert-statement)
- [NumPy: isfinite](https://numpy.org/doc/2.2/reference/generated/numpy.isfinite.html)
- [pandas: DataFrame.duplicated](https://pandas.pydata.org/pandas-docs/version/2.3/reference/api/pandas.DataFrame.duplicated.html)
