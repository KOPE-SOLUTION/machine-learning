# %% Cell 1 — นำเข้าเครื่องมือ
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# %% [markdown]
# **อ่านโค้ด:** pandas ใช้จัดตาราง ส่วน scikit-learn เตรียมเครื่องมือ 4 อย่าง: `load_iris` โหลดข้อมูล, `train_test_split` แบ่งข้อมูล, `DummyClassifier` สร้าง baseline และ `accuracy_score` วัดผล
#
# **ผลที่ได้:** ไม่มีข้อความแสดงเป็นเรื่องปกติ เพราะ cell นี้นำเข้าเครื่องมือ ถ้าไม่มี error จึงไปต่อได้

# %% Cell 2 — อ่านข้อมูล Iris และแยก feature กับ target
data = load_iris(as_frame=True)
X = data.data
y = data.target

print(X.head(3).to_string(index=False))
print("First three labels:", y.head(3).to_list())
print("Classes:", dict(enumerate(data.target_names.tolist())))

# %% [markdown]
# **อ่านโค้ด**
#
# - `load_iris(as_frame=True)` โหลดชุดข้อมูล Iris ที่มากับ scikit-learn โดยให้ข้อมูลอยู่ในรูปแบบ pandas ตัวแปร `data` เก็บทั้งตาราง คำตอบ และชื่อสายพันธุ์
# - `X = data.data` เลือกตารางขนาดกลีบ 4 คอลัมน์ ส่วน `y = data.target` เลือกคำตอบที่ตรงกับแต่ละแถว X/y เป็นชื่อที่นิยมใช้ ไม่ใช่คำสงวนของ Python
# - `head(3)` แสดง 3 แถวแรก; `to_string(index=False)` จัดตารางเป็นข้อความโดยซ่อน index และ `to_list()` แสดงคำตอบเป็นรายการ
# - `dict(enumerate(data.target_names.tolist()))` จับคู่รหัสที่เริ่มจาก 0 กับชื่อสายพันธุ์
#
# **ผลที่ได้:** แถวแรกมีค่า `5.1, 3.5, 1.4, 0.2` หน่วยเซนติเมตร และคำตอบเป็น 0 = setosa ทั้งสามแถวแรกเป็น setosa แต่ข้อมูลทั้งชุดมี 3 สายพันธุ์

# %% Cell 3 — ตรวจรูปร่างและลำดับแถวของ X/y
print("X shape:", X.shape)
print("y shape:", y.shape)
assert X.shape == (150, 4)
assert y.shape == (150,)
assert X.index.equals(y.index)
print("Shape and index checks passed")

# %% [markdown]
# **อ่านโค้ด**
#
# - `X.shape` เป็น `(150, 4)` หมายถึง 150 ตัวอย่าง ตัวอย่างละ 4 features ส่วน `y.shape` เป็น `(150,)` หมายถึงคำตอบหนึ่งมิติ 150 ค่า
# - `assert` ตรวจเงื่อนไข ถ้าผิดจะหยุดด้วย `AssertionError` ตัวเลข 150 และ 4 เป็นขนาดเฉพาะของ Iris
# - `X.index.equals(y.index)` ตรวจว่า index และลำดับแถวตรงกัน ช่วยตรวจการจับคู่ แต่ไม่ได้พิสูจน์ว่าคำตอบในต้นฉบับถูกทุกค่า
#
# **ผลที่ได้:** `Shape and index checks passed`

# %% Cell 4 — แบ่งข้อมูลพร้อมคำตอบเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# %% [markdown]
# **อ่านโค้ด**
#
# - `train_test_split(X, y, ...)` แบ่ง X/y พร้อมกันเพื่อรักษาคู่ข้อมูลกับคำตอบ คืนค่าตามลำดับ `X_train, X_test, y_train, y_test`
# - `test_size=0.2` กัน 20% ไว้ทดสอบ จึงเหลือ train 120 ดอกและ test 30 ดอก สัดส่วนนี้เป็นตัวเลือกของตัวอย่าง ไม่ใช่กฎตายตัว
# - `stratify=y` ช่วยรักษาสัดส่วนคลาสให้ใกล้เคียงข้อมูลเดิม ไม่ได้เพิ่มคลาสที่มีน้อยให้เท่าคลาสอื่น
# - `random_state=42` ทำให้แบ่งซ้ำได้เมื่อข้อมูลและเงื่อนไขเดิม เลข 42 ไม่ได้ทำให้โมเดลแม่นขึ้น
#
# **ผลที่ได้:** X_train `(120, 4)`, X_test `(30, 4)`, y_train `(120,)`, y_test `(30,)`
#
# การแบ่งแบบนี้ใช้ฝึกแนวคิดกับ Iris ข้อมูลตามเวลาหรือคนเดิมหลายแถวต้องพิจารณาวิธีแบ่งเพิ่มเติมในบท 17

# %% Cell 5 — ตรวจจำนวนคลาสหลังแบ่งข้อมูล
class_counts = pd.DataFrame({
    "train": y_train.value_counts().sort_index(),
    "test": y_test.value_counts().sort_index(),
})
class_counts.index.name = "class"
print(class_counts.to_string())

# %% [markdown]
# **อ่านโค้ด:** `value_counts()` นับแต่ละรหัสสายพันธุ์ และ `sort_index()` เรียงรหัส 0, 1, 2 ก่อนแสดง train/test คู่กัน ส่วน `class_counts.index.name` ตั้งชื่อหัวแถวว่า class
#
# **ผลที่ได้:** ทุกคลาสมี train 40 ดอกและ test 10 ดอก ซึ่งรักษาสัดส่วนเท่ากันเหมือนข้อมูลเดิม

# %% Cell 6 — ฝึก baseline จากชุด train
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
print("Baseline trained")

# %% [markdown]
# **อ่านโค้ด**
#
# - `DummyClassifier(strategy="most_frequent")` สร้างตัวทำนายและกำหนดให้ตอบคลาสที่พบมากที่สุดใน train
# - `fit(X_train, y_train)` คือขั้นเรียนจากชุดฝึก สำหรับ baseline นี้สิ่งที่เรียนคือความถี่ของคลาสใน y_train ไม่ได้เรียนจากค่าขนาดกลีบ
# - เมื่อจำนวนเท่ากันทั้ง 3 คลาส ตัวอย่างนี้เลือกคลาส 0 ดังนั้นคำทำนายที่จะได้คือ setosa ทุกดอก
#
# **ผลที่ได้:** `Baseline trained` หมายถึงฝึกเสร็จ ยังไม่ได้วัดว่าทาย test ถูกกี่ดอก

# %% Cell 7 — ทำนายโดยส่งเฉพาะ X_test
pred = baseline.predict(X_test)
print("Predictions:", pred)
print("Prediction shape:", pred.shape)

# %% [markdown]
# **อ่านโค้ด:** `predict(X_test)` รับเฉพาะข้อมูลที่ต้องการทำนายและคืนคำตอบตามลำดับแถวของ X_test โดยไม่รับ y_test
#
# **ผลที่ได้:** `pred` เป็น array รูปร่าง `(30,)` และมีค่า 0 ทั้ง 30 ค่า เป็นผลตามวิธี most_frequent ไม่ใช่โค้ดค้างหรือสุ่มผิด

# %% Cell 8 — เปรียบเทียบคำทำนายกับคำตอบจริง
results = pd.DataFrame({"actual": y_test, "predicted": pred})
results["correct"] = results["actual"] == results["predicted"]
print(results.head(10).to_string())

# %% [markdown]
# **อ่านโค้ด**
#
# - `actual` คือคำตอบจริงจาก y_test ส่วน `predicted` คือคำทำนายจาก pred โดย pred เรียงตาม X_test และ y_test ที่แบ่งมาพร้อมกัน
# - `results["actual"] == results["predicted"]` เปรียบเทียบทีละแถว ได้ True เมื่อทายถูก และ False เมื่อทายผิด
# - `head(10)` แสดงเพียง 10 แถวแรก แต่การวัดผลใน Cell 9 ใช้ครบทั้ง 30 ดอก เลข index ด้านซ้ายเป็นหมายเลขแถวเดิมของ Iris
#
# **ผลที่ได้:** แถว index 38 มี actual=0, predicted=0 จึงถูก ส่วนแถว 127 มี actual=2, predicted=0 จึงผิด การนำคำตอบมาแสดงหลัง predict ไม่ได้ส่งเฉลยกลับเข้าโมเดล

# %% Cell 9 — วัดผลด้วย accuracy
accuracy = accuracy_score(y_test, pred)
print(f"Baseline test accuracy: {accuracy:.3f}")
print(f"Correct percentage: {accuracy:.1%}")

# %% [markdown]
# **อ่านโค้ด:** `accuracy_score(y_test, pred)` เทียบคำตอบจริงกับคำทำนายครบทั้งชุด `:.3f` แสดงทศนิยม 3 ตำแหน่ง ส่วน `:.1%` แสดงเปอร์เซ็นต์ทศนิยม 1 ตำแหน่ง ทั้งสองบรรทัดเป็นคะแนนค่าเดียวกัน
#
# **ผลที่ได้:** `Baseline test accuracy: 0.333` และ `Correct percentage: 33.3%`

# %% Cell 10 — คำนวณ accuracy เองจากจำนวนที่ทายถูก
correct_count = int((pred == y_test).sum())
total_count = len(y_test)
manual_accuracy = correct_count / total_count
print("Correct test predictions:", correct_count, "of", total_count)
print(f"Manual accuracy: {manual_accuracy:.3f}")
print("Same result:", manual_accuracy == accuracy)

# %% [markdown]
# **อ่านโค้ด:** `pred == y_test` เปรียบเทียบตามลำดับของดอกไม้ และ `.sum()` นับ True เป็น 1 จึงนับจำนวนที่ทายถูกได้ จากนั้นหารด้วยจำนวนดอกไม้ที่ทดสอบ
#
# **ผลที่ได้:** ทายถูก `10 of 30`, `Manual accuracy: 0.333` และ `Same result: True` ตรงกับ accuracy_score เพราะใช้คำตอบและคำทำนายชุดเดียวกัน
