# %% Cell 1 — สร้างข้อมูลและแยก train/validation/test
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=1500, n_features=8, n_informative=5,
    weights=[0.9, 0.1], flip_y=0.02, random_state=42,
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
)

# %% [markdown]
# **อ่านโค้ด**
#
# - `make_classification` สร้างข้อมูลจำลอง 1,500 แถว 8 feature โดย `n_informative=5` กำหนดจำนวน feature ที่ใช้สร้างสัญญาณแยกคลาส
# - `weights=[0.9, 0.1]` ตั้งสัดส่วนคลาสก่อนเติมความผิดพลาดของ label ด้วย `flip_y=0.02` สัดส่วนสุดท้ายจึงอาจไม่ใช่ 90/10 พอดี
# - แบ่งรอบแรกกัน test 20% เหลือ `X_dev` 1,200 แถวสำหรับพัฒนาโมเดล
# - แบ่งรอบสองกัน validation 25% **ของ X_dev** คือ 300 แถว จึงเหลือ train 900 แถว
#
# **ผลที่ได้:** ตอนนี้มีข้อมูล train 900 แถว, validation 300 แถว และ test 300 แถว ยังไม่ได้ฝึกหรือทำนาย จึงยังไม่มีคะแนนแสดงใต้ cell

# %% Cell 2 — เปรียบเทียบโมเดลบน validation
models = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
}
valid_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    valid_scores[name] = f1_score(y_valid, pred, zero_division=0)
    print(name, {
        "accuracy": round(accuracy_score(y_valid, pred), 3),
        "precision": round(precision_score(y_valid, pred, zero_division=0), 3),
        "recall": round(recall_score(y_valid, pred), 3),
        "f1": round(valid_scores[name], 3),
    })

# %% [markdown]
# **อ่านโค้ด**
#
# - `models` เก็บชื่อกับโมเดลแต่ละตัว เพื่อวนฝึกและประเมินด้วยข้อมูลชุดเดียวกัน
# - `fit(X_train, y_train)` ใช้ชุดฝึก ส่วน `predict(X_valid)` ใช้ทำนายชุด validation
# - `valid_scores[name]` เก็บ F1 ของแต่ละโมเดล เพื่อใช้เลือกใน Cell ถัดไป
# - `zero_division=0` ให้รายงาน 0 เมื่อสูตรของ metric นั้นหารด้วยศูนย์ ไม่ได้เปลี่ยนคำทำนายของโมเดล
#
# **การตั้งค่าของโมเดลใน cell นี้**
#
# - `StandardScaler` เรียนค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานจาก train ภายใน Pipeline แล้วใช้ค่าเดิมแปลง validation
# - `max_iter=1000` เป็นเพดานรอบของตัวแก้ปัญหา ไม่ได้บังคับให้ฝึกครบ 1,000 รอบ และไม่ใช่ threshold
# - `LogisticRegression` รุ่นที่ใช้มี L2 regularization และ `C=1.0` เป็นค่าตั้งต้น เพื่อควบคุมขนาดน้ำหนัก ค่า C น้อยลงควบคุมเข้มขึ้น บทนี้ยังไม่ค้นหาค่า C
#
# **อ่านผล:** บรรทัด baseline และ logistic เป็นคะแนนบน validation 300 แถวของการทดลองนี้ จึงไม่ใช่ 90% และ 85% จากตัวอย่างข้อความก่อนลงมือทำ

# %% Cell 3 — ฝึกตัวเลือกที่ชนะและประเมิน test
chosen = max(valid_scores, key=valid_scores.get)
final = models[chosen].fit(X_dev, y_dev)
test_pred = final.predict(X_test)
positive_column = list(final.classes_).index(1)
test_probability = final.predict_proba(X_test)[:, positive_column]
print("Selected:", chosen)
print("Test actual counts (class 0, class 1):", (y_test == 0).sum(), (y_test == 1).sum())
print("Test predicted counts (class 0, class 1):", (test_pred == 0).sum(), (test_pred == 1).sum())
print("Test confusion matrix (rows=true, columns=predicted):\n",
      confusion_matrix(y_test, test_pred, labels=[0, 1]))
print(f"Test accuracy: {accuracy_score(y_test, test_pred):.3f}")
print(f"Test F1: {f1_score(y_test, test_pred):.3f}")
print(f"Test average precision: {average_precision_score(y_test, test_probability):.3f}")

# %% [markdown]
# **อ่านโค้ด**
#
# - `max(valid_scores, key=valid_scores.get)` เลือก **ชื่อโมเดล** ที่มี F1 บน validation สูงสุด
# - `models[chosen].fit(X_dev, y_dev)` ฝึกโมเดลที่เลือกใหม่ด้วย train รวม validation ก่อนประเมิน test
# - `list(final.classes_).index(1)` หาว่าคลาส 1 อยู่คอลัมน์ใดของผล `predict_proba`
# - `[:, positive_column]` เลือก probability ของคลาส 1 ทุกแถว เพื่อใช้คำนวณ average precision
# - `confusion_matrix(..., labels=[0, 1])` กำหนดลำดับคลาสของแถวและคอลัมน์ ส่วน F1 ใช้ label จาก `predict` ต่างจาก AP ที่ใช้ probability
#
# **อ่านสรุปก่อนตาราง:** `Test actual counts` นับคำตอบจริง ส่วน `Test predicted counts` นับว่าทายคลาส 0 และ 1 อย่างละกี่แถว เช่น `(test_pred == 1).sum()` นับแถวที่ทายเป็น 1 จำนวนที่ทายเป็น 1 ยังไม่ใช่จำนวนที่ทายถูก ต้องอ่านคู่กับค่าจริงใน confusion matrix
#
# `Test accuracy` ใช้คำทำนาย `test_pred` ชุดเดียวกับตาราง จึงเท่ากับผลรวมแนวทแยงหารด้วยจำนวนแถว test
