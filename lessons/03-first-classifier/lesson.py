# %% Cell 1 — โหลดและแบ่งข้อมูล
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, stratify=data.target, random_state=42
)

# %% [markdown]
# **อ่านโค้ด**
#
# - `load_iris()` คืนข้อมูลพร้อมรายละเอียดชุดข้อมูล โดย `data.data` เป็น array ของ feature สี่ค่า และ `data.target` เป็นรหัสคำตอบ
# - บทนี้ไม่ได้ขอ `as_frame=True` จึงใช้ array แทน DataFrame แบบบท 02
# - `train_test_split` คืน X_train, X_test, y_train, y_test ตามลำดับ โดยจับคู่ข้อมูลกับคำตอบให้ตรงกัน
# - `test_size=0.2` กัน test 30 ดอก ส่วน `stratify=data.target` รักษาสัดส่วนคลาส และ seed 42 ใช้ทำซ้ำ

# %% Cell 2 — ฝึก Pipeline และเทียบ baseline
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Baseline accuracy: {accuracy_score(y_test, baseline.predict(X_test)):.3f}")
print(f"KNN test accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, target_names=data.target_names))

# %% [markdown]
# **อ่านโค้ด**
#
# - `make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))` ต่อการปรับสเกลกับ KNN ตามลำดับ โดย KNN ใช้เพื่อนบ้าน 5 ตัว
# - `model.fit(X_train, y_train)` ให้ scaler เรียนสถิติจาก train แล้วส่งข้อมูลที่ปรับสเกลให้ KNN
# - `DummyClassifier(...).fit(...)` สร้างและฝึก baseline ในบรรทัดเดียว โดย `fit` คืนโมเดลที่ฝึกแล้ว ส่วน `strategy="most_frequent"` ให้ทายคลาสพบบ่อยใน train
# - `model.predict(X_test)` ใช้ scaler ที่ fit แล้วแปลง test ก่อนทำนาย ไม่ได้เรียนสถิติใหม่จาก test
# - `accuracy_score(y_test, pred)` เทียบคำตอบจริงกับคำทำนาย ส่วน `classification_report` แยกคะแนนรายคลาส โดย `target_names` ใส่ชื่อสายพันธุ์ในรายงาน

# %% Cell 3 — ทำนายดอกไม้ดอกใหม่
new_flower = [[5.1, 3.5, 1.4, 0.2]]
label = model.predict(new_flower)[0]
print("New flower:", data.target_names[label])

# %% [markdown]
# **อ่านโค้ด**
#
# - `[[5.1, 3.5, 1.4, 0.2]]` ใช้วงเล็บสองชั้นให้ได้ข้อมูลหนึ่งแถว สี่คอลัมน์ ตามรูปแบบ `(n_samples, n_features)` ที่โมเดลรับ
# - ค่าทั้งสี่เรียงเป็นความยาว/ความกว้างกลีบเลี้ยง แล้วตามด้วยความยาว/ความกว้างกลีบดอก หน่วยเซนติเมตร
# - `predict(new_flower)[0]` เลือกรหัสคำตอบของดอกแรกจากผลทำนายที่เป็น array
# - `data.target_names[label]` แปลงรหัสนั้นเป็นชื่อสายพันธุ์ ซึ่งตัวอย่างนี้ได้ setosa
