# %% Cell 1 — เตรียมข้อความและคำตอบ
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts = [
    "team wins the football match", "player scores a goal", "coach trains the team",
    "football fans celebrate victory", "the match ends in a draw", "goal keeper saves the team",
    "basketball team wins tonight", "player joins the football club", "coach plans match tactics",
    "fans watch the final game", "the club signs a player", "training helps the team win",
    "software update fixes a bug", "python code runs on computer", "developer tests the software",
    "new computer chip released", "database stores application data", "the code has a bug",
    "software developer writes tests", "computer program processes data", "python library update released",
    "database server runs software", "chip powers the new computer", "developer fixes python code",
]
y = np.array(["sport"] * 12 + ["tech"] * 12)
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.25, stratify=y, random_state=42
)
print("Train:", len(X_train), "| Test:", len(X_test))

# %% [markdown]
# **อ่านโค้ด**
#
# - `texts` เก็บข้อความดิบ 24 ข้อความ: 12 ข้อความแรกเป็นกีฬา และ 12 ข้อความหลังเป็นเทคโนโลยี ส่วน `y` สร้างคำตอบให้ตรงกับลำดับนั้น
# - `train_test_split` แบ่งข้อความพร้อมคำตอบก่อนเรียน vocabulary: train 18 ข้อความ (คลาสละ 9) และ test 6 ข้อความ (คลาสละ 3) โดย `stratify=y` รักษาสัดส่วนคลาส
# - `random_state=42` ทำให้รันซ้ำแล้วได้ชุดแบ่งเดิม ส่วน `display` ใช้แสดงตารางใน Jupyter แม้มีคำสั่งอื่นตามท้าย cell
#
# **ผลที่ได้:** `Train: 18 | Test: 6`

# %% Cell 2 — ฝึก Pipeline และประเมินกับ test
model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

vectorizer = model.named_steps["countvectorizer"]
words = vectorizer.get_feature_names_out()
print("Vocabulary size:", len(words))

# %% [markdown]
# **อ่านโค้ด**
#
# - `make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))` ต่อ **ตัวนับคำ → โมเดลจำแนกประเภท** โดย `alpha=1.0` เพิ่มจำนวนสมมติ 1 ให้ทุกคำในแต่ละคลาสตอนคำนวณความน่าจะเป็น
# - `fit(X_train, y_train)` เรียน vocabulary จากข้อความ train แล้วใช้จำนวนคำกับคำตอบฝึก Naive Bayes ส่วน `predict(X_test)` ใช้ vocabulary และโมเดลที่เรียนไว้ โดยไม่ fit ใหม่
# - `classification_report` อ่านเหมือนบท 04; `zero_division=0` ให้ค่า 0 แทนกรณีที่ตัวหารของ metric เป็นศูนย์ ไม่ได้เปลี่ยนคำทำนาย
# - `model.named_steps["countvectorizer"]` หยิบตัวนับคำที่ฝึกแล้วออกมาตรวจ และ `get_feature_names_out()` คืนชื่อคำตามลำดับคอลัมน์
#
# **ผลที่ได้:** vocabulary มี 49 คำ และทาย test ถูกทั้ง 6 ข้อความ จึงได้ accuracy 1.00 ผลนี้เป็นเพียงชุดตัวอย่างเล็กที่แต่งขึ้น ไม่ใช่หลักฐานว่าระบบจริงจะถูก 100%

# %% Cell 3 — ดูจำนวนคำและทำนายข้อความใหม่
examples = ["team team goal", "developer fixes software", "recipe with vegetables"]
vectors = vectorizer.transform(examples)
print("Feature matrix shape:", vectors.shape)

used_columns = np.asarray(vectors.sum(axis=0)).ravel() > 0
counts = pd.DataFrame(
    vectors[:, used_columns].toarray(),
    index=examples,
    columns=words[used_columns],
)
display(counts)

results = pd.DataFrame({
    "text": examples,
    "prediction": model.predict(examples),
    "known_tokens": np.asarray(vectors.sum(axis=1)).ravel(),
})
probabilities = pd.DataFrame(
    model.predict_proba(examples),
    columns=model.classes_,
)
display(pd.concat([results, probabilities], axis=1).round(3))

# %% [markdown]
# **อ่านโค้ด**
#
# - `vectorizer.transform(examples)` นับคำด้วย vocabulary เดิม ได้ sparse matrix ขนาด `(3, 49)`: 3 ข้อความ × 49 คำ โดย sparse matrix เก็บข้อมูลแบบประหยัดพื้นที่เมื่อส่วนใหญ่เป็นศูนย์
# - `vectors.sum(axis=0)` รวมตามคอลัมน์; `np.asarray(...).ravel() > 0` จัดเป็น array หนึ่งมิติแล้วเลือกคำที่ปรากฏอย่างน้อยหนึ่งครั้งใน 3 ตัวอย่าง ได้ 5 คอลัมน์สำหรับแสดงผล
# - `vectors[:, used_columns].toarray()` แปลงเฉพาะตารางเล็กเป็น array แล้วใช้ชื่อข้อความและคำกำกับแถว/คอลัมน์ การเลือกนี้ใช้เพื่อแสดงผลเท่านั้น โมเดลยังรับ 49 features
# - `vectors.sum(axis=1)` รวมตามแถว ได้ `known_tokens` หรือจำนวนคำที่รู้จักต่อข้อความ **นับคำซ้ำด้วย**: `[3, 3, 0]`
# - `predict` คืนชื่อคลาส ส่วน `predict_proba` คืนความน่าจะเป็นตามโมเดล ใช้ `model.classes_` ตั้งชื่อคอลัมน์ให้ถูกลำดับ: `sport, tech`
# - `pd.concat(..., axis=1)` ต่อผลสองตารางตามแนวนอน และ `round(3)` ปัดเลขเพื่อแสดงผล ไม่ได้เปลี่ยนโมเดล
