# %% Cell 1 — เตรียมข้อความและประเภทที่กำกับไว้
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
# - `texts` เก็บข้อความดิบ 24 ข้อความ: 12 ข้อความแรกเป็นกีฬา และ 12 ข้อความหลังเป็นเทคโนโลยี ส่วน `y` เก็บประเภทข้อความ (class labels) เป็น sport และ tech ให้ตรงกับลำดับข้อความ
# - `train_test_split` แบ่งข้อความกับประเภทที่กำกับไว้เป็นคู่ ๆ ก่อนสร้างรายการคำ (vocabulary): train 18 ข้อความ (คลาสละ 9) และ test 6 ข้อความ (คลาสละ 3) โดย `stratify=y` รักษาสัดส่วนคลาส
# - `random_state=42` ทำให้รันซ้ำแล้วได้ชุดแบ่งเดิม ส่วน `display` ใช้แสดงตารางใน Jupyter แม้มีคำสั่งอื่นตามท้าย cell
#
# **ผลที่ได้:** `Train: 18 | Test: 6`

# %% Cell 2 — ฝึก Pipeline และประเมินกับ test
model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
display(pd.DataFrame({"text": X_test, "actual": y_test, "prediction": y_pred}))
print(classification_report(y_test, y_pred, zero_division=0))

vectorizer = model.named_steps["countvectorizer"]
words = vectorizer.get_feature_names_out()
print("Vocabulary size:", len(words))

# %% [markdown]
# **อ่านโค้ด**
#
# - `make_pipeline(...)` ต่อ **CountVectorizer → MultinomialNB** ส่วน `alpha=1.0` เพิ่มจำนวนสมมติ 1 ให้ทุกคำในแต่ละคลาสตามวิธี smoothing
# - `model.fit(X_train, y_train)` เริ่มจากให้ CountVectorizer **fit สร้าง vocabulary แล้ว transform ข้อความเป็นตารางจำนวนคำ** จากนั้นจึงให้ Naive Bayes **fit ตารางจำนวนคำคู่กับ labels ใน y_train** เพื่อเรียน prior และความน่าจะเป็นของคำในแต่ละคลาส
# - `model.predict(X_test)` ให้ตัวนับคำใช้ vocabulary เดิมแปลงข้อความ test ก่อนให้โมเดลเลือก sport หรือ tech โดยไม่ fit ใหม่
# - ตาราง `actual` กับ `prediction` ให้ดูว่าแต่ละข้อความทายตรงกับคลาสจริงหรือไม่ ก่อนอ่าน `classification_report` ตามบท 04; `zero_division=0` ให้ค่า 0 ในกรณีที่ตัวหารของ metric เป็นศูนย์
# - `model.named_steps["countvectorizer"]` เข้าถึงตัวนับคำที่ fit แล้ว ส่วน `get_feature_names_out()` คืนชื่อคำตามลำดับคอลัมน์ เก็บใน `words` เพื่อใช้กำกับตารางใน Cell 3
#
# **ผลที่ได้:** `Vocabulary size: 49` หมายถึงคำไม่ซ้ำ 49 คำ จึงมี 49 features ส่วน test ทายถูกทั้ง 6 ข้อความ ได้ accuracy `6/6 = 1.00` คะแนนนี้มาจากข้อความที่แต่งขึ้นเพียง 6 ข้อความ จึงยังไม่ยืนยันผลกับข้อความจริงที่หลากหลาย

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
# - `examples` มีข้อความกีฬา เทคโนโลยี และอาหาร เพื่อดูทั้งกรณีที่พบคำใน vocabulary และกรณีที่ไม่พบเลย
# - `vectorizer.transform(examples)` นับคำลงคอลัมน์เดิม ได้ **Feature matrix หรือตาราง features** ขนาด `(3, 49)`: 3 ข้อความ × 49 คำ ใช้รูปแบบ sparse matrix เพื่อประหยัดพื้นที่สำหรับตารางที่ส่วนใหญ่เป็นศูนย์
# - `used_columns` เลือกเฉพาะคำที่พบในตัวอย่างมาแสดง: `sum(axis=0)` รวมจำนวนตามคอลัมน์ และ `np.asarray(...).ravel() > 0` ทำเป็นรายการ True/False หนึ่งมิติ คำที่มีผลรวมมากกว่า 0 ได้ True
# - `vectors[:, used_columns].toarray()` เลือก 5 คอลัมน์นั้นแล้วแปลงเป็น array สำหรับแสดงตาราง โมเดลยังใช้ features ครบ 49 คอลัมน์ ไม่ได้ฝึกใหม่ด้วย 5 คอลัมน์
# - `known_tokens` มาจาก `vectors.sum(axis=1)` ซึ่งรวมจำนวนคำต่อข้อความ **นับคำซ้ำด้วย** จึงได้ `[3, 3, 0]` เช่น team สองครั้งกับ goal หนึ่งครั้งรวมเป็น 3; ค่า 0 หมายถึงไม่มีคำใดตรงกับ vocabulary
# - `model.predict(examples)` รับข้อความดิบแล้วให้ Pipeline นับคำก่อนเลือกคลาส ส่วน `predict_proba` ให้ความน่าจะเป็นของแต่ละคลาส ใช้ `model.classes_` กำกับคอลัมน์ sport, tech ให้ตรงกับลำดับผลลัพธ์
# - `pd.concat(..., axis=1)` ต่อข้อความ คำทำนาย และความน่าจะเป็นให้อยู่ในตารางเดียว ส่วน `round(3)` ปัดเลขเฉพาะการแสดงผล
