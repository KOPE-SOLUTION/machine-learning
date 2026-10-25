# %% Cell 1 — เตรียมข้อความและคำตอบ
import numpy as np
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

# %% [markdown]
# **อ่านโค้ด**
#
# - `texts` เก็บข้อความดิบหนึ่งข้อความต่อสมาชิก โดย 12 ข้อความแรกเป็นกีฬา และ 12 ข้อความหลังเป็นเทคโนโลยี
# - `np.array(["sport"] * 12 + ["tech"] * 12)` สร้าง label ให้เรียงตรงกับข้อความ จึงต้องรักษาลำดับให้สัมพันธ์กัน
# - `train_test_split(..., test_size=0.25, stratify=y)` แบ่งข้อความพร้อมคำตอบเป็น train 18 และ test 6 โดยรักษาสัดส่วนคลาส
# - ยังไม่ต้องแปลงข้อความเป็นตัวเลขเอง เพราะจะให้ Pipeline ทำใน Cell ถัดไป

# %% Cell 2 — ฝึกตัวนับคำและตรวจคำทำนาย
model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test), zero_division=0))
examples = ["team scores a goal", "developer fixes software", "recipe with vegetables"]
vectorizer = model.named_steps["countvectorizer"]
vectors = vectorizer.transform(examples)
print("Vocabulary size:", len(vectorizer.get_feature_names_out()))
used_columns = np.asarray(vectors.sum(axis=0)).ravel() > 0
print("Words present in examples:", vectorizer.get_feature_names_out()[used_columns])
print("Counts for these words:\n", vectors[:, used_columns].toarray())
for text, label, token_count in zip(examples, model.predict(examples), vectors.sum(axis=1).A1):
    print(f"{text!r} -> {label}; known-token count={token_count}")
print("Class order:", model.classes_)
print("Probabilities (not guaranteed calibrated):\n", model.predict_proba(examples).round(3))

# %% [markdown]
# **อ่านโค้ด**
#
# - `make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))` ต่อการนับคำกับโมเดล โดย `fit` เรียน vocabulary และโมเดลจาก train เท่านั้น
# - `model.named_steps["countvectorizer"]` หยิบตัวนับคำที่ fit แล้วออกมาตรวจ ส่วน `transform(examples)` นับคำในข้อความใหม่โดยใช้ vocabulary เดิม
# - `vectors.sum(axis=0)` รวมจำนวนคำตามคอลัมน์ แล้ว `np.asarray(...).ravel() > 0` ทำ mask เลือกเฉพาะคำที่พบในตัวอย่าง
# - `vectors[:, used_columns].toarray()` เลือกคอลัมน์เหล่านั้นและแปลง sparse matrix เป็น array เพื่อแสดงตารางเล็ก ๆ
# - `vectors.sum(axis=1).A1` รวมจำนวนคำที่รู้จักต่อข้อความ แล้วจัดผลเป็น array หนึ่งมิติ เพื่อใช้ `zip` จับคู่กับข้อความและคำทำนาย
# - `model.classes_` บอกลำดับคอลัมน์ของ `predict_proba` ส่วนข้อความอาหารที่ไม่มีคำรู้จักยังได้คำทำนาย จึงต้องอ่าน known-token count ประกอบด้วย
