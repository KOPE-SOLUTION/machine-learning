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

# %% [markdown]
# **อ่านโค้ด**
#
# - `make_pipeline(...)` ต่อ **CountVectorizer → MultinomialNB** ส่วน `alpha=1.0` เพิ่มจำนวนสมมติ 1 ให้ทุกคำในแต่ละคลาสตามวิธี smoothing
# - `model.fit(X_train, y_train)` ให้ CountVectorizer สร้าง vocabulary และแปลงข้อความฝึกเป็นจำนวนคำก่อน จากนั้น Naive Bayes จึงเรียนจากจำนวนคำคู่กับ labels ใน `y_train`
# - `model.predict(X_test)` ใช้ vocabulary และโมเดลที่ฝึกไว้ โดยไม่ fit ใหม่ ตาราง `actual` กับ `prediction` ให้เทียบประเภทจริงกับคำทำนายก่อนอ่าน `classification_report` ตามบท 04
# - `zero_division=0` ให้ค่า 0 ในกรณีที่ตัวหารของ metric เป็นศูนย์
#
# **ผลที่ได้:** test ทายถูกทั้ง 6 ข้อความ ได้ accuracy `6/6 = 1.00` คะแนนนี้มาจากข้อความที่แต่งขึ้นเพียง 6 ข้อความ จึงยังไม่ยืนยันผลกับข้อความจริงที่หลากหลาย
#
# ต่อไปเปิดดูข้อความฝึกและจำนวนคำที่ตัวนับคำใช้ เพื่อให้เห็นที่มาของตัวเลขก่อนแทนสูตร

# %% Cell 3 — นับคำในข้อความฝึกและสรุปแยกคลาส
vectorizer = model.named_steps["countvectorizer"]
words = vectorizer.get_feature_names_out()
train_vectors = vectorizer.transform(X_train)
train_word_counts = pd.DataFrame(train_vectors.toarray(), columns=words)

train_counts = pd.DataFrame({
    "text": X_train,
    "label": y_train,
    "token_count": train_word_counts.sum(axis=1),
    "team": train_word_counts["team"],
    "goal": train_word_counts["goal"],
})
for label in model.classes_:
    print(label)
    class_rows = train_counts.loc[
        train_counts["label"] == label,
        ["text", "token_count", "team", "goal"],
    ].reset_index(drop=True)
    class_rows.loc[len(class_rows)] = [
        "รวม", *class_rows[["token_count", "team", "goal"]].sum()
    ]
    display(class_rows)

class_summary = train_counts.groupby("label").agg(
    messages=("text", "size"),
    token_count=("token_count", "sum"),
    team=("team", "sum"),
    goal=("goal", "sum"),
)
class_summary["prior"] = class_summary["messages"] / len(X_train)
display(class_summary)

# %% [markdown]
# **อ่านโค้ด**
#
# - `model.named_steps["countvectorizer"]` ดึงตัวนับคำที่ฝึกแล้วออกจาก Pipeline ส่วน `get_feature_names_out()` คืนชื่อคำตามลำดับคอลัมน์
# - `transform(X_train)` นับคำด้วย vocabulary เดิม แล้ว `train_word_counts` แสดงหนึ่งแถวต่อข้อความและหนึ่งคอลัมน์ต่อคำ
# - `token_count` รวมจำนวนคำทุกคอลัมน์ในแถวนั้น **นับคำซ้ำด้วย** ส่วน `team` และ `goal` แสดงจำนวนครั้งของสองคำที่จะใช้ในตัวอย่างคำนวณ
# - `class_rows` เลือกข้อความของแต่ละคลาสมาแสดง แล้วเพิ่มแถวรวมเพื่อให้เห็นจำนวนที่จะนำไปใช้ในสูตร
# - `groupby("label").agg(...)` แยก sport/tech แล้วนับจำนวนข้อความด้วย `size` และรวมจำนวนคำด้วย `sum` ส่วน `prior` คือจำนวนข้อความของคลาสนั้นหารด้วยจำนวนข้อความฝึกทั้งหมด
#
# **ผลการนับจากข้อความฝึก:** `player scores a goal` นับได้ 3 คำ คือ `player`, `scores`, `goal` เพราะค่าเริ่มต้นของ CountVectorizer ไม่นับคำหนึ่งตัวอักษรอย่าง `a`
#
# **sport — 9 ข้อความ**
#
# | text | token_count | team | goal |
# | --- | --- | --- | --- |
# | fans watch the final game | 5 | 0 | 0 |
# | coach trains the team | 4 | 1 | 0 |
# | goal keeper saves the team | 5 | 1 | 1 |
# | training helps the team win | 5 | 1 | 0 |
# | coach plans match tactics | 4 | 0 | 0 |
# | player scores a goal | 3 | 0 | 1 |
# | team wins the football match | 5 | 1 | 0 |
# | the club signs a player | 4 | 0 | 0 |
# | the match ends in a draw | 5 | 0 | 0 |
# | **รวม** | **40** | **4** | **2** |
#
# **tech — 9 ข้อความ**
#
# | text | token_count | team | goal |
# | --- | --- | --- | --- |
# | database stores application data | 4 | 0 | 0 |
# | python code runs on computer | 5 | 0 | 0 |
# | python library update released | 4 | 0 | 0 |
# | new computer chip released | 4 | 0 | 0 |
# | chip powers the new computer | 5 | 0 | 0 |
# | software developer writes tests | 4 | 0 | 0 |
# | database server runs software | 4 | 0 | 0 |
# | the code has a bug | 4 | 0 | 0 |
# | software update fixes a bug | 4 | 0 | 0 |
# | **รวม** | **38** | **0** | **0** |
#
# **ตารางสรุปแยกคลาส**
#
# | label | messages | token_count | team | goal | prior |
# | --- | --- | --- | --- | --- | --- |
# | sport | 9 | 40 | 4 | 2 | 0.5 |
# | tech | 9 | 38 | 0 | 0 | 0.5 |
#
# อ่านแถว sport: มี **9 ข้อความ** และคำปรากฏรวม **40 ครั้ง** โดยเป็น team **4 ครั้ง** และ goal **2 ครั้ง** ส่วน tech มีคำรวม **38 ครั้ง** และไม่พบทั้ง team กับ goal
#
# **Prior มาจากจำนวนข้อความ:** ทั้งสองคลาสมีคลาสละ `9 / (9 + 9) = 9/18 = 0.5` ส่วนการหาความน่าจะเป็นของคำใช้จำนวนครั้งของคำในคอลัมน์ `token_count`, `team` และ `goal`

# %% Cell 4 — แสดง vocabulary และจำนวนคำไม่ซ้ำ
print("Total counted tokens:", train_counts["token_count"].sum())
print("Vocabulary size:", len(words))
for start in range(0, len(words), 7):
    print(", ".join(words[start:start + 7]))

# %% [markdown]
# **อ่านโค้ด**
#
# - ผลรวม `token_count` นับทุกครั้งที่คำปรากฏใน train ทั้งสองคลาส ส่วน `len(words)` นับคำไม่ซ้ำใน vocabulary ที่เรียนจาก train
# - ลูปแสดงรายการคำครั้งละ 7 คำ เพื่อให้อ่านรายการทั้งหมดได้ง่าย
#
# **ผลที่ได้:** คำปรากฏรวม `40 + 38 = 78` ครั้ง แต่เมื่อตัดคำซ้ำเหลือ **49 คำ** ตามรายการนี้:
#
# ```text
# application, bug, chip, club, coach, code, computer
# data, database, developer, draw, ends, fans, final
# fixes, football, game, goal, has, helps, in
# keeper, library, match, new, on, plans, player
# powers, python, released, runs, saves, scores, server
# signs, software, stores, tactics, team, tests, the
# training, trains, update, watch, win, wins, writes
# ```
#
# **เลข 49 ในสูตรมาจาก vocabulary นี้** ซึ่งรวมคำไม่ซ้ำจากทั้ง sport และ tech เมื่อใช้ `alpha=1` จึงเพิ่มจำนวนสมมติคำละ 1 ครบ 49 คำภายในแต่ละคลาส
#
# ### จากตารางนับคำไปสู่ Smoothing
#
# ![เปรียบเทียบ goal ใน tech: ไม่เพิ่ม alpha ได้ 0/38 = 0 ทำให้คะแนน tech เป็นศูนย์เมื่อข้อความมี goal; เพิ่ม alpha เท่ากับ 1 ได้ (0+1)/(38+49) = 1/87 ประมาณ 0.0115 ตัวส่วนเพิ่ม 49 เพราะ vocabulary มี 49 คำและเพิ่มคำละ 1](../../assets/figures/07-naive-bayes-text-smoothing.png)
#
# [เปิดภาพ SVG](../../assets/figures/07-naive-bayes-text-smoothing.svg)
#
# **ที่มาของ `(0+1)/(38+49)`:** แถว `tech` ในตารางสรุปให้ `goal=0` และ `token_count=38` ส่วน vocabulary มี 49 คำ และเลข 1 คือ `alpha` ที่กำหนดตอนสร้างโมเดล
#
# **คำที่ไม่มีใน vocabulary เลย** เช่น `vegetables` จะถูก CountVectorizer ข้าม ส่วน smoothing ใช้กับคำที่มีอยู่ใน vocabulary
#
# <details>
# <summary>อ่านสูตรและการเลือก alpha เพิ่มเติม</summary>
#
# เช่น goal อยู่ใน vocabulary แต่ไม่เคยพบในข้อความ tech หากไม่ทำ smoothing จะได้ความน่าจะเป็นของ goal ใน tech เป็นศูนย์ เมื่อข้อความใหม่มี goal คะแนนของ tech จึงถูกคูณด้วยศูนย์ แม้คำอื่นอาจสนับสนุน tech อยู่
#
# **Additive smoothing** เพิ่มจำนวนสมมติ `alpha` ให้ทุกคำใน vocabulary ของแต่ละคลาส เมื่อ `alpha=1` จำนวน 0 จะกลายเป็น 1 ในสูตร และจำนวน 3 จะกลายเป็น 4 จำนวนสมมติเพิ่มเฉพาะในสูตร ไม่ได้เพิ่มข้อความฝึก
#
# สูตรคือ **(จำนวนครั้งของคำนั้นในคลาส + alpha) ÷ (จำนวนคำทั้งหมดในคลาส + alpha × จำนวนคำไม่ซ้ำใน vocabulary)** โดยจำนวนคำทั้งหมดนับคำซ้ำด้วย ตัวส่วนต้องเพิ่มด้วย เพราะเราเพิ่มจำนวนสมมติให้ทุกคำ
#
# **แทนค่าจากชุดฝึกของบทนี้:** นับคำในข้อความ tech รวมได้ 38 ครั้ง และ vocabulary จาก train ทั้งสองคลาสมีคำไม่ซ้ำ 49 คำ เมื่อเพิ่มคำละ 1 ตัวส่วนจึงเป็น `38 + (1 × 49) = 87` ส่วน goal ใน tech พบ 0 ครั้ง จึงได้ `(0 + 1) / 87 = 1/87 ≈ 0.0115`
#
# ค่า 0.0115 เป็นความน่าจะเป็นของ **คำ goal ภายในคลาส tech** การหาความน่าจะเป็นว่า **ข้อความเป็น tech** ต้องนำ prior มาคูณกับความน่าจะเป็นของคำที่รู้จัก แล้วหารคะแนนแต่ละคลาสด้วยผลรวมคะแนนทุกคลาส
#
# `alpha` ของบทนี้ควบคุม smoothing **คนละความหมายกับ alpha ของ Ridge ในบท 06** เมื่อเพิ่มมากขึ้น ความน่าจะเป็นของคำภายในแต่ละคลาสจะใกล้กันขึ้น บทนี้กำหนด 1.0 ไว้เรียนหลักการ หากจะเลือกค่าให้ใช้ validation/CV โดยเก็บ test ไว้ประเมินท้ายสุด
#
# </details>
#
# ### ตัวอย่างแทนค่า: ใช้ Prior ทำนาย team team goal
#
# อ่านตามลำดับ: **เตรียมค่าจากชุดฝึก → คูณ Prior กับความน่าจะเป็นของคำ → หารด้วยผลรวมคะแนน**
#
# ![ขั้นที่ 1: ชุดฝึกมี sport 9 และ tech 9 ข้อความ จึงได้ Prior คลาสละ 9/18 = 0.5; vocabulary 49 คำ ใช้ alpha เท่ากับ 1; sport มีคำรวม 40 ครั้ง พบ team 4 และ goal 2 จึงได้โอกาส 5/89 และ 3/89; tech มีคำรวม 38 ครั้ง ไม่พบทั้งสองคำ จึงได้โอกาสคำละ 1/87](../../assets/figures/07-naive-bayes-text-prediction-inputs.png)
#
# [เปิดภาพที่มาของตัวเลข (SVG)](../../assets/figures/07-naive-bayes-text-prediction-inputs.svg)
#
# ![ขั้นที่ 2: team team goal มี team สองครั้งและ goal หนึ่งครั้ง คะแนน sport = 0.5 × (5/89) × (5/89) × (3/89) ประมาณ 0.00005319 ส่วน tech = 0.5 × (1/87) × (1/87) × (1/87) ประมาณ 0.0000007593; ขั้นที่ 3 หารแต่ละคะแนนด้วยผลรวมประมาณ 0.00005395 ได้ sport 0.986 และ tech 0.014](../../assets/figures/07-naive-bayes-text-prediction-scores.png)
#
# [เปิดภาพคำนวณคำทำนาย (SVG)](../../assets/figures/07-naive-bayes-text-prediction-scores.svg)
#
# **ผล:** `predict` เลือก **sport** ส่วน `predict_proba` ให้ sport ≈ **0.986** และ tech ≈ **0.014** แม้ Prior เริ่มเท่ากัน แต่คำที่พบสนับสนุน sport มากกว่า
#
# <details>
# <summary>อ่านวิธีแทนค่าทีละขั้นแบบข้อความ</summary>
#
# **1. หาความน่าจะเป็นของคำด้วย smoothing**
#
# จากข้อความใน train ของบทนี้ ฝั่ง sport นับคำรวมได้ 40 ครั้ง โดยพบ team 4 ครั้งและ goal 2 ครั้ง ส่วน tech นับคำรวมได้ 38 ครั้ง และไม่พบทั้ง team กับ goal ใช้ vocabulary 49 คำและ `alpha=1` จึงได้:
#
# | คลาส | ความน่าจะเป็นของ team ภายในคลาส | ความน่าจะเป็นของ goal ภายในคลาส |
# |---|---|---|
# | sport | (4 + 1) / (40 + 49) = 5/89 | (2 + 1) / (40 + 49) = 3/89 |
# | tech | (0 + 1) / (38 + 49) = 1/87 | (0 + 1) / (38 + 49) = 1/87 |
#
# **2. นำ Prior มาคูณหลักฐานจากคำ**
#
# ข้อความ `team team goal` มี team 2 ครั้งและ goal 1 ครั้ง จึงคูณความน่าจะเป็นของ team ซ้ำสองครั้ง โดยแต่ละคลาสใช้ค่าของตัวเอง:
#
# ```text
# คะแนน sport = 0.5 × (5/89) × (5/89) × (3/89)
#             ≈ 0.00005319
#
# คะแนน tech  = 0.5 × (1/87) × (1/87) × (1/87)
#             ≈ 0.0000007593
# ```
#
# คะแนน sport สูงกว่า จึงเลือกคำทำนาย **sport** ได้แล้ว ตัวเลขสองค่านี้เป็น **คะแนนสำหรับเปรียบเทียบคลาส** ซึ่งยังไม่ได้ปรับให้รวมเป็น 1
#
# **3. เปลี่ยนคะแนนเป็นความน่าจะเป็นของคลาส**
#
# นำคะแนนทั้งสองมาบวกกัน แล้วหารคะแนนของแต่ละคลาสด้วยผลรวมเดียวกัน:
#
# ```text
# ผลรวมคะแนน ≈ 0.00005319 + 0.0000007593 ≈ 0.00005395
#
# ความน่าจะเป็นของ sport ≈ 0.00005319   / 0.00005395 ≈ 0.986
# ความน่าจะเป็นของ tech  ≈ 0.0000007593 / 0.00005395 ≈ 0.014
# ```
#
# นี่คือผลของ `predict_proba` สำหรับข้อความนี้เมื่อแสดงทศนิยม 3 ตำแหน่ง แม้ prior เริ่มเท่ากันที่ 0.5 แต่หลักฐานจากคำต่างกัน จึงได้ความน่าจะเป็นสุดท้ายต่างกัน ส่วน `predict` คืนชื่อคลาส **sport**
#
# </details>

# %% Cell 5 — ดูจำนวนคำและทำนายข้อความใหม่
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
