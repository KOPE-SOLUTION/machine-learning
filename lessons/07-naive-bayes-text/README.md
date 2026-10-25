# 07 · จำแนกข้อความด้วย Naive Bayes

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เปลี่ยนข้อความเป็นจำนวนคำ แล้วจำแนก **sport / tech** ด้วย Multinomial Naive Bayes

**ใช้สิ่งที่เรียนแล้ว:** Pipeline, train/test และ classification report จากบท 03–06

## ใจความใหม่

**Vocabulary:** รายการคำที่ `CountVectorizer` เรียนจาก train

**Bag-of-words:** แทนข้อความด้วยจำนวนครั้งที่พบแต่ละคำ โดยไม่เก็บลำดับคำ

ตัวอย่าง: vocabulary `goal, team, wins` ทำให้ “team team goal” เป็น `[1, 2, 0]`

### Naive Bayes คืออะไร

**Naive Bayes** เป็นกลุ่มอัลกอริทึม **Supervised Learning สำหรับ Classification** ใช้กฎของ Bayes รวมข้อมูลเกี่ยวกับคลาสกับหลักฐานจาก feature เพื่อคำนวณความน่าจะเป็นของคลาสหลังเห็นข้อมูล

คำว่า **naive** หมายถึงสมมติฐานว่า feature เป็นอิสระต่อกันเมื่อทราบคลาส ซึ่งไม่ตรงกับภาษาจริงทั้งหมด ส่วน **Multinomial Naive Bayes** เป็นรุ่นที่เหมาะกับข้อมูลนับ เช่น จำนวนครั้งที่พบคำในบทนี้

**หลักการทำงานกับข้อความ:**

1. เรียนสัดส่วนคลาสใน train เรียกว่า **prior** และความถี่ของแต่ละคำภายในคลาส
2. แปลงข้อความที่ต้องการทำนายเป็นจำนวนคำด้วย vocabulary เดิม
3. รวม prior กับหลักฐานจากคำ แล้วเลือกคลาสที่ได้ความน่าจะเป็นสูงสุดตามโมเดล

**ตัวอย่าง:** ถ้าคำว่า `goal` พบในข้อความ sport บ่อยกว่า tech คำนี้จะเป็นหลักฐานสนับสนุน sport แต่คำตอบยังขึ้นกับคำอื่นและ prior ด้วย

- **จุดเด่น:** ฝึกและทำนายได้เร็ว เหมาะเป็นโมเดลเริ่มต้นสำหรับข้อความที่มี feature คำจำนวนมาก
- **ข้อจำกัด:** วิธีนับคำนี้ไม่เก็บลำดับหรือบริบท และ probability จากโมเดลอาจไม่ตรงกับความถี่ที่เกิดขึ้นจริง แม้จำแนกประเภทได้ดี

### สิ่งที่ต้องรู้เมื่อใช้ MultinomialNB

**MultinomialNB:** ใช้ feature แบบนับที่ไม่ติดลบ โดยมีรายละเอียดสำคัญดังนี้:

- **Smoothing:** ค่า `alpha` ช่วยให้คำใน vocabulary ที่ไม่เคยพบในคลาสหนึ่งยังมีน้ำหนัก จึงไม่ทำให้คะแนนคลาสนั้นเป็นศูนย์ทันที
- **คำที่ไม่รู้จัก:** คำที่ไม่มีใน vocabulary เลยจะถูกข้าม
- **Known-token count:** จำนวนคำที่รู้จักในข้อความ โดยนับคำซ้ำด้วย

ตัวนับคำคืน sparse matrix เพื่อประหยัดพื้นที่ โค้ดใช้ `toarray()` แสดงเฉพาะตัวอย่างเล็ก ๆ

ข้อมูลแต่งขึ้น 24 ประโยค แบ่งเป็น train 18 และ test 6 ประโยค ใช้เรียนวิธีทำ ไม่ใช่ผลประเมินระบบจริง

สำหรับภาษาไทย ต้องพิจารณาการตัดคำหรือ character n-grams เพิ่ม

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรัน **Cell 1–2 ตามลำดับ** หรือเปิด [Notebook ของบทนี้](lesson.ipynb) ใช้ kernel เดิมได้เลย อ่านคำอธิบายใต้โค้ดแต่ละช่วงประกอบผลที่ได้

<!-- CODE:CELLS -->

<!-- CODE:START -->
### Cell 1 — เตรียมข้อความและคำตอบ

```python
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
```

**อ่านโค้ด**

- `texts` เก็บข้อความดิบหนึ่งข้อความต่อสมาชิก โดย 12 ข้อความแรกเป็นกีฬา และ 12 ข้อความหลังเป็นเทคโนโลยี
- `np.array(["sport"] * 12 + ["tech"] * 12)` สร้าง label ให้เรียงตรงกับข้อความ จึงต้องรักษาลำดับให้สัมพันธ์กัน
- `train_test_split(..., test_size=0.25, stratify=y)` แบ่งข้อความพร้อมคำตอบเป็น train 18 และ test 6 โดยรักษาสัดส่วนคลาส
- ยังไม่ต้องแปลงข้อความเป็นตัวเลขเอง เพราะจะให้ Pipeline ทำใน Cell ถัดไป

### Cell 2 — ฝึกตัวนับคำและตรวจคำทำนาย

```python
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
```

**อ่านโค้ด**

- `make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))` ต่อการนับคำกับโมเดล โดย `fit` เรียน vocabulary และโมเดลจาก train เท่านั้น
- `model.named_steps["countvectorizer"]` หยิบตัวนับคำที่ fit แล้วออกมาตรวจ ส่วน `transform(examples)` นับคำในข้อความใหม่โดยใช้ vocabulary เดิม
- `vectors.sum(axis=0)` รวมจำนวนคำตามคอลัมน์ แล้ว `np.asarray(...).ravel() > 0` ทำ mask เลือกเฉพาะคำที่พบในตัวอย่าง
- `vectors[:, used_columns].toarray()` เลือกคอลัมน์เหล่านั้นและแปลง sparse matrix เป็น array เพื่อแสดงตารางเล็ก ๆ
- `vectors.sum(axis=1).A1` รวมจำนวนคำที่รู้จักต่อข้อความ แล้วจัดผลเป็น array หนึ่งมิติ เพื่อใช้ `zip` จับคู่กับข้อความและคำทำนาย
- `model.classes_` บอกลำดับคอลัมน์ของ `predict_proba` ส่วนข้อความอาหารที่ไม่มีคำรู้จักยังได้คำทำนาย จึงต้องอ่าน known-token count ประกอบด้วย
<!-- CODE:END -->

## อ่านผล

ดูรายการคำที่ใช้เป็น feature และจำนวนครั้งที่พบแต่ละคำ

**ข้อความอาหาร:** มี known-token count=0 แต่โมเดลยังทาย sport หรือ tech เพราะมีเพียงสองทางเลือกและยังใช้ prior ได้

โมเดลจึงไม่ได้ตรวจข้อความนอกเรื่องให้เอง และความน่าจะเป็นที่รายงานสูงก็ยังทายผิดได้

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** ใช้ `CountVectorizer(analyzer="char", ngram_range=(2, 4))` กับข้อความภาษาไทยที่เตรียมเพิ่ม

**แนวคำตอบ:** character n-grams ใช้ส่วนย่อยของตัวอักษร จึงไม่พึ่งช่องว่างแบ่งคำ

ต้องวัดผลกับข้อความที่แยกจาก train และกันข้อความใกล้ซ้ำไม่ให้ข้ามชุด

</details>

## อ้างอิง

- [Handbook: Naive Bayes](https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html)
- [scikit-learn: Naive Bayes](https://scikit-learn.org/1.7/modules/naive_bayes.html)
- [Text feature extraction](https://scikit-learn.org/1.7/modules/feature_extraction.html#text-feature-extraction)
