# 07 · จำแนกข้อความด้วย Naive Bayes

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เปลี่ยนข้อความเป็นจำนวนคำ แล้วจำแนก **sport / tech** ด้วย Multinomial Naive Bayes

**ใช้สิ่งที่เรียนแล้ว:** Pipeline, train/test และ classification report จากบท 03–06

## ใจความใหม่

`CountVectorizer` เรียน **vocabulary** (รายการคำ) จาก train แล้วนับคำในแต่ละข้อความ เรียกว่า bag-of-words เช่น vocabulary `goal, team, wins` ทำให้ “team team goal” เป็น `[1, 2, 0]` วิธีนี้ไม่เก็บลำดับคำ

**Naive Bayes** ใช้สัดส่วนคลาสเดิม (prior) ร่วมกับความถี่คำในแต่ละคลาสเพื่อประมาณโอกาสเป็นคลาสหลังเห็นข้อความ สมมติฐาน naive ถือว่า feature เป็นอิสระเมื่อทราบคลาส ซึ่งไม่ตรงกับภาษาจริงทั้งหมด

`MultinomialNB` ใช้ feature แบบนับที่ไม่ติดลบ ส่วน `alpha` ทำ **smoothing** ให้คำใน vocabulary ที่ไม่เคยพบในคลาสหนึ่งยังมีน้ำหนัก ไม่ทำให้คะแนนคลาสนั้นเป็นศูนย์ทันที คำที่ไม่มีใน vocabulary เลยจะถูกข้าม

ตัวนับคำคืน sparse matrix เพื่อประหยัดพื้นที่ โค้ดแสดงด้วย `toarray()` เฉพาะตัวอย่างเล็ก ๆ และใช้ **known-token count** นับคำที่รู้จักรวมคำซ้ำ

ข้อมูลแต่งขึ้น 24 ประโยค (train 18 / test 6) ใช้เรียนวิธีทำ ไม่ใช่ผลประเมินระบบจริง ภาษาไทยต้องพิจารณาการตัดคำหรือ character n-grams เพิ่ม

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

<!-- CODE:START -->
```python
"""จำแนกข้อความสั้นด้วยตัวนับคำและ Multinomial Naive Bayes"""
# %% ข้อความแต่งขึ้นเพื่อฝึก API ใช้ภาษาอังกฤษเพื่อลดประเด็นการตัดคำ
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

# %% คำศัพท์ถูกสร้างจาก train เท่านั้น
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
<!-- CODE:END -->

## อ่านผล

ดูคำที่ใช้เป็น feature พร้อมจำนวนครั้งที่พบ ข้อความอาหารมี **known-token count=0** แต่ยังได้ sport หรือ tech เพราะโมเดลมีเพียงสองทางเลือกและยังอาศัย prior ได้ จึงไม่ได้ตรวจข้อความนอกเรื่องให้เอง และ probability สูงก็ผิดได้

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** ใช้ `CountVectorizer(analyzer="char", ngram_range=(2, 4))` กับข้อความภาษาไทยที่เตรียมเพิ่ม

**แนวคำตอบ:** วิธีนี้ใช้ส่วนย่อยของตัวอักษรจึงไม่พึ่งช่องว่างแบ่งคำ แต่ต้องวัดผลกับข้อความที่แยกจาก train และกันข้อความใกล้ซ้ำไม่ให้ข้ามชุด

</details>

## อ้างอิง

- [Handbook: Naive Bayes](https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html)
- [scikit-learn: Naive Bayes](https://scikit-learn.org/1.7/modules/naive_bayes.html)
- [Text feature extraction](https://scikit-learn.org/1.7/modules/feature_extraction.html#text-feature-extraction)
