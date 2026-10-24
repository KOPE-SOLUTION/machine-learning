# 07 · จำแนกข้อความด้วย Naive Bayes

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 45–60 นาที  
สิ่งที่จะทำได้: แปลงข้อความเป็น bag-of-words อธิบาย smoothing และข้อจำกัดของข้อความนอกเรื่องได้

## จากประโยคเป็นเวกเตอร์

CountVectorizer สร้างคำศัพท์จาก train แล้วนับจำนวนคำในแต่ละข้อความ ลำดับคำหายไปในรูปแบบ bag-of-words ดังนั้นประโยคที่ใช้คำเหมือนกันแต่สื่อความตรงข้ามอาจมีเวกเตอร์คล้ายกัน

Naive Bayes เปรียบเทียบ P(คลาส | คำ) โดยใช้ prior ของคลาสและความถี่คำ สมมติฐาน naive คือ feature เป็นอิสระเมื่อกำหนดคลาสแล้ว ซึ่งไม่ตรงกับภาษาธรรมชาติทั้งหมด แต่ยังสร้าง baseline ที่ฝึกเร็วได้ MultinomialNB เหมาะกับ feature แบบนับที่ไม่ติดลบ ส่วน GaussianNB เป็นอีกสมมติฐานสำหรับ feature ต่อเนื่อง

Smoothing ด้วย alpha เพิ่มน้ำหนักให้คำที่มีจำนวนน้อยหรือเป็นศูนย์ เพื่อไม่ให้คำเดียวทำให้คะแนนของทั้งคลาสเป็นศูนย์ เลือก alpha ด้วย validation หากมีข้อมูลพอ

## ไม่รู้จักคำ ไม่ได้แปลว่ารู้คำตอบ

ข้อความเกี่ยวกับอาหารในตัวอย่างอาจไม่มีคำใน vocabulary เลย โมเดลยังคืน sport หรือ tech เพราะมีเพียงสองทางเลือกและอาศัย prior ได้ จึงต้องดู known-token count และออกแบบกรณีนอกขอบเขตเพิ่มเติม

ข้อความทั้งหมดแต่งขึ้นและมีเพียง 24 รายการ ผลบนชุดทดสอบเล็กนี้แสดงวิธีใช้ API เท่านั้น ไม่ใช่ benchmark ของระบบจำแนกข้อความจริง สำหรับภาษาไทยต้องพิจารณาการตัดคำหรือ character n-grams เพิ่มเติม

## ลงมือรัน

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/07-naive-bayes-text/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` คัดลอกบล็อกต่อไปนี้ได้ทั้งชุด แต่ละส่วน `# %%` คือช่วงการทดลองเดียวกันกับ notebook

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
vectors = model.named_steps["countvectorizer"].transform(examples)
for text, label, token_count in zip(examples, model.predict(examples), vectors.sum(axis=1).A1):
    print(f"{text!r} -> {label}; known-token count={token_count}")
print("Class order:", model.classes_)
print("Probabilities (not guaranteed calibrated):\n", model.predict_proba(examples).round(3))
```
<!-- CODE:END -->

## อ่านผลที่ได้

รายงานสองคลาส พร้อมจำนวนคำที่อยู่ใน vocabulary ของข้อความใหม่ ข้อความอาหารมี known-token count เท่ากับศูนย์ ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## แบบฝึกหัด

ลอง CountVectorizer(analyzer='char', ngram_range=(2, 4)) แล้วออกแบบชุดข้อความภาษาไทยอย่างน้อยคลาสละ 20 รายการ โดยแยกข้อความใกล้ซ้ำให้ไม่ข้าม train/test

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

character n-grams ใช้ส่วนย่อยของตัวอักษรจึงไม่พึ่งช่องว่างแบ่งคำ แต่จำนวน feature และความไวต่อการสะกดเปลี่ยนไป ต้องประเมินกับข้อความคนละแหล่งและดูข้อผิดพลาดจริง ไม่สรุปว่าเหมาะกว่าจากตัวอย่างเล็ก

</details>

## อ่านต่อ

- [Handbook: Naive Bayes](https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html)
- [scikit-learn: Naive Bayes](https://scikit-learn.org/1.7/modules/naive_bayes.html)
- [Text feature extraction](https://scikit-learn.org/1.7/modules/feature_extraction.html#text-feature-extraction)

