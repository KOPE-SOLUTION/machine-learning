# 07 · จำแนกข้อความด้วย Naive Bayes

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 45–60 นาที  
สิ่งที่จะทำได้: แปลงข้อความเป็น bag-of-words อธิบาย smoothing และข้อจำกัดของข้อความนอกเรื่องได้

## ก่อนเริ่มบทนี้

ต่อจาก Pipeline และ one-hot ในบท 06 บทนี้เปลี่ยนจากหนึ่งหมวดหมู่ต่อแถวเป็นคำหลายคำในหนึ่งข้อความ โดยยังฝึกและทำนายด้วย API เดิม

## จากประโยคเป็นเวกเตอร์

CountVectorizer สร้างคำศัพท์จาก train แล้วนับจำนวนคำในแต่ละข้อความ ลำดับคำหายไปในรูปแบบ bag-of-words ดังนั้นประโยคที่ใช้คำเหมือนกันแต่สื่อความตรงข้ามอาจมีเวกเตอร์คล้ายกัน

Naive Bayes เปรียบเทียบ P(คลาส | คำ) โดยใช้ prior ของคลาสและความถี่คำ สมมติฐาน naive คือ feature เป็นอิสระเมื่อกำหนดคลาสแล้ว ซึ่งไม่ตรงกับภาษาธรรมชาติทั้งหมด แต่ยังสร้าง baseline ที่ฝึกเร็วได้ MultinomialNB เหมาะกับ feature แบบนับที่ไม่ติดลบ ส่วน GaussianNB เป็นอีกสมมติฐานสำหรับ feature ต่อเนื่อง

Smoothing ด้วย alpha เพิ่มน้ำหนักให้คำที่มีจำนวนน้อยหรือเป็นศูนย์ เพื่อไม่ให้คำเดียวทำให้คะแนนของทั้งคลาสเป็นศูนย์ เลือก alpha ด้วย validation หากมีข้อมูลพอ

## ไม่รู้จักคำ ไม่ได้แปลว่ารู้คำตอบ

ข้อความเกี่ยวกับอาหารในตัวอย่างอาจไม่มีคำใน vocabulary เลย โมเดลยังคืน sport หรือ tech เพราะมีเพียงสองทางเลือกและอาศัย prior ได้ จึงต้องดู known-token count และออกแบบกรณีนอกขอบเขตเพิ่มเติม

ข้อความทั้งหมดแต่งขึ้นและมีเพียง 24 รายการ ผลบนชุดทดสอบเล็กนี้แสดงวิธีใช้ API เท่านั้น ไม่ใช่ benchmark ของระบบจำแนกข้อความจริง สำหรับภาษาไทยต้องพิจารณาการตัดคำหรือ character n-grams เพิ่มเติม

## ตัวอย่าง bag-of-words ขนาดเล็ก

ถ้าคำศัพท์มี `goal, team, wins` ข้อความ “team wins” จะเป็น `[0, 1, 1]` ส่วน “team team goal” จะเป็น `[1, 2, 0]` คอลัมน์แทนคำและค่าคือจำนวนครั้งที่พบ ดังนั้น “wins team” จะเหมือน “team wins” ในตัวแทนแบบนี้

**Vocabulary** คือรายการคำที่เรียนจาก train; **prior** คือสัดส่วนคลาสก่อนอ่านข้อความ; **likelihood** บอกความสอดคล้องของคำกับคลาส; **posterior** คือคะแนนความน่าจะเป็นของคลาสหลังเห็นคำ สัญลักษณ์ `P(คลาส | คำ)` อ่านว่าโอกาสเป็นคลาสนั้นเมื่อทราบคำ ไม่ใช่โอกาสพบคำเมื่อทราบคลาส

โมเดลรวมหลักฐานจากความถี่คำในแต่ละคลาส คำที่ไม่เคยพบในคลาสหนึ่งยังมีน้ำหนักเล็กน้อยจาก smoothing ทำให้คะแนนไม่เป็นศูนย์ทันที ส่วนคำที่ไม่เคยอยู่ใน vocabulary ของ train เลยจะไม่สร้างคอลัมน์ใหม่ตอน predict

## ตามรอยโค้ด

ข้อมูล 24 ประโยคแบ่งเป็น train 18 และ test 6 รายการ `CountVectorizer` คืน **sparse matrix** ซึ่งเก็บเฉพาะค่าที่ไม่เป็นศูนย์เพื่อลดพื้นที่ โค้ดแสดงตัวอย่างเล็กด้วย `toarray()` ให้เห็นค่าจริง แต่ไม่ควรแปลง sparse matrix ขนาดใหญ่ทั้งหมดแบบนี้

`named_steps` ใช้หยิบตัวนับคำที่ fit แล้วออกมาตรวจ ไม่ได้สร้าง vocabulary ใหม่ ผลรวมแต่ละแถวคือ known-token count ซึ่งนับจำนวนคำที่รู้จักรวมคำซ้ำ ไม่ใช่จำนวนคำไม่ซ้ำ Probability ที่สูงก็อาจผิดได้เมื่อข้อความอยู่นอกขอบเขต เช่นเรื่องอาหาร

## คัดลอกโค้ดไปรัน

คัดลอกบล็อก `python` ด้านล่างทั้งชุดไป Code cell ใน Jupyter หรือ Colab แล้วกด Shift+Enter หากเปิด `lesson.ipynb` ของบทนี้อยู่แล้ว ให้รัน cell ที่มีตามลำดับ ไม่ต้องคัดลอกซ้ำ ดู [วิธีเริ่มเรียน](../../docs/start-here.md) หากยังไม่ได้เตรียมแพ็กเกจ

<details>
<summary>ทางเลือก: รันไฟล์ Python จาก repo ในเครื่อง</summary>

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/07-naive-bayes-text/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` คัดลอกบล็อกต่อไปนี้ได้ทั้งชุด แต่ละส่วน `# %%` คือช่วงการทดลองเดียวกันกับ notebook

</details>

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

## อ่านผลที่ได้

รายงานสองคลาส พร้อมจำนวนคำที่อยู่ใน vocabulary ของข้อความใหม่ ข้อความอาหารมี known-token count เท่ากับศูนย์ ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## แบบฝึกหัด

เริ่มจากเทียบ known-token count ของข้อความสามรายการกับ vocabulary ที่พิมพ์ออกมา อธิบายว่าทำไมข้อความอาหารยังได้ label ทั้งที่ count=0

แบบฝึกหัดต่อยอด: ลอง CountVectorizer(analyzer='char', ngram_range=(2, 4)) แล้วออกแบบชุดข้อความภาษาไทยอย่างน้อยคลาสละ 20 รายการ โดยแยกข้อความใกล้ซ้ำให้ไม่ข้าม train/test

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

เมื่อไม่มีคำที่รู้จัก โมเดลยังมี prior ของสองคลาสให้เลือก จึงไม่ได้ตรวจพบข้อความนอกเรื่องโดยอัตโนมัติ

character n-grams ใช้ส่วนย่อยของตัวอักษรจึงไม่พึ่งช่องว่างแบ่งคำ แต่จำนวน feature และความไวต่อการสะกดเปลี่ยนไป ต้องประเมินกับข้อความคนละแหล่งและดูข้อผิดพลาดจริง ไม่สรุปว่าเหมาะกว่าจากตัวอย่างเล็ก

</details>

## อ่านต่อ

- [Handbook: Naive Bayes](https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html)
- [scikit-learn: Naive Bayes](https://scikit-learn.org/1.7/modules/naive_bayes.html)
- [Text feature extraction](https://scikit-learn.org/1.7/modules/feature_extraction.html#text-feature-extraction)

