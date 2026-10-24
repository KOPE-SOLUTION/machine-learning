# 01 · Machine Learning คืออะไร และทำไมต้องมี baseline

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 30–45 นาที  
สิ่งที่จะทำได้: แยก classification, regression, clustering และกำหนดหน่วยข้อมูลกับวิธีประเมินได้

## ก่อนเริ่มบทนี้

ใช้พื้นฐานการอ่านแถว/คอลัมน์จากบท 00 เป้าหมายบทนี้คือเข้าใจวงจรฝึกและประเมิน ก่อนสนใจว่าอัลกอริทึมใดแม่นกว่า

## จากการวิเคราะห์ข้อมูลสู่การทำนาย

เมื่อใช้ notebook จัดรูปข้อมูล คำนวณด้วย NumPy และสื่อสารผลด้วยกราฟได้แล้ว คุณพร้อมนำทักษะเหล่านี้มาต่อยอดใน ML สิ่งที่เพิ่มใน ML คือการเรียนความสัมพันธ์จากตัวอย่างและตรวจว่าความสัมพันธ์นั้นใช้กับข้อมูลใหม่ได้หรือไม่ จึงต้องแยกขั้นสำรวจ การเรียนพารามิเตอร์ และการประเมินออกจากกัน

## จากกฎที่เขียนเองสู่กฎที่เรียนจากตัวอย่าง

สมมติว่าต้องแยกสายพันธุ์ดอกไม้จากขนาดกลีบ เราอาจเขียนเงื่อนไขหลายข้อเอง หรือให้โมเดลเรียนความสัมพันธ์จากดอกไม้ที่รู้สายพันธุ์แล้ว เมื่อพบดอกใหม่จึงใช้ความสัมพันธ์นั้นทำนาย การเรียนรู้จะมีประโยชน์เมื่อใช้กับตัวอย่างที่ไม่ได้เห็นระหว่างฝึกได้ด้วย

| รูปแบบ | คำตอบที่ต้องการ | ตัวอย่าง |
|---|---|---|
| Classification | หมวดหมู่ | สายพันธุ์ดอกไม้ |
| Regression | ตัวเลขต่อเนื่อง | ระยะเวลาจัดส่ง |
| Clustering | กลุ่มที่ค้นพบจากข้อมูล | กลุ่มพฤติกรรมการใช้งานโดยไม่มี label |

Supervised learning ใช้ทั้ง X และ y ส่วน unsupervised learning ค้นโครงสร้างจาก X ความหมายของกลุ่มที่ค้นพบต้องอาศัยการตีความเพิ่มเติม

## เขียนโจทย์ก่อนเลือกโมเดล

ตอบสี่ข้อ: หนึ่งแถวแทนอะไร, ทำนายอะไร, รู้ข้อมูลใดบ้าง ณ เวลาทำนาย, และความผิดพลาดแบบไหนสำคัญ สำหรับบทนี้หนึ่งแถวคือดอกไม้หนึ่งดอก input เป็นขนาด 4 ค่า output เป็นหนึ่งใน 3 สายพันธุ์ และเริ่มวัดด้วย accuracy

Baseline คือคำตอบอ้างอิงที่สร้างง่าย เช่นทายคลาสที่พบบ่อยที่สุด หากโมเดลซับซ้อนไม่ดีกว่า baseline ให้ตรวจข้อมูลและวิธีวัดก่อนเพิ่มความซับซ้อน Baseline ต้องเรียนจาก train เช่นเดียวกับโมเดลจริง

`random_state=42` ทำให้การสุ่มแบ่งข้อมูลทำซ้ำได้ ส่วน `stratify=y` ช่วยรักษาสัดส่วนคลาสในแต่ละชุด สิ่งนี้เหมาะกับตัวอย่างอิสระแบบ Iris; ข้อมูลเวลาและคนเดิมหลายแถวต้องแยกอีกวิธีในบท 17

## ฝึกกับตรวจคำตอบเป็นคนละขั้น

**Model** คือกฎหรือความสัมพันธ์ที่ใช้แปลงข้อมูลเป็นคำทำนาย ส่วน **algorithm** คือวิธีสร้างกฎนั้นจากตัวอย่าง ใน supervised learning เรามีคำตอบเก่าไว้ให้เรียน แต่ตอนใช้งานจริงจะส่งเฉพาะ feature ของรายการใหม่

การวัดผลมีคำตอบจริงเก็บไว้เช่นกัน แต่แยกไว้ตรวจภายหลัง ลองมอง train เป็นชุดฝึกหัดพร้อมเฉลย และ test เป็นข้อสอบที่ยังไม่ใช้ปรับวิธีทำ การจำเฉลย train ได้ไม่เท่ากับทำข้อสอบใหม่ได้ นี่คือเหตุผลที่ต้องสนใจ **generalization** หรือการใช้ความรู้กับข้อมูลใหม่

| ขั้นในโค้ด | สิ่งที่เกิดขึ้น |
|---|---|
| `load_iris(return_X_y=True)` | ได้ array ขนาด `(150, 4)` และคำตอบ `(150,)` ที่จับคู่แถวกันแล้ว |
| `train_test_split(X, y, ...)` | แบ่งทั้งคู่พร้อมกัน คืน `X_train, X_test, y_train, y_test` ตามลำดับ |
| `baseline.fit(X_train, y_train)` | เรียนว่าคลาสใดพบบ่อยที่สุดใน train |
| `baseline.predict(X_test)` | ทายคลาสนั้นทุกแถว โดยไม่ได้รับ `y_test` |
| `accuracy_score(y_test, pred)` | นับจำนวนที่ทายตรงแล้วหารด้วยจำนวนแถว test |

ตัวอย่างนี้ train มีคลาสละ 40 แถวเท่ากัน `DummyClassifier` เลือกคลาสหนึ่งตามลำดับเมื่อเสมอ และทายถูก 10 จาก 30 แถว จึงได้ accuracy `10/30 ≈ 0.333` โมเดลไม่ได้ใช้ค่าขนาดดอกไม้ในการตัดสิน นี่คือจุดเริ่มต้นที่ KNN ในบท 03 ต้องนำมาเปรียบเทียบ

## ลงมือรัน

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/01-ml-workflow/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` คัดลอกบล็อกต่อไปนี้ได้ทั้งชุด แต่ละส่วน `# %%` คือช่วงการทดลองเดียวกันกับ notebook

<!-- CODE:START -->
```python
"""ตั้งโจทย์ แยกข้อมูล และวัด baseline"""
# %% หนึ่งแถวคือดอกไม้หนึ่งดอก
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %% ทายคลาสที่พบมากที่สุดเสมอ
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
pred = baseline.predict(X_test)
print("Train rows:", len(X_train), "Test rows:", len(X_test))
print(f"Baseline test accuracy: {accuracy_score(y_test, pred):.3f}")
print("First five predictions:", pred[:5])
print("First five true labels:", y_test[:5])
print("Correct test predictions:", int((pred == y_test).sum()), "of", len(y_test))
```
<!-- CODE:END -->

## อ่านผลที่ได้

train 120 แถว test 30 แถว และ baseline accuracy ประมาณ 0.333 เพราะสามคลาสมีสัดส่วนเท่ากัน ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## แบบฝึกหัด

ตั้งโจทย์ ML จากงานหนึ่งอย่างที่สนใจ โดยระบุหน่วยข้อมูล X, y, เวลาทำนาย, baseline และ metric จากนั้นยกตัวอย่าง feature ที่จะเป็นข้อมูลอนาคตหนึ่งรายการ

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

เช่นทำนายเวลาส่งของเมื่อรับคำสั่งซื้อ: หนึ่งแถวเป็นหนึ่งคำสั่งซื้อ ใช้ระยะทางและบริการที่เลือก ทายเวลาจัดส่งจริง baseline เป็นค่ามัธยฐานจาก train วัด MAE ส่วนเวลาที่ส่งสำเร็จเป็นข้อมูลอนาคต ใช้เป็น feature ไม่ได้

</details>

## อ่านต่อ

- [Handbook: What Is Machine Learning?](https://jakevdp.github.io/PythonDataScienceHandbook/05.01-what-is-machine-learning.html)
- [scikit-learn: DummyClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.dummy.DummyClassifier.html)

