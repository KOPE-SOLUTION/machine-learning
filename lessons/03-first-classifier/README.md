# 03 · โมเดลแรกด้วย scikit-learn และ KNN

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 40–60 นาที  
สิ่งที่จะทำได้: ใช้ fit/predict เข้าใจระยะทาง และสร้าง Pipeline ที่ scale จาก train เท่านั้นได้

## โมเดลจำแนกด้วยเพื่อนบ้านใกล้ที่สุด

K-nearest neighbors หรือ KNN มองตัวอย่างใกล้ข้อมูลใหม่ที่สุด k ตัวแล้วลงคะแนนคลาส หาก k เล็ก โมเดลไวต่อจุดรบกวน; หาก k ใหญ่ ขอบเขตอาจหยาบจนแยกกลุ่มเล็กไม่ได้ การเก็บข้อมูลฝึกมีต้นทุน และการทำนายอาจช้าขึ้นเมื่อข้อมูลมากขึ้น

ระยะทางแบบ Euclidean คือรากที่สองของผลรวมกำลังสองของความต่างแต่ละ feature คอลัมน์ที่มีสเกลใหญ่จึงอาจครอบงำระยะทาง `StandardScaler` แปลงค่าโดยใช้ z = (x − μ_train) / σ_train ไม่ได้ทำให้ข้อมูลทุกชนิดมีการแจกแจงปกติ

## API ที่ใช้ซ้ำได้

`fit(X_train, y_train)` เรียนจากข้อมูลฝึก; `predict(X_new)` ทำนายคำตอบ; `transform(X)` แปลง feature โดยใช้สิ่งที่เรียนไว้แล้ว ใน `Pipeline` การเรียก fit จะจัดลำดับการแปลงและฝึก ส่วน predict จะใช้ตัวแปลงเดิมก่อนทำนาย

บล็อกนี้กำหนด k=5 ไว้ล่วงหน้าเพื่อเรียน API หากต้องการเลือก k ให้ใช้ CV บน train ตามบท 10 การลองหลายค่าแล้วเลือกจากคะแนน test ทำให้ test ไม่ใช่ข้อมูลใหม่สำหรับกระบวนการตัดสินใจอีกต่อไป

ข้อมูลดอกใหม่ต้องมีจำนวนคอลัมน์ ลำดับ และหน่วยเหมือนตอนฝึก โมเดลไม่ได้รู้เองว่าคอลัมน์ที่หนึ่งถูกสลับกับคอลัมน์ที่สาม

## ลงมือรัน

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/03-first-classifier/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` คัดลอกบล็อกต่อไปนี้ได้ทั้งชุด แต่ละส่วน `# %%` คือช่วงการทดลองเดียวกันกับ notebook

<!-- CODE:START -->
```python
"""Pipeline แรก: scale -> K-nearest neighbors"""
# %% เตรียมข้อมูล
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

# %% scaler เรียนค่าเฉลี่ย/ส่วนเบี่ยงเบนจาก train เท่านั้น
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Baseline accuracy: {accuracy_score(y_test, baseline.predict(X_test)):.3f}")
print(f"KNN test accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, target_names=data.target_names))

# %% ข้อมูลใหม่: 4 คอลัมน์ในลำดับเดิม หน่วยเซนติเมตร
new_flower = [[5.1, 3.5, 1.4, 0.2]]
label = model.predict(new_flower)[0]
print("New flower:", data.target_names[label])
```
<!-- CODE:END -->

## อ่านผลที่ได้

ได้ accuracy ของ KNN พร้อม baseline และรายงาน precision/recall/F1 ของแต่ละสายพันธุ์ ตัวอย่างดอกใหม่ถูกทำนายเป็น setosa ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## แบบฝึกหัด

อธิบายว่าทำไม scaler.fit_transform(X) ก่อน train_test_split จึงผิด แล้วเขียนลำดับที่ถูกต้องด้วยคำพูด

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

ค่าเฉลี่ยและส่วนเบี่ยงเบนจะมีข้อมูลจาก test ปนเข้ามา ต้อง split ก่อน แล้ว fit Pipeline บน train เท่านั้น สำหรับ test และข้อมูลใหม่เรียก predict ซึ่งใช้ transform จากสถิติเดิม

</details>

## อ่านต่อ

- [Handbook: Introducing Scikit-Learn](https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html)
- [scikit-learn: Nearest Neighbors](https://scikit-learn.org/1.7/modules/neighbors.html)
- [Common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html)

