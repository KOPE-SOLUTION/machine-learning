# 09 · Decision Trees, Random Forests และ overfitting

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 45–65 นาที  
สิ่งที่จะทำได้: อ่าน train/CV gap อธิบายการรวมต้นไม้ และแยก feature importance ออกจากเหตุและผลได้

## ก่อนเริ่มบทนี้

ใช้ CV ที่อธิบายในบท 08 และการอ่านคะแนนบท 04 เป้าหมายคือสังเกตความต่างระหว่างการจำข้อมูลฝึกกับการใช้กฎกับข้อมูลที่ไม่ได้ใช้ fit

## ต้นไม้เรียนคำถามทีละข้อ

Decision tree แบ่งข้อมูลด้วยเงื่อนไข เช่น feature_3 ≤ ค่าหนึ่ง แล้วทำซ้ำในแต่ละกิ่ง ใบของต้นไม้เก็บคำตอบที่ใช้ทำนาย การแบ่งลึกช่วยจำรายละเอียดของ train แต่ยังจำ noise ได้ด้วย ต้นไม้เดี่ยวจึงอาจเปลี่ยนมากเมื่อข้อมูลเปลี่ยนเล็กน้อย

Random Forest ฝึกต้นไม้หลายต้นด้วยการสุ่มตัวอย่างและสุ่ม feature บางส่วน แล้วรวมคำตอบ การลดความสัมพันธ์ระหว่างต้นไม้ช่วยลดความแปรปรวนของผลรวม แต่ยัง overfit ได้และไม่แก้ปัญหา leakage ให้เอง

## อ่านช่องว่างของคะแนน

โค้ดใช้ชุด CV เดียวกันเปรียบเทียบต้นไม้ตื้น ต้นไม้ไม่จำกัดความลึก และป่าไม้ หาก train สูงมากแต่ CV ต่ำกว่าอย่างชัดเจน แปลว่าโมเดลอาจจำรายละเอียดที่ไม่ช่วยกับข้อมูลใหม่ พิจารณา max_depth หรือ min_samples_leaf ก่อนเพิ่มจำนวนต้นไม้

Permutation importance สลับค่าของ feature ทีละคอลัมน์แล้ววัดว่าคะแนนตกเท่าไร ภายใต้โมเดลและชุดข้อมูลนั้น ถ้า feature สัมพันธ์กัน โมเดลอาจใช้แทนกันจนแต่ละคอลัมน์ดูไม่สำคัญ ค่านี้จึงไม่ใช่ผลกระทบเชิงสาเหตุ

บทนี้คำนวณ importance บน test เพื่อรายงานโมเดลสุดท้ายเท่านั้น หากนำผลนี้ไปเลือก feature หรือปรับโมเดล ต้องใช้ validation หรือเตรียม final test ใหม่ที่ยังไม่แตะ

## คิดตามเส้นทางของต้นไม้

ตัวอย่างคำถาม “ระยะทาง ≤ 10 หรือไม่” แบ่งแถวออกเป็นสองกิ่ง แต่ละกิ่งถามต่อจนถึง **leaf** หรือใบ ต้นไม้เรียนทั้งคอลัมน์และจุดแบ่งจาก train แล้วใช้สัดส่วนคลาสในใบตัดสินคำตอบ **Depth** คือความลึกของเส้นทาง ส่วน `min_samples_leaf` กำหนดจำนวนตัวอย่างขั้นต่ำที่ยอมให้เหลือในใบ

ใน Random Forest ต้นไม้แต่ละต้นเห็นตัวอย่างที่สุ่มแบบคืนกลับและลอง feature บางส่วนในแต่ละจุดแบ่ง รุ่น classifier ของ scikit-learn เฉลี่ยความน่าจะเป็นจากต้นไม้แล้วเลือกคลาสที่สูงสุด ตัวแปร `n_estimators=120` หมายถึงจำนวนต้นไม้ ไม่ใช่จำนวนข้อมูล

## อ่านตาราง train/CV ให้ถูก

`cross_validate(..., return_train_score=True)` รายงานสองคะแนนต่อ fold: `train_score` ของส่วนที่ใช้ fit และ `test_score` ของส่วนที่กันไว้ใน **CV รอบนั้น** ชื่อ key นี้ไม่ได้หมายถึง final `X_test` ที่เรากันไว้ก่อนเริ่ม

**Overfitting** คือเรียนรายละเอียด train ที่ไม่ช่วยกับข้อมูลใหม่ เช่น train สูงแต่ CV ต่ำกว่า ส่วน **underfitting** คือยังจับรูปแบบสำคัญไม่พอ มักเห็นทั้ง train/CV ต่ำ ให้ดู baseline และโจทย์ร่วมด้วย ไม่ตัดสินจากช่องว่างอย่างเดียว ต้นไม้ในตัวอย่างไม่จำเป็นต้อง scale เพราะการแบ่งใช้เกณฑ์ของทีละคอลัมน์

Permutation importance ที่เท่ากับ 0.05 หมายถึง accuracy ลดลงเฉลี่ย 0.05 หรือ 5 จุดเปอร์เซ็นต์เมื่อสลับคอลัมน์นั้น ค่าใกล้ศูนย์ไม่ได้พิสูจน์ว่า feature ไม่มีประโยชน์ในทุกโมเดล และค่าติดลบเล็กน้อยอาจเกิดจากความแปรปรวนของชุดประเมิน

## ลงมือรัน

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/09-trees-and-forests/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` บล็อกนี้รันได้ทั้งชุดโดยไม่ต้องรันบทก่อนหน้า

<!-- CODE:START -->
```python
"""ดู overfitting ของต้นไม้และการเฉลี่ยผลด้วย Random Forest"""
# %% เปรียบเทียบทุกตัวบน CV ชุดเดียวกัน
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(
    n_samples=800, n_features=10, n_informative=4, n_redundant=2,
    flip_y=0.08, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
models = {
    "shallow_tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "deep_tree": DecisionTreeClassifier(random_state=42),
    "forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=1),
}
scores = {}
for name, model in models.items():
    result = cross_validate(model, X_train, y_train, cv=cv, scoring="accuracy", return_train_score=True)
    scores[name] = result["test_score"].mean()
    print(f"{name}: train={result['train_score'].mean():.3f}, CV={scores[name]:.3f}")

# %% อธิบายโมเดลที่เลือกแล้ว ผลบน test ใช้รายงาน ไม่ใช้เลือก feature รอบนี้
chosen = max(scores, key=scores.get)
model = models[chosen].fit(X_train, y_train)
print("Selected:", chosen, "test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))
importance = permutation_importance(model, X_test, y_test, scoring="accuracy", n_repeats=8, random_state=42)
order = np.argsort(importance.importances_mean)[::-1][:5]
for i in order:
    print(f"feature_{i}: accuracy drop={importance.importances_mean[i]:.3f} +/- {importance.importances_std[i]:.3f}")
```
<!-- CODE:END -->

## อ่านผลที่ได้

ได้คะแนน train/CV ของสามโมเดล ตามด้วย test accuracy ของตัวที่เลือกและ accuracy drop ของ feature สำคัญ ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## แบบฝึกหัด

เปลี่ยน min_samples_leaf เป็น 1, 5, 20 แล้ววางแผนเทียบด้วย CV อธิบายว่าทำไมเพิ่ม n_estimators อย่างเดียวอาจไม่แก้คะแนน validation ต่ำ

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

min_samples_leaf ใหญ่ขึ้นจำกัดการสร้างใบจากข้อมูลน้อยและอาจลด overfitting จำนวนต้นไม้มากขึ้นทำให้ผลรวมเสถียรขึ้น แต่ไม่ชดเชย feature ที่ไม่มีข้อมูล โจทย์ผิด หรือข้อมูลรั่ว

</details>

## อ่านต่อ

- [Handbook: Decision Trees and Random Forests](https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html)
- [scikit-learn: Ensembles](https://scikit-learn.org/1.7/modules/ensemble.html)
- [Permutation importance](https://scikit-learn.org/1.7/modules/permutation_importance.html)

