# 09 · Decision Trees, Random Forests และ overfitting

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เทียบ **Decision Tree กับ Random Forest** และดูว่าโมเดลจำข้อมูลฝึกมากเกินไปหรือไม่

**ใช้สิ่งที่เรียนแล้ว:** metric และ CV จากบท 04/08 ไม่ทวนขั้นตอนแบ่ง fold

## ใจความใหม่

**Decision Tree:** เรียนเงื่อนไข เช่น `feature_3 ≤ ค่าหนึ่ง` แล้วแบ่งต่อเป็นกิ่งจนถึง **leaf** หรือใบที่ใช้ทำนาย

- `max_depth` จำกัดความลึกของต้นไม้
- `min_samples_leaf` กำหนดจำนวนตัวอย่างขั้นต่ำในแต่ละใบ

การแบ่งตามคอลัมน์แบบนี้ไม่จำเป็นต้อง scale

**Random Forest:** ฝึกต้นไม้หลายต้น โดยสุ่มแถวแบบคืนกลับและสุ่ม feature ที่แต่ละจุดแบ่ง

รุ่น classifier ของ scikit-learn เฉลี่ย probability จากต้นไม้แล้วเลือกคลาส ค่า `n_estimators=120` หมายถึงใช้ต้นไม้ 120 ต้น

การรวมต้นไม้ช่วยลดความแปรปรวน แต่ยังเกิด overfitting ได้

| คะแนนที่เห็น | สิ่งที่อาจเกิดขึ้น |
|---|---|
| train สูง แต่ CV ต่ำกว่าอย่างชัดเจน | **Overfitting:** จำรายละเอียดที่ไม่ช่วยกับข้อมูลใหม่ |
| train และ CV ต่ำทั้งคู่ | **Underfitting:** จับรูปแบบได้ไม่พอ หรือ feature ให้ข้อมูลไม่พอ |

ใช้โจทย์และ baseline ประกอบการตีความคะแนนด้วย

**`test_score` ใน `cross_validate`:** หมายถึงคะแนนของ validation fold ไม่ใช่คะแนน final test

**Permutation importance:** สลับค่าของ feature ทีละคอลัมน์ แล้วดูว่าคะแนนลดลงเท่าไร

ตัวอย่าง: accuracy drop=0.05 หมายถึงลดลง 5 จุดเปอร์เซ็นต์ ค่านี้ไม่ใช่ผลเชิงสาเหตุ และ feature ที่ใช้แทนกันได้อาจดูสำคัญน้อย

บทนี้ใช้ importance บน test เพื่อรายงานเท่านั้น หากจะใช้เลือก feature ต้องใช้ validation และกัน final test สำหรับกระบวนการใหม่

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

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

## อ่านผล

เทียบคะแนน train และ CV ของต้นไม้ตื้น ต้นไม้ลึก และ Random Forest จากนั้นอ่าน test accuracy กับ importance ของโมเดลที่เลือก

**Importance ใกล้ศูนย์:** ยังสรุปไม่ได้ว่า feature นั้นไร้ประโยชน์ในทุกโมเดล

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** วางแผนเทียบ `min_samples_leaf=1, 5, 20` ด้วย CV

**แนวคำตอบ:** การเพิ่มจำนวนตัวอย่างขั้นต่ำในใบอาจลด overfitting แต่ถ้ามากเกินไป โมเดลอาจจับรูปแบบได้ไม่พอ

การเพิ่มจำนวนต้นไม้อย่างเดียวไม่แก้ปัญหา feature ที่ไม่มีข้อมูลหรือ leakage

</details>

## อ้างอิง

- [Handbook: Decision Trees and Random Forests](https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html)
- [scikit-learn: Ensembles](https://scikit-learn.org/1.7/modules/ensemble.html)
- [Permutation importance](https://scikit-learn.org/1.7/modules/permutation_importance.html)
