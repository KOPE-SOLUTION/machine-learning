# 14 · Gaussian Mixtures และ Kernel Density Estimation

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้ประมาณ **ความหนาแน่นของข้อมูล** ด้วย Gaussian Mixture Model (GMM) และ Kernel Density Estimation (KDE)

**ใช้สิ่งที่เรียนแล้ว:** clustering จากบท 13, CV จากบท 08–10 และ histogram แบบ density จาก Matplotlib

## ใจความใหม่

| วิธี | สร้างความหนาแน่นอย่างไร | สิ่งที่เลือกในบท |
|---|---|---|
| GMM | ผสม Gaussian รูประฆังหลาย component โดยเรียนจุดกลาง ความกว้าง และน้ำหนัก | จำนวน component ที่มี BIC ต่ำสุดบน train |
| KDE | วาง Gaussian kernel รอบแต่ละตัวอย่างแล้วรวมเป็นเส้น | bandwidth ที่ได้คะแนน CV ดีที่สุดบน train |

**Responsibility:** ความน่าจะเป็นที่จุดหนึ่งเป็นสมาชิกของแต่ละ component ใน GMM

ตัวอย่าง: `[0.2, 0.8]` หมายถึงโอกาสเป็นสมาชิก 20% และ 80% ตามลำดับ การให้สัดส่วนแบบนี้เรียกว่า **soft assignment**

GMM จึงต่างจาก K-means ที่เลือกกลุ่มเดียว และ component ไม่จำเป็นต้องตรงกับกลุ่มจริง

- **BIC:** ชั่งความพอดีของโมเดลกับความซับซ้อน ค่าต่ำกว่าดีกว่าเมื่อเทียบบนข้อมูลเดียวกัน
- **Bandwidth:** คุมความเรียบของ KDE ค่าน้อยตามรายละเอียดมาก ส่วนค่ามากอาจเรียบจนกลบหลายยอด

## Density และ probability

**Density:** ความสูงของกราฟความหนาแน่น ซึ่งอาจมากกว่า 1 ได้ ต้องหาพื้นที่ใต้กราฟในช่วงที่สนใจจึงจะได้ probability ของช่วงนั้น

ตัวอย่างการกระจายสม่ำเสมอในช่วง 0–0.5:

- กราฟสูง 2 และมีพื้นที่รวม `0.5 × 2 = 1`
- โอกาสอยู่ในช่วง 0–0.1 เท่ากับ `0.1 × 2 = 0.2` หรือ 20%

จึงอ่านความสูงของกราฟเป็นเปอร์เซ็นต์ตรง ๆ ไม่ได้

**`score_samples`:** คืน log density ต่อแถว ใช้ `np.exp` แปลงกลับเป็น density เพื่อวาดกราฟ

ค่าเฉลี่ย test log density ยิ่งมากยิ่งดีเมื่อเทียบบนข้อมูล หน่วย และมิติเดียวกัน ไม่ควรเทียบข้ามสเกลโดยตรง

KDE ใช้ KFold เพราะไม่มี label ให้ stratify คะแนนค้นหาเป็นผลรวม log density ของแต่ละ validation fold ซึ่งมีจำนวนแถวเท่ากันในบทนี้

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

<!-- CODE:START -->
```python
"""ประมาณความหนาแน่นด้วย Gaussian mixtures และ KDE"""
# %% ข้อมูลหนึ่งมิติจากสองกลุ่มจำลอง
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KernelDensity

rng = np.random.default_rng(42)
X = np.concatenate([rng.normal(-2, 0.6, 250), rng.normal(2, 1.0, 350)]).reshape(-1, 1)
X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)

# %% เลือกจำนวน component ด้วย BIC บน train (ยิ่งต่ำยิ่งดี)
mixtures = [GaussianMixture(n_components=k, n_init=5, random_state=42).fit(X_train) for k in range(1, 5)]
gmm = min(mixtures, key=lambda model: model.bic(X_train))
for model in mixtures:
    print(f"GMM k={model.n_components}: train BIC={model.bic(X_train):.1f}")
print("Responsibilities for x=0:", gmm.predict_proba([[0.0]]).round(3))

# %% เลือก bandwidth ด้วย CV บน train
kde_search = GridSearchCV(
    KernelDensity(kernel="gaussian"), {"bandwidth": [0.15, 0.3, 0.6, 1.0]},
    cv=KFold(5, shuffle=True, random_state=42),
)
kde_search.fit(X_train)
kde = kde_search.best_estimator_
print("KDE bandwidth:", kde.bandwidth)
print(f"GMM test mean log density: {gmm.score_samples(X_test).mean():.3f}")
print(f"KDE test mean log density: {kde.score_samples(X_test).mean():.3f}")

# %% score_samples คืน log density จึงใช้ exp เพื่อวาดกราฟ
grid = np.linspace(-5, 6, 400).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(X_train[:, 0], bins=30, density=True, alpha=0.3, label="Training histogram")
ax.plot(grid[:, 0], np.exp(gmm.score_samples(grid)), label="GMM")
ax.plot(grid[:, 0], np.exp(kde.score_samples(grid)), label="KDE")
ax.set(xlabel="Value", ylabel="Probability density")
ax.legend()
fig.tight_layout()
out = Path("outputs/14-mixtures-and-density")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "density.png", dpi=140)
plt.show()
plt.close(fig)
```
<!-- CODE:END -->

## อ่านผล

- **GMM:** ดู BIC ของ k=1–4 และ responsibilities ของแต่ละจุด ซึ่งรวมกันใกล้ 1
- **KDE:** ดู bandwidth ที่เลือก
- **กราฟ:** เทียบเส้น GMM และ KDE กับ histogram ของ train
- **ผล test:** รายงาน log density เฉลี่ยต่อหนึ่งตัวอย่าง

![ผลการทดลอง 14-mixtures-and-density](../../assets/figures/14-mixtures-and-density-density.png)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองคิด:** responsibilities รวมเป็น 1 แล้วความสูงกราฟ density ต้องเท่ากับ 1 ด้วยหรือไม่?

**เฉลย:** ความสูงกราฟไม่จำเป็นต้องเท่ากับ 1 เพราะเป็นคนละค่ากัน:

- **Responsibilities:** probability ของการเป็นสมาชิกแต่ละ component จึงรวมกันเป็น 1
- **Density:** ต้องหาพื้นที่ใต้กราฟในช่วงที่สนใจจึงจะได้ probability

สำหรับตัวแปรต่อเนื่อง โอกาสของจุดที่ระบุค่าเดียวเป๊ะ ๆ เป็นศูนย์

</details>

## อ้างอิง

- [Handbook: Gaussian Mixture Models](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html)
- [Handbook: KDE](https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html)
- [Gaussian mixtures](https://scikit-learn.org/1.7/modules/mixture.html)
- [Density estimation](https://scikit-learn.org/1.7/modules/density.html)
