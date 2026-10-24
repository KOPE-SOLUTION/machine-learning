# 14 · Gaussian Mixtures และ Kernel Density Estimation

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 50–70 นาที  
สิ่งที่จะทำได้: แยก hard/soft assignment และแยก probability density ออกจาก probability ได้

## ก่อนเริ่มบทนี้

ต่อจาก K-means บท 13 และ histogram แบบ density ของ Matplotlib บทนี้เริ่มประมาณว่าค่าช่วงใดพบได้หนาแน่น โดยไม่ได้ทำนาย class label

## มองข้อมูลเป็นส่วนผสมของการกระจาย

Gaussian Mixture Model หรือ GMM อธิบายข้อมูลด้วยการกระจาย Gaussian หลาย component ที่มีน้ำหนักรวมกันเป็นหนึ่ง แต่ละจุดมี responsibility หรือความน่าจะเป็นของการเป็นสมาชิก component ภายใต้โมเดล จึงเป็น soft assignment ต่างจากการเลือกกลุ่มเดียวของ K-means

GMM เรียนพารามิเตอร์สลับระหว่างการประมาณสมาชิกและปรับการกระจาย โค้ดเลือกจำนวน component ด้วย BIC บน train ซึ่งเพิ่มโทษให้โมเดลซับซ้อน ค่า BIC ต่ำกว่าดีกว่าในตัวเลือกที่เทียบ ไม่ใช่หลักฐานว่าจำนวน component คือจำนวนกลุ่มจริงเสมอ

## KDE วาง kernel รอบแต่ละตัวอย่าง

Kernel Density Estimation สร้างความหนาแน่นจาก kernel หลายตัว bandwidth เล็กทำให้เส้นละเอียดและอาจตาม noise; ใหญ่ทำให้เรียบและอาจกลบหลายยอด บทนี้เลือก bandwidth ด้วย CV บน train ก่อนรายงานคะแนน test

Density ที่จุดหนึ่งอาจมากกว่า 1 ได้ เพราะความน่าจะเป็นของช่วงคือพื้นที่ใต้กราฟในช่วงนั้น ในกรณีต่อเนื่อง probability ของจุดที่ระบุเป๊ะ ๆ เป็นศูนย์ score_samples คืน log density จึงต้อง exp ก่อนวาดความหนาแน่น

ค่าเฉลี่ย test log density ยิ่งมากยิ่งดีเมื่อเทียบข้อมูล หน่วย และมิติเดียวกัน ห้ามเทียบข้าม preprocessing ที่เปลี่ยนหน่วยโดยไม่คำนึงถึงการแปลงความหนาแน่น และไม่ควรใช้ค่าต่ำเป็นข้อสรุปว่าเป็นความผิดปกติที่เป็นอันตรายโดยไม่มีบริบท

## Gaussian และ component คืออะไร

Gaussian คือการกระจายรูประฆังที่อธิบายด้วยค่าเฉลี่ย (จุดกลาง) และความแปรปรวน (ความกว้าง) ตัวอย่างมีข้อมูลหนาแน่นสองบริเวณแถว −2 กับ 2 GMM จึงลองผสมระฆังหลายลูก แต่ **component** เป็นส่วนหนึ่งของโมเดล ไม่จำเป็นต้องเท่ากับกลุ่มจริงในทุกข้อมูล

**Responsibility** ของ x=0 เช่น `[0.2, 0.8]` หมายถึงโมเดลให้โอกาสเป็นสมาชิก component แต่ละตัว 20% และ 80% เมื่อทราบ x=0 ไม่ใช่โอกาสที่ข้อมูลจะมีค่าเท่ากับศูนย์ ลำดับ component สลับได้เช่นเดียวกับ cluster ID

**BIC** (Bayesian information criterion) ชั่งความพอดีกับข้อมูลกับจำนวนพารามิเตอร์ ค่าเล็กกว่าดีกว่าเมื่อเทียบตัวเลือกบนข้อมูลชุดเดียวกัน รอบแรกเพียงอ่านตาราง k=1..4 และเลือกค่าต่ำสุด ยังไม่ต้องพิสูจน์สูตร BIC

## จาก histogram สู่เส้น density

KDE วางระฆังเล็กที่แต่ละจุดแล้วรวมเป็นเส้น `bandwidth` คือสเกลความกว้างของระฆัง ส่วน GMM เรียนระฆังจำนวนน้อยพร้อมตำแหน่งและความกว้างของแต่ละลูก ทั้งสองจึงใช้วิธีอธิบายรูปข้อมูลต่างกัน

Density มีหน่วยกลับกับ x และอาจสูงกว่า 1 เช่นการกระจายสม่ำเสมอในช่วง 0 ถึง 0.5 มีความสูง 2 แต่พื้นที่รวม `0.5×2=1` โอกาสอยู่ในช่วง 0 ถึง 0.1 คือ `0.1×2=0.2` จึงอ่านความสูงเป็นเปอร์เซ็นต์ตรง ๆ ไม่ได้

`score_samples` คืน log density ของแต่ละแถว และ `np.exp` แปลงกลับเป็นความสูงกราฟ การเลือก bandwidth ใช้ KFold เพราะไม่มี label ให้ stratify; คะแนนเริ่มต้นของ KernelDensity เป็นผลรวม log density บน validation แต่ละ fold ในตัวอย่างมีจำนวนแถวเท่ากัน ส่วนรายงาน test ใช้ค่าเฉลี่ยเพื่ออ่านต่อหนึ่งตัวอย่าง

## คัดลอกโค้ดไปรัน

สร้างไฟล์ `.ipynb` ใหม่ คัดลอกโค้ดด้านล่างทั้งชุดลง Code cell แล้วกด **Shift+Enter** ใช้โปรแกรมเดิมที่เรียน Matplotlib ได้เลย ดู [วิธีเริ่มเรียน](../../docs/start-here.md) หากต้องการทำตามทีละขั้น

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

## อ่านผลที่ได้

เห็น train BIC แต่ละ k, responsibilities รวมใกล้หนึ่ง, bandwidth ที่เลือก และกราฟ GMM/KDE เทียบ histogram ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## ภาพตัวอย่างจากการรัน

![ผลการทดลอง 14-mixtures-and-density](../../assets/figures/14-mixtures-and-density-density.png)

ภาพนี้สร้างจากโค้ดและ seed ในบท เพื่อช่วยอ่านผลก่อนทดลองด้วยตัวเอง

## แบบฝึกหัด

คำนวณผลรวม responsibilities ของ x=0 และอธิบายว่าต่างจากค่าความสูงกราฟ density ณ x=0 อย่างไร

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

responsibilities เป็น probability แบบมีเงื่อนไขบน component จึงรวมเป็นหนึ่ง ส่วนความสูงกราฟเป็น density ของค่า x มีหน่วยกลับกับ x และต้องอินทิเกรตช่วงเพื่อได้ probability

</details>

## อ่านต่อ

- [Handbook: Gaussian Mixture Models](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html)
- [Handbook: KDE](https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html)
- [Gaussian mixtures](https://scikit-learn.org/1.7/modules/mixture.html)
- [Density estimation](https://scikit-learn.org/1.7/modules/density.html)

