# 05 · Linear Regression และการอ่านความผิดพลาด

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 45–60 นาที  
สิ่งที่จะทำได้: ตีความ slope/intercept เปรียบเทียบ MAE/RMSE/R² และอ่าน residual ได้

## เรียนเส้นตรงจากตัวอย่าง

สำหรับหนึ่ง feature โมเดลเขียนเป็น ŷ = b + wx โดย w เป็น slope และ b เป็น intercept ข้อมูลจำลองในบทนี้ใช้ y = 12 + 3.5x + noise จึงเปรียบเทียบค่าที่เรียนได้กับสมการกำเนิดได้ ข้อมูลจริงมักไม่เปิดเผยสมการนี้

Ordinary least squares เลือกค่าพารามิเตอร์ให้ผลรวมกำลังสองของ residual ต่ำที่สุด โดย residual = y − ŷ คำว่า linear หมายถึงเป็นเชิงเส้นในพารามิเตอร์; สามารถสร้าง x² เป็น feature แล้วใช้ LinearRegression ได้ แต่โมเดลจะซับซ้อนขึ้นและต้องตรวจ overfitting

## วัดในหน่วยที่อ่านรู้เรื่อง

MAE คือค่าเฉลี่ย |y − ŷ| ส่วน RMSE คือรากที่สองของค่าเฉลี่ย (y − ŷ)² จึงลงโทษความผิดพลาดใหญ่หนักกว่า ทั้งสองมีหน่วยเดียวกับ y ส่วน R² เทียบกับความแปรปรวนของ y ในชุดประเมิน ค่า R² อาจติดลบและไม่ใช่เปอร์เซ็นต์ความถูกต้อง

Baseline ที่ทายค่าเฉลี่ย train ช่วยตอบว่าเส้นตรงใช้ข้อมูล x ได้มีประโยชน์เพียงใด ค่าเฉลี่ยเหมาะกับ squared error ส่วนค่ามัธยฐานมักเหมาะเมื่อเป้าหมายเป็น absolute error

Residual ที่กระจายรอบศูนย์โดยไม่มีรูปแบบชัดสนับสนุนว่าโมเดลจับรูปแบบหลักได้ หากเห็นความโค้งหรือความกว้างเพิ่มตามค่า prediction ให้ตรวจข้อสมมติและ feature ก่อนตีความ coefficient ว่าเป็นเหตุและผล

## ลงมือรัน

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/05-linear-regression/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` คัดลอกบล็อกต่อไปนี้ได้ทั้งชุด แต่ละส่วน `# %%` คือช่วงการทดลองเดียวกันกับ notebook

<!-- CODE:START -->
```python
"""ทำนายตัวเลขด้วยเส้นตรง และดู residual"""
# %% สร้างข้อมูลจำลองที่รู้สมการจริง
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = rng.uniform(0, 10, size=(240, 1))
y = 12 + 3.5 * X[:, 0] + rng.normal(0, 3, size=len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %% เปรียบเทียบกับการทายค่าเฉลี่ยจาก train
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Baseline MAE: {mean_absolute_error(y_test, baseline.predict(X_test)):.3f}")
print(f"Linear MAE: {mean_absolute_error(y_test, pred):.3f}")
print(f"Linear RMSE: {root_mean_squared_error(y_test, pred):.3f}")
print(f"Linear R2: {r2_score(y_test, pred):.3f}")
print(f"Learned intercept={model.intercept_:.3f}, slope={model.coef_[0]:.3f}")

# %% residual = ค่าจริง - ค่าทำนาย
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
grid = np.linspace(0, 10, 100).reshape(-1, 1)
axes[0].scatter(X_test[:, 0], y_test, label="Test observations", alpha=0.7)
axes[0].plot(grid[:, 0], model.predict(grid), color="tab:red", label="Fitted line")
axes[0].set(xlabel="Input x", ylabel="Target y")
axes[0].legend()
axes[1].scatter(pred, y_test - pred, alpha=0.7)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set(xlabel="Prediction", ylabel="Residual")
fig.tight_layout()
out = Path("outputs/05-linear-regression")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "regression.png", dpi=140)
plt.show()
plt.close(fig)
```
<!-- CODE:END -->

## อ่านผลที่ได้

slope ใกล้ 3.5 intercept ใกล้ 12 โดยไม่จำเป็นต้องเท่าพอดี กราฟมีเส้นทำนายกับ residual และบันทึก regression.png ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## ภาพตัวอย่างจากการรัน

![ผลการทดลอง 05-linear-regression](../../assets/figures/05-linear-regression-regression.png)

ภาพนี้สร้างจากโค้ดและ seed ในบท เพื่อช่วยอ่านผลก่อนทดลองด้วยตัวเอง

## แบบฝึกหัด

สร้างข้อมูลเพิ่มด้วยส่วนโค้ง 0.8 * X[:, 0] ** 2 แล้วดู residual ของเส้นตรง ลองอธิบายสิ่งที่เส้นตรงยังจับไม่ได้ โดยใช้ validation ถ้าจะเลือกโมเดลใหม่

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

residual มักมีรูปโค้งที่สม่ำเสมอ แสดงว่าโมเดลยังพลาดรูปแบบของค่าเฉลี่ยตาม x อาจลอง PolynomialFeatures ใน Pipeline และเลือก degree ด้วย CV อย่าใช้ test เลือก degree

</details>

## อ่านต่อ

- [Handbook: Linear Regression](https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html)
- [scikit-learn: Linear models](https://scikit-learn.org/1.7/modules/linear_model.html)

