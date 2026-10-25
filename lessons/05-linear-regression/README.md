# 05 · Linear Regression และการอ่านความผิดพลาด

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เปลี่ยนจากทายคลาสเป็น **ทายตัวเลข** ด้วยเส้นตรง และอ่านความผิดพลาดในหน่วยของคำตอบ

**ใช้สิ่งที่เรียนแล้ว:** train/test และ baseline จากบท 01; การสร้าง array และกราฟใช้ NumPy/Matplotlib เดิม

## ใจความใหม่

**Linear Regression:** ใช้สมการ **ŷ = b + wx** ทำนายตัวเลข โดย `coef_[0]` คือความชัน w และ `intercept_` คือ b

การ fit แบบ **least squares** เลือก w และ b ให้ผลรวมความผิดพลาดกำลังสองต่ำที่สุด

**ตัวอย่าง:** b=12, w=3.5 และ x=2 จะทายได้ 19 หากค่าจริงคือ 21 แปลว่าทายต่ำไป 2 หน่วย

**Residual:** ค่าจริงลบคำทำนาย (`y − ŷ`) ตัวอย่างข้างต้นจึงมี residual เท่ากับ 2

ข้อมูลในบทสร้างจากสมการเส้นตรงนี้แล้วเติม noise

| Metric | ความหมาย |
|---|---|
| MAE | ค่าเฉลี่ยของขนาดความผิดพลาด ยิ่งต่ำยิ่งดี |
| RMSE | รากของค่าเฉลี่ยความผิดพลาดกำลังสอง ให้น้ำหนักความผิดพลาดใหญ่มากกว่า MAE |
| R² | เทียบ squared error กับการทายค่าเฉลี่ยของชุดประเมิน โดย 1 คือทายตรงทั้งหมด, 0 คือเท่าค่าเฉลี่ยนั้น และติดลบได้ |

- **หน่วย:** MAE และ RMSE มีหน่วยเดียวกับ y ส่วน R² ไม่ใช่เปอร์เซ็นต์ความถูกต้อง
- **Baseline:** โค้ดเรียนค่าเฉลี่ยจาก train ซึ่งอาจต่างจากค่าเฉลี่ยชุดประเมินที่ใช้ในนิยาม R²
- **กราฟ residual:** ควรกระจายรอบศูนย์ หากเหลือรูปโค้ง โมเดลอาจยังจับรูปแบบไม่ครบ

กราฟนี้ไม่ได้ยืนยันเหตุและผล หากนำผล test ไปปรับโมเดลต่อ ต้องกันข้อมูลที่ยังไม่ใช้ตัดสินใจไว้ประเมินใหม่

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

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

## อ่านผล

ได้ slope ใกล้ 3.5 และ intercept ใกล้ 12 พร้อมคะแนน MAE, RMSE และ R²

- **กราฟซ้าย:** เทียบข้อมูลกับเส้นทำนาย
- **กราฟขวา:** แสดง residual โดยจุดเหนือศูนย์หมายถึงทายต่ำกว่าค่าจริง

ไฟล์ภาพ: `outputs/05-linear-regression/regression.png`

![ผลการทดลอง 05-linear-regression](../../assets/figures/05-linear-regression-regression.png)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** เพิ่มส่วนโค้ง `0.8 * X[:, 0] ** 2` ในสมการข้อมูล แล้วดู residual

**แนวคำตอบ:** เส้นตรงอาจเหลือรูปโค้งในกราฟ residual

หากจะลอง `PolynomialFeatures` ให้เลือก degree ด้วย validation หรือ CV โดยไม่ใช้ test เลือกค่า

</details>

## อ้างอิง

- [Handbook: Linear Regression](https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html)
- [scikit-learn: Linear models](https://scikit-learn.org/1.7/modules/linear_model.html)
