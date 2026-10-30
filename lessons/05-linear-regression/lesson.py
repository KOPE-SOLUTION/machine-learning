# %% Cell 1 — สร้างข้อมูลสำหรับทำนายตัวเลข
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# %% [markdown]
# **อ่านโค้ด**
#
# - `np.random.default_rng(42)` สร้างตัวสุ่มที่ทำซ้ำได้ เพื่อให้ตัวอย่างและ noise เหมือนเดิมเมื่อรันใหม่
# - `size=(240, 1)` สร้าง X เป็น 240 แถว หนึ่ง feature ส่วน `X[:, 0]` หยิบค่าคอลัมน์นั้นมาเป็นเวกเตอร์สำหรับสร้าง y
# - `12 + 3.5 * X[:, 0]` คือสมการจริงที่ตั้งไว้ แล้วเติม noise จาก `rng.normal(0, 3, ...)` ซึ่งมีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 3
# - `train_test_split(..., test_size=0.25)` กัน test 25% ได้ train 180 แถว และ test 60 แถว
# - สมการ `12 + 3.5 * x` ใช้สร้างข้อมูลจำลองเท่านั้น โมเดลจะเรียน b และ w จาก train เอง ไม่ได้รับค่า 12 และ 3.5 ตอน fit
#
# **ผลที่ได้:** Cell นี้สร้างตัวแปร จึงยังไม่มี output แสดงใต้ cell

# %% Cell 2 — ฝึกเส้นตรงและวัดความผิดพลาด
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("Actual y (first 5):", y_test[:5].round(2))
print("Predicted y (first 5):", pred[:5].round(2))
print(f"Baseline MAE: {mean_absolute_error(y_test, baseline.predict(X_test)):.3f}")
print(f"Linear MAE: {mean_absolute_error(y_test, pred):.3f}")
print(f"Linear RMSE: {root_mean_squared_error(y_test, pred):.3f}")
print(f"Linear R2: {r2_score(y_test, pred):.3f}")
print(f"Learned intercept={model.intercept_:.3f}, slope={model.coef_[0]:.3f}")

# %% [markdown]
# **อ่านโค้ด**
#
# - `DummyRegressor(strategy="mean")` เรียนค่าเฉลี่ย y_train แล้วใช้ค่านั้นทายทุกแถว เป็น baseline ของโจทย์ตัวเลข
# - `LinearRegression().fit(...)` สร้างโมเดลและใช้ Ordinary Least Squares หา b และ w ให้ผลรวม residual² บน train ต่ำที่สุด จากนั้น `predict(X_test)` คืนตัวเลขทำนายหนึ่งค่าต่อแถว ค่าที่ได้จากข้อมูลจำลองหลักนี้จะต่างจากกรณีศึกษาฝุ่นสามแถว
# - ฟังก์ชัน MAE, RMSE และ R² รับ `y_test` กับ `pred` เพื่อเทียบคำตอบจริงกับคำทำนายชุดเดียวกัน
# - `model.intercept_` คือ b ที่เรียนได้ ส่วน `model.coef_[0]` คือ w ของ feature แรก ควรใกล้ 12 และ 3.5 ตามสมการจำลอง
#
# **อ่านผลตามลำดับ:** สองบรรทัดแรกให้เทียบค่าจริงกับคำทำนายห้าแถวแรกในตำแหน่งเดียวกัน ส่วน MAE, RMSE และ R² ถัดมาคำนวณจาก test ทั้ง 60 แถว การปัดเลขด้วย `round(2)` ใช้แสดงผลเท่านั้น

# %% Cell 3 — วาดเส้นทำนายและ residual
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
figure_path = out / "regression.png"
fig.savefig(figure_path, dpi=140)
print("Saved figure:", figure_path.resolve())
plt.show()
plt.close(fig)

# %% [markdown]
# **อ่านโค้ด**
#
# - `np.linspace(...).reshape(-1, 1)` สร้างจุด x สำหรับวาดเส้น แล้วจัดให้เป็นข้อมูลสองมิติที่โมเดลรับ โดย `-1` ให้ NumPy คำนวณจำนวนแถวเอง
# - `model.predict(grid)` สร้างค่าบนเส้นที่ฝึกแล้ว ไม่ได้ฝึกโมเดลใหม่ตามจุด grid
# - `y_test - pred` คือ residual ใช้วาดเทียบกับคำทำนายในกราฟขวา จุดเหนือศูนย์หมายถึงทายต่ำกว่าค่าจริง ส่วนจุดใต้ศูนย์คือทายสูงกว่าค่าจริง
# - `axhline(0)` วางเส้นอ้างอิง residual ศูนย์ ส่วน `savefig` บันทึกภาพก่อนปิด figure ตามวิธีในบท 02
# - `Path("outputs/05-linear-regression")` อ้างอิงจากโฟลเดอร์ที่ notebook ทำงาน ส่วน `figure_path.resolve()` แสดงตำแหน่งเต็มของไฟล์ที่บันทึก
#
# **ผลที่ได้:** ตำแหน่งไฟล์จาก `Saved figure:` และกราฟสองช่อง โดย `plt.show()` แสดงกราฟก่อน `plt.close(fig)` ปิด figure
