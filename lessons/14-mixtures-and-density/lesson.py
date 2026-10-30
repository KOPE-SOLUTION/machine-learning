# %% Cell 1 — สร้างข้อมูลหนึ่งมิติ
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KernelDensity

rng = np.random.default_rng(42)
X = np.concatenate([rng.normal(-2, 0.6, 250), rng.normal(2, 1.0, 350)]).reshape(-1, 1)
X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)

# %% [markdown]
# **อ่านโค้ด**
#
# - `rng.normal(-2, 0.6, 250)` และ `rng.normal(2, 1.0, 350)` สุ่มข้อมูลสองส่วนที่มีค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานต่างกัน
# - `np.concatenate(...)` ต่อทั้งสองส่วนเป็น 600 ค่า แล้ว `reshape(-1, 1)` จัดเป็น 600 แถว หนึ่ง feature
# - `train_test_split(..., test_size=0.25)` แยก train 450 แถวและ test 150 แถว บทนี้ไม่มี y เพราะกำลังประมาณการกระจายของ X

# %% Cell 2 — เลือก GMM ด้วย BIC
mixtures = [GaussianMixture(n_components=k, n_init=5, random_state=42).fit(X_train) for k in range(1, 5)]
gmm = min(mixtures, key=lambda model: model.bic(X_train))
for model in mixtures:
    print(f"GMM k={model.n_components}: train BIC={model.bic(X_train):.1f}")
print("Responsibilities for x=0:", gmm.predict_proba([[0.0]]).round(3))

# %% [markdown]
# **อ่านโค้ด**
#
# - list comprehension สร้างและ fit GMM ที่มี `n_components` ตั้งแต่ 1 ถึง 4 บน train ชุดเดียวกัน
# - `n_init=5` ให้แต่ละโมเดลลองห้าการเริ่มต้น เพื่อเลือกผลฝึกที่ดีที่สุดของจำนวน component นั้น
# - `min(mixtures, key=lambda model: model.bic(X_train))` เลือก **ตัวโมเดล** ที่มี BIC ต่ำสุด โดย `lambda` ระบุค่าที่นำมาเทียบ
# - `gmm.predict_proba([[0.0]])` คืน responsibility ของ x=0 ต่อแต่ละ component ผลในแถวนี้รวมกันเป็น 1 โดยประมาณ

# %% Cell 3 — เลือก KDE ด้วย CV
kde_search = GridSearchCV(
    KernelDensity(kernel="gaussian"), {"bandwidth": [0.15, 0.3, 0.6, 1.0]},
    cv=KFold(5, shuffle=True, random_state=42),
)
kde_search.fit(X_train)
kde = kde_search.best_estimator_
print("KDE bandwidth:", kde.bandwidth)
print(f"GMM test mean log density: {gmm.score_samples(X_test).mean():.3f}")
print(f"KDE test mean log density: {kde.score_samples(X_test).mean():.3f}")

# %% [markdown]
# **อ่านโค้ด**
#
# - `KernelDensity(kernel="gaussian")` เลือก kernel รูป Gaussian ส่วน dictionary `bandwidth` ระบุสี่ค่าที่ให้ GridSearchCV ทดลอง
# - `KFold(5, shuffle=True, ...)` แบ่ง train ห้าส่วนเพื่อเลือก bandwidth โดยไม่ต้องมี label
# - เมื่อไม่กำหนด `scoring` การค้นหาจะใช้ `KernelDensity.score` ซึ่งคืนผลรวม log density บน validation ของแต่ละ fold
# - `best_estimator_` คือ KDE ที่เลือกและ refit บน train แล้ว
# - `score_samples(X_test).mean()` เฉลี่ย log density ต่อแถวบน test เพื่อเปรียบเทียบ GMM กับ KDE
#
# **อ่านคะแนน CV กับ test แยกกัน:** KDE ใช้ KFold เพราะไม่มี label ให้ stratify คะแนนค้นหาเป็นผลรวม log density ของ validation fold ที่มี 90 แถวเท่ากันทุกรอบ ส่วนสองบรรทัดท้ายเป็นค่าเฉลี่ยต่อแถวบน test 150 แถว จึงไม่ใช้ตัวเลขสองรูปแบบนี้เทียบกันตรง ๆ

# %% Cell 4 — วาดความหนาแน่นของสองโมเดล
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

# %% [markdown]
# **อ่านโค้ด**
#
# - `grid` เป็นตำแหน่ง x ที่ต้องการคำนวณความหนาแน่น จัดเป็นสองมิติตามรูปข้อมูลที่โมเดลรับ
# - `score_samples(grid)` คืน log density จึงใช้ `np.exp(...)` แปลงเป็น density ก่อนวาดเส้น
# - `hist(..., density=True)` ปรับ histogram ให้พื้นที่รวมเป็น 1 เพื่อเทียบกับเส้น density ได้
# - กราฟใช้ train ส่วนผล test log density แสดงแยกใน Cell ก่อนหน้า
