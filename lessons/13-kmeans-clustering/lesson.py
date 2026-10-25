# %% Cell 1 — ลองจำนวนกลุ่มและเลือกจาก silhouette
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

X, truth = make_blobs(n_samples=450, centers=3, cluster_std=1.4, random_state=42)
scale = StandardScaler()
Z = scale.fit_transform(X)
models, scores = {}, {}
for k in range(2, 7):
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    models[k] = model
    scores[k] = silhouette_score(Z, model.labels_)
    print(f"k={k}: inertia={model.inertia_:.1f}, silhouette={scores[k]:.3f}")
chosen = max(scores, key=scores.get)
model = models[chosen]
print("Selected k:", chosen)
print("ARI against hidden synthetic labels (after selection):", round(adjusted_rand_score(truth, model.labels_), 3))
print("New point cluster ID:", model.predict(scale.transform([[0.0, 0.0]]))[0])

# %% [markdown]
# **อ่านโค้ด**
#
# - `make_blobs(..., centers=3, ...)` สร้างข้อมูลจำลองจากสามกลุ่ม พร้อมเก็บ label จริงใน `truth` แต่ไม่ส่ง label นี้ให้ K-means
# - `scale.fit_transform(X)` เรียนสถิติและปรับสเกลเป็น `Z` สำหรับการสำรวจกลุ่มในข้อมูลชุดนี้
# - `range(2, 7)` ลองจำนวนกลุ่มตั้งแต่ 2 ถึง 6 ส่วน `n_init=10` ให้แต่ละค่า k ลองสิบจุดเริ่มต้น
# - `model.labels_` คือหมายเลขกลุ่มของข้อมูลที่ใช้ fit ส่วน `model.inertia_` คือผลรวมระยะกำลังสองภายในกลุ่ม
# - `scores[k] = silhouette_score(...)` เก็บคะแนนเพื่อเลือก k ที่ได้คะแนนสูงสุดด้วย `max(scores, key=scores.get)` แล้วจึงใช้ `truth` คำนวณ ARI
# - `scale.transform([[0.0, 0.0]])` ปรับสเกลจุดใหม่ด้วยสถิติเดิมก่อนเรียก `predict` และ `[0]` เลือกหมายเลขกลุ่มของจุดนั้น

# %% Cell 2 — แสดงกลุ่มและศูนย์กลาง
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(Z[:, 0], Z[:, 1], c=model.labels_, cmap="tab10", s=18, alpha=0.7)
ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", marker="X", s=150)
ax.set(xlabel="Scaled feature 1", ylabel="Scaled feature 2", title=f"K-means: k={chosen}")
fig.tight_layout()
out = Path("outputs/13-kmeans-clustering")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "clusters.png", dpi=140)
plt.show()
plt.close(fig)

# %% [markdown]
# **อ่านโค้ด**
#
# - `c=model.labels_` ระบายสีจุดตามกลุ่มที่ K-means หาได้ ไม่ได้ใช้ `truth` ระบายสี
# - `model.cluster_centers_` เก็บพิกัด centroid ของแต่ละกลุ่มในสเกลเดียวกับ `Z`
# - `[:, 0]` กับ `[:, 1]` เลือกสองแกนของ centroid แล้ว `marker="X"` วาดเป็นกากบาทดำ
#
# หมายเลขกลุ่มเป็นเพียงรหัส จึงใช้ตัดสินลำดับหรือคุณค่าของกลุ่มไม่ได้
