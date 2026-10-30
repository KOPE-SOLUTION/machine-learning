# %% Cell 1 — เลือกภาพตัวอย่างและสร้าง embedding
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import train_test_split

X_all, y_all = load_digits(return_X_y=True)
X, _, y, _ = train_test_split(X_all, y_all, train_size=600, stratify=y_all, random_state=42)
X = X / 16.0  # ขอบเขต pixel 0..16 เป็นที่ทราบจากนิยามชุดข้อมูล
Z_pca = PCA(n_components=2).fit_transform(X)
Z_tsne = TSNE(
    n_components=2, perplexity=30, init="pca", learning_rate="auto",
    max_iter=750, random_state=42,
).fit_transform(X)
print(f"PCA neighborhood trustworthiness: {trustworthiness(X, Z_pca, n_neighbors=10):.3f}")
print(f"t-SNE neighborhood trustworthiness: {trustworthiness(X, Z_tsne, n_neighbors=10):.3f}")

# %% [markdown]
# **อ่านโค้ด**
#
# - `train_test_split(..., train_size=600, stratify=y_all)` ใช้เลือกตัวอย่าง 600 ภาพโดยรักษาสัดส่วนตัวเลข ส่วน `_` รับค่าที่ไม่ใช้ต่อ ครั้งนี้ไม่ได้แบ่งเพื่อวัด test accuracy
# - `X / 16.0` ปรับค่าพิกเซลจากช่วง 0–16 เป็น 0–1 โดยใช้ขอบเขตที่ทราบจากนิยามชุดข้อมูล
# - `fit_transform(X)` เรียนพิกัดใหม่แล้วแปลงภาพชุดนี้ทันที ทั้ง PCA และ t-SNE คืนสองคอลัมน์ตาม `n_components=2`
# - `perplexity=30` กำหนดระดับการพิจารณาเพื่อนบ้านใกล้ ๆ ไม่ใช่จำนวนกลุ่ม ส่วน `init="pca"` ใช้พิกัดจาก PCA เป็นจุดเริ่ม และ `max_iter=750` ให้ปรับตำแหน่งได้สูงสุด 750 รอบ
# - `learning_rate="auto"` ให้ t-SNE เลือกขนาดก้าวตามวิธีของไลบรารี ส่วน `random_state=42` ช่วยให้ทำซ้ำได้
# - `trustworthiness(X, Z, n_neighbors=10)` เทียบเพื่อนบ้านสิบตัวในข้อมูลเดิมกับพิกัดใหม่ โดยไม่ได้ใช้ y คำนวณ
#
# **รูปของผลลัพธ์:** `Z_pca` และ `Z_tsne` มีขนาด `(600, 2)` คือพิกัดสองค่าของ 600 ภาพ ค่า trustworthiness ที่พิมพ์ประเมินความสอดคล้องของเพื่อนบ้าน ไม่ใช่สัดส่วนภาพที่จำแนกถูก

# %% Cell 2 — เปรียบเทียบภาพสองวิธี
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, Z, title in zip(axes, (Z_pca, Z_tsne), ("PCA", "t-SNE")):
    points = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=12, alpha=0.75)
    ax.set(title=title, xlabel="Embedding axis 1", ylabel="Embedding axis 2")
fig.colorbar(points, ax=axes.ravel().tolist(), ticks=range(10), label="Digit label")
out = Path("outputs/12-manifold-learning")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "embeddings.png", dpi=140, bbox_inches="tight")
plt.show()
plt.close(fig)

# %% [markdown]
# **อ่านโค้ด**
#
# - `zip(axes, (Z_pca, Z_tsne), ("PCA", "t-SNE"))` จับคู่แต่ละช่องกราฟกับพิกัดและชื่อวิธีที่ตรงกัน
# - `Z[:, 0]` และ `Z[:, 1]` คือแกน embedding ส่วน `c=y` ใส่สีจาก label เพื่อช่วยตรวจการปนกันของตัวเลข
# - `axes.ravel().tolist()` รวมช่องกราฟเป็นรายการสำหรับวาง colorbar ร่วมกัน
# - `bbox_inches="tight"` ช่วยให้การบันทึกภาพครอบคลุมองค์ประกอบรอบกราฟ เช่น colorbar
#
# ตำแหน่งซ้าย/ขวาของกลุ่มใน t-SNE ไม่ใช่ลำดับหรือความหมายของตัวเลข
