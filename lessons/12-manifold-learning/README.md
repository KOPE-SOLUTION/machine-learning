# 12 · Manifold Learning และการอ่านภาพ t-SNE

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เปรียบเทียบภาพ **PCA กับ t-SNE** เพื่อสำรวจข้อมูลหลายมิติ

**ใช้สิ่งที่เรียนแล้ว:** PCA จากบท 11 และ scatter/colorbar จาก Matplotlib

## ใจความใหม่

**Manifold** คือโครงสร้างที่ใช้พิกัดน้อยกว่าจำนวน feature เช่น กระดาษสองมิติที่ม้วนอยู่ในพื้นที่สามมิติ ส่วน **embedding** คือตำแหน่งใหม่ที่ใช้แทนแต่ละตัวอย่าง

t-SNE จัดพิกัดโดยเน้นความสัมพันธ์ของเพื่อนบ้านใกล้กัน ต่างจาก PCA ที่หาแกนเชิงเส้นเพื่อเก็บ variance `perplexity` คุมขอบเขตเพื่อนบ้านโดยประมาณ ต้องน้อยกว่าจำนวนตัวอย่าง และไม่ใช่จำนวนกลุ่ม

โค้ดสุ่ม 600 ภาพแล้วแปลงเป็นพิกัด `(600, 2)` สีใช้ label เดิมเพื่ออ่านภาพ ไม่ส่ง y เข้า `fit_transform` การสุ่มครั้งนี้เป็นการเลือกตัวอย่างเพื่อสำรวจ ไม่มีการวัด train/test accuracy

| สิ่งที่เห็น | ตีความอย่างไร |
|---|---|
| จุดใกล้กัน | ตรวจความคล้ายในข้อมูลต้นฉบับประกอบ |
| ระยะระหว่างเกาะ ขนาดเกาะ ทิศแกน | ไม่อ่านเป็นระยะ ขนาดกลุ่ม หรือความหมายจริงโดยตรง |
| เกาะแยกชัด | ยังไม่พิสูจน์จำนวนคลาสจริงหรือความแม่นยำ |

**Trustworthiness** วัดว่ามีเพื่อนบ้านใกล้ที่เกิดขึ้นผิดจากข้อมูลเดิมมากเพียงใด ค่าสูงใกล้ 1 ดีกว่าตามเกณฑ์นี้ ไม่ใช่ accuracy และไม่รับรองระยะไกลทั้งหมด

`TSNE` ของ scikit-learn ไม่มี `transform` สำหรับจุดใหม่แบบ PCA บทนี้จึงเป็นงานสำรวจ ไม่ใช่ขั้นทำนายภาพใหม่

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

<!-- CODE:START -->
```python
"""เปรียบเทียบ PCA และ t-SNE สำหรับภาพสำรวจข้อมูล"""
# %% งานสำรวจ: ไม่มี classifier หรือคะแนนทำนายข้อมูลใหม่
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

# %% สีคือ label เดิมที่ใช้แสดงผล ไม่ได้ส่งให้ fit_transform
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
```
<!-- CODE:END -->

## อ่านผล

ได้ภาพ PCA/t-SNE พร้อม trustworthiness สองค่า ผล t-SNE อาจเปลี่ยนตาม seed และ perplexity บันทึก `outputs/12-manifold-learning/embeddings.png`

![ผลการทดลอง 12-manifold-learning](../../assets/figures/12-manifold-learning-embeddings.png)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** เปลี่ยน perplexity เป็น 10 และ 50 โดยใช้ 600 ภาพชุดเดิม

**แนวคำตอบ:** เปรียบเทียบสมาชิกเพื่อนบ้านและการปนของ label แล้วตรวจข้อมูลเดิม ไม่เลือกจากภาพเกาะที่ดูสวยที่สุดหรือชื่อกลุ่มตามตำแหน่งซ้าย/ขวา

</details>

## อ้างอิง

- [Handbook: Manifold Learning](https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html)
- [scikit-learn: Manifold learning](https://scikit-learn.org/1.7/modules/manifold.html#t-sne)
