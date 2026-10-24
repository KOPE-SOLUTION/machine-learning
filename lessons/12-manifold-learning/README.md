# 12 · Manifold Learning และการอ่านภาพ t-SNE

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 40–60 นาที  
สิ่งที่จะทำได้: แยกการสำรวจภาพออกจากการทำนาย และระบุสิ่งที่ภาพ embedding ไม่ได้บอกได้

## เหตุใดการฉายลงระนาบจึงสูญเสียข้อมูล

ข้อมูลที่มีหลายมิติอาจอยู่บนโครงสร้างที่โค้ง การฉายเชิงเส้นอย่าง PCA อาจวางจุดที่ห่างตามโครงสร้างจริงใกล้กัน ส่วน manifold learning พยายามรักษาความสัมพันธ์บางอย่าง เช่นความเป็นเพื่อนบ้าน

t-SNE เปรียบเทียบความคล้ายกันระหว่างจุดและหาตำแหน่งใหม่ในสองมิติ โดยเน้นเพื่อนบ้านใกล้กัน perplexity ควบคุมสเกลของบริเวณเพื่อนบ้านและต้องน้อยกว่าจำนวนตัวอย่าง ค่าเริ่มต้นและ random seed มีผลต่อผลลัพธ์

## ขอบเขตของการตีความ

ระยะห่างระหว่างเกาะในภาพ t-SNE ขนาดเกาะ และทิศของแกน ไม่ควรตีความตรง ๆ ว่าเป็นระยะหรือปริมาณจริงในข้อมูลต้นฉบับ การเห็นเกาะแยกไม่ใช่หลักฐานว่ามีคลาสธรรมชาติเท่าจำนวนเกาะ

บทนี้สุ่มตัวอย่าง 600 ภาพเพื่อสำรวจ สีเป็น label เดิมที่เติมหลังคำนวณ embedding; ไม่ส่ง y ให้ตัวลดมิติ Trustworthiness ประเมินความน่าเชื่อถือของเพื่อนบ้านบางส่วน ไม่ใช่คะแนนจำแนกหรือหลักฐานว่าความสัมพันธ์ระดับไกลถูกเก็บไว้

TSNE ใน scikit-learn ไม่มี transform สำหรับจุดใหม่แบบเดียวกับ PCA จึงไม่ควรนำไปต่อเป็นขั้นตอนทำนายข้อมูลใหม่ตามตัวอย่าง Pipeline บทก่อนโดยตรง

## ลงมือรัน

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/12-manifold-learning/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` บล็อกนี้รันได้ทั้งชุดโดยไม่ต้องรันบทก่อนหน้า

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

## อ่านผลที่ได้

ได้ภาพ PCA เทียบ t-SNE พร้อม neighborhood trustworthiness สองค่า บทนี้ไม่มี test accuracy เพราะเป็นงานสำรวจ ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## ภาพตัวอย่างจากการรัน

![ผลการทดลอง 12-manifold-learning](../../assets/figures/12-manifold-learning-embeddings.png)

ภาพนี้สร้างจากโค้ดและ seed ในบท เพื่อช่วยอ่านผลก่อนทดลองด้วยตัวเอง

## แบบฝึกหัด

ลอง perplexity=10 และ 50 โดยคงชุดข้อมูลเดิม เขียนสิ่งที่เหมือนและต่าง ห้ามตั้งชื่อกลุ่มจากตำแหน่งซ้าย/ขวาอย่างเดียว

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

เปรียบเทียบสมาชิกเพื่อนบ้านและการปนของ label ที่แสดงได้ แต่ตำแหน่งและขนาดกลุ่มเปลี่ยนได้ตามพารามิเตอร์ ต้องตรวจข้อมูลต้นฉบับและความเสถียร ไม่ใช้ภาพที่ดูดีที่สุดเป็นหลักฐานความแม่นยำ

</details>

## อ่านต่อ

- [Handbook: Manifold Learning](https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html)
- [scikit-learn: Manifold learning](https://scikit-learn.org/1.7/modules/manifold.html#t-sne)

