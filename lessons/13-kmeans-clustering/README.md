# 13 · K-means: ค้นกลุ่มโดยไม่มี label

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 45–60 นาที  
สิ่งที่จะทำได้: เข้าใจ centroid/inertia เลือก k อย่างมีเหตุผล และแยก cluster ID ออกจาก class label ได้

## วนระหว่างการจัดกลุ่มกับย้ายศูนย์กลาง

K-means เริ่มด้วย centroid จำนวน k จุด จากนั้นให้แต่ละตัวอย่างอยู่กับ centroid ที่ใกล้ที่สุด แล้วคำนวณ centroid ใหม่จากสมาชิก ทำซ้ำจนเข้าเกณฑ์หยุด จุดเริ่มต้นมีผลจึงกำหนด n_init=10 ให้ลองหลายการเริ่มต้น

Inertia คือผลรวมระยะห่างกำลังสองถึง centroid ของแต่ละจุด เมื่อเพิ่ม k ค่านี้มักลดลง จึงเลือก k จาก inertia ต่ำสุดอย่างเดียวไม่ได้ Silhouette เปรียบเทียบความใกล้ภายในกลุ่มกับกลุ่มอื่น ค่าอยู่ประมาณ −1 ถึง 1 ยิ่งสูงยิ่งแยกชัดตามระยะที่ใช้ แต่ยังมีอคติด้านรูปทรงของกลุ่ม

## กลุ่มเชิงเรขาคณิตไม่ใช่หมวดหมู่ที่ยืนยันแล้ว

ตัวอย่างนี้ scale ก่อนจัดกลุ่มเพราะระยะทางไวต่อหน่วย K-means เหมาะกับกลุ่มค่อนข้างกะทัดรัดรอบ centroid มากกว่ากลุ่มโค้งยาวหรือความหนาแน่นต่างกันมาก

truth จากข้อมูลจำลองไม่ถูกใช้ตอน fit หรือเลือก k เราเปิดดู ARI หลังเลือกเพื่อฝึกวิธีประเมินภายนอกเท่านั้น หมายเลขกลุ่มอาจสลับได้เมื่อรันใหม่ แม้สมาชิกเหมือนเดิม จึงเปรียบเทียบ label ด้วย accuracy ตรง ๆ ไม่ได้

การเลือก k บนข้อมูลชุดนี้เป็นงานสำรวจ หากจะนำไปใช้งานต้องดูความเสถียรเมื่อข้อมูลเปลี่ยน ความหมายของแต่ละกลุ่ม และผลต่อการตัดสินใจจริงด้วย

## ลงมือรัน

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/13-kmeans-clustering/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` บล็อกนี้รันได้ทั้งชุดโดยไม่ต้องรันบทก่อนหน้า

<!-- CODE:START -->
```python
"""แบ่งกลุ่มจุดโดยไม่ให้ K-means เห็นคำตอบ"""
# %% เลือก k จาก silhouette เป็นการสำรวจ ไม่ใช่ test accuracy
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

# %% หมายเลขกลุ่มไม่มีลำดับหรือความหมายทางธุรกิจโดยตัวเอง
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
```
<!-- CODE:END -->

## อ่านผลที่ได้

ได้ inertia/silhouette ของ k=2..6, k ที่เลือก, ARI หลังเลือก และภาพ centroid ของกลุ่ม ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## ภาพตัวอย่างจากการรัน

![ผลการทดลอง 13-kmeans-clustering](../../assets/figures/13-kmeans-clustering-clusters.png)

ภาพนี้สร้างจากโค้ดและ seed ในบท เพื่อช่วยอ่านผลก่อนทดลองด้วยตัวเอง

## แบบฝึกหัด

เปลี่ยนข้อมูลเป็น make_moons แล้วพิจารณาว่า centroid เหมาะกับรูปกลุ่มหรือไม่ ค้นอ่าน DBSCAN เป็นทางเลือกและระบุ hyperparameter ที่ต้องศึกษา

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

กลุ่มโค้งอาจถูก K-means ตัดเป็นส่วนตามระยะถึง centroid DBSCAN ใช้ความหนาแน่นและมี eps กับ min_samples แต่ต้องเลือกสเกลและระยะอย่างเหมาะสมเช่นกัน

</details>

## อ่านต่อ

- [Handbook: k-Means Clustering](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html)
- [scikit-learn: Clustering](https://scikit-learn.org/1.7/modules/clustering.html)

