# 13 · K-means: ค้นกลุ่มโดยไม่มี label

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้ใช้ **K-means** ค้นกลุ่มจาก X โดยไม่ใช้ label ฝึกหรือเลือกจำนวนกลุ่ม

**ใช้สิ่งที่เรียนแล้ว:** ระยะทางและ scaling จากบท 03; **KNN** ใช้ k เพื่อนบ้านทายคลาส ส่วน **K-means** ใช้ k ศูนย์กลางค้นกลุ่ม

## ใจความใหม่

K-means วนสองขั้น: จัดแต่ละจุดเข้าศูนย์กลางที่ใกล้ที่สุด → ย้ายศูนย์กลางไปค่าเฉลี่ยของสมาชิก จุดเฉลี่ยนี้เรียก **centroid**

เช่น จุด 1, 2, 3 มี centroid=2 และ **inertia** = (1−2)² + (2−2)² + (3−2)² = 2 เป็นผลรวมระยะกำลังสองถึงศูนย์กลางของกลุ่ม `n_init=10` ให้ลองสิบการเริ่มต้นแล้วเลือก inertia ต่ำสุด ไม่ใช่สิบกลุ่ม

| คะแนน | ใช้อ่านอะไร |
|---|---|
| Inertia | ความกระชับรอบ centroid; เพิ่ม k แล้วมักลด จึงเลือกค่าต่ำสุดอย่างเดียวไม่ได้ |
| Silhouette | −1 ถึง 1; ใกล้ 1 แยกชัด ใกล้ 0 อยู่รอยต่อ ติดลบอาจใกล้อีกกลุ่มมากกว่า |
| ARI | เทียบการแบ่งกับ label ภายนอกโดยไม่สนหมายเลขกลุ่ม; 1 คือตรงกัน, 0 ใกล้ระดับสุ่ม และติดลบได้ |

โค้ดเลือก k จาก silhouette ก่อน แล้วค่อยเปิด label จำลอง (`truth`) ดู ARI หมายเลขกลุ่มสลับได้ จึงใช้ accuracy เทียบ cluster ID กับ class label ตรง ๆ ไม่ได้

K-means ไวต่อสเกลและเหมาะกับกลุ่มกะทัดรัดมากกว่ากลุ่มโค้งยาว คะแนนเหล่านี้ยังไม่ยืนยันว่ากลุ่มมีความหมายหรือเสถียรในงานจริง

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

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

## อ่านผล

เห็น inertia/silhouette ของ k=2–6 และ k ที่เลือก กากบาทดำคือ centroid ในพิกัดที่ scale แล้ว `predict` ของจุดใหม่เลือก centroid ที่ใกล้ที่สุด; cluster 2 ไม่ได้ดีกว่าหรือใหญ่กว่า cluster 1

![ผลการทดลอง 13-kmeans-clustering](../../assets/figures/13-kmeans-clustering-clusters.png)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** เปลี่ยนข้อมูลเป็น `make_moons` แล้วอ่านรูปร่างกลุ่ม

**แนวคำตอบ:** K-means อาจตัดกลุ่มโค้งตามระยะถึง centroid ทางเลือกอย่าง DBSCAN ใช้ความหนาแน่น แต่ยังต้องเลือก `eps`, `min_samples` และสเกลอย่างเหมาะสม

</details>

## อ้างอิง

- [Handbook: k-Means Clustering](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html)
- [scikit-learn: Clustering](https://scikit-learn.org/1.7/modules/clustering.html)
