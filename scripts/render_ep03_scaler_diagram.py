"""Explain the origin of each number in EP03's StandardScaler example."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from render_model_diagrams import configure, page, box, label, arrow, footer, save
from render_model_diagrams import BLUE, NAVY, ORANGE, MUTED, PURPLE, plot_style


def source_values():
    train = np.array([[3.0], [3.0], [7.0], [7.0]])
    scaler = StandardScaler().fit(train)
    assert np.isclose(train.mean(), 5)
    assert np.isclose(train.std(ddof=0), 2)
    assert np.allclose(scaler.mean_, [5])
    assert np.allclose(scaler.scale_, [2])
    assert np.allclose(scaler.transform([[3], [5], [7]]).ravel(), [-1, 0, 1])
    assert np.allclose(scaler.inverse_transform([[1]]), [[7]])

    configure()
    fig, p = page("StandardScaler · รู้ที่มาของ 7, 5 และ 2",
                  "ตัวอย่างสมมติ: ความยาวกลีบดอกหนึ่งคอลัมน์ · หน่วยเดิมเป็นเซนติเมตร")

    box(p, .05, .53, .43, .29, color="#EDF4FF")
    label(p, .265, .77, "1  เรียนสถิติจาก train สี่ดอก", 21, BLUE, True, ha="center")
    for i, value in enumerate([3, 3, 7, 7]):
        box(p, .075 + i*.10, .65, .08, .07, f"{value} ซม.", color="white", size=21)
    label(p, .265, .607, "ค่าเฉลี่ย = (3 + 3 + 7 + 7) / 4 = 5 ซม.",
          18, NAVY, ha="center")
    label(p, .265, .559, "ส่วนเบี่ยงเบนมาตรฐาน = 2 ซม.",
          19, BLUE, True, ha="center")

    box(p, .57, .53, .38, .29, color="#FFF0DF")
    label(p, .76, .77, "2  วัดดอกใหม่", 22, ORANGE, True, ha="center")
    label(p, .76, .668, "กลีบดอกยาว 7 ซม.", 30, ORANGE, True, ha="center")
    label(p, .76, .575, "7 คือค่า x ที่ต้องการแปลง", 20, NAVY, ha="center")

    arrow(p, (.265, .53), (.265, .46), BLUE, 2.5)
    arrow(p, (.76, .53), (.76, .46), ORANGE, 2.5)
    box(p, .08, .28, .84, .17, color="white")
    label(p, .5, .412, "3  นำค่าดอกใหม่มาแปลงด้วยสถิติจาก train", 19, MUTED, ha="center")
    label(p, .5, .335, "z = (7 − 5) / 2 = 1", 34, NAVY, True, ha="center")

    label(p, .265, .211, "7 − 5 = 2 ซม. สูงกว่าค่าเฉลี่ย", 18, ha="center")
    label(p, .755, .211, "2 ÷ 2 = 1 ส่วนเบี่ยงเบนมาตรฐาน", 18, ha="center")
    footer(p,
           "ค่า 1 เป็น feature หลังปรับสเกล → ส่งต่อให้ KNN ใช้ร่วมกับ feature อื่น",
           "1 หมายถึงสูงกว่าค่าเฉลี่ย train หนึ่งส่วนเบี่ยงเบนมาตรฐาน · ยังไม่ใช่คำทำนาย")
    save(fig, p, "03-first-classifier-scaling")


def distance_effect():
    # Synthetic 2-feature example; mean/std of length are exactly 5 and 2.
    # A is nearest before scaling; B is nearest after scaling.
    train = np.array([[7, 1], [5, 1.5], [7, .5],
                      [1, 1], [5, .9], [5, 1.1]], dtype=float)
    query = np.array([[7, 1.5]], dtype=float)
    scaler = StandardScaler().fit(train)
    scaled_train, scaled_query = scaler.transform(train), scaler.transform(query)
    raw_dist = np.linalg.norm(train - query, axis=1)
    scaled_dist = np.linalg.norm(scaled_train - scaled_query, axis=1)
    assert np.allclose(scaler.mean_, [5, 1])
    assert np.isclose(scaler.scale_[0], 2)
    assert np.isclose(scaled_query[0, 0], 1)
    assert raw_dist.argmin() == 0 and scaled_dist.argmin() == 1

    fig, p = page("ค่า 1 ไปไหนต่อ? → ใช้วัดระยะใน KNN",
                  "ตัวอย่างสมมติสอง feature · จุดฝึกชุดเดิม 6 จุด · ดาวคือดอกใหม่ · วงแหวนคือจุดที่ใกล้ที่สุด")
    for left, values, q, distances, scaled in [
        (.06, train, query[0], raw_dist, False),
        (.55, scaled_train, scaled_query[0], scaled_dist, True),
    ]:
        label(p, left+.195, .79,
              "หลังปรับสเกล · B ใกล้ที่สุด" if scaled else "ก่อนปรับสเกล · A ใกล้ที่สุด",
              23, BLUE if scaled else ORANGE, True, ha="center")
        label(p, left+.195, .735,
              "ดอกใหม่ [1, 1.70] · ค่าหลังแปลง" if scaled else "ดอกใหม่ [7, 1.5] · หน่วย cm",
              18, PURPLE, ha="center")
        ax = fig.add_axes([left, .325, .39, .355])
        plot_style(ax, "ความยาว หลังปรับสเกล" if scaled else "ความยาว (cm)",
                      "ความกว้าง หลังปรับสเกล" if scaled else "ความกว้าง (cm)")
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(values[2:, 0], values[2:, 1], c="#95A4B6", s=65, zorder=3)
        for index, color, offset in [(0, BLUE, (8, -15)), (1, ORANGE, (-17, 12))]:
            ax.scatter(*values[index], c=color, s=95, zorder=4)
            ax.annotate("AB"[index], values[index], xytext=offset, textcoords="offset points",
                        fontsize=16, color=color, weight="bold")
        nearest = distances.argmin()
        ax.plot([q[0], values[nearest, 0]], [q[1], values[nearest, 1]],
                color=PURPLE, linewidth=2, linestyle="--")
        ax.scatter(*values[nearest], facecolors="none", edgecolors=PURPLE,
                   s=330, linewidths=2, zorder=5)
        ax.scatter(*q, marker="*", color=PURPLE, s=290, zorder=6)
        if scaled:
            ax.set(xlim=(-2.5, 1.7), ylim=(-2.1, 2.1))
            ax.set_xticks([-2, -1, 0, 1])
        else:
            ax.set(xlim=(0, 8), ylim=(-.5, 3.5))
            ax.set_yticks([0, 1, 2, 3])
        label(p, left+.195, .235,
              f"ระยะถึง A = {distances[0]:.2f}   |   ถึง B = {distances[1]:.2f}",
              20, NAVY, True, ha="center")
    label(p, .5, .169,
          "หารแต่ละคอลัมน์ด้วยสเกลของตนเอง → น้ำหนักของความต่างในระยะทางเปลี่ยน",
          19, MUTED, ha="center")
    footer(p, "ภาพเน้นเพื่อนบ้านอันดับแรก · ในบทใช้ 4 feature และเลือก 5 เพื่อนบ้านมาโหวต",
           "Pipeline ทำให้เอง: ปรับสเกล → วัดระยะ → เลือกเพื่อนบ้าน → ทำนายสายพันธุ์")
    save(fig, p, "03-first-classifier-scaling-distance")


def main():
    configure()
    source_values()
    distance_effect()


if __name__ == "__main__":
    main()
