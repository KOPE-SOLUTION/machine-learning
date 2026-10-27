"""Explain the origin of each number in EP03's StandardScaler example."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from render_model_diagrams import configure, page, box, label, arrow, footer, save
from render_model_diagrams import BLUE, NAVY, ORANGE, MUTED


def main():
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
           "ผลลัพธ์ 1 บอกตำแหน่งเทียบกับ train · ความยาวจริงของดอกยังเป็น 7 ซม.",
           "fit จาก train → ใช้สถิติเดิมแปลงข้อมูลใหม่ · ตัวอย่างนี้ไม่ใช่ค่าสถิติจริงของ Iris")
    save(fig, p, "03-first-classifier-scaling")


if __name__ == "__main__":
    main()
