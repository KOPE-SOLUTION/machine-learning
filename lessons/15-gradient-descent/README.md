# 15 · เขียน Gradient Descent ด้วย NumPy

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 50–75 นาที  
สิ่งที่จะทำได้: คำนวณ loss/gradient อัปเดต w,b และตรวจคำตอบกับ least squares ได้

## เปิดดูขั้นตอนการเรียนพารามิเตอร์

บทนี้สร้าง ŷ = wx + b แล้วลด mean squared error ด้วยการอัปเดตทีละก้าว เหมาะสำหรับเชื่อมแนวคิดกับโค้ดและบทเสริม Mojo ส่วน LinearRegression ที่ใช้ก่อนหน้านี้ไม่จำเป็นต้องใช้ gradient descent แบบนี้ในการ fit

กำหนด error_i = wx_i + b − y_i จะได้

- MSE = (1/n) Σ error_i²
- ∂MSE/∂w = (2/n) Σ error_i x_i
- ∂MSE/∂b = (2/n) Σ error_i
- w ใหม่ = w เดิม − learning_rate × gradient_w
- b ใหม่ = b เดิม − learning_rate × gradient_b

คำนวณ gradient ของทั้งสองค่าจากพารามิเตอร์รอบเดิมก่อนอัปเดต มิฉะนั้นอาจได้วิธีคำนวณต่างจากที่ตั้งใจ

## เลือกขนาดก้าว

Learning rate เล็กเกินไปอาจต้องใช้หลายรอบ; ใหญ่เกินไปอาจแกว่งหรือ diverge สเกล x มีผลต่อความชันของ loss ตัวอย่างจึงใช้ x ในช่วง −1 ถึง 1 และเทียบพารามิเตอร์กับ least squares บน train เดียวกัน

การที่ training loss ลดลงยังไม่รับรองผลกับข้อมูลใหม่ เราจึงรายงาน test MAE ด้วย ฟังก์ชันตรวจ shape และค่าที่ไม่เป็น finite เพื่อให้ความผิดพลาดของ input ชัดเจน

## ลงมือรัน

จากโฟลเดอร์หลักของ repo:

```bash
uv run --frozen python lessons/15-gradient-descent/lesson.py
```

หากใช้ venv ให้ใช้ `python` แทน `uv run --frozen python` บล็อกนี้รันได้ทั้งชุดโดยไม่ต้องรันบทก่อนหน้า

<!-- CODE:START -->
```python
"""เรียนรู้พารามิเตอร์ด้วย gradient descent จาก NumPy"""
# %% ฟังก์ชันนี้ใช้ซ้ำในแบบทดสอบและบทเสริม Mojo
import numpy as np


def fit_gradient_descent(x, y, learning_rate=0.1, steps=1000):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or len(x) == 0:
        raise ValueError("x and y must be nonempty one-dimensional arrays of equal length")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must be finite")
    if learning_rate <= 0 or steps < 1:
        raise ValueError("learning_rate and steps must be positive")
    w, b = 0.0, 0.0
    history = []
    for _ in range(steps):
        error = w * x + b - y
        dw = 2.0 * np.mean(error * x)
        db = 2.0 * np.mean(error)
        w -= learning_rate * dw
        b -= learning_rate * db
        history.append(np.mean((w * x + b - y) ** 2))
    return w, b, np.asarray(history)


def main():
    from pathlib import Path
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, 200)
    y = 3 * x + 2 + rng.normal(0, 0.08, len(x))
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    w, b, history = fit_gradient_descent(x_train, y_train)
    reference = LinearRegression().fit(x_train.reshape(-1, 1), y_train)
    print(f"Gradient descent: w={w:.4f}, b={b:.4f}")
    print(f"Least squares: w={reference.coef_[0]:.4f}, b={reference.intercept_:.4f}")
    print(f"Test MAE: {mean_absolute_error(y_test, w * x_test + b):.4f}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(history)
    ax.set(xlabel="Update step", ylabel="Training MSE", title="Learning the line")
    fig.tight_layout()
    out = Path("outputs/15-gradient-descent")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "loss.png", dpi=140)
    plt.show()
    plt.close(fig)


# %% เรียกการทดลอง
if __name__ == "__main__":
    main()
```
<!-- CODE:END -->

## อ่านผลที่ได้

w ใกล้ 3, b ใกล้ 2 พารามิเตอร์ใกล้คำตอบ least squares และ training MSE ลดลงก่อนเข้าใกล้ระดับ noise ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## ภาพตัวอย่างจากการรัน

![ผลการทดลอง 15-gradient-descent](../../assets/figures/15-gradient-descent-loss.png)

ภาพนี้สร้างจากโค้ดและ seed ในบท เพื่อช่วยอ่านผลก่อนทดลองด้วยตัวเอง

## แบบฝึกหัด

ลอง learning_rate=0.001, 0.1, 2.0 บนข้อมูลเดิม บันทึกว่า loss ลดช้า ลู่เข้า หรือไม่เสถียร พร้อมอธิบายว่าการเพิ่ม steps ช่วยกรณีใด

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

ก้าวเล็กอาจได้ประโยชน์จาก steps เพิ่ม แต่ก้าวที่ทำให้ diverge ต้องลด learning rate หรือปรับสเกล ไม่ใช่เพิ่ม steps อย่างเดียว ปิดการทดลองเมื่อ loss ไม่ finite และอย่ารายงานพารามิเตอร์นั้นเป็นโมเดลสำเร็จ

</details>

## อ่านต่อ

- [Google MLCC: Gradient descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent)
- [Linear models](https://scikit-learn.org/1.7/modules/linear_model.html)

