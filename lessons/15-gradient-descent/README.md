# 15 · เขียน Gradient Descent ด้วย NumPy

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เขียน **Gradient Descent** เพื่อเรียน w และ b ของเส้นตรงด้วย NumPy

**ใช้สิ่งที่เรียนแล้ว:** สมการเส้นตรงและ MAE จากบท 05; vectorization และการวาดกราฟใช้พื้นฐานเดิม

## ใจความใหม่

- **Loss:** ความผิดพลาดที่ต้องการลด บทนี้ใช้ MSE
- **Gradient:** บอกทิศและอัตราการเปลี่ยน loss เมื่อเปลี่ยนพารามิเตอร์
- **Learning rate:** กำหนดขนาดก้าวในการอัปเดตพารามิเตอร์ไปทิศตรงข้ามกับ gradient

กำหนด `error = wx + b − y` ซึ่งกลับเครื่องหมายจาก residual ในบท 05 แต่เมื่อยกกำลังสองจะได้ค่าเท่ากัน

| ค่าที่คำนวณ | สูตรใน NumPy |
|---|---|
| MSE | `np.mean(error ** 2)` |
| gradient ของ w | `dw = 2 * np.mean(error * x)` |
| gradient ของ b | `db = 2 * np.mean(error)` |
| อัปเดต | `w -= learning_rate * dw` และ `b -= learning_rate * db` |

ต้องคำนวณ dw และ db จาก w และ b รอบเดิมให้ครบก่อนเปลี่ยนค่า

**Batch gradient descent:** ใช้ทุกตัวอย่างในชุดฝึกคำนวณ gradient ต่อหนึ่งก้าว

**ตัวอย่างหนึ่งก้าว:** ให้ x=1, y=5 และ learning rate=0.1

1. เริ่ม w=b=0 จะได้ error=−5
2. คำนวณ dw=db=−10 แล้วอัปเดตได้ w=b=1
3. คำทำนายใหม่เป็น 2 ทำให้ squared error ลดจาก 25 เป็น 9

ตัวอย่างจุดเดียวนี้ใช้ฝึกคำนวณ ยังแยกค่า w และ b ที่เหมาะสมไม่ได้

- **ก้าวเล็ก:** loss อาจลดช้า
- **ก้าวใหญ่เกินไป:** ค่าอาจแกว่งหรือล้น
- **สเกล x:** มีผลต่อขนาดก้าวที่เหมาะสม

โค้ดตรวจค่าที่ไม่ finite และหยุดเมื่อคำนวณล้มเหลว โดย `history` เก็บ MSE หลังแต่ละก้าว

เราเทียบ w และ b กับ least squares บน train เดียวกัน แล้ววัด test MAE เพื่อดูผลกับข้อมูลใหม่

`LinearRegression.fit` ไม่จำเป็นต้องใช้ gradient descent แบบที่เขียนในบทนี้

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

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
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    w, b = 0.0, 0.0
    history = []
    try:
        with np.errstate(over="raise", invalid="raise"):
            for _ in range(steps):
                error = w * x + b - y
                dw = 2.0 * np.mean(error * x)
                db = 2.0 * np.mean(error)
                # ใช้ gradient ทั้งคู่จาก w,b รอบเดิมก่อนเปลี่ยนค่า
                w -= learning_rate * dw
                b -= learning_rate * db
                loss = np.mean((w * x + b - y) ** 2)
                if not np.isfinite(loss):
                    raise FloatingPointError("Non-finite loss")
                history.append(loss)
    except FloatingPointError as error:
        raise FloatingPointError("Gradient descent diverged; reduce learning_rate or scale x") from error
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

## อ่านผล

ได้ w ใกล้ 3 และ b ใกล้ 2 ซึ่งใกล้เคียงกับผล least squares

กราฟ loss ลดลงก่อนเข้าใกล้ระดับ noise ให้ดู test MAE ควบคู่ เพราะ training loss ต่ำยังไม่รับรองผลกับข้อมูลใหม่

ไฟล์ภาพ: `outputs/15-gradient-descent/loss.png`

![ผลการทดลอง 15-gradient-descent](../../assets/figures/15-gradient-descent-loss.png)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** ทดลอง learning rate 0.001, 0.1, 2.0 บนข้อมูลเดิม

**แนวคำตอบ:**

- หากก้าวเล็กและ loss ยังลดอยู่ การเพิ่มจำนวนรอบอาจช่วยได้
- หากค่า diverge ให้ลด learning rate หรือปรับสเกล การเพิ่มรอบอย่างเดียวไม่ช่วย และไม่ควรนำค่าที่ล้นไปทำนายต่อ

</details>

## อ้างอิง

- [Google MLCC: Gradient descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent)
- [Linear models](https://scikit-learn.org/1.7/modules/linear_model.html)
