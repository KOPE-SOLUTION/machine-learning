# 15 · ฝึก Linear Regression ด้วย Gradient Descent และ NumPy

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เขียน **Gradient Descent** เพื่อเรียน w และ b ของเส้นตรงด้วย NumPy

**ใช้สิ่งที่เรียนแล้ว:** สมการเส้นตรงและ MAE จากบท 05; vectorization และการวาดกราฟใช้พื้นฐานเดิม

## โมเดลที่ฝึก: Linear Regression

โมเดลคือ **Linear Regression** จาก [บท 05](../05-linear-regression/README.md) ใช้สมการ `ŷ = wx + b` คำนวณค่าทำนาย ŷ จากข้อมูลนำเข้า x ส่วน **Gradient Descent เป็นวิธีหาค่า w และ b** ให้โมเดลนี้

โค้ดจึงมีทั้งเส้นตรงที่ฝึกด้วย NumPy และ `LinearRegression()` ของ scikit-learn สำหรับเทียบผลบน train เดียวกัน ทั้งสองเป็นโมเดลเชิงเส้น แต่ใช้วิธีหาพารามิเตอร์ต่างกัน ความสัมพันธ์ที่ไม่เหมาะกับเส้นตรงยังเป็นข้อจำกัด แม้ฝึกจน loss ลดลงแล้ว

## Gradient Descent คืออะไร

**Gradient Descent (GD)** คือวิธีปรับค่าที่โมเดลใช้ทำนายทีละก้าว เพื่อให้ค่าความผิดพลาดที่กำหนดไว้ลดลง งานหาค่าที่เหมาะสมเช่นนี้เรียกว่า **Optimization** ในบทนี้ค่าที่ต้องปรับคือ w และ b ของ Linear Regression

**หลักการ:** ตั้งค่า w และ b เริ่มต้น ใช้ทำนายแล้วคำนวณความผิดพลาด จากนั้นคำนวณว่าการเพิ่มหรือลด w และ b เล็กน้อยจะเปลี่ยนความผิดพลาดไปทางใด แล้วปรับค่าไปทางที่ช่วยลดความผิดพลาด ทำซ้ำโดยใช้ขนาดก้าวที่เหมาะสม

- **จุดเด่น:** นำแนวคิดเดียวกันไปใช้กับโมเดลอื่นที่คำนวณ gradient ได้ โดยไม่ต้องมีสูตรหาพารามิเตอร์คำตอบสำเร็จรูป
- **ข้อจำกัด:** ต้องเลือก learning rate และจำนวนรอบ สเกลมีผลต่อการเรียน และ loss ที่ลดลงบน train ยังไม่ยืนยันว่าจะทำนายข้อมูลใหม่ได้ดี

### ค่าที่ใช้คำนวณ

- **Loss:** ตัวเลขที่สรุปความผิดพลาดระหว่างฝึก บทนี้ใช้ **MSE — Mean Squared Error** โดยยกกำลังสองความผิดพลาดแต่ละแถวแล้วหาค่าเฉลี่ย
- **Gradient:** ตัวเลขที่บอกว่า loss จะเปลี่ยนอย่างไรเมื่อเพิ่มพารามิเตอร์เล็กน้อย เช่น gradient ของ w เป็นบวก หมายถึงเพิ่ม w เล็กน้อยแล้ว loss มีแนวโน้มสูงขึ้น จึงปรับ w ลดลง
- **Learning rate:** ตัวคูณที่กำหนดว่าจะปรับมากเพียงใดในแต่ละก้าว ขนาดที่ปรับจริงขึ้นกับทั้ง learning rate และ gradient

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

1. เริ่ม w=b=0 จึงทาย ŷ=0 ค่าจริง y=5 ทำให้ error=0−5=−5
2. จากสูตร ได้ dw = 2 × (−5) × 1 = −10 และ db = 2 × (−5) = −10 จึงปรับ w = 0 − 0.1 × (−10) = 1 และ b = 0 − 0.1 × (−10) = 1
3. คำทำนายใหม่เป็น 2 ทำให้ squared error ลดจาก 25 เป็น 9

![Gradient Descent หนึ่งก้าวจาก w และ b เท่ากับศูนย์ เมื่อ x เท่ากับ 1 และ y เท่ากับ 5 ได้ gradient ลบ 10 ทั้งคู่ อัปเดตเป็น w และ b เท่ากับ 1 และ squared error ลดจาก 25 เป็น 9](../../assets/figures/15-gradient-descent-update.png)

**อ่านภาพ:** เส้นประคือโมเดลก่อนอัปเดต เส้นน้ำเงินคือหลังอัปเดตหนึ่งก้าว จึงทายที่ x=1 ได้ 2 และเข้าใกล้ค่าจริง 5 ตัวอย่างนี้ตรงกับการคำนวณด้านบน ส่วนโค้ดฝึกจากหลายจุดใน train · [เปิดภาพ SVG](../../assets/figures/15-gradient-descent-update.svg)

จุดเดียวนี้ใช้ฝึกคำนวณหนึ่งก้าวเท่านั้น เพราะมีหลายคู่ w และ b ที่ทาย y=5 เมื่อ x=1 ได้เหมือนกัน เช่น w=1, b=4 หรือ w=3, b=2

- **ก้าวเล็ก:** loss อาจลดช้า
- **ก้าวใหญ่เกินไป:** อาจปรับข้ามค่าที่เหมาะสมไปมา ทำให้ loss เพิ่มขึ้น หรือค่าตัวเลขใหญ่เกินกว่าที่คอมพิวเตอร์จะเก็บได้
- **สเกล x:** มีผลต่อขนาดก้าวที่เหมาะสม

เราเทียบ w และ b กับ `LinearRegression.fit` จากบท 05 บน train ชุดเดียวกัน ทั้งสองวิธีมุ่งลดความผิดพลาดกำลังสอง: Gradient Descent ในบทนี้ลดค่าเฉลี่ยหรือ MSE ส่วน Ordinary Least Squares ลดผลรวมหรือ SSE เมื่อใช้ข้อมูลจำนวนเท่ากัน ทั้งสองเกณฑ์จะได้ w และ b ที่เหมาะสมชุดเดียวกัน

ความต่างอยู่ที่วิธีหาคำตอบ: โค้ด NumPy ค่อย ๆ ปรับ w และ b ส่วน `LinearRegression.fit` สำหรับข้อมูลในบทนี้ใช้ตัวแก้ปัญหาพีชคณิตเชิงเส้น จึงต้องตรวจว่า GD ปรับไปใกล้คำตอบเพียงพอแล้วหรือยัง และอ่าน test MAE เพื่อดูผลกับข้อมูลใหม่

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรัน **Cell 1–3 ตามลำดับ** หรือเปิด [Notebook ของบทนี้](lesson.ipynb) ใช้ kernel เดิมได้เลย อ่านคำอธิบายใต้โค้ดแต่ละช่วงประกอบผลที่ได้

<!-- CODE:CELLS -->

<!-- CODE:START -->
### Cell 1 — ประกาศฟังก์ชัน Gradient Descent

```python
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
```

**อ่านโค้ด**

- `def fit_gradient_descent(...)` ประกาศฟังก์ชันไว้เรียกภายหลัง การรัน Cell นี้ยังไม่เริ่มฝึก
- `np.asarray(..., dtype=np.float64)` จัด x และ y เป็น array ตัวเลข จากนั้นตรวจว่ามีจำนวนแถวเท่ากัน ไม่ว่าง ไม่มี NaN หรือค่าบวก/ลบอนันต์ และการตั้งค่าจำนวนก้าวกับ learning rate ใช้งานได้ ก่อนเริ่มฝึก
- `w, b = 0.0, 0.0` เป็นจุดเริ่ม ส่วน `error = w * x + b - y` คำนวณคำทำนายลบค่าจริงจากทุกแถว
- `dw` และ `db` คำนวณ gradient จาก w และ b รอบเดียวกัน แล้วจึงอัปเดตด้วย `w -= learning_rate * dw` และ `b -= learning_rate * db`
- `history.append(loss)` เก็บ MSE **หลังอัปเดต** แต่ละรอบ เพื่อนำไปวาดกราฟ
- `np.errstate(over="raise", invalid="raise")` ให้หยุดเมื่อคำนวณล้นหรือใช้ค่าที่ผิด แล้วส่งข้อความแนะนำให้ลด learning rate หรือปรับสเกล
- `return w, b, np.asarray(history)` คืนพารามิเตอร์สองค่าและประวัติ loss ให้ผู้เรียก

**ค่าตั้งต้นของฟังก์ชัน:** `learning_rate=0.1` ให้ใช้ 0.1 คูณ gradient เพื่อคำนวณว่าจะปรับ w และ b เท่าไร ส่วน `steps=1000` คือปรับซ้ำ 1,000 ก้าว เมื่อเรียกโดยไม่ส่งค่าสองตัวนี้ จะใช้ค่าตั้งต้นดังกล่าว การประกาศฟังก์ชันใน cell นี้ยังไม่สร้างผลทำนายหรือกราฟ

### Cell 2 — ประกาศขั้นตอนทดลองและวาดกราฟ

```python
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
```

**อ่านโค้ด**

- `main()` รวมขั้นตอนทดลองไว้ แต่ Cell นี้เพียงประกาศฟังก์ชัน ยังไม่เรียกทำงาน
- `y = 3 * x + 2 + ...` สร้างข้อมูลจากเส้นจริงที่รู้ w=3 และ b=2 แล้วเติม noise เล็กน้อย
- `w, b, history = fit_gradient_descent(...)` รับผลสามส่วนจากฟังก์ชันที่สร้างใน Cell 1
- `x_train.reshape(-1, 1)` จัด x เป็นหนึ่งคอลัมน์ให้ `LinearRegression` รับได้ เพื่อเทียบกับ least squares บนข้อมูลเดียวกัน
- `w * x_test + b` คือคำทำนายที่นำไปวัด test MAE
- `ax.semilogy(history)` วาด MSE โดยแกน y ใช้สเกล log เพื่อเห็นการลดลงหลายระดับได้ชัด

### Cell 3 — เรียกการทดลอง

```python
if __name__ == "__main__":
    main()
```

**อ่านโค้ด:** `if __name__ == "__main__":` เรียก `main()` เมื่อรันไฟล์นี้โดยตรงหรือรัน Cell นี้ใน notebook

หากไฟล์อื่น import ฟังก์ชันไปใช้ เงื่อนไขนี้จะไม่เรียกการทดลองซ้ำ เมื่อรันใน notebook ต้องรัน Cell 1–2 ก่อน เพื่อประกาศฟังก์ชันให้ครบ
<!-- CODE:END -->

## อ่านผล

ได้ w ใกล้ 3 และ b ใกล้ 2 ซึ่งใกล้เคียงกับผล least squares

กราฟแสดง MSE บน train หลังปรับแต่ละรอบ ช่วงแรกค่าลดลงแล้วเปลี่ยนเพียงเล็กน้อย เพราะเส้นตรงยังทายจุดที่มีความผันผวนจากข้อมูลจำลองได้ไม่ตรงทั้งหมด ให้ดู test MAE ควบคู่เพื่อประเมินข้อมูลที่ไม่ได้ใช้ฝึก

ไฟล์ภาพ: `outputs/15-gradient-descent/loss.png`

![ผลการทดลอง 15-gradient-descent](../../assets/figures/15-gradient-descent-loss.png)

## แบบฝึกท้ายบท

**ลองทำ:** ทดลอง learning rate 0.001, 0.1, 2.0 บนข้อมูลเดิม

<details>
<summary>ดูแนวคำตอบ</summary>

**แนวคำตอบ:**

- หากก้าวเล็กและ loss ยังลดอยู่ การเพิ่มจำนวนรอบอาจช่วยได้
- หาก loss เพิ่มขึ้นเรื่อย ๆ หรือ w/b โตจนคำนวณต่อไม่ได้ เรียกว่า diverge ให้ลด learning rate หรือปรับสเกลก่อน การเพิ่มจำนวนรอบอย่างเดียวไม่แก้ปัญหา

</details>

## อ้างอิง

- [Google MLCC: Gradient descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent)
- [Linear models](https://scikit-learn.org/1.7/modules/linear_model.html)
