# %% Cell 1 — ประกาศฟังก์ชัน Gradient Descent
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

# %% [markdown]
# **อ่านโค้ด**
#
# - `def fit_gradient_descent(...)` ประกาศฟังก์ชันไว้เรียกภายหลัง การรัน Cell นี้ยังไม่เริ่มฝึก
# - `np.asarray(..., dtype=np.float64)` จัด x และ y เป็น array ตัวเลข จากนั้นตรวจว่ามีจำนวนแถวเท่ากัน ไม่ว่าง ไม่มี NaN หรือค่าบวก/ลบอนันต์ และการตั้งค่าจำนวนก้าวกับ learning rate ใช้งานได้ ก่อนเริ่มฝึก
# - `w, b = 0.0, 0.0` เป็นจุดเริ่ม ส่วน `error = w * x + b - y` คำนวณคำทำนายลบค่าจริงจากทุกแถว
# - `dw` และ `db` คำนวณ gradient จาก w และ b รอบเดียวกัน แล้วจึงอัปเดตด้วย `w -= learning_rate * dw` และ `b -= learning_rate * db`
# - `history.append(loss)` เก็บ MSE **หลังอัปเดต** แต่ละรอบ เพื่อนำไปวาดกราฟ
# - `np.errstate(over="raise", invalid="raise")` ให้หยุดเมื่อคำนวณล้นหรือใช้ค่าที่ผิด แล้วส่งข้อความแนะนำให้ลด learning rate หรือปรับสเกล
# - `return w, b, np.asarray(history)` คืนพารามิเตอร์สองค่าและประวัติ loss ให้ผู้เรียก
#
# **ค่าตั้งต้นของฟังก์ชัน:** `learning_rate=0.1` ให้ใช้ 0.1 คูณ gradient เพื่อคำนวณว่าจะปรับ w และ b เท่าไร ส่วน `steps=1000` คือปรับซ้ำ 1,000 ก้าว เมื่อเรียกโดยไม่ส่งค่าสองตัวนี้ จะใช้ค่าตั้งต้นดังกล่าว การประกาศฟังก์ชันใน cell นี้ยังไม่สร้างผลทำนายหรือกราฟ

# %% Cell 2 — ประกาศขั้นตอนทดลองและวาดกราฟ
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

# %% [markdown]
# **อ่านโค้ด**
#
# - `main()` รวมขั้นตอนทดลองไว้ แต่ Cell นี้เพียงประกาศฟังก์ชัน ยังไม่เรียกทำงาน
# - `y = 3 * x + 2 + ...` สร้างข้อมูลจากเส้นจริงที่รู้ w=3 และ b=2 แล้วเติม noise เล็กน้อย
# - `w, b, history = fit_gradient_descent(...)` รับผลสามส่วนจากฟังก์ชันที่สร้างใน Cell 1
# - `x_train.reshape(-1, 1)` จัด x เป็นหนึ่งคอลัมน์ให้ `LinearRegression` รับได้ เพื่อเทียบกับ least squares บนข้อมูลเดียวกัน
# - `w * x_test + b` คือคำทำนายที่นำไปวัด test MAE
# - `ax.semilogy(history)` วาด MSE โดยแกน y ใช้สเกล log เพื่อเห็นการลดลงหลายระดับได้ชัด

# %% Cell 3 — เรียกการทดลอง
if __name__ == "__main__":
    main()

# %% [markdown]
# **อ่านโค้ด:** `if __name__ == "__main__":` เรียก `main()` เมื่อรันไฟล์นี้โดยตรงหรือรัน Cell นี้ใน notebook
#
# หากไฟล์อื่น import ฟังก์ชันไปใช้ เงื่อนไขนี้จะไม่เรียกการทดลองซ้ำ เมื่อรันใน notebook ต้องรัน Cell 1–2 ก่อน เพื่อประกาศฟังก์ชันให้ครบ
