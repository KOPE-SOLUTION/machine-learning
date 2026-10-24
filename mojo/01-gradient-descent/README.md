# M01 · เขียน Gradient Descent ด้วย Mojo

[กลับเส้นทาง Mojo](../README.md) · [Mojo](main.mojo) · [Python reference](reference.py)

เวลาเรียนประมาณ 45–60 นาที เรียน [บท 15](../../lessons/15-gradient-descent/README.md) ก่อนเริ่ม

**ทดสอบแล้วด้วย Mojo 1.2.0 บน Ubuntu 24.04 / WSL และ CPython 3.12.3** ใช้ environment ตาม [คู่มือติดตั้ง](../README.md) เพื่อให้รุ่นตรงกัน

## แนวคิด

ใช้ข้อมูลห้าจุดที่อยู่บนเส้น y=3x+2 และอัปเดต w,b ด้วย MSE gradient เหมือนบท Python แต่ระบุ Float64 ให้ตัวเลขและ List[Float64] ให้ข้อมูล `var` ประกาศตัวแปรที่เปลี่ยนค่าได้ และ `main` เป็นจุดเริ่มโปรแกรมตาม syntax ของเอกสารที่อ้างอิง

การสะสม dw,db ด้วย loop ทำให้เห็นว่าการคูณ/บวกแต่ละตัวอย่างประกอบกันเป็น gradient อย่างไร เราคำนวณทั้งคู่จากพารามิเตอร์รอบเดิมก่อนอัปเดต

## เชื่อมจาก NumPy มาสู่ loop

| ที่คุ้นเคยใน Python | ที่ใช้ในตัวอย่าง Mojo |
|---|---|
| array ของ float64 | `List[Float64]` เก็บค่าชนิดเดียวกันในตัวอย่างนี้ |
| การเปลี่ยนค่า w, b | `var` ประกาศตัวแปรที่แก้ค่าได้ |
| `np.mean(error * x)` | วนทีละจุด บวก `error * x[i] / n` จนครบ |
| การแปลงจำนวนเต็มเป็นทศนิยม | `Float64(n)` ก่อนหาร |
| รันไฟล์ Python | compiler แปล `.mojo` แล้วเริ่มที่ `main()` |

คัดลอกบล็อก Mojo ไปไฟล์ `main.mojo` ก่อนรัน ไม่วางใน cell Python ข้อมูลห้าจุดนี้ไม่มี noise จึงคาด w≈3, b≈2 และ MSE ใกล้ศูนย์ บท 15 ใช้ข้อมูลมี noise และ test แยก จึงไม่ควรคาด MSE ศูนย์แบบเดียวกัน

## รันจาก root ของ repo

หลัง activate environment ตามคู่มือ Mojo:

```bash
mojo mojo/01-gradient-descent/main.mojo
python mojo/01-gradient-descent/reference.py
```

## โค้ด

<!-- MOJO:START -->
```mojo
# Verified with Mojo 1.2.0 on Ubuntu 24.04 / WSL; see mojo/README.md.
def main():
    var x: List[Float64] = [-1.0, -0.5, 0.0, 0.5, 1.0]
    var y: List[Float64] = [-1.0, 0.5, 2.0, 3.5, 5.0]
    var w: Float64 = 0.0
    var b: Float64 = 0.0
    var learning_rate: Float64 = 0.1
    var n = len(x)

    for _ in range(300):
        var dw: Float64 = 0.0
        var db: Float64 = 0.0
        for i in range(n):
            var error = w * x[i] + b - y[i]
            dw += 2.0 * error * x[i] / Float64(n)
            db += 2.0 * error / Float64(n)
        w -= learning_rate * dw
        b -= learning_rate * db

    var mse: Float64 = 0.0
    for i in range(n):
        var error = w * x[i] + b - y[i]
        mse += error * error / Float64(n)

    print("w:", w)
    print("b:", b)
    print("training MSE:", mse)
```
<!-- MOJO:END -->

## Python สำหรับเทียบผล

คัดลอกบล็อกนี้ไปรันด้วย Python ใน environment เดียวกันได้ครบชุด เปรียบเทียบคำตอบก่อนทดลองเปลี่ยนค่าหรือจับเวลา

<!-- PYTHON:START -->
```python
"""Python reference using the same data, precision, update order, and iterations."""
import numpy as np

x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
y = np.array([-1.0, 0.5, 2.0, 3.5, 5.0], dtype=np.float64)
w, b = 0.0, 0.0
for step in range(300):
    error = w * x + b - y
    dw = 2.0 * np.mean(error * x)
    db = 2.0 * np.mean(error)
    w -= 0.1 * dw
    b -= 0.1 * db
np.testing.assert_allclose([w, b], [3.0, 2.0], atol=1e-8)
print("w:", w)
print("b:", b)
print("training MSE:", np.mean((w * x + b - y) ** 2))
```
<!-- PYTHON:END -->

## อ่านผลและฝึกต่อ

ค่าที่คาดตามสมการคือ w ใกล้ 3 และ b ใกล้ 2 โดยมี training MSE ใกล้ศูนย์ Python reference ตรวจพารามิเตอร์ด้วย tolerance 1e-8; ผลนี้ไม่ใช่คะแนน generalization เพราะทุกจุดอยู่บนสมการที่เรากำหนดและใช้ฝึกทั้งหมด

ลองลดจำนวนรอบเป็น 10 แล้วเปรียบเทียบกับ 300 รอบ ทั้งสองภาษาใช้ข้อมูล, Float64, learning rate และ update order เดียวกันหรือไม่? หากต่างต้องแก้ความถูกต้องก่อนจับเวลา

<details><summary>แนวคำตอบ</summary>

10 รอบอาจยังเรียนไม่ถึงพารามิเตอร์จริง 300 รอบให้เวลาเข้าใกล้ค่าที่เหมาะสมมากขึ้นในโจทย์นี้ ความต่างระดับ floating point ยอมรับได้ตาม tolerance ที่สัมพันธ์กับโจทย์ อย่าตัดสินว่า implementation ถูกจากการแสดงทศนิยมเพียงสองตำแหน่ง

</details>

## อ้างอิง

- [Mojo quickstart: variables, types and loops](https://mojolang.org/docs/manual/quickstart/)
- [Google MLCC: Gradient descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent)

