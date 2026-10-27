# M01 · ฝึก Linear Regression ด้วย Gradient Descent ใน Mojo

[กลับเส้นทาง Mojo](../README.md) · [Mojo](main.mojo) · [Python reference](reference.py)

เวลาเรียนประมาณ 45–60 นาที เรียน [บท 15](../../lessons/15-gradient-descent/README.md) ก่อนเริ่ม

**ทดสอบแล้วด้วย Mojo 1.2.0 บน Ubuntu 24.04 / WSL และ CPython 3.12.3** ใช้ environment ตาม [คู่มือติดตั้ง](../README.md) เพื่อให้รุ่นตรงกัน

## โมเดลที่ฝึกในบทนี้

โมเดลคือ **Linear Regression** ใช้ `ŷ = wx + b` ทำนาย y จาก x ส่วน Gradient Descent เป็นวิธีปรับ w/b โค้ด Mojo เรียนจากตัวอย่างห้าจุดบนเส้น `y = 3x + 2` แล้วตรวจว่าค่าที่เรียนเข้าใกล้ w=3 และ b=2 ทบทวนความหมายโมเดลและการอัปเดตใน [บท 15](../../lessons/15-gradient-descent/README.md)

## สิ่งที่เพิ่มจากบท 15

ใช้สูตร gradient descent เดิมกับห้าจุดบนเส้น y=3x+2 รอบนี้เขียนเป็น typed loop ใน Mojo จึงไม่ทวนสูตร MSE/gradient

| สิ่งที่ใช้ใน Mojo | หน้าที่ |
|---|---|
| `var` | ประกาศตัวแปรที่เปลี่ยนค่าได้ |
| `Float64` / `List[Float64]` | ระบุชนิดตัวเลขและรายการข้อมูล |
| loop สะสม dw/db | ทำหน้าที่เดียวกับค่าเฉลี่ยแบบ vectorized ใน NumPy |
| `Float64(n)` | แปลงจำนวนตัวอย่างก่อนหาร |
| `main()` | จุดเริ่มโปรแกรม |

คำนวณ dw/db จาก w/b รอบเดิมก่อนอัปเดตเหมือนบท 15 คัดลอกโค้ด Mojo ไปไฟล์ `main.mojo` ไม่วางใน cell Python

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

รัน Python ใน environment เดียวกันเพื่อเทียบคำตอบ

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

## อ่านผล

คาด w≈3, b≈2 และ training MSE ใกล้ศูนย์ เพราะข้อมูลไม่มี noise และใช้ฝึกทุกจุด ผลนี้ไม่ใช่คะแนนกับข้อมูลใหม่ Python reference ตรวจพารามิเตอร์ด้วย tolerance 1e-8

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** ลดจาก 300 เหลือ 10 รอบแล้วเทียบสองภาษา

**แนวคำตอบ:** 10 รอบอาจยังไม่ลู่เข้า ตรวจข้อมูล ชนิดตัวเลข learning rate และลำดับอัปเดตให้ตรงกันก่อนเทียบความเร็ว ไม่ตัดสินความถูกต้องจากทศนิยมแค่สองตำแหน่ง

</details>

## อ้างอิง

- [Mojo quickstart: variables, types and loops](https://mojolang.org/docs/manual/quickstart/)
- [Google MLCC: Gradient descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent)
