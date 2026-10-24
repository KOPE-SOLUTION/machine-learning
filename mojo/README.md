# เส้นทางเสริม: ML กับ Mojo

[กลับหลักสูตร](../README.md)

เรียนบท 03, 05 และ 15 ก่อน ส่วนนี้มีเป้าหมายให้เห็นว่าการคำนวณตัวเลขแบบมีชนิดข้อมูลและการเรียก Python library มาเชื่อมกับหลักสูตรอย่างไร ไม่มี GPU หรือบริการ AI ภายนอก

| บท | เป้าหมาย |
|---|---|
| [M01: Gradient descent](01-gradient-descent/README.md) | เขียนการอัปเดตพารามิเตอร์ด้วย typed loop |
| [M02: Python interoperability](02-python-interop/README.md) | เรียก scikit-learn Pipeline จาก Mojo |

## สถานะของตัวอย่าง

**ทั้งสองบท compile และรันผ่านด้วย Mojo 1.2.0 บน Ubuntu 24.04 / WSL (x86_64), CPython 3.12.3** ตรวจเมื่อ 24 ตุลาคม 2026 โดยเทียบผลกับ Python reference ใน environment เดียวกัน

- M01: w≈3, b≈2 และ MSE ใกล้ศูนย์ ตรงกับ Python ภายใน tolerance ที่กำหนด
- M02: accuracy 0.933333 และคำทำนายห้าตัวแรก `[0, 2, 1, 1, 0]` ตรงกับ Python

ไฟล์ `requirements.lock` ล็อก dependency ของเส้นทาง Mojo แยกจากหลักสูตร Python ผลตรวจนี้เป็นของ Linux x86_64 ผ่าน WSL; ยังไม่ได้ตรวจบน macOS หรือ CPU สถาปัตยกรรมอื่น

## เตรียม environment แยก

ใช้ terminal ภายใน WSL บน Windows ตาม [system requirements](https://mojolang.org/docs/requirements/) และ [คู่มือติดตั้งทางการ](https://mojolang.org/install/) เริ่มจากโฟลเดอร์หลักของ repo หลังติดตั้ง uv แล้ว:

```bash
uv venv .venv-mojo --python 3.12
uv pip sync --python .venv-mojo/bin/python mojo/requirements.lock
source .venv-mojo/bin/activate
mojo --version
python --version
```

ถ้าไม่มี Python 3.12 ให้ใช้ `uv python install 3.12` แล้วรันคำสั่งสร้าง environment อีกครั้ง การ activate นี้เลือก Mojo และ Python จาก `.venv-mojo` สำหรับ terminal ปัจจุบัน ไม่ติดตั้ง compiler ทับ environment ของหลักสูตรหลัก

## รันตามลำดับ

```bash
mojo mojo/01-gradient-descent/main.mojo
python mojo/01-gradient-descent/reference.py
mojo mojo/02-python-interop/main.mojo
python mojo/02-python-interop/reference.py
python scripts/verify_mojo.py
```

คำสั่งสุดท้ายรันทั้งสองภาษาและตรวจผลเทียบอัตโนมัติ บันทึกหลักฐานใน `outputs/verification/mojo/` เมื่อจะกลับไปบท Python ใช้ `deactivate` แล้วรัน environment หลักตาม [setup](../docs/setup.md)

หาก Mojo หา Python library ไม่เจอ ให้ตรวจว่า `which mojo` และ `which python` ชี้ `.venv-mojo/bin/` ทั้งคู่ แล้วรัน `python -c "import sklearn; print(sklearn.__version__)"` อ่าน [Calling Python from Mojo](https://mojolang.org/docs/manual/python/python-from-mojo/) หากยังมีปัญหา ไม่ชี้ WSL ไปยัง Python executable ของ Windows

## อ่านผลอย่างถูกต้อง

การเรียก scikit-learn จาก Mojo ยังเรียก Python library ผ่าน CPython ไม่ได้แปลง training kernel เป็น Mojo อัตโนมัติ การใช้ typed loop ก็ไม่ได้รับประกันว่าเร็วกว่าการคำนวณ NumPy ที่ใช้ native kernels อยู่แล้ว

หากทดลอง benchmark ให้ตรวจคำตอบเท่ากันก่อน ระบุ dtype/ขนาดข้อมูล แยก compile/startup ออกจากเวลาคำนวณ warm up รันซ้ำหลายครั้ง รายงาน median และรวมต้นทุนข้ามภาษาเมื่อสิ่งนั้นเกิดในการใช้งานจริง หลักสูตรนี้ยังไม่มีผล benchmark ที่ยืนยันว่า Mojo เร็วกว่า

