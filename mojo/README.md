# เส้นทางเสริม: ML กับ Mojo

[กลับหลักสูตร](../README.md)

เรียนบท 03, 05 และ 15 ก่อน ส่วนนี้มีเป้าหมายให้เห็นว่าการคำนวณตัวเลขแบบมีชนิดข้อมูลและการเรียก Python library มาเชื่อมกับหลักสูตรอย่างไร ไม่มี GPU หรือบริการ AI ภายนอก

| บท | เป้าหมาย |
|---|---|
| [M01: Gradient descent](01-gradient-descent/README.md) | เขียนการอัปเดตพารามิเตอร์ด้วย typed loop |
| [M02: Python interoperability](02-python-interop/README.md) | เรียก scikit-learn Pipeline จาก Mojo |

## สถานะของตัวอย่าง

**ยังไม่ได้ compile/run ไฟล์ .mojo ในการตรวจรอบนี้** การตรวจฉบับแรกทำบน Windows; ฉบับปรับพื้นฐานผู้เรียนย้าย repo และตรวจ Python ผ่าน Ubuntu 24.04 บน WSL แล้ว แต่ยังไม่ได้ติดตั้ง/ตรวจ Mojo compiler ตัวอย่างอ้างอิง syntax จากเอกสาร Mojo ที่แสดงรุ่น 1.2.0 ณ วันที่ตรวจ 24 ตุลาคม 2026; ไม่ใช่การยืนยันความเข้ากันได้ของ compiler ที่ติดตั้งบนเครื่องผู้เรียน

Python reference ทั้งสองไฟล์รันตรวจแยกได้ และใช้เทียบความถูกต้องก่อนวัดความเร็ว หาก compiler ใช้คนละรุ่นให้ตรวจ import, function syntax และ Python interop ตามเอกสารรุ่นนั้นก่อน ไม่ควรแก้ Python environment หลักเพื่อลองไลบรารีเสริม

## เตรียม environment แยก

เอกสารทางการระบุ Linux/macOS และ Windows ผ่าน WSL ตรวจ [system requirements](https://mojolang.org/docs/requirements/) และ [คู่มือติดตั้ง](https://mojolang.org/install/) ก่อน คำสั่งต่อไปนี้สำหรับ shell บนระบบที่รองรับ; บน Windows ให้ใช้ shell ภายใน WSL ไม่ใช่ PowerShell

จากโฟลเดอร์ repo ในระบบที่รองรับ หลังติดตั้ง uv แล้ว:

```bash
uv venv .venv-mojo --python 3.13
uv pip install --python .venv-mojo/bin/python mojo==1.2.0 numpy==2.2.6 scikit-learn==1.7.1 scipy==1.16.1 joblib==1.5.1
.venv-mojo/bin/mojo --version
.venv-mojo/bin/python --version
```

แพ็กเกจ Mojo ถูกระบุรุ่นตามเอกสารที่ใช้เขียนตัวอย่าง แต่ environment เส้นทางนี้ยังไม่มี lockfile ที่ผ่านการ resolve และทดสอบจริง หากแพ็กเกจ/รุ่นนั้นไม่มีสำหรับแพลตฟอร์มของคุณ ให้ยึดคู่มือติดตั้งทางการและบันทึกรุ่นที่ได้; บทหลัก Python ยังเรียนต่อได้ตามปกติ

เพื่อให้ Mojo และ Python ใช้ environment เดียวกัน:

```bash
source .venv-mojo/bin/activate
mojo mojo/01-gradient-descent/main.mojo
python mojo/01-gradient-descent/reference.py
mojo mojo/02-python-interop/main.mojo
python mojo/02-python-interop/reference.py
```

Mojo ที่เรียก Python ต้องหา CPython dynamic library และแพ็กเกจเจอ หาก import Python ไม่สำเร็จ ให้ตรวจ [Calling Python from Mojo](https://mojolang.org/docs/manual/python/python-from-mojo/) ไม่ตั้ง path ไปยัง Python ของ Windows จาก WSL

## อ่านผลอย่างถูกต้อง

การเรียก scikit-learn จาก Mojo ยังเรียก Python library ผ่าน CPython ไม่ได้แปลง training kernel เป็น Mojo อัตโนมัติ การใช้ typed loop ก็ไม่ได้รับประกันว่าเร็วกว่าการคำนวณ NumPy ที่ใช้ native kernels อยู่แล้ว

หากทดลอง benchmark ให้ตรวจคำตอบเท่ากันก่อน ระบุ dtype/ขนาดข้อมูล แยก compile/startup ออกจากเวลาคำนวณ warm up รันซ้ำหลายครั้ง รายงาน median และรวมต้นทุนข้ามภาษาเมื่อสิ่งนั้นเกิดในการใช้งานจริง หลักสูตรนี้ยังไม่มีผล benchmark ที่ยืนยันว่า Mojo เร็วกว่า

