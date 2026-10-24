# การตรวจซ้ำและผลที่วัดได้

[กลับหน้าแรก](../README.md)

ฉบับปรับคำอธิบายทุกบทสำหรับผู้ที่ใช้ NumPy, pandas และ Matplotlib ได้แล้วและเริ่มเรียน ML ตรวจเมื่อ **24 ตุลาคม 2026 (Asia/Bangkok)** บน **Ubuntu 24.04 ผ่าน WSL, CPython 3.12.3** ใช้แพ็กเกจหลักตาม uv.lock และ Mojo ใน environment แยก

## ผลการรันฉบับนี้

| รายการ | ผล |
|---|---|
| Python lesson.py | 18/18 ผ่าน: 17 บท ML และ 1 บทตรวจความพร้อม |
| บล็อก Python ที่คัดลอกจาก Markdown | 21/21 ผ่าน: 18 บทหลัก + 2 Python references ของ Mojo + ตัวอย่าง math primer |
| Jupyter notebook | 18/18 ผ่านจาก kernel ใหม่ของแต่ละบท |
| Behavioral tests | 6/6 ผ่าน: gradient, invalid settings, overflow, pixel schema, train-only scaling และ serialization |
| Mojo compiler | 2/2 บท compile/run ผ่านด้วย Mojo 1.2.0 (2016d0cd) |
| Mojo เทียบ Python reference | พารามิเตอร์/MSE และผลจำแนกตรงกันภายใน tolerance ที่กำหนด |
| CLI predict.py | ผ่าน: อ่าน CSV สามแถวและทาย [5, 2, 8] |
| README/notebook synchronization, local links, notebook schema | ผ่าน |

โหมด `readmes` ดึงบล็อก Python จาก Markdown จริงแล้วรันใน process ใหม่แต่ละบล็อก จึงไม่อาศัยตัวแปรจากบทก่อน สำหรับ Mojo ตรวจให้โค้ดใน README ตรงกับไฟล์ .mojo ก่อน compile และรันเทียบ reference

หลักฐานรุ่นแพ็กเกจ ผลแต่ละบท และ SHA-256 ของโค้ด/เอกสารอยู่ใน [verification-results.json](verification-results.json) ที่ `current_wsl_run` และ `mojo_compiler` หลักฐานฉบับก่อนอยู่ใน `previous_wsl_revision` กับ `previous_windows_revision`; ไม่ถือเป็นการรันฉบับปัจจุบันบนทุกแพลตฟอร์ม

## ตัวอย่างผลที่ใช้ตรวจความเข้าใจ

- บท 01: baseline ทายถูก 10 จาก 30 ดอก หรือ accuracy ประมาณ 0.333
- บท 03: KNN accuracy ประมาณ 0.933 พร้อม baseline สำหรับเปรียบเทียบ
- บท 04: test confusion matrix เป็น [[263, 4], [17, 16]] อ่านได้ว่า TP=16, FN=17 และ F1 ประมาณ 0.604
- บท 06: baseline MAE ประมาณ 53.511 เทียบ Ridge ประมาณ 6.494 ในหน่วยค่าบริการจำลอง พร้อมตาราง feature หลังแปลง
- บท 10: ตาราง CV มีหกคู่พารามิเตอร์ `mean_test_score` หมายถึง validation ภายใน CV ไม่ใช่ final test
- บท 16: train 1,437 ภาพ / test 360 ภาพ, test accuracy ประมาณ 0.9806 และ macro F1 ประมาณ 0.9805
- บท 17: random split มีคนซ้ำสองฝั่ง 41 คนและ accuracy 1.000; group split ไม่มีคนซ้ำและ accuracy ประมาณ 0.383 จึงเห็นว่าคะแนนตอบคำถามการใช้งานต่างกัน
- M01: w≈3, b≈2 และ training MSE ประมาณ 1.59e-27 ตรงกับ Python
- M02: accuracy ประมาณ 0.933333 และคำทำนายห้าตัวแรก [0, 2, 1, 1, 0] ตรงกับ Python

ค่าคะแนนใช้ตรวจความสมเหตุสมผล ไม่ต้องให้ floating point ตรงทุกหลัก และไม่ใช่ผลรับรองกับข้อมูลภายนอก อ่านข้อจำกัดโครงงานใน [model card](model-card.md)

## ขอบเขตผลตรวจ

ผลในหน้านี้เป็นการรันบน WSL ของฉบับนี้ ยังไม่ได้รันฉบับนี้ใน Colab หรือ macOS ส่วนผล Windows/Linux และ Python รุ่นอื่นบน GitHub ให้ดู [Actions](https://github.com/KOPE-SOLUTION/machine-learning/actions) ของ commit ที่ต้องการตรวจ มี job สำหรับ Python 3.11/3.13 และ Mojo 1.2.0 แยกกัน

ประวัติการเปิดลิงก์ภายนอก 66 URL เก็บไว้ใน `previous_external_link_check` ไม่ใช่การตรวจลิงก์ทั้งหมดซ้ำรอบนี้ ลิงก์วิดีโอพื้นฐานใช้ตามช่องที่ระบุใน [prerequisites](prerequisites.md)

## รันตรวจเอง

จากโฟลเดอร์หลักใน WSL หลังติดตั้ง uv:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync --frozen
uv run --frozen python scripts/sync_materials.py --check
uv run --frozen python scripts/check_materials.py
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python scripts/verify_course.py --mode all
uv run --frozen python lessons/16-digits-project/predict.py outputs/16-digits-project/example-input.csv
```

หากใช้ venv โดยตรง ให้แทน `uv run --frozen python` ด้วย `.venv-wsl/bin/python` ส่วน Mojo ติดตั้ง environment ตาม [คู่มือเสริม](../mojo/README.md) แล้วรัน:

```bash
.venv-mojo/bin/python scripts/verify_mojo.py
```

verify_course ใช้ backend กราฟ Agg และจำกัด thread แต่ละ script/README block มี timeout 180 วินาที ส่วน notebook มี timeout ต่อ cell 180 วินาที หาก log มีคำเตือนว่า Agg ไม่แสดงหน้าต่างกราฟ ให้เปิด PNG ที่บันทึกใน outputs/ แทน Notebook ที่รันแล้วและ log อยู่ใน outputs/verification/ ส่วน notebook ใน lessons/ ว่าง output เพื่อให้ทดลองใหม่ได้
