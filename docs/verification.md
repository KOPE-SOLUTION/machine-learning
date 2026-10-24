# การตรวจซ้ำและผลที่วัดได้

[กลับหน้าแรก](../README.md) · [เริ่มเรียนจาก notebook](start-here.md)

ตรวจการปรับบทนำและบทเรียนแรกเมื่อ **24 ตุลาคม 2026 (Asia/Bangkok)** บน Windows, CPython 3.13.7 โดยใช้ Python/Jupyter ที่ติดตั้งอยู่เดิม โค้ดที่เปลี่ยนคือบท 00 และบท 01 ส่วนบทอื่นและ dependency ไม่เปลี่ยนในรอบนี้

## ผลตรวจของฉบับนี้

| รายการ | ผล |
|---|---|
| ไฟล์ Python ของบท 00–01 | 2/2 ผ่าน |
| Python block ที่คัดลอกจาก README บท 00–01 | 2/2 ผ่าน |
| Notebook จาก kernel ใหม่ | 2/2 ผ่าน: บท 00 มี 1 code cell, บท 01 มี 10 code cells |
| ผลลัพธ์ที่ตรวจ | บท 00 แสดงข้อความพร้อมใช้งาน; บท 01 ได้ train 120, test 30 และทายถูก 10/30 หรือ accuracy 0.333 |
| README/notebook synchronization | ผ่านทั้ง repo |
| Local links และ notebook schema | ผ่าน: 33 Markdown files |

ทั้งหกรูปแบบการรันใช้ working directory ชั่วคราวที่ว่าง เพื่อยืนยันว่าไม่อาศัยไฟล์หรือสถานะจากบทก่อนหน้า ตัวตรวจใช้ kernel ของ Python ที่ระบุโดยตรง และไม่เขียนผลทับ notebook ต้นฉบับ รายละเอียด รุ่นแพ็กเกจ และ SHA-256 ของไฟล์ที่รันอยู่ใน [verification-results.json](verification-results.json)

## ผลตรวจเต็มหลักสูตรก่อนปรับบทนำ

[รายงานฉบับ cdd842c](https://github.com/KOPE-SOLUTION/machine-learning/blob/cdd842c75aaa0114b254099997196b8cd33a0f30/docs/verification-results.json) บันทึกการตรวจเดิมเมื่อ 24 ตุลาคม 2026: ติดตั้ง requirements ด้วย pip ใน environment Windows ใหม่ รัน 18 scripts, 21 Python blocks และ 18 notebooks ผ่านทั้งหมด พร้อม behavioral tests 6 ข้อ ส่วน Mojo ผ่าน 2 บทใน environment Linux ใหม่ด้วย Mojo 1.2.0 (2016d0cd) บน Ubuntu 24.04 ผ่าน WSL

ผลเดิมเป็นหลักฐานของฉบับนั้น ส่วนรอบนี้รันซ้ำเฉพาะบท 00–01 ที่แก้ไขและตรวจความสอดคล้องของเอกสารทั้ง repo ผลตรวจเต็มชุดของ commit ล่าสุดให้ดู [GitHub Actions](https://github.com/KOPE-SOLUTION/machine-learning/actions)

`requirements.txt` ระบุรุ่น dependency หลัก ส่วน dependency ทางอ้อมเลือกตามความเข้ากันได้ของ Python และระบบปฏิบัติการ จึงบันทึกรุ่นที่ทดสอบจริงแยกไว้ในรายงาน

## ตัวอย่างผลที่ใช้ตรวจความเข้าใจ

บท 01 ยืนยันซ้ำในรอบนี้ ส่วนผลบทอื่นด้านล่างมาจากรายงานเต็มหลักสูตรข้างต้น และโค้ดของบทเหล่านั้นยังไม่เปลี่ยน

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

ผลบท 00–01 รอบนี้มาจาก Windows ยังไม่ได้รันซ้ำบน Colab, macOS หรือ Linux ในรอบนี้ ผล Python 3.11/3.13 บน Windows/Linux และงาน Mojo บน GitHub ให้ดู [Actions](https://github.com/KOPE-SOLUTION/machine-learning/actions) ของ commit ที่ต้องการตรวจ

ตัวตรวจเต็มหลักสูตรใช้ CPU และ backend กราฟ Agg เพื่อบันทึกภาพ ไม่มีการเปลี่ยนไดรเวอร์หรือทดสอบ CUDA ผลนี้ไม่ใช่ benchmark เปรียบเทียบระบบปฏิบัติการ

## รันตรวจเอง

จาก terminal ที่โฟลเดอร์หลักของ repo หลังติดตั้ง Python:

```bash
python -m pip install -r requirements.txt
python -m pip check
python scripts/sync_materials.py --check
python scripts/check_materials.py
python -m unittest discover -s tests -v
python scripts/verify_course.py --mode all
```

คำสั่งสุดท้ายรันไฟล์ Python, Python block ใน Markdown และ notebook ครบชุด บันทึก log และผลที่รันแล้วใน `outputs/verification/` โดยไม่แก้ output ของ notebook ต้นฉบับใน `lessons/`

หากใช้ environment แยก ให้เรียก Python ของ environment นั้นแทน `python` ส่วน Mojo ติดตั้งตาม [คู่มือเสริม](../mojo/README.md) แล้วใช้:

```bash
.venv-mojo/bin/python scripts/verify_mojo.py
```
