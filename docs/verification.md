# การตรวจซ้ำและผลที่วัดได้

[กลับหน้าแรก](../README.md) · [เริ่มเรียนจาก notebook](start-here.md)

ตรวจเมื่อ **24 ตุลาคม 2026 (Asia/Bangkok)** โดยสร้าง environment ชั่วคราวใหม่ด้วย Python แล้วติดตั้ง `requirements.txt` ด้วย **pip** บน Windows, CPython 3.13.7 การแยก environment ในรอบตรวจมีไว้ยืนยันว่าติดตั้งจากรายการแพ็กเกจได้ครบ ไม่ใช่ขั้นตอนบังคับสำหรับผู้เรียนที่ใช้ notebook ได้อยู่แล้ว

## ผลตรวจของฉบับนี้

| รายการ | ผล |
|---|---|
| ติดตั้ง requirements.txt ด้วย pip | ผ่านใน environment Windows ใหม่ และ `pip check` ผ่าน |
| ไฟล์ Python ของบทเรียน | 18/18 ผ่าน |
| Python block ที่คัดลอกจาก Markdown | 21/21 ผ่าน: 18 บทหลัก, 2 Python references ของ Mojo และ math primer |
| Notebook จาก kernel ใหม่ | 18/18 ผ่าน |
| Behavioral tests | 6/6 ผ่าน |
| README/notebook synchronization, local links และ notebook schema | ผ่าน: 33 Markdown files |
| ติดตั้ง Mojo ด้วย pip | ผ่านใน environment Linux ใหม่, CPython 3.12.3 และ `pip check` ผ่าน |
| Mojo compiler และ Python references | 2/2 บทผ่านด้วย Mojo 1.2.0 (2016d0cd) บน Ubuntu 24.04 ผ่าน WSL |

โค้ดบทเรียน ML ไม่เปลี่ยนจากฉบับก่อน การตรวจนี้ยืนยันการติดตั้งและรันด้วยรายการแพ็กเกจของ repo โดยไม่ติดตั้งแพ็กเกจทับ Python/Jupyter ที่ใช้เรียนอยู่ ผลแต่ละบท รุ่นแพ็กเกจ และ SHA-256 ของไฟล์อยู่ใน [verification-results.json](verification-results.json) ส่วนรายงานฉบับก่อนเก็บในประวัติ Git

`requirements.txt` ระบุรุ่น dependency หลัก ส่วน dependency ทางอ้อมเลือกตามความเข้ากันได้ของ Python และระบบปฏิบัติการ จึงบันทึกรุ่นที่ทดสอบจริงแยกไว้ในรายงาน

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

ผล Python รอบนี้มาจาก Windows และผล Mojo มาจาก Linux ผ่าน WSL ยังไม่ได้ตรวจบริการ Colab หรือ macOS ในรอบนี้ ผล Python 3.11/3.13 บน Windows/Linux และงาน Mojo บน GitHub ให้ดู [Actions](https://github.com/KOPE-SOLUTION/machine-learning/actions) ของ commit ที่ต้องการตรวจ

การตรวจอัตโนมัติใช้ CPU และ backend กราฟ Agg เพื่อบันทึกภาพ ไม่มีการเปลี่ยนไดรเวอร์หรือทดสอบ CUDA ผลนี้ไม่ใช่ benchmark เปรียบเทียบระบบปฏิบัติการ

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
