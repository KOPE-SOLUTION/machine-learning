# การตรวจซ้ำและผลที่วัดได้

[กลับหน้าแรก](../README.md) · [สร้าง notebook แล้วรันตามบทเรียน](start-here.md)

ฉบับนี้ปรับทางเริ่มเรียนให้ต่อเนื่องจากหลักสูตร Matplotlib: สร้าง `.ipynb` แล้วคัดลอกโค้ดมารัน บท 00 อ่านข้อมูล X/y โดยไม่มีการตรวจ environment และ notebook ที่แจกไม่มี cell เปลี่ยนโฟลเดอร์อัตโนมัติ ตรวจเมื่อ **24 ตุลาคม 2026 (Asia/Bangkok)** บน **Ubuntu 24.04 ผ่าน WSL, CPython 3.12.11**

## ผลตรวจของฉบับนี้

| รายการ | ผล |
|---|---|
| Notebook บท 00–17 | 18/18 ผ่าน: kernel ใหม่และโฟลเดอร์ว่างนอก repo แยกกันทุกบท |
| Cell ติดตั้ง scikit-learn ในคู่มือเริ่มเรียน | ผ่านใน environment ชั่วคราวที่มี NumPy/pandas/Matplotlib/IPython อยู่ก่อน |
| Python block บท 00 จาก README | ผ่านใน IPython หลังติดตั้งด้วย cell ข้างต้น; `pip check` ผ่าน |
| README/notebook synchronization, local links, notebook schema | ผ่าน: 33 Markdown files |
| โค้ด ML บท 01–17 | ไม่เปลี่ยนจาก commit `d61f247`; ตัดเฉพาะ cell ช่วยตั้ง working directory ออกจาก notebook |
| Environment ที่ใช้เรียนอยู่ | ไม่มีการติดตั้งเพิ่มระหว่างทดสอบคำสั่งติดตั้ง; ใช้ environment ชั่วคราว |

แต่ละ notebook เริ่มจากโฟลเดอร์ว่างนอก checkout ไม่มีไฟล์ข้อมูลหรือผลจากบทก่อน และไม่มี `PYTHONPATH` ชี้กลับมาที่ repo จึงตรวจว่ารันได้โดยไม่อาศัยโครงสร้างโฟลเดอร์ของหลักสูตร ใช้ CPU และ backend กราฟ Agg ในการตรวจอัตโนมัติ ส่วนภาพในหน้า notebook ของผู้เรียนขึ้นกับ backend ของโปรแกรมที่ใช้

หลักฐานและวิธีตรวจอยู่ใน [verification-results.json](verification-results.json) ที่ `simple_notebook_revision` ส่วน `material_sha256`, `lesson_source_sha256` และ `documentation_sha256` ระบุไฟล์ฉบับปัจจุบัน Hash ของ notebook ใช้ไฟล์แจกที่ยังไม่มี output หรือการตั้งค่า kernel ส่วนตัว

## หลักฐานก่อนการปรับครั้งนี้

`readme_learning_revision` เป็นผลรอบก่อน: Python block จาก README 18 บทรันผ่านใน process ใหม่และโฟลเดอร์ว่างแยกกัน ไม่ใช่การรัน notebook ฉบับปัจจุบัน

`native_wsl_readiness` เป็นผลเต็มหลักสูตรที่ commit `74058ee`: script 18 บท, Python block 21 ชุด, notebook 18 บท, behavioral tests 6 ข้อ และ Mojo 2 บท ผ่านบน CPython 3.12.11 รวมทั้ง Jupyter API และหน้า login ผ่าน localhost จาก Windows ตอบ HTTP 200

ผลบน CPython 3.12.3 อยู่ที่ `previous_setup_wsl_run` และ `setup_verification` ส่วน `mojo_compiler` เก็บผล compile/run และเทียบ Python ของรอบก่อน โค้ด Mojo ไม่เปลี่ยนและไม่ได้รัน compiler ซ้ำในรอบนี้

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

ยังไม่ได้รันฉบับนี้ในบริการ Colab หรือ macOS จึงไม่รับรองความเร็ว รุ่น Python หรือแพ็กเกจที่บริการจะจัดให้แต่ละครั้ง ส่วนผล Windows/Linux และ Python รุ่นอื่นบน GitHub ให้ดู [Actions](https://github.com/KOPE-SOLUTION/machine-learning/actions) ของ commit ที่ต้องการตรวจ มี job สำหรับ Python 3.11/3.13 และ Mojo 1.2.0 แยกกัน

การตรวจนี้ไม่ได้ติดตั้ง WSL ใหม่ เปลี่ยนไดรเวอร์ หรือทดสอบ CUDA workload หลักสูตร Python ใช้ CPU ได้ทั้งหมด ส่วน Mojo มีคู่มือติดตั้งแยก

ผล HTTP ของลิงก์ภายนอกใน `new_external_link_check` และ `previous_external_link_check` เป็นประวัติรอบก่อน ไม่ใช่การตรวจทุก URL ซ้ำรอบนี้ ลิงก์วิดีโอพื้นฐานใช้ตามช่องที่ระบุใน [prerequisites](prerequisites.md)

## รันตรวจเอง

จากโฟลเดอร์หลักใน WSL หลังติดตั้ง uv:

```bash
uv sync --frozen --python 3.12
uv run --frozen python scripts/sync_materials.py --check
uv run --frozen python scripts/check_materials.py
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python scripts/verify_course.py --mode all
uv run --frozen python lessons/16-digits-project/predict.py outputs/16-digits-project/example-input.csv
```

หากใช้ venv โดยตรง ให้แทน `uv run --frozen python` ด้วย `.venv/bin/python` ส่วน Mojo ติดตั้ง environment ตาม [คู่มือเสริม](../mojo/README.md) แล้วรัน:

```bash
.venv-mojo/bin/python scripts/verify_mojo.py
```

verify_course ใช้ backend กราฟ Agg และจำกัด thread แต่ละ script/README block มี timeout 180 วินาที ส่วน notebook มี timeout ต่อ cell 180 วินาที หาก log มีคำเตือนว่า Agg ไม่แสดงหน้าต่างกราฟ ให้เปิด PNG ที่บันทึกใน outputs/ แทน Notebook ที่รันแล้วและ log อยู่ใน outputs/verification/ ส่วน notebook ใน lessons/ ว่าง output เพื่อให้ทดลองใหม่ได้
