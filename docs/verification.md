# การตรวจซ้ำและผลที่วัดได้

[กลับหน้าแรก](../README.md) · [เริ่มเรียนจาก README](start-here.md)

ฉบับนี้ปรับทางเริ่มเรียนเป็นอ่าน README แล้วคัดลอกโค้ดไป notebook โดยใช้ Jupyter เป็นตัวอย่างและให้ Colab เป็นทางเลือก ตรวจเมื่อ **24 ตุลาคม 2026 (Asia/Bangkok)** บน **Ubuntu 24.04 ผ่าน WSL, CPython 3.12.11** ในระบบไฟล์ Linux ใช้แพ็กเกจหลักตาม uv.lock

## ผลตรวจของการปรับทางเริ่มเรียนครั้งนี้

| รายการ | ผล |
|---|---|
| Python block จาก README บท 00–17 | 18/18 ผ่าน แต่ละบทเริ่มจากโฟลเดอร์ว่างแยกกันนอก repo และ process ใหม่ |
| ความครบของโค้ดที่ก๊อปวาง | รันได้โดยไม่มีไฟล์ repo หรือตัวแปรของบทก่อนอยู่ใน working directory; มีแพ็กเกจหลักติดตั้งแล้ว |
| Cell `%pip` ในคู่มือเริ่มเรียน | ผ่านใน IPython ของ environment ชั่วคราว จากนั้นรันบท 00 ใน process ใหม่และ `pip check` ผ่าน |
| โค้ดบทเรียนและ code cell ใน notebook | ทั้ง 18 บทไม่เปลี่ยนจาก commit `74058ee`; แก้เฉพาะคำอธิบายและวิธีเริ่มเรียน |
| README/notebook synchronization, local links, notebook schema | ผ่าน: 33 Markdown files |
| Environment ที่ใช้เรียนอยู่ | ไม่ติดตั้งแพ็กเกจทับระหว่างตรวจ cell `%pip`; ใช้ environment ชั่วคราวแยกต่างหาก |

การตรวจบล็อก README รอบนี้ดึงข้อความระหว่าง code fence `python` จากเอกสารจริง รันด้วย CPU และ backend กราฟ Agg ในโฟลเดอร์ชั่วคราวคนละแห่งต่อบท ลบ `PYTHONPATH` จาก environment เพื่อไม่อาศัยโมดูลใน checkout ผลนี้ตรวจความครบของโค้ดและการสร้างไฟล์ ไม่ได้วัดความเร็วใน browser หรือรับรองทุก runtime ของ Colab

หลักฐานแต่ละบทและวิธีตรวจอยู่ใน [verification-results.json](verification-results.json) ที่ `readme_learning_revision` ส่วน `material_sha256` และ `documentation_sha256` ระบุไฟล์ฉบับที่แก้แล้ว โดย hash ของ notebook ใช้ไฟล์แจกที่ยังไม่มี output หรือการตั้งค่า kernel ของผู้ใช้

## ผลตรวจเต็มหลักสูตรก่อนแก้คำอธิบาย

ก่อนการแก้ครั้งนี้ checkout บน Linux ที่ commit `74058ee` ใช้ CPython 3.12.11 และผ่าน script 18 บท, Python block ใน Markdown 21 ชุด, notebook 18 บท, behavioral tests 6 ข้อ และ Mojo 2 บท Jupyter API บน WSL และหน้า login ผ่าน localhost จาก Windows ตอบ HTTP 200 หลักฐานสรุปอยู่ที่ `native_wsl_readiness`

รอบปัจจุบันไม่ได้รัน notebook ทั้งหมดหรือ compiler Mojo ซ้ำ เพราะตรวจเทียบแล้ว code cell, ไฟล์ Python และ Mojo ไม่เปลี่ยน ผลทดสอบฉบับก่อนบน CPython 3.12.3 เก็บที่ `previous_setup_wsl_run`, `setup_verification` และ `mojo_compiler` รวมถึงผล venv + pip, CLI predict และการเทียบ Mojo กับ Python จึงแยกจากการตรวจ README รอบนี้อย่างชัดเจน

ผลตั้ง environment รอบก่อนครอบคลุม uv บนโฟลเดอร์ทดลอง `/mnt/c` และ venv + pip บน Linux `/tmp` โดยรันบท 00, JupyterLab และ pip check ตามวิธีที่ใช้ การติดตั้ง pip บน `/mnt/c` ครั้งแรกถูกหยุดแล้วเริ่มใหม่บน Linux จนผ่าน; ไม่ใช่ benchmark เปรียบเทียบความเร็วอย่างเป็นระบบ

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
