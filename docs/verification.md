# การตรวจซ้ำและผลที่วัดได้

[กลับหน้าแรก](../README.md)

ฉบับปรับพื้นฐานผู้เรียนให้ต่อจาก Handbook ส่วนที่ 1–4 ตรวจเมื่อ **24 ตุลาคม 2026 (Asia/Bangkok)** บน **Ubuntu 24.04 ผ่าน WSL, CPython 3.12.3** ใช้แพ็กเกจตาม uv.lock และ uv 0.8.18 ใน environment แยกจาก Windows

ไฟล์หลักสูตรและผลการรันอยู่ในโฟลเดอร์ repo จริง การสร้าง/ปรับไฟล์ในรอบนี้ทำผ่าน WSL ส่วนหลักฐานจากฉบับก่อนที่รันบน Windows 11 / CPython 3.13.7 เก็บแยกเป็น previous_windows_revision ไม่ถือเป็นการตรวจโค้ดบท 02 ฉบับใหม่บน Windows

## ผ่านการรันจริงใน WSL

| รายการ | ผล |
|---|---|
| Python lesson.py | 18/18 ผ่าน: 17 บท ML และ 1 บทตรวจความพร้อม |
| Jupyter notebook | 18/18 ผ่านจาก kernel ใหม่ของแต่ละบท |
| Behavioral tests | 4/4 ผ่าน: gradient, input validation, train-only scaling และ serialization |
| Python reference ของ Mojo | 2/2 ผ่าน |
| CLI predict.py ของโครงงาน | ผ่าน: อ่าน CSV สามแถวและทาย [5, 2, 8] |
| README code synchronization / local links / notebook schema | ผ่าน รวมเอกสาร prerequisites ฉบับใหม่ |
| ลิงก์อ้างอิงภายนอก | 66/66 URL เปิดได้ ณ วันที่ตรวจ |
| บท 02 ฉบับปรับปรุง | ผ่านการรันและสร้าง feature-audit.csv กับกราฟใหม่ |

หลักฐานผลการรัน environment, metric และ SHA-256 ของ lesson.py อยู่ใน [verification-results.json](verification-results.json) ที่ current_wsl_run ส่วน previous_windows_revision คือประวัติฉบับก่อนปรับเนื้อหา

## ตัวอย่างผลที่สังเกตได้

- บท 02 ใช้ feature ขนาดดอกไม้สี่ค่า และตัด target, species_name, sample_id ออกจาก X
- บท 02 กัน train 120 แถว / test 30 แถวก่อนใช้ EDA ตัดสินใจเกี่ยวกับโมเดล
- Iris KNN: test accuracy 0.9333 และ Python interop reference ทายห้าตัวแรก [0, 2, 1, 1, 0]
- Digits: train 1,437 ภาพ / test 360 ภาพ, baseline accuracy 0.1000
- Digits: best CV macro F1 0.9804, final test accuracy 0.9806, final test macro F1 0.9805
- Gradient descent reference ห้าจุด: w≈3, b≈2 และ MSE ใกล้ศูนย์

ค่าคะแนนเป็นจุดตรวจความสมเหตุสมผล ไม่ใช่ข้อกำหนดให้ floating point ตรงทุกหลัก หรือการรับรองผลกับข้อมูลภายนอก อ่านข้อจำกัดใน [model card](model-card.md)

## ยังไม่ได้ตรวจ

- Mojo compiler และการรัน .mojo ทั้งสองไฟล์ แม้ตอนนี้ใช้ WSL ทำงานกับ repo ได้แล้ว
- Colab, macOS และ Python 3.11
- โค้ดบท 02 ฉบับปรับปรุงบน Windows native; ฉบับนี้ตรวจใหม่ด้วย WSL
- GitHub Actions บน remote: มี workflow เตรียมไว้ แต่ยังไม่มีผล CI ของ repo ที่เผยแพร่

## รันตรวจเอง

สำหรับ uv ที่ติดตั้งใน WSL:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync --frozen
uv run --frozen python scripts/sync_materials.py --check
uv run --frozen python scripts/check_materials.py
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python scripts/verify_course.py --mode all
uv run --frozen python mojo/01-gradient-descent/reference.py
uv run --frozen python mojo/02-python-interop/reference.py
uv run --frozen python lessons/16-digits-project/predict.py outputs/16-digits-project/example-input.csv
```

หากใช้ venv โดยตรง ให้แทน uv run --frozen python ด้วย .venv-wsl/bin/python สำหรับ Windows native ใช้ environment ของ Windows ตาม [setup](setup.md)

verify_course ใช้ backend กราฟ Agg และจำกัด thread แต่ละ script มี timeout 180 วินาที ส่วน notebook มี timeout ต่อ cell 180 วินาที Notebook ที่รันแล้วและ log อยู่ใน outputs/verification/ ส่วน notebook ใน lessons/ ว่าง output เพื่อให้ทดลองใหม่ได้
