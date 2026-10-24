# ติดตั้งและแก้ปัญหา

[กลับหน้าแรก](../README.md)

เอกสารนี้ใช้เตรียม environment สำหรับหลักสูตร ML โดยถือว่าผู้เรียนใช้ Jupyter notebook, NumPy, pandas และ Matplotlib ได้แล้ว หาก environment พร้อม ให้ตรวจบท 00 แล้วเริ่มบท 01

## วิธี A: uv — ใช้ lockfile

เปิด terminal ในโฟลเดอร์ที่มี pyproject.toml ใช้ Python 3.11–3.13 รุ่น 64-bit:

```bash
uv sync --frozen
uv run --frozen python lessons/00-setup/lesson.py
uv run --frozen jupyter lab
```

uv สร้าง .venv ภายใน repo และไม่ต้อง activate เอง หากเครื่องยังไม่มี Python รุ่นที่รองรับ สามารถใช้ `uv python install 3.13` แล้ว `uv sync --frozen --python 3.13` ได้หลังติดตั้ง uv ดู [เอกสาร uv](https://docs.astral.sh/uv/guides/projects/)

## วิธี B: Python venv + pip

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe lessons/00-setup/lesson.py
.\.venv\Scripts\python.exe -m jupyterlab
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python lessons/00-setup/lesson.py
.venv/bin/python -m jupyterlab
```

ไม่ต้องเปลี่ยน PowerShell execution policy เพราะไม่ได้ใช้ Activate.ps1 หากเลือกวิธีนี้ คำสั่งรันในบทเรียนให้แทน `uv run --frozen python` ด้วย path ของ Python ใน .venv ข้างต้น requirements.txt ล็อก dependency หลัก ส่วนการล็อกทั้ง dependency ทางอ้อมใช้วิธี uv

## ใช้ Windows ร่วมกับ WSL

หากทำงานผ่าน WSL ให้เปิด terminal ของ Linux แล้วเข้าโฟลเดอร์ repo ผ่าน `/mnt/c/...` การสร้างหรือแก้ไฟล์ใน repo จะเกิดผ่าน WSL ได้ ส่วน environment ของ Windows และ Linux ต้องแยกกัน เพราะ executable และแพ็กเกจ native ใช้ร่วมกันไม่ได้

หากติดตั้ง uv ใน WSL แล้ว ใช้ชื่อ environment แยกจาก Windows:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync --frozen
uv run --frozen python lessons/00-setup/lesson.py
uv run --frozen jupyter lab
```

หากใช้ venv + pip แทน uv:

```bash
python3 -m venv .venv-wsl
.venv-wsl/bin/python -m pip install -r requirements.txt
.venv-wsl/bin/python lessons/00-setup/lesson.py
.venv-wsl/bin/python -m jupyterlab
```

ใช้ environment เดียวกันตลอดการรัน notebook และสคริปต์ หาก kernel ของ Jupyter ใช้ Python คนละฝั่งกับที่ติดตั้งแพ็กเกจ ให้เปลี่ยน kernel ก่อนทดลอง

## Notebook และ kernel

เปิด JupyterLab จาก environment เดียวกับที่ติดตั้งแพ็กเกจแล้ว เลือก Python 3 และ Restart Kernel → Run All เมื่อทดสอบใหม่ หากเปิดจาก VS Code ให้เลือก interpreter ใน .venv ของ repo หรือ .venv-wsl เมื่อใช้ WSL

## คัดลอกโค้ดจาก README

1. ติดตั้งแพ็กเกจด้วยวิธีใดวิธีหนึ่งข้างต้น แล้วเปิด notebook ด้วย environment นั้น
2. คัดลอกบล็อกที่ระบุภาษา `python` ทั้งชุด รวม import และฟังก์ชัน ไป cell เดียวใน notebook เปล่าแล้วรัน หรือบันทึกเป็นไฟล์ `.py` แล้วรันจาก terminal
3. บล็อก `bash`/`powershell` เป็นคำสั่ง terminal ไม่วางใน cell Python ตรง ๆ ส่วนบล็อก `mojo` ให้บันทึกเป็นไฟล์ `.mojo` ตามคู่มือเสริม
4. รูปและไฟล์ที่สร้างจะอยู่ใน `outputs/` ใต้ working directory ปัจจุบัน หากต้องการให้ตรงเส้นทางในคู่มือ ให้เริ่มที่ root ของ repo
5. หากทดลองเปลี่ยนตัวแปร ให้ Restart Kernel → Run All ก่อนสรุปผล เพื่อไม่ใช้ object ที่ fit จากการทดลองก่อนหน้าโดยไม่ตั้งใจ

README แต่ละบทมีข้อมูลตัวอย่างอยู่ในโค้ดหรือ loader ของ scikit-learn ครบ ไม่ต้องหา CSV เพิ่ม บท 16 สร้าง input CSV ตัวอย่างหลังฝึกเสร็จ ให้รันคำสั่งทำนายหลังเห็นไฟล์นั้นแล้ว

Notebook มี cell แรกสำหรับกลับไป working directory ของ repo เมื่อเปิดจากโฟลเดอร์บทเรียน โค้ดตัวอย่างไม่จำเป็นต้องรันบทก่อนหน้า หากคัดลอกไป notebook เปล่า ให้รัน Python block ทั้งชุดตามลำดับ

## Google Colab

ดาวน์โหลด lesson.ipynb ของบทที่สนใจ แล้วเลือก Upload notebook ใน [Colab](https://colab.research.google.com/) เพิ่ม cell แรกด้วยคำสั่ง:

```python
%pip install numpy==2.2.6 pandas==2.3.2 scikit-learn==1.7.1 matplotlib==3.10.3 scipy==1.16.1 joblib==1.5.1
```

ถ้า Colab แจ้งว่าต้อง restart session ให้ restart ก่อน Run All runtime ควรใช้ Python 3.11–3.13; Colab อาจเปลี่ยนเวอร์ชันในอนาคต จึงใช้ environment ในเครื่องเป็นทางเลือกหลักที่ล็อกได้ แนวทาง Colab ยังไม่ได้ทดสอบในรอบตรวจนี้

ไฟล์ใน outputs/ ของ Colab เป็นไฟล์ชั่วคราว ต้องดาวน์โหลดก่อนจบ session บท 16 รันฝึกใน notebook ได้ครบ แต่ CLI predict.py ต้องดาวน์โหลดไฟล์ lesson.py และ predict.py ให้อยู่ด้วยกันด้วย เส้นทาง Mojo ไม่รวมอยู่ในขั้นตอน Colab นี้

## เมื่อเกิดข้อผิดพลาด

| อาการ | ตรวจและแก้ |
|---|---|
| No module named sklearn | ใช้ Python จาก .venv ที่ติดตั้งแล้ว ชื่อแพ็กเกจติดตั้งคือ scikit-learn |
| เปิด notebook แล้ว import ไม่ผ่าน | ตรวจ kernel ว่าเป็น environment ของ repo |
| File not found | เริ่มคำสั่งที่โฟลเดอร์หลักซึ่งมี pyproject.toml |
| รูปไม่เด้งเป็นหน้าต่าง | เปิดไฟล์ PNG ใน outputs/ หรือใช้ notebook |
| รัน terminal แล้วค้างหลังวาดรูป | ปิดหน้าต่างกราฟเพื่อให้สคริปต์ทำงานต่อ |
| คะแนนต่างทศนิยมเล็กน้อย | ตรวจเวอร์ชัน แพลตฟอร์ม และ seed; ผล floating point อาจต่างได้ |
| uv --frozen แจ้งไม่มี lockfile | ตรวจว่าแตก/ดาวน์โหลด repo ครบ รวม uv.lock |
| TLS/เครือข่ายติดตั้งไม่ได้ | ตรวจ proxy/เครือข่ายแล้วติดตั้งใหม่ ไม่ปิดการตรวจ certificate |

ใช้ `python -c "import sys; print(sys.executable)"` ภายใน environment เพื่อตรวจ interpreter และใช้บท 00 ตรวจเวอร์ชันแพ็กเกจ

## แก้โค้ดแล้วปรับ README/notebook

ไฟล์ lesson.py เป็นแหล่งโค้ดหลัก หากแก้โค้ดให้รัน:

```bash
uv run --frozen python scripts/sync_materials.py
uv run --frozen python scripts/sync_materials.py --check
uv run --frozen python scripts/verify_course.py --mode readmes
```

คำสั่งสร้าง notebook ใหม่จะล้าง output และการแก้ cell ที่ทำเฉพาะใน notebook หากต้องการเก็บการทดลองส่วนตัว ให้ Save As ชื่ออื่นก่อน


โหมด `readmes` รันบล็อก Python จาก README ทุกบทและตัวอย่างคณิตศาสตร์ใน process ใหม่แต่ละบล็อก ส่วน `--mode all` รวมสคริปต์ บล็อก README และ notebook การตรวจ Mojo ใช้ `python scripts/verify_mojo.py` จาก environment เสริมตาม [คู่มือ Mojo](../mojo/README.md)
