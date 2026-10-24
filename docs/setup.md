# ติดตั้งไลบรารีและเปิด notebook

[กลับหน้าแรก](../README.md) · [สร้าง notebook แล้วเริ่มเรียน](start-here.md)

ถ้าเปิด `.ipynb` และรัน NumPy, pandas และ Matplotlib ได้อยู่แล้ว ใช้โปรแกรมเดิมต่อได้เลย หลักสูตรเพิ่ม scikit-learn สำหรับเรียน ML และใช้ CPU ได้ทั้งหมด

## เพิ่ม scikit-learn เมื่อยังไม่มี

หากพบ `No module named sklearn` ให้รันใน Code cell:

```python
%pip install scikit-learn==1.7.1
```

ติดตั้งแล้ว restart kernel และรันโค้ดใหม่ตั้งแต่ต้น ชื่อตอนติดตั้งคือ `scikit-learn` แต่ตอน import ใช้ `sklearn` หากรันผ่านอยู่แล้วไม่ต้องติดตั้งซ้ำ

## เครื่องใหม่ที่ยังไม่มีไลบรารี

ติดตั้ง [Python](https://www.python.org/downloads/) รุ่น 3.11–3.13 แล้วเปิด terminal เพื่อติดตั้งไลบรารีที่ใช้ในหลักสูตร:

```bash
python -m pip install numpy==2.2.6 pandas==2.3.2 matplotlib==3.10.3 scikit-learn==1.7.1 scipy==1.16.1 joblib==1.5.1 jupyterlab==4.4.6 ipykernel==6.30.1
python -m jupyterlab
```

คำสั่ง `python -m jupyterlab` เปิด Jupyter ใน browser จากนั้นสร้าง notebook ใหม่ เช่น `ml-00.ipynb` แล้วคัดลอกโค้ดจาก [บท 00](../lessons/00-setup/README.md) มารัน คำสั่งส่วนนี้ใช้ใน terminal; `%pip` ด้านบนใช้ใน notebook

หากใช้ VS Code เปิด `.ipynb` แล้วเลือก Python/kernel ที่ติดตั้งไลบรารีไว้ บน macOS หรือ Linux บางเครื่องใช้ชื่อคำสั่ง `python3` แทน `python`

## ใช้ไฟล์จาก repo

ถ้า clone หรือดาวน์โหลด repo มาแล้ว เปิด terminal ที่โฟลเดอร์หลักและติดตั้งจากรายการแพ็กเกจได้:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

`requirements.txt` ระบุรุ่นไลบรารีหลัก รวมเครื่องมือ notebook และตัวตรวจหลักสูตร แพ็กเกจทางอ้อมอาจต่างกันตาม Python และระบบปฏิบัติการ อ่าน [ผลทดสอบ](verification.md) เพื่อดูรุ่นที่ตรวจจริง

ผู้ที่ต้องการรันไฟล์ `.py` แทน notebook ใช้ Python ตัวเดียวกัน:

```bash
python lessons/01-ml-workflow/lesson.py
```

บท 16 มีคำสั่งฝึกและทำนายจากไฟล์ CSV เพิ่มเติม:

```bash
python lessons/16-digits-project/lesson.py
python lessons/16-digits-project/predict.py outputs/16-digits-project/example-input.csv
```

CSV ไม่มี header และเรียง pixel แบบ row-major ตามภาพ 8×8 ตัวอย่างนี้มาจาก holdout เพื่อสาธิตการโหลดโมเดล ไม่ใช่การประเมินกับข้อมูลภายนอก

## เมื่อรันไม่ได้

| อาการ | วิธีตรวจ |
|---|---|
| `No module named sklearn` | ติดตั้ง scikit-learn ใน notebook ที่ใช้ แล้ว restart kernel |
| ติดตั้งแล้วแต่ import ไม่ได้ | ตรวจว่า notebook เลือก Python ตัวเดียวกับที่ติดตั้ง; รัน `import sys; print(sys.executable)` ใน cell เพื่อดูตำแหน่ง |
| `NameError` | คัดลอก import และฟังก์ชันครบ แล้วรันตั้งแต่ cell แรก |
| บันทึก notebook หรือรูปไม่ได้ | เลือกโฟลเดอร์ที่ Python เขียนได้ เช่น `C:\Users\ชื่อผู้ใช้\Projects\machine-learning` บน Windows |
| `externally-managed-environment` บน Linux | ใช้ environment แยกตามทางเลือกด้านล่าง |

<details>
<summary>ทางเลือก: แยกแพ็กเกจออกจากโปรเจกต์อื่นด้วย venv</summary>

ส่วนนี้ใช้เมื่อต้องการแยกแพ็กเกจของแต่ละงาน ไม่ใช่ขั้นตอนที่ต้องทำก่อนเรียนทุกบท

Windows PowerShell จากโฟลเดอร์ repo:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m jupyterlab
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab
```

บน Ubuntu หากไม่มีโมดูล venv ให้ติดตั้ง `python3-venv` ก่อน การเรียก Python ด้วยตำแหน่งข้างต้นไม่ต้อง activate และเมื่อย้ายเครื่องควรสร้าง environment ใหม่จาก requirements.txt แทนการคัดลอกโฟลเดอร์ `.venv`

</details>

## อ้างอิง

- [pip: ติดตั้งแพ็กเกจและใช้ requirements files](https://pip.pypa.io/en/stable/user_guide/)
- [IPython: คำสั่ง %pip ใน notebook](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-pip)
- [Python: venv](https://docs.python.org/3.13/library/venv.html)

Google Colab ใช้ขั้นตอนสร้าง notebook ตาม [คู่มือเริ่มเรียน](start-here.md) ส่วนผู้ที่เลือก Linux บน Windows หรือบทเสริม Mojo มี [คู่มือ WSL](wsl.md) และ [คู่มือ Mojo](../mojo/README.md) แยกไว้
