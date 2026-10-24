# 00 · ตั้ง environment และตรวจข้อมูลก่อนเริ่ม ML

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลา: 30–45 นาทีสำหรับตั้งเครื่องครั้งแรก บวกเวลาดาวน์โหลด · 10–15 นาทีถ้าติดตั้งพร้อมแล้ว

พื้นฐาน: ใช้ Python, notebook, NumPy, pandas และ Matplotlib ได้แล้ว ดู [ทักษะพื้นฐานและวิดีโอทบทวน](../../docs/prerequisites.md)

## ก่อนเริ่มบทนี้

บทนี้เริ่มจากแยก environment ของโปรเจกต์ ตรวจว่า terminal และ notebook ใช้ Python ตัวเดียวกัน แล้วนำทักษะ DataFrame เดิมมาอ่านข้อมูลของ scikit-learn ยังไม่มีการฝึกโมเดล เมื่อจบจะพร้อมเริ่มบท 01

## ทำไมต้องแยก environment

หากติดตั้งทุกอย่างลง Python ตัวเดียว การเปลี่ยนรุ่นแพ็กเกจเพื่องานหนึ่งอาจทำให้อีกงานรันไม่ได้ เราจึงสร้างโฟลเดอร์ `.venv` ให้โปรเจกต์นี้มี Python และแพ็กเกจของตัวเอง โฟลเดอร์นี้ไม่ส่งขึ้น Git; ส่งรายการแพ็กเกจและ lockfile เพื่อให้สร้างใหม่ได้

`venv` สร้าง environment, `pip` ติดตั้งแพ็กเกจ ส่วน `uv` จัดการทั้ง Python, environment และแพ็กเกจจากไฟล์โปรเจกต์ได้ เลือกหนึ่งวิธีด้านล่าง ไม่ต้องทำทั้งสองวิธี บน Windows ใช้ Ubuntu WSL 2 ตาม [คู่มือ WSL](../../docs/wsl.md) แล้ว clone ลง `~/projects/machine-learning` ก่อนเริ่ม

### ทางเลือก A: uv

ติดตั้ง uv ตาม [คู่มือทีละขั้น](../../docs/setup.md) แล้วเปิด Ubuntu terminal:

```bash
cd ~/projects/machine-learning
uv python install 3.12
uv sync --frozen --python 3.12
uv run --frozen python -c "import sys; print(sys.executable)"
uv run --frozen jupyter lab
```

`uv sync` อ่าน `uv.lock` แล้วสร้าง `.venv` พร้อมติดตั้งแพ็กเกจ `--frozen` ใช้ lockfile ที่มีอยู่ ส่วน `uv run` เรียกโปรแกรมใน environment นี้ให้ จึงไม่ต้อง activate เปิด URL ที่ Jupyter แสดงใน browser แล้วเปิด notebook บท 00

### ทางเลือก B: ฝึกสร้าง venv ด้วยตัวเอง

บน Ubuntu 24.04 ซึ่งมี Python 3.12 ให้ใช้คำสั่งต่อไปนี้จาก terminal (ไม่วางใน cell Python):

```bash
sudo apt update
sudo apt install -y python3-venv
cd ~/projects/machine-learning
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable)"
python -m jupyterlab
```

1. ตรวจ Python ว่าเป็นรุ่น 3.11–3.13 ที่หลักสูตรรองรับ ก่อนใช้สร้าง environment
2. `python3 -m venv .venv` สร้างโฟลเดอร์ environment ใน repo
3. `source .venv/bin/activate` ทำให้คำสั่ง `python` ใน terminal นี้เลือก Python ของโปรเจกต์
4. `python -m pip install ...` ติดตั้งแพ็กเกจลง Python ตัวเดียวกัน
5. `sys.executable` ควรชี้เข้า `.venv/bin/` เช่น `python` หรือ `python3` ก่อนเปิด Jupyter

หลังปิด Jupyter ด้วย Ctrl+C ใช้ `deactivate` เพื่อออกจาก environment ครั้งหน้าเข้า repo และ activate ใหม่ การ activate เปลี่ยน PATH เฉพาะ terminal ปัจจุบัน ไม่ได้ย้ายหรือลบแพ็กเกจ สำหรับ Windows Python และ macOS ดู [คู่มือติดตั้ง](../../docs/setup.md)

### Interpreter กับ notebook kernel ต้องตรงกัน

Interpreter คือ Python ที่รันโปรแกรม ส่วน kernel คือ process ที่รัน cell ใน notebook หากเปิด notebook ด้วย Python อีกตัว อาจ import ไม่ผ่านแม้เคยติดตั้งแล้ว โค้ดตรวจด้านล่างพิมพ์ `sys.executable` ให้เทียบกับ terminal ได้ เลือก kernel ของ `.venv` แล้ว Restart Kernel → Run All เมื่อต้องการทดสอบใหม่

## ตรวจสิ่งที่หลักสูตรต้องใช้

เมื่อตั้ง environment เสร็จแล้ว รันโปรแกรมด้านล่างจาก kernel ใหม่ บทนี้ตรวจ interpreter และข้อมูลที่ scikit-learn คืนมา โดยไม่ทบทวนวิธีใช้ notebook หรือ indexing ตั้งแต่ต้น

สัญญา input ของ estimator ส่วนใหญ่คือ X มี shape (n_samples, n_features) และ y มีคำตอบหนึ่งค่าต่อแถว NumPy shape ถูกต้องอย่างเดียวยังไม่พอ: ลำดับตัวอย่างใน X/y ต้องตรงกัน และคอลัมน์ตอน predict ต้องมีลำดับ/หน่วยเหมือนตอน fit

`load_iris(as_frame=True)` ทำให้ตรวจชื่อคอลัมน์และ index ได้ก่อนส่งข้อมูลเข้าโมเดล ส่วนรหัส 0,1,2 เป็น label ของสายพันธุ์ ไม่ใช่ค่าที่ควรนำมาเฉลี่ยหรือตีความเป็นลำดับ

## คำใหม่ที่ใช้กับตารางเดิม

ใน ML เราเรียกหนึ่งแถวว่า **sample** หรือตัวอย่าง และคอลัมน์ที่ใช้ทำนายว่า **feature** ส่วนคำตอบที่ต้องการทำนายคือ **target** เช่น หนึ่งแถวเป็นดอกไม้หนึ่งดอก ใช้ขนาดกลีบเป็น feature และสายพันธุ์เป็น target

| ชื่อในโค้ด | ความหมาย | รูปร่างในบทนี้ |
|---|---|---|
| `X` | ตาราง feature ที่ให้โมเดลดู | 150 แถว × 4 คอลัมน์ |
| `y` | คำตอบที่ตรงกับแต่ละแถว | 150 ค่า |
| `data.target_names` | ชื่อที่รหัส 0, 1, 2 แทนอยู่ | setosa, versicolor, virginica |

`sklearn` คือชื่อที่ใช้ import ไลบรารี **scikit-learn** ซึ่งมีเครื่องมือเตรียมข้อมูลและโมเดล ML ส่วน `estimator` คือ object ที่มีวิธี `fit` สำหรับเรียนจากข้อมูล จะได้ลองใช้จริงในบท 01–03

`assert` ตรวจเงื่อนไข: ถ้าถูกโปรแกรมทำต่อ ถ้าผิดจะเกิด `AssertionError` จึงไม่ต้องรอให้โมเดลทำนายผิดก่อนรู้ว่า shape ไม่ตรง อ่านชื่อคอลัมน์จาก `X.head()` จะพบ sepal (กลีบเลี้ยง) และ petal (กลีบดอก) อย่างละความยาว/ความกว้าง หน่วยเซนติเมตร

## รันจาก root ของ repo

```bash
uv run --frozen python lessons/00-setup/lesson.py
```

หากใช้ venv ให้ใช้ Python ของ environment นั้นแทน `uv run --frozen python`

<!-- CODE:START -->
```python
"""ตรวจสภาพแวดล้อมและรูปร่างข้อมูลก่อนเริ่มเรียน"""
# %% ตรวจรุ่นเครื่องมือ
import importlib.metadata
import platform
import sys

print("Interpreter:", sys.executable)
print("Python:", platform.python_version())
for package in ("numpy", "pandas", "scikit-learn", "matplotlib"):
    print(f"{package}: {importlib.metadata.version(package)}")

# %% อ่านข้อมูลที่ติดมากับ scikit-learn
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
X, y = data.data, data.target
assert X.shape == (150, 4)
assert y.shape == (150,)
print("X shape:", X.shape, "y shape:", y.shape)
print("Classes:", data.target_names)
print(X.head(3).to_string(index=False))
```
<!-- CODE:END -->

## ผลตรวจที่คาด

เห็น X shape (150,4), y shape (150,), ชื่อสามสายพันธุ์และรุ่นแพ็กเกจที่ตรงกับ requirements.txt เมื่อผ่านแล้วให้เริ่ม [บท 01](../01-ml-workflow/README.md)

## ตรวจความเข้าใจเรื่อง environment

ติดตั้ง scikit-learn ผ่าน terminal สำเร็จ แต่ notebook แจ้งว่าไม่มี `sklearn` ควรตรวจอะไรเป็นอันดับแรก? และเหตุใด clone repo ใหม่จึงต้องติดตั้งแพ็กเกจอีกครั้ง?

<details>
<summary>แนวคำตอบ</summary>

เทียบ `sys.executable` ของ terminal กับ notebook เพื่อดูว่าใช้ environment เดียวกันหรือไม่ แล้วเลือก kernel ให้ถูก `.venv` ไม่อยู่ใน Git และมี path/ไฟล์เฉพาะเครื่อง จึงสร้างใหม่จาก `uv.lock` หรือ `requirements.txt` หลัง clone ไม่จำเป็นต้องติดตั้งแพ็กเกจทับ Python ของระบบ

</details>

## ตรวจความเข้าใจเรื่องแถวข้อมูล

หากสุ่มลำดับเฉพาะ y แล้วแปลง X และ y เป็น NumPy array ก่อน fit จะเกิดอะไรขึ้น แม้ shape ยังถูกต้อง? อธิบายความต่างระหว่างการจับคู่ด้วย index กับการจับคู่ตามตำแหน่ง

<details>
<summary>แนวคำตอบ</summary>

โมเดลอาจเรียนจากคำตอบของคนละตัวอย่างโดยไม่เกิด shape error เพราะ array จับคู่ตามตำแหน่ง ควรเลือก/แยก X,y จากตารางหรือ index ชุดเดียวกัน และตรวจ alignment ก่อนแปลงเป็น array การสุ่มหรือ reset_index เพียงฝั่งเดียวไม่ได้แก้ความหมายของแถว

</details>

## อ่านต่อ

- [Python: venv](https://docs.python.org/3.12/library/venv.html)
- [uv: working on projects](https://docs.astral.sh/uv/guides/projects/)
- [scikit-learn: load_iris](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_iris.html)
- [ทักษะพื้นฐานและวิดีโอทบทวน](../../docs/prerequisites.md)
- [คู่มือติดตั้งของหลักสูตร](../../docs/setup.md)
