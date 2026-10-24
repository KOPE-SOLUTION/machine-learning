# ใช้ Ubuntu WSL เรียน ML บน Windows

[กลับหน้าแรก](../README.md) · [สร้าง environment](setup.md)

สำหรับหลักสูตรนี้ใช้ Ubuntu 24.04 บน **WSL 2**, Python 3.12 และ CPU ได้ครบทั้งบทหลักและตัวอย่าง Mojo เสริม เก็บโปรเจกต์ใน `~/projects/machine-learning` ของ Ubuntu แล้วใช้ Python, Git และ terminal ฝั่ง Ubuntu ตลอดงาน

## เก็บไฟล์ไว้ตรงไหนดี

| ตำแหน่ง | ความหมายและผลต่อการทำงาน |
|---|---|
| `~/projects/machine-learning` | อยู่ในระบบไฟล์ Linux ของ WSL เหมาะกับการใช้เครื่องมือ Linux; ลดการอ่านเขียนข้ามระบบไฟล์ |
| `/mnt/c/Users/.../machine-learning` | อยู่บนไดรฟ์ Windows แม้เรียกคำสั่งผ่าน WSL; ยังมีเรื่อง permission และการอ่านเขียนข้ามระบบไฟล์ |

[Microsoft แนะนำให้เก็บไฟล์ในระบบเดียวกับเครื่องมือที่ใช้](https://learn.microsoft.com/en-us/windows/wsl/filesystems) ประโยชน์ที่คาดได้คือการทำงานกับไฟล์จำนวนมาก เช่น ติดตั้งแพ็กเกจและ Git คล่องขึ้น แต่ไม่ได้หมายความว่าเวลาฝึกโมเดลทุกตัวจะเร็วขึ้น การแก้ปัญหา permission ของ Windows ด้วยตำแหน่งใหม่ก็ไม่ได้รับรองว่าไม่มีข้อจำกัดจากนโยบายเครื่องอื่น

ไฟล์ Linux ยังใช้พื้นที่บนดิสก์ของเครื่อง Windows ผ่าน virtual disk ของ WSL ต้องเผื่อพื้นที่สำหรับ repo, environment, cache และผลทดลอง หากเก็บทั้ง checkout เก่าและใหม่พร้อมกัน จะใช้พื้นที่เพิ่ม

## 1. เตรียม Ubuntu WSL 2

หากมี Ubuntu 24.04 / WSL 2 แล้ว ให้ข้ามการติดตั้ง เปิด **PowerShell** เพื่อตรวจ:

```powershell
wsl --list --verbose
```

ดูคอลัมน์ `VERSION` ของ Ubuntu ให้เป็น `2` สำหรับเครื่องที่ยังไม่มี Ubuntu 24.04 ให้เปิด PowerShell แบบ administrator แล้วใช้:

```powershell
wsl --install -d Ubuntu-24.04
```

restart Windows หากระบบร้องขอ จากนั้นเปิด Ubuntu และสร้างชื่อผู้ใช้/รหัสผ่าน Linux ตามหน้าจอ รหัสผ่านนี้ใช้กับ `sudo` ภายใน Ubuntu ได้ รายละเอียดและกรณี WSL รุ่นเก่าดู [คู่มือติดตั้ง Microsoft](https://learn.microsoft.com/en-us/windows/wsl/install)

หากเครื่องมีหลาย distro ให้เลือก Ubuntu โดยตรงจาก PowerShell:

```powershell
wsl -d Ubuntu-24.04
```

คำสั่ง `wsl` ที่ไม่ระบุ `-d` จะเปิด distro เริ่มต้นของเครื่อง ซึ่งอาจเป็นชื่ออื่น

## 2. Clone ลงระบบไฟล์ Linux

เปิด **Ubuntu terminal** แล้วรันทีละบล็อก `~` หมายถึง home ของผู้ใช้ Linux, `cd` ใช้เปลี่ยนโฟลเดอร์ และ `pwd` แสดงตำแหน่งปัจจุบัน repo คือสำเนาไฟล์หลักสูตรพร้อมประวัติ Git ที่ clone มา:

```bash
sudo apt update
sudo apt install -y git curl ca-certificates
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/KOPE-SOLUTION/machine-learning.git
cd machine-learning
pwd
git remote -v
```

`pwd` ควรเป็น `/home/ชื่อผู้ใช้/projects/machine-learning` หากเห็น `/mnt/c` ยังเป็นโฟลเดอร์บน Windows คำสั่ง clone เหมาะกับโฟลเดอร์ปลายทางที่ยังไม่มี; ถ้ามี repo อยู่แล้วให้เข้าโฟลเดอร์เดิมและตรวจ `git status` ก่อนอัปเดต ไม่ clone ทับ

repo เป็นสาธารณะ ผู้เรียน clone เพื่ออ่าน/รันได้โดยไม่ต้อง login GitHub ส่วนการ push ต้องมีสิทธิ์และตั้ง authentication ของ Git ใน WSL แยกตามวิธีที่ใช้

จากนั้นไป [คู่มือตั้ง environment](setup.md) เลือก uv หรือ venv + pip สร้าง environment **ใหม่ในตำแหน่งใหม่** อย่าคัดลอก `.venv`, `.venv-wsl` หรือ `.venv-mojo` จาก checkout เก่า เพราะ environment มี path ของ interpreter และอาจมี binary คนละระบบปฏิบัติการ

## 3. ย้ายจากโฟลเดอร์ Windows โดยไม่ทำงานหาย

clone ใหม่ได้โดยเก็บโฟลเดอร์เก่าไว้ก่อน ให้ตรวจจาก checkout เก่าใน Ubuntu ตาม path ของเครื่องตนเอง:

```bash
git status --short
git fetch origin
git log --oneline origin/main..HEAD
git ls-files --others --exclude-standard
git ls-files --others --ignored --exclude-standard outputs
```

- `git status` และรายการ untracked บอกการแก้ไขที่ยังไม่อยู่ใน commit
- `git log origin/main..HEAD` บอก commit ของ branch ปัจจุบันที่ยังไม่มีใน `origin/main`; หากมี branch อื่นที่ต้องเก็บ ให้ตรวจและ push branch นั้นด้วย
- GitHub ไม่เก็บสิ่งที่ถูก ignore: ตัวอย่าง `outputs/` อาจมีโมเดล รูป และผลทดลองที่ต้องการเก็บ; `.env` และไฟล์ส่วนตัวก็ไม่กลับมาพร้อม clone ตรวจของจริงนอกเหนือจากรายการ outputs ด้วย
- สำรอง notebook ที่แก้เองและผลทดลองที่ต้องการก่อน ไม่ต้องสำรอง environment เพื่อย้าย เพราะติดตั้งใหม่ได้

ใน checkout ใหม่ ให้สร้าง environment แล้วรันบท 00, บท 03 และเปิด notebook จาก kernel ใหม่จนผ่าน ตรวจไฟล์ส่วนตัวที่สำรองไว้ครบก่อนตัดสินใจลบ checkout เก่าผ่าน File Explorer การ clone สำเร็จหรือ `git status` สะอาดอย่างเดียวไม่ได้ยืนยันว่าไฟล์ที่ถูก ignore ถูกสำรองแล้ว

## 4. เปิดไฟล์และ editor

จาก Ubuntu เปิด File Explorer ที่ตำแหน่งปัจจุบันได้ด้วย:

```bash
explorer.exe .
```

เข้าผ่าน `\\wsl$` ในแถบที่อยู่ File Explorer ได้เช่นกัน ถ้าใช้ VS Code ให้ติดตั้ง WSL extension แล้วเปิดโฟลเดอร์ในโหมด WSL ตาม [คู่มือ VS Code](https://code.visualstudio.com/docs/remote/wsl) ตรวจว่ามุมล่างแสดงการเชื่อมต่อ WSL และเลือก `.venv/bin/python` ของ repo ทั้ง interpreter/kernel

ชื่อไฟล์ Linux แยกตัวพิมพ์เล็ก/ใหญ่ เช่น `Data.csv` กับ `data.csv` ต่างกัน คำสั่ง Linux ใช้ `/` และ Python ใน environment อยู่ที่ `.venv/bin/python`; Windows ใช้ `.venv\Scripts\python.exe` จึงควรเลือกทำงานฝั่งเดียวตลอดการทดลอง

## GPU, CUDA และข้อจำกัด

**บทเรียนใน repo นี้ใช้ CPU รวมถึง Mojo ทั้งสองบท ไม่ต้องติดตั้ง CUDA เพื่อเริ่มเรียน** การมีการ์ด NVIDIA ไม่ทำให้โค้ดเหล่านี้เปลี่ยนไปใช้ GPU โดยอัตโนมัติ

หากต่อยอดเป็นงาน deep learning ภายหลัง ให้แยกโปรเจกต์/environment แล้วตรวจคู่มือติดตั้งของ framework รุ่นที่จะใช้ ความเข้ากันได้เกี่ยวข้องกับ GPU, ไดรเวอร์ Windows, WSL 2 และ CUDA runtime ของ framework:

| ส่วน | หน้าที่ |
|---|---|
| NVIDIA driver บน Windows | ให้ระบบและ WSL 2 ติดต่อ GPU |
| CUDA runtime ของ framework | ไลบรารีที่โปรแกรม GPU เรียกใช้; วิธีติดตั้งขึ้นกับ framework และรุ่น |
| CUDA Toolkit / `nvcc` | เครื่องมือสำหรับคอมไพล์โค้ด CUDA หรือ extension ที่ต้อง build; ไม่จำเป็นต้องมีเพื่อรันทุกโปรแกรม GPU |
| uv / venv | แยก Python และแพ็กเกจ; ไม่จัดการไดรเวอร์ Windows |

[NVIDIA ระบุให้ใช้ไดรเวอร์ Windows ที่รองรับ WSL และไม่ติดตั้ง Linux display driver ใน WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) หากต้องคอมไพล์ CUDA ให้ทำตามตัวเลือก Toolkit สำหรับ WSL ในคู่มือทางการ อย่าติดตั้งแพ็กเกจรวมไดรเวอร์ Linux ตามคู่มือ Ubuntu บนเครื่องจริงโดยตรง

ตรวจการมองเห็น GPU จาก Ubuntu ได้ด้วย:

```bash
nvidia-smi
```

ถ้าหาคำสั่งไม่พบ ลอง path ที่ NVIDIA ระบุ:

```bash
/usr/lib/wsl/lib/nvidia-smi
```

การเห็นชื่อ GPU ยืนยันเพียงว่าระบบมองเห็นการ์ด ยังต้องรันทดสอบกับ framework เพื่อยืนยันงาน CUDA ช่อง `CUDA Version` หรือ `CUDA UMD Version` (ขึ้นกับรุ่น) ใน nvidia-smi บอกระดับ CUDA ที่ไดรเวอร์รองรับ ไม่ได้ยืนยันว่าได้ติดตั้ง CUDA Toolkit รุ่นนั้น และ `nvidia-smi` ใน WSL รองรับบางความสามารถต่างจาก Linux บนเครื่องจริง

GPU ยังมีขนาด VRAM จำกัดและ WSL มีข้อจำกัดบางอย่างด้าน memory/profiling ตามรุ่นซอฟต์แวร์ สำหรับงานในหลักสูตรนี้ ให้เริ่ม CPU และ environment เดียวก่อน แล้วเพิ่มการตั้งค่า GPU เมื่อมีบทเรียนที่ใช้จริง

## อ้างอิง

- [Microsoft: WSL filesystems](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
- [Microsoft: WSL networking](https://learn.microsoft.com/en-us/windows/wsl/networking) — เปิดเว็บแอป Linux จาก browser Windows
- [Microsoft: WSL disk space](https://learn.microsoft.com/en-us/windows/wsl/disk-space)
- [Python: venv portability](https://docs.python.org/3.12/library/venv.html)
- [NVIDIA: CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [NVIDIA: nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html) — ความหมายของรุ่น driver/CUDA ที่รายงาน
