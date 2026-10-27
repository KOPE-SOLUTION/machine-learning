# สร้างภาพประกอบหลักสูตร

ส่วนนี้ใช้สำหรับแก้และสร้างไฟล์ภาพใน repository ผู้เรียนรัน Notebook ตามบทเรียนได้โดยไม่ต้องติดตั้งชุดนี้

## เตรียมเครื่องมือสร้างภาพ

ใช้ Python 3.11 และติดตั้ง dependency สำหรับสร้างภาพ:

```bash
python -m pip install -r scripts/requirements-figures.txt
```

เครื่องต้องมีฟอนต์ไทยอย่าง Tahoma, Noto Sans Thai หรือ Leelawadee UI สคริปต์จะเลือกฟอนต์ที่ติดตั้งไว้

## สร้างภาพและตรวจข้อความไทย

```bash
python scripts/check_thai_figure_text.py
python scripts/render_ep03_scaler_diagram.py
python scripts/render_model_diagrams.py
```

สคริปต์ `render_ep*.py` สร้างภาพของบทนั้น ส่วน `render_model_diagrams.py` สร้างภาพอธิบายโมเดลหลายบท ไฟล์ผลลัพธ์อยู่ใน `assets/figures/`

`thai_figure_text.py` ใช้ HarfBuzz จัดตำแหน่งสระและวรรณยุกต์ไทยก่อนแปลงเป็นเส้นขอบตัวอักษร จึงวางไม้เอกเหนือสระได้ถูกต้องทั้ง PNG และ SVG ผู้เปิด SVG ไม่ต้องติดตั้งฟอนต์ที่ใช้สร้างภาพ ข้อความภาษาอังกฤษล้วนและสูตร MathText ใช้ตัววาดเดิมของ Matplotlib

เมื่อแก้ภาพให้เปิดตรวจ PNG และ SVG ด้วย โดยเฉพาะข้อความที่มีสระบนร่วมกับวรรณยุกต์ เช่น **ที่**, **นี่**, **ค่าเฉลี่ย** รวมถึงหัวข้อหนาและชื่อแกนแนวตั้ง

## อ้างอิงเครื่องมือ

- [HarfBuzz: shaping](https://harfbuzz.github.io/shaping-and-shape-plans.html)
- [uharfbuzz](https://github.com/harfbuzz/uharfbuzz)
