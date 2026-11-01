# 06 · Ridge Regression กับการเตรียมตัวเลข หมวดหมู่ และค่าที่หาย

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

บทนี้เตรียมตารางที่มีตัวเลข หมวดหมู่ และค่าหายด้วย **ColumnTransformer** แล้วเรียน **Ridge Regression** เพื่อทำนายค่าบริการส่งสินค้าจำลอง

**ใช้สิ่งที่เรียนแล้ว:** pandas, Pipeline/StandardScaler จาก [บท 03](../03-first-classifier/README.md) และ regression/MAE จาก [บท 05](../05-linear-regression/README.md)

## เตรียมข้อมูลคนละชนิดให้โมเดลใช้ร่วมกัน

**Feature Engineering — การสร้างหรือปรับรูปแบบ feature** คือการสร้างหรือแปลงข้อมูลนำเข้าให้โมเดลใช้ได้เหมาะสม เช่น สร้างระยะทางจากพิกัด หรือเปลี่ยนประเภทบริการเป็นตัวเลข บทนี้เน้นเตรียมคอลัมน์ที่มีอยู่ให้โมเดลรับได้

**ColumnTransformer** แยกคอลัมน์ไปเตรียมตามชนิดข้อมูล เช่น ส่งระยะทางกับน้ำหนักไปเติมค่าหายและปรับสเกล ส่วนประเภทบริการแปลงเป็นคอลัมน์ 0/1 แล้วนำผลมาต่อเป็นตารางตัวเลขชุดเดียวให้โมเดลใช้

![ตัวเลขผ่านการเติม median และปรับสเกล ส่วน service ผ่านการเติมค่าที่พบบ่อยและ one-hot ก่อนรวมเป็น 4 features เข้า Ridge พร้อมตัวอย่างหมวดหมู่ใหม่](../../assets/figures/06-feature-engineering-preprocessing.png)

*อ่านจากข้อมูลดิบ → เติมค่าหายและแปลงหมวดหมู่ → Ridge → ค่าบริการหนึ่งตัวเลข ตัวอย่าง median ด้านล่างสมมติขึ้นเพื่ออธิบายการเติมค่า* · [เปิดภาพ SVG](../../assets/figures/06-feature-engineering-preprocessing.svg)

### SimpleImputer: คำนวณค่าทดแทนจาก train แล้วนำไปเติมช่องว่าง

- **ตัวเลข:** เติมด้วยมัธยฐานของคอลัมน์ เช่น train มี `[1, 3, 5, NaN]` จะคำนวณมัธยฐานจากค่าที่มีอยู่ ได้ 3 แล้วเก็บค่านี้ไว้เติมช่องว่าง ทั้งใน train และข้อมูลใหม่
- **หมวดหมู่:** เติมด้วยค่าที่พบบ่อย เช่น `[express, express, standard]` จะเลือก express ซึ่งพบมากที่สุด แล้วเก็บค่านี้ไว้เติมช่องว่าง

**แบ่ง train/test ก่อน fit ตัวเติมค่า** เพื่อไม่ให้ข้อมูล test มาช่วยกำหนดค่าที่ใช้ฝึก โมเดลในบทนี้จึงรวมการเตรียมข้อมูลไว้ใน Pipeline

ข้อมูลจำลองจะมีค่าน้ำหนักหาย แต่ service ยังครบ ขั้นตอนเติม service เตรียมไว้รองรับช่องว่างที่เป็น `np.nan` ไม่ได้เปลี่ยนข้อความที่ไม่เคยพบให้เป็นค่าที่พบบ่อย

### One-hot encoding: เปลี่ยนหมวดหมู่เป็นคอลัมน์ 0/1

ใช้คอลัมน์แยกแทนการกำหนดรหัสที่ทำให้โมเดลตีความว่าประเภทบริการมีลำดับหรือระยะห่างเชิงตัวเลข โดยต้องตรวจชื่อและลำดับคอลัมน์หลังแปลงก่อนอ่านค่า

| service | service_express | service_standard |
|---|---:|---:|
| express | 1 | 0 |
| standard | 0 | 1 |
| same_day ที่ไม่เคยพบ | 0 | 0 |

แถว same_day เป็นผลของ `OneHotEncoder(handle_unknown="ignore")`: เมื่อ encoder เรียนรู้เฉพาะ express และ standard จาก train แล้วพบ same_day จะให้ทั้ง service_express และ service_standard เป็น 0 **ไม่มีคอลัมน์ same_day เพิ่มขึ้นระหว่าง predict** และโมเดลยังไม่เคยเรียนค่าบริการของ same_day

**จุดเด่นของ ColumnTransformer:** เตรียมตัวเลขและหมวดหมู่ด้วยวิธีต่างกันได้ภายใน Pipeline เดียว

**ข้อจำกัด:** ต้องเลือกคอลัมน์และวิธีเตรียมให้เหมาะเอง การเติม median ไม่ได้กู้ค่าน้ำหนักจริงกลับมา ส่วน one-hot อาจสร้างคอลัมน์จำนวนมากเมื่อมีหลายหมวดหมู่

## Ridge Regression คืออะไร และเพิ่มอะไรจาก Linear Regression

**Ridge Regression — การถดถอยแบบริดจ์** เป็นโมเดล Supervised Learning สำหรับงาน Regression หรือทำนายตัวเลข เช่น ค่าบริการส่งสินค้า โมเดลเรียนจากข้อมูลนำเข้า (features) คู่กับค่าที่ต้องการทำนาย (target)

Ridge ใช้สมการเส้นตรงเช่นเดียวกับ Linear Regression แต่เปลี่ยนวิธีเลือก **coefficient (สัมประสิทธิ์ หรือน้ำหนักที่คูณกับ feature)** ระหว่างฝึก: นอกจากลดความผิดพลาดแล้ว ยังควบคุมขนาดน้ำหนักด้วย **L2 regularization**

### จากข้อมูลที่เตรียมแล้ว ไปเป็นคำทำนาย

ColumnTransformer เตรียมข้อมูล ส่วน Ridge เรียนค่าน้ำหนักและทำนาย หลังเตรียมตารางค่าจัดส่งจะได้ 4 features:

- `z_distance` และ `z_weight` คือระยะทางและน้ำหนักหลังเติมค่าหายแล้วปรับสเกลด้วย StandardScaler
- `I_express` และ `I_standard` คือคอลัมน์ 0/1 ที่บอกว่าเป็นบริการแต่ละชนิดหรือไม่
- `w₁, w₂, w₃, w₄` คือน้ำหนักที่ Ridge เรียนให้แต่ละ feature ส่วน `b` คือ intercept หรือค่าคงที่ที่บวกในสมการ

เมื่อต้องทำนายค่าบริการหนึ่งรายการ Ridge คำนวณ:

![ค่าทำนายเท่ากับ b บวก w1 คูณ z ของระยะทาง บวก w2 คูณ z ของน้ำหนัก บวก w3 คูณ I express บวก w4 คูณ I standard](../../assets/figures/06-feature-engineering-formula-prediction.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-prediction.svg)

**ตอน fit** เรียนน้ำหนักทั้ง 4 ค่าและ intercept ร่วมกันจาก train ส่วน **ตอน predict** ใช้ค่าที่เรียนไว้แล้วคำนวณสมการนี้ โมเดลไม่ได้รับสูตรที่เราใช้สร้างค่าบริการจำลอง

### ตอน fit: ลด SSE พร้อมควบคุมขนาดน้ำหนัก

ใน EP5 **Ordinary Least Squares (OLS)** เลือกน้ำหนักและ intercept ให้ **SSE — ผลรวมความผิดพลาดกำลังสองบน train** ต่ำที่สุด ส่วน Ridge เพิ่ม **พจน์ลงโทษ (penalty term)** ซึ่งเป็นค่าที่เพิ่มเพื่อควบคุมขนาดน้ำหนัก

ให้ `J` เป็นค่ารวมที่ Ridge ต้องลดระหว่างฝึก, `α` (อัลฟา หรือ `alpha` ในโค้ด) เป็นตัวกำหนดความเข้มของการควบคุม และ `Σ` (ซิกมา) หมายถึงนำค่ามาบวกกัน:

![Ridge ลด SSE บน train บวกพจน์ลงโทษ ตัวอย่างน้ำหนัก 2 และลบ 1 เมื่อ alpha เท่ากับ 1 ได้พจน์ลงโทษ 5](../../assets/figures/06-feature-engineering-ridge.png)

*ตัวเลขสมมติเพื่ออ่านสองส่วนของสูตร ไม่ใช่น้ำหนักจากข้อมูลค่าจัดส่ง* · [เปิดภาพ SVG](../../assets/figures/06-feature-engineering-ridge.svg)

`yᵢ` คือค่าจริงของแถวที่ i, `ŷᵢ` คือค่าทำนาย, `n` คือจำนวนแถว train และ `p` คือจำนวน features หลังแปลง **intercept ไม่รวมในพจน์ลงโทษ** คำว่า L2 ใน Ridge หมายถึงใช้ผลรวมกำลังสองของน้ำหนักในสูตรนี้

ตัวอย่างสมมติ: น้ำหนัก `[2, -1]` และ `alpha=1` ให้พจน์ลงโทษ `1 × (2² + (-1)²) = 5` ถ้า SSE เท่ากับ 100 ค่ารวมคือ `J = 100 + 5 = 105`


**แยกการฝึกกับการประเมินผล:** Ridge ลด `J` บน train เพื่อหาน้ำหนัก ส่วนบทนี้ใช้ **MAE บน test** ประเมินว่าค่าบริการที่ทำนายคลาดเคลื่อนเฉลี่ยเท่าไร พจน์ลงโทษไม่ได้ถูกบวกเข้า MAE และค่ารวม J ไม่ใช่คะแนนความแม่นยำ

### ตัวอย่างคำนวณ: ทำไมน้ำหนักจึงเปลี่ยนจาก 2 เป็น 1

ใช้ข้อมูลสมมติขนาดเล็กเพียง 3 แถว เพื่อเห็นขั้นตอนคำนวณ โดย `x` เป็น feature หนึ่งคอลัมน์ และ `y` เป็น target:

| x | y |
|---:|---:|
| −1 | 1 |
| 0 | 3 |
| 1 | 5 |

ตัวอย่างนี้ใช้ x ตามตารางโดยตรง ไม่ผ่าน StandardScaler และเป็นคนละชุดกับข้อมูลค่าจัดส่ง 400 แถว

**1. เริ่มจาก OLS ที่เรียนใน EP5**

ใช้ `x̄` (เอ็กซ์บาร์) แทนค่าเฉลี่ย x และ `ȳ` (วายบาร์) แทนค่าเฉลี่ย y:

- `x̄ = (−1 + 0 + 1) / 3 = 0` และ `ȳ = (1 + 3 + 5) / 3 = 3`
- ผลรวม `(x − x̄)(y − ȳ)` คือ `(−1)(−2) + (0)(0) + (1)(2) = 4`
- ผลรวม `(x − x̄)²` คือ `(−1)² + 0² + 1² = 2`
- OLS จึงได้ `w = 4 / 2 = 2` และ `b = ȳ − wx̄ = 3 − 2(0) = 3`

สมการ `ŷ = 3 + 2x` ทำนายได้ `[1, 3, 5]` ตรงกับ y ทุกแถว จึงมี SSE = 0

**2. เพิ่ม L2 โดยกำหนด alpha = 2**

สำหรับ Ridge ที่มี feature เดียวและเรียน intercept สูตรหาน้ำหนักเปลี่ยนเป็น:

![w เท่ากับผลรวม x ลบค่าเฉลี่ย x คูณ y ลบค่าเฉลี่ย y หารด้วยผลรวมกำลังสองของ x ลบค่าเฉลี่ย x บวก alpha ส่วน b เท่ากับค่าเฉลี่ย y ลบ w คูณค่าเฉลี่ย x](../../assets/figures/06-feature-engineering-formula-weights.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-weights.svg)

แทนค่าผลรวมที่เพิ่งคำนวณ: `w = 4 / (2 + 2) = 1` และ `b = 3 − 1(0) = 3` จึงได้ `ŷ = 3 + x` ที่มาของการบวก alpha ในตัวหารอยู่ในหัวข้อพิสูจน์แบบยุบด้านล่าง

เมื่อนำ x ทั้งสามค่ามาทำนาย ได้ `[2, 3, 4]` ความผิดพลาด `y − ŷ` คือ `[−1, 0, 1]` จึงมี `SSE = (−1)² + 0² + 1² = 2` และพจน์ลงโทษ `2 × 1² = 2`

**3. เทียบทั้งสองเส้นด้วยสูตร Ridge ที่ alpha = 2 เท่ากัน**

| ค่าน้ำหนักที่นำมาเปรียบเทียบ | SSE บน train | พจน์ลงโทษที่ alpha = 2 | J = SSE + พจน์ลงโทษ |
|---|---:|---:|---:|
| เส้น OLS: b = 3, w = 2 | 0 | 2 × 2² = 8 | 8 |
| เส้น Ridge: b = 3, w = 1 | 2 | 2 × 1² = 2 | 4 |

Ridge ยอมให้ SSE เพิ่มจาก 0 เป็น 2 เพราะพจน์ลงโทษลดจาก 8 เป็น 2 ทำให้ค่ารวมลดจาก 8 เป็น 4 นี่คือเหตุผลที่เลือกน้ำหนักต่างจาก OLS **ไม่ได้แปลว่า Ridge ทำนายข้อมูลใหม่ได้ดีกว่าเสมอ** ตัวอย่างนี้ยังไม่มี test และข้อมูลทั้งสามจุดวางอยู่บนเส้นตรงพอดี

![ข้อมูลสมมติ 3 จุด เส้น OLS มีความชัน 2 ส่วน Ridge ที่ alpha เท่ากับ 2 มีความชัน 1 เมื่อคิดสูตร Ridge เดียวกันค่ารวมลดจาก 8 เป็น 4](../../assets/figures/06-feature-engineering-ridge-tradeoff.png)

*เส้นประแนวตั้งแสดง residual ของ Ridge ส่วนแท่งทางขวาแยก SSE กับพจน์ลงโทษที่ alpha = 2 เท่ากัน* · [เปิดภาพ SVG](../../assets/figures/06-feature-engineering-ridge-tradeoff.svg)

<details>
<summary>อ่านเพิ่มเติม: พิสูจน์สูตร w และ b ของ Ridge ทีละขั้น พร้อมเหตุผล</summary>

ใช้ feature เดียว, `n` แถว, `alpha > 0`, ให้น้ำหนักความผิดพลาดของทุกแถวเท่ากัน และไม่ลงโทษ intercept สูตรนี้ต่อยอดจาก OLS ใน EP5

**1. เขียนสิ่งที่ต้องลดให้ครบ ก่อนหาอนุพันธ์**

แทน `ŷᵢ = b + wxᵢ` ลงใน SSE แล้วบวกพจน์ลงโทษของน้ำหนักหนึ่งค่า:

![J ของ b และ w เท่ากับผลรวมกำลังสองของ y ลบ b ลบ wx บวก alpha คูณ w กำลังสอง](../../assets/figures/06-feature-engineering-formula-proof-objective.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-proof-objective.svg)

ให้ `rᵢ = yᵢ − b − wxᵢ` เป็น residual เพื่อเขียนขั้นตอนถัดไปให้สั้นลง

**2. หา b โดยดูว่า J เปลี่ยนอย่างไรเมื่อขยับ b แต่ตรึง w ไว้**

อนุพันธ์ของ `rᵢ` เทียบกับ b คือ −1 จึงได้อนุพันธ์ของ `rᵢ²` เป็น `2rᵢ(−1)` ส่วน `alpha × w²` ไม่มี b จึงให้อนุพันธ์เทียบกับ b เป็น 0:

![อนุพันธ์ย่อยของ J เทียบกับ b เท่ากับลบสองคูณผลรวม y ลบ b ลบ wx แล้วตั้งเท่ากับศูนย์](../../assets/figures/06-feature-engineering-formula-proof-intercept-derivative.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-proof-intercept-derivative.svg)

หารทั้งสองข้างด้วย −2 เพื่อตัดตัวคูณร่วม แล้วแยกผลรวม โดย b ซ้ำ n ครั้ง:

![ผลรวม y ลบ nb ลบ w คูณผลรวม x เท่ากับศูนย์ จัดรูปได้ nb เท่ากับผลรวม y ลบ w คูณผลรวม x](../../assets/figures/06-feature-engineering-formula-proof-intercept-isolate.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-proof-intercept-isolate.svg)

หารด้วย n เพื่อให้เหลือ b และเขียนผลรวมที่หารด้วยจำนวนแถวเป็นค่าเฉลี่ย:

![b เท่ากับค่าเฉลี่ย y ลบ w คูณค่าเฉลี่ย x](../../assets/figures/06-feature-engineering-formula-proof-intercept-mean.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-proof-intercept-mean.svg)

**3. หา w โดยดูว่า J เปลี่ยนอย่างไรเมื่อขยับ w แต่ตรึง b ไว้**

อนุพันธ์ของ `rᵢ` เทียบกับ w คือ `−xᵢ` ส่วนอนุพันธ์ของ `alpha × w²` คือ `2alpha × w` จึงได้:

![อนุพันธ์ย่อยของ J เทียบกับ w เท่ากับลบสองคูณผลรวม x คูณ residual บวกสอง alpha w แล้วตั้งเท่ากับศูนย์](../../assets/figures/06-feature-engineering-formula-proof-weight-derivative.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-proof-weight-derivative.svg)

ย้ายพจน์ผลรวมไปอีกข้าง แล้วหารด้วย 2 เพื่อลดตัวคูณร่วม:

![ผลรวม x คูณ y ลบ b ลบ wx เท่ากับ alpha คูณ w](../../assets/figures/06-feature-engineering-formula-proof-weight-balance.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-proof-weight-balance.svg)

**4. แทน b เพื่อให้เหลือตัวแปรที่ต้องหาเพียง w**

แทน `b = ȳ − wx̄` ทำให้ residual เขียนเป็น `(yᵢ − ȳ) − w(xᵢ − x̄)` แล้วกระจายผลรวม:

![หลังแทน b ได้ผลรวม x คูณ y ลบค่าเฉลี่ย y ลบ w คูณผลรวม x คูณ x ลบค่าเฉลี่ย x เท่ากับ alpha w](../../assets/figures/06-feature-engineering-formula-proof-centered-residual.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-proof-centered-residual.svg)

ผลรวมของค่าที่ลบค่าเฉลี่ยแล้วเป็นศูนย์: `Σ(yᵢ − ȳ) = 0` และ `Σ(xᵢ − x̄) = 0` ดังนั้นการแทน `xᵢ = (xᵢ − x̄) + x̄` ทำให้พจน์ที่คูณ x̄ หายไป:

![ผลรวม x คูณ y ลบค่าเฉลี่ย y เท่ากับผลรวม x ลบค่าเฉลี่ย x คูณ y ลบค่าเฉลี่ย y](../../assets/figures/06-feature-engineering-formula-proof-centered-xy.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-proof-centered-xy.svg)

![ผลรวม x คูณ x ลบค่าเฉลี่ย x เท่ากับผลรวมกำลังสองของ x ลบค่าเฉลี่ย x](../../assets/figures/06-feature-engineering-formula-proof-centered-xx.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-proof-centered-xx.svg)

**5. รวบพจน์ที่มี w แล้วหารเพื่อหาน้ำหนัก**

![ผลรวมผลคูณของค่าที่ลบค่าเฉลี่ยแล้ว เท่ากับ w คูณผลรวมกำลังสองของ x ลบค่าเฉลี่ย x บวก alpha w ดึง w ออกได้ w คูณวงเล็บผลรวมกำลังสองบวก alpha](../../assets/figures/06-feature-engineering-formula-proof-factor-weight.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-proof-factor-weight.svg)

วงเล็บทางขวาเป็นบวกเมื่อ alpha > 0 จึงหารทั้งสองข้างด้วยวงเล็บนี้ได้:

![w เท่ากับผลรวม x ลบค่าเฉลี่ย x คูณ y ลบค่าเฉลี่ย y หารด้วยผลรวมกำลังสองของ x ลบค่าเฉลี่ย x บวก alpha ส่วน b เท่ากับค่าเฉลี่ย y ลบ w คูณค่าเฉลี่ย x](../../assets/figures/06-feature-engineering-formula-weights.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-weights.svg)

สำหรับสามแถวในตัวอย่าง: ตัวเศษ = 4, ผลรวมกำลังสองในตัวหาร = 2 และ alpha = 2 จึงได้ `w = 4 / 4 = 1` และ `b = 3`

**6. ทำไมจุดนี้จึงเป็นค่าต่ำสุด ไม่ใช่เพียงจุดที่อนุพันธ์เป็นศูนย์**

เมื่อแทน `b = ȳ − wx̄` กลับเข้า J แล้ว จะได้กราฟพาราโบลาในตัวแปร w ซึ่งมีสัมประสิทธิ์หน้า `w²` เท่ากับ `Σ(xᵢ − x̄)² + alpha` ที่เป็นบวก กราฟจึงหงาย จุดที่อนุพันธ์เป็นศูนย์จึงเป็นจุดต่ำสุดของกราฟนี้ ส่วน b ที่แทนไว้ก็เป็นค่าที่ทำให้ J ต่ำสุดสำหรับแต่ละ w อยู่แล้ว

ตรวจด้วยตัวเลขชุดเดิมได้อีกทาง: ให้ `b = 3 + u` และ `w = 1 + v` โดย u และ v คือการขยับจากคำตอบที่หาได้ จะมี residual เป็น `[−1 − u + v, −u, 1 − u − v]` จึงได้:

![เมื่อขยับ b เป็น 3 บวก u และ w เป็น 1 บวก v จะได้ J เท่ากับ 4 บวก 3u กำลังสอง บวก 4v กำลังสอง จึงไม่น้อยกว่า 4](../../assets/figures/06-feature-engineering-formula-proof-minimum.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-proof-minimum.svg)

ค่ารวมเท่ากับ 4 เมื่อ `u = v = 0` เท่านั้น ยืนยันว่า `b = 3, w = 1` ให้ค่าต่ำสุด **อนุพันธ์เป็นศูนย์อย่างเดียวไม่พอ ต้องตรวจลักษณะฟังก์ชันด้วย**

**เมื่อมีหลาย features:** สูตรเศษส่วนข้างต้นใช้กับ feature เดียว ห้ามนำไปหาน้ำหนักทีละคอลัมน์อย่างอิสระในตาราง 4 features เพราะ Ridge ต้องหาน้ำหนักทุกค่าร่วมกัน scikit-learn ใช้วิธีแก้สมการเชิงตัวเลขตาม solver; ค่าเริ่มต้น `solver="auto"` เลือกวิธีให้ จึงไม่จำเป็นต้องใช้ gradient descent ทุกครั้ง

</details>

### alpha เลือกอย่างไร และทำไมต้องปรับสเกล

**น้ำหนักกับ alpha เป็นคนละอย่าง:** `coef_` และ `intercept_` เป็นค่าที่โมเดลเรียนระหว่าง fit ส่วน `alpha` เป็นค่าตั้งก่อนฝึก หรือ **hyperparameter** ไม่ใช่เปอร์เซ็นต์ความแม่นยำและไม่ใช่อัตราการเรียนรู้

- alpha = 0: ไม่มีพจน์ลงโทษ เป้าหมายเหมือน OLS; ถ้าต้องการกรณีนี้ให้ใช้ `LinearRegression`
- alpha มากขึ้น: ให้ความสำคัญกับการควบคุมขนาดน้ำหนักมากขึ้น หากมากเกินไปอาจเรียนความสัมพันธ์ได้น้อยจนเกิด **underfitting**
- เลือก alpha ด้วย validation หรือ **Cross-validation (CV)** บน train โดยนำ Pipeline ทั้งชุดเข้า CV เพื่อให้การเติมค่าและปรับสเกลเรียนใหม่เฉพาะส่วนฝึกของแต่ละรอบ แล้วใช้ test ประเมินท้ายสุด ค่า alpha = 2 ในตัวอย่างคำนวณเลือกเพื่อให้คิดเลขง่าย ไม่ใช่ค่าที่แนะนำให้ใช้กับทุกชุดข้อมูล

**เหตุผลที่ปรับสเกล:** น้ำหนักที่ใช้คูณระยะทางหน่วยกิโลเมตรกับหน่วยเมตรมีขนาดต่างกัน แม้ทำนายค่าเดียวกัน เช่น `2 × ระยะทางกิโลเมตร` เท่ากับ `0.002 × ระยะทางเมตร` แต่น้ำหนักยกกำลังสองต่างกันมาก จึงมีผลต่อ L2 เราปรับสเกลคอลัมน์ตัวเลขด้วยสถิติจาก train ก่อนให้ Ridge เรียนรู้

ในบทนี้ปรับสเกลเฉพาะตัวเลข ส่วน one-hot คงค่า 0/1 ไว้ **น้ำหนักของตัวเลขจึงหมายถึงผลต่อคำทำนายเมื่อค่า z เพิ่ม 1 หรือข้อมูลดิบเพิ่มหนึ่งส่วนเบี่ยงเบนมาตรฐาน โดยตรึง feature อื่นไว้** ไม่ใช่เมื่อเพิ่ม 1 กิโลเมตรหรือ 1 กิโลกรัม

**จุดเด่น:** L2 ช่วยให้ค่าน้ำหนักเสถียรขึ้นเมื่อ features สัมพันธ์กัน และอาจช่วยลด overfitting ส่วน **ข้อจำกัด:** ยังใช้ความสัมพันธ์เชิงเส้นใน features ที่ป้อนเข้าไป และไม่ได้รับประกันว่าจะชนะ OLS บน test ทั้งยังไม่ได้มุ่งตัด features ด้วยการทำให้น้ำหนักเป็นศูนย์


## ลงมือทำ

สร้าง Notebook ใหม่แล้วรัน **Cell 1–3 ตามลำดับ** หรือเปิด [Notebook ของบทนี้](lesson.ipynb) ใช้ kernel เดิมที่มี scikit-learn 1.7.1 ได้เลย

ภาพด้านบนฝังไว้ใน README และ notebook แล้ว ส่วนโค้ดบทนี้แสดงตารางและคำทำนาย จึงไม่มีไฟล์กราฟให้บันทึก

<!-- CODE:CELLS -->

<!-- CODE:START -->
### Cell 1 — สร้างตารางตัวเลข หมวดหมู่ และค่าหาย

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rng = np.random.default_rng(42)
n = 400
distance = rng.uniform(1, 100, n)
weight = rng.uniform(0.2, 8, n)
service = rng.choice(["standard", "express"], n)
y = 30 + 2 * distance + 5 * weight + 40 * (service == "express") + rng.normal(0, 8, n)
X = pd.DataFrame({"distance_km": distance, "weight_kg": weight, "service": service})
X.loc[rng.choice(n, 30, replace=False), "weight_kg"] = np.nan
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print("Train/Test rows:", len(X_train), len(X_test))
print("Missing values (train):", X_train.isna().sum().to_dict())
print("First three training rows:\n", X_train.iloc[:3].round(3).to_string(index=False))
```

**อ่านโค้ด**

- `y = 30 + ...` สร้างค่าบริการจำลองจากระยะทาง น้ำหนัก ประเภทบริการ และ noise ไม่ใช่อัตราค่าจัดส่งจริง และไม่ได้ส่งสูตรนี้ให้โมเดล
- `40 * (service == "express")` เพิ่มค่าบริการ 40 เฉพาะ express โดย True/False ทำหน้าที่เป็น 1/0
- `pd.DataFrame(...)` เก็บระยะทาง น้ำหนัก และประเภทบริการเป็นข้อมูลนำเข้า `X` (features) ส่วนค่าบริการที่สร้างไว้เก็บใน `y` (target) ชื่อคอลัมน์ใน X ช่วยให้ ColumnTransformer เลือกแต่ละกลุ่มไปแปลงได้
- `rng.choice(n, 30, replace=False)` เลือก 30 แถวไม่ซ้ำ แล้วใส่ `np.nan` ในน้ำหนัก สร้างค่าบริการ `y` ก่อนลบค่าน้ำหนัก เพื่อจำลองกรณีที่ทราบค่าบริการแล้ว แต่บันทึกน้ำหนักพัสดุไว้ไม่ครบ
- `test_size=0.25` กัน 100 แถวไว้ทดสอบ อีก 300 แถวใช้ฝึก โดยยังไม่คำนวณค่าทดแทนหรือ fit ตัวแปลงใด
- ผลท้าย cell แสดงจำนวนแถว ค่าหายของ train และข้อมูล 3 แถวก่อนแปลง เพื่อเทียบกับ Cell 2

### Cell 2 — เตรียมข้อมูล ฝึก Ridge และดูน้ำหนักที่เรียนได้

```python
numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
categorical = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
)
preprocess = ColumnTransformer([
    ("numeric", numeric, ["distance_km", "weight_kg"]),
    ("categorical", categorical, ["service"]),
])
model = make_pipeline(preprocess, Ridge(alpha=1.0))
model.fit(X_train, y_train)
baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
print(f"Baseline MAE: {mean_absolute_error(y_test, baseline.predict(X_test)):.3f}")
print(f"Test MAE (synthetic cost units): {mean_absolute_error(y_test, model.predict(X_test)):.3f}")
feature_names = preprocess.get_feature_names_out()
print("Transformed columns:", feature_names)
preview = preprocess.transform(X_train.iloc[:3])
print("First three transformed rows:\n",
      pd.DataFrame(preview, columns=feature_names).round(3).to_string(index=False))

ridge = model.named_steps["ridge"]
coefficients = pd.DataFrame({"feature": feature_names, "coefficient": ridge.coef_})
print(f"Learned intercept: {ridge.intercept_:.3f}")
print("Learned coefficients:\n", coefficients.round(3).to_string(index=False))
```

**อ่านโค้ด**

- `numeric` เติม median ก่อนปรับสเกล ส่วน `categorical` เติมค่าที่พบบ่อยก่อนแยกหมวดหมู่เป็น one-hot
- `sparse_output=False` ให้ encoder คืนอาร์เรย์ปกติ เพื่อแสดงตารางตัวอย่างได้ตรงไปตรงมา เหมาะกับสองหมวดหมู่ในบทนี้; ถ้ามีหมวดหมู่จำนวนมากควรพิจารณารูปแบบ sparse เพื่อประหยัดหน่วยความจำ
- รายการใน `ColumnTransformer` มีสามส่วน: ชื่อขั้นตอน, ตัวแปลง และรายชื่อคอลัมน์ ชื่อ `numeric` และ `categorical` จะปรากฏเป็นคำนำหน้าในชื่อ feature หลังแปลง
- `model.fit(X_train, y_train)` คำนวณค่าที่ใช้เติมช่องว่าง ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานสำหรับปรับสเกล และรายการประเภทบริการจาก train แล้วฝึก Ridge ส่วน `model.predict(X_test)` ใช้ค่าที่เก็บไว้เหล่านี้กับ test
- `get_feature_names_out()` ให้ชื่อ 4 คอลัมน์ตามลำดับจริง: ตัวเลข 2 คอลัมน์ และ service ที่แยกเป็น express/standard อีก 2 คอลัมน์
- `preprocess.transform(X_train.iloc[:3])` แปลงสามแถวเดิมใน Cell 1 โดยไม่ fit ซ้ำ ตัวเลขอาจติดลบหลัง scale; คอลัมน์ one-hot ยังคงเป็น 0/1 เพราะไม่ผ่าน StandardScaler
- `DummyRegressor(strategy="median")` ใช้มัธยฐานของค่าบริการใน `y_train` ทายทุกแถว เป็น baseline สำหรับเทียบ MAE ส่วนมัธยฐานของน้ำหนักพัสดุใช้เติมค่าน้ำหนักที่หาย จึงเป็นคนละค่าและคนละหน้าที่
- `model.named_steps["ridge"]` เข้าถึง Ridge ที่ฝึกแล้วใน Pipeline ไม่ได้สร้างโมเดลใหม่ ชื่อ `"ridge"` มาจาก `make_pipeline` ที่ตั้งชื่อขั้นตอนตามชื่อคลาส
- `ridge.coef_` เก็บน้ำหนักทั้ง 4 ค่าตามลำดับเดียวกับ `feature_names` ส่วน `ridge.intercept_` คือค่าคงที่ b ตารางนี้ช่วยจับคู่ได้ว่าน้ำหนักใดใช้กับคอลัมน์ไหน

**การตั้งค่าใน cell นี้:** `handle_unknown="ignore"` ทำให้บริการที่ไม่เคยพบมีค่า one-hot เป็น 0 ทุกคอลัมน์เดิม ส่วน `Ridge(alpha=1.0)` กำหนดความเข้มของ L2 ไว้ล่วงหน้า หากจะเลือกค่า alpha ให้ใช้ validation/CV ตาม [บท 04](../04-validation-and-metrics/README.md) โดยนำ Pipeline ทั้งชุดเข้า CV และเก็บ test ไว้ประเมินท้ายสุด ค่า alpha เป็นค่าตั้ง ส่วน `coef_` กับ `intercept_` เป็นค่าที่เรียนจาก train

**อ่านคะแนน:** คำทำนายเป็นค่าบริการหนึ่งตัวเลขต่อแถว MAE เปรียบเทียบค่าบริการที่ทำนายกับค่าบริการจริงใน test ทั้ง 100 แถว ส่วนตารางสามแถวที่แสดงเป็นข้อมูลนำเข้าหลังแปลง ไม่ใช่ค่าบริการที่โมเดลทำนาย

### Cell 3 — ลองข้อมูลใหม่ที่มีหมวดหมู่ไม่เคยพบ

```python
new_order = pd.DataFrame({
    "distance_km": [20.0],
    "weight_kg": [np.nan],
    "service": ["same_day"],
})
new_features = pd.DataFrame(
    preprocess.transform(new_order), columns=feature_names
)
print("New order after preprocessing:\n", new_features.round(3).to_string(index=False))
print("Unseen-category prediction:", model.predict(new_order).round(2))
manual_prediction = ridge.intercept_ + new_features.to_numpy() @ ridge.coef_
print("Prediction from coefficients:", manual_prediction.round(2))
```

**อ่านโค้ด**

- `new_order` มีชื่อคอลัมน์ตรงกับตอนฝึก แต่ใส่น้ำหนักเป็น `np.nan` และบริการเป็น `same_day`
- `preprocess.transform(new_order)` เติมน้ำหนักที่หายด้วยมัธยฐานจาก train แล้วปรับสเกลด้วยค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานเดิม ส่วน service_express และ service_standard เป็น 0 ทั้งคู่ เพราะ same_day ไม่อยู่ในรายการที่เรียนไว้
- ค่าศูนย์ทั้งคู่หมายถึงหมวดหมู่ที่ encoder ไม่เคยพบในตัวอย่างนี้ ไม่ได้หมายถึงบริการฟรี และไม่มีการเรียนค่าน้ำหนักของ same_day ขึ้นมาใหม่
- `model.predict(new_order)` รับตารางดิบ แล้วเตรียมข้อมูลด้วย Pipeline เดิมก่อนทำนาย ไม่ต้องส่ง `new_features` เข้าไปอีก
- `new_features.to_numpy() @ ridge.coef_` นำ features ที่แปลงแล้วมาคูณน้ำหนักทีละคอลัมน์แล้วรวมกัน จากนั้นบวก `ridge.intercept_` จึงเป็นการแทนค่าสมการ Ridge โดยตรง ผลต้องตรงกับ `model.predict(new_order)` เมื่อใช้ตัวเลขเต็มความละเอียด
- `round(2)` ปัดเฉพาะการแสดงคำทำนาย การรันผ่านยังไม่ยืนยันว่าคำทำนายสำหรับบริการใหม่น่าเชื่อถือ
<!-- CODE:END -->

## อ่านผล

เมื่อใช้ข้อมูลและค่าตั้งต้นตามบทนี้กับ scikit-learn 1.7.1 จะได้:

| ผลที่ตรวจ | ค่าที่ได้ |
|---|---|
| แถว train / test | 300 / 100 |
| ค่าน้ำหนักที่หายใน train | 23 จาก 30 ค่าหายทั้งหมด |
| Baseline MAE | 53.511 |
| Ridge Test MAE | 6.494 |
| คำทำนายสำหรับ same_day | 109.89 |

MAE ของ Ridge ต่ำกว่า baseline บน test ชุดนี้ โดยหน่วยเป็นค่าบริการจำลอง ไม่ใช่อัตราค่าจัดส่งจริง และไม่ควรเทียบกับ MAE ของบท 05 ซึ่งใช้ข้อมูลคนละชุด

หลังแปลงจะมี 4 features โดย express เป็น `[1, 0]`, standard เป็น `[0, 1]` และ same_day เป็น `[0, 0]` เฉพาะสองคอลัมน์ service

คำทำนาย 109.89 แสดงว่า Pipeline รับข้อมูลนี้ได้ แต่ยังไม่มีค่าบริการจริงของรายการ same_day นี้สำหรับตรวจความคลาดเคลื่อน ควรตรวจจับหมวดหมู่ใหม่และวางวิธีรับมือก่อนใช้งานจริง


### อ่านน้ำหนักที่เรียนได้ แล้วตามไปถึงคำทำนาย

จากตารางใน Cell 2 จะได้ intercept ประมาณ **168.432** และน้ำหนักตามลำดับ features ประมาณ **[56.380, 11.288, 20.473, −20.473]**:

- ระยะทางหลังปรับสเกลเพิ่ม 1 คำทำนายเพิ่มประมาณ 56.380 หน่วยค่าบริการ โดยตรึง feature อื่นไว้ **ไม่ใช่ระยะทางเพิ่ม 1 กิโลเมตร** น้ำหนักพัสดุก็อ่านในหน่วยหลังปรับสเกลเช่นกัน
- เปลี่ยนบริการจาก standard เป็น express โดยคงระยะทางและน้ำหนักเดิม จะเปลี่ยนส่วนของบริการจาก −20.473 เป็น +20.473 ผลต่างจึงประมาณ **40.946** น้ำหนักติดลบของ standard ไม่ได้หมายถึงค่าบริการรวมติดลบ
- intercept เป็นค่าคงที่ในสมการที่ใช้ features หลังแปลง จึงไม่ใช่ค่าเริ่มต้น 30 ในสูตรสร้าง y โดยตรง พจน์ลงโทษครอบคลุมน้ำหนักทั้งตัวเลขและ one-hot แต่ไม่รวม intercept

สำหรับข้อมูลใหม่ใน Cell 3 ระยะทาง 20 กิโลเมตรและน้ำหนักที่เติมแล้ว กลายเป็น z ประมาณ `−1.042254` และ `0.019219` ส่วน same_day ให้ one-hot เป็น `[0, 0]` จึงคำนวณได้:

![แทนค่า 168.432 บวก 56.380 คูณลบ 1.042254 บวก 11.288 คูณ 0.019219 บวก 20.473 คูณศูนย์ ลบ 20.473 คูณศูนย์ ได้ค่าบริการประมาณ 109.89](../../assets/figures/06-feature-engineering-formula-fitted-prediction.png)

[เปิดภาพ SVG](../../assets/figures/06-feature-engineering-formula-fitted-prediction.svg)

**ไม่ต้อง fit ซ้ำหลังได้ค่า z:** ส่งค่าที่แปลงแล้วให้สมการใช้น้ำหนักเดิมต่อได้เลย โค้ด `manual_prediction` คำนวณด้วยค่าจริงที่ยังไม่ปัดเศษ และได้ 109.89 ตรงกับ Pipeline การคำนวณตรงกันยืนยันขั้นตอน predict แต่ยังไม่ยืนยันคุณภาพคำทำนายสำหรับบริการใหม่


## แบบฝึกท้ายบท

**ลองทำ:** เพิ่มคอลัมน์จำนวนชิ้น แล้วออกแบบการรับ service ที่ไม่เคยเห็น

<details>
<summary>ดูแนวคำตอบ</summary>

**แนวคำตอบ:**

- เพิ่มชื่อคอลัมน์จำนวนชิ้นในรายการ numeric ของ ColumnTransformer
- ตรวจ service ที่ไม่เคยเห็น แล้วกำหนดว่าจะให้ผู้ใช้แก้ข้อมูลหรือใช้วิธีสำรอง การป้องกัน error ด้วย encoder เพียงอย่างเดียวยังไม่พอ

</details>

## อ้างอิง

- [Handbook: Feature Engineering](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html)
- [Pipelines and composite estimators](https://scikit-learn.org/1.7/modules/compose.html)
- [SimpleImputer](https://scikit-learn.org/1.7/modules/generated/sklearn.impute.SimpleImputer.html)
- [OneHotEncoder](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [ColumnTransformer](https://scikit-learn.org/1.7/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Ridge Regression: objective, alpha, solver และค่าน้ำหนักที่เรียนได้](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Ridge.html)
- [Ridge Regression: หลักการและผลของ regularization](https://scikit-learn.org/1.7/modules/linear_model.html#ridge-regression-and-classification)
- [StandardScaler: การปรับสเกลก่อนใช้ L2](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Ridge: penalty term และผลของ alpha](https://scikit-learn.org/1.7/auto_examples/linear_model/plot_ridge_coeffs.html)
- [ตัวอย่างศัพท์ภาษาไทย: การถดถอยที่ปรับด้วยฟังก์ชันการลงโทษ — วารสารวิทยาศาสตร์และเทคโนโลยี](https://li01.tci-thaijo.org/index.php/tstj/article/download/181573/165385)
