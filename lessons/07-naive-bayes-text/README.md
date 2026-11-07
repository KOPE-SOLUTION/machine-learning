# 07 · จำแนกข้อความด้วย Multinomial Naive Bayes

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เปลี่ยนข้อความเป็นจำนวนคำ แล้วจำแนกว่าเป็น **กีฬา (sport)** หรือ **เทคโนโลยี (tech)** ด้วย Multinomial Naive Bayes

**ใช้สิ่งที่เรียนแล้ว:** Pipeline, train/test และ classification report จากบท 03–06 บทนี้เพิ่มวิธีแปลงข้อความและหลักการของ Naive Bayes

ข้อความฝึกในบทนี้แต่งขึ้น 24 ข้อความ แบ่งเป็น train 18 ข้อความ (sport และ tech อย่างละ 9) และ test 6 ข้อความ ผลจำแนกเป็นชื่อ **sport หรือ tech หนึ่งค่าต่อข้อความ** ก่อนนำไปเทียบกับประเภทข้อความที่กำกับไว้

## ต่อจาก EP6: Feature Engineering สำหรับข้อความ

ใน [EP6](../06-feature-engineering/README.md) เราเตรียมข้อมูลให้ Ridge ใช้ได้ เช่น เปลี่ยนประเภทบริการเป็นคอลัมน์ one-hot ส่วน EP7 เปลี่ยนข้อความเป็น **คอลัมน์จำนวนครั้งของแต่ละคำ** เพื่อให้ Naive Bayes ใช้จำแนกประเภทข้อความ

การแปลงข้อความเป็น features เป็นส่วนหนึ่งของ **Feature Engineering** โดยเรียกงานนี้ให้เฉพาะเจาะจงว่า **Feature Extraction — การสกัดคุณลักษณะ** คือดึงข้อมูลจากข้อความออกมาเป็น features ตัวเลข ในบทนี้เลือกใช้จำนวนครั้งที่พบคำ

| สิ่งที่ใช้ | หน้าที่ในบทนี้ |
|---|---|
| **Bag-of-words** | แนวคิดแทนข้อความด้วยจำนวนครั้งของคำ โดยไม่เก็บลำดับคำ |
| **CountVectorizer** | เครื่องมือเตรียมข้อมูลที่สร้าง vocabulary และแปลงข้อความเป็นตารางจำนวนคำตามแนวคิด Bag-of-words |
| **Multinomial Naive Bayes** | โมเดลที่เรียนจากตารางจำนวนคำคู่กับ labels แล้วจำแนกข้อความเป็น sport หรือ tech |

**ลำดับการใช้งาน:** ข้อความ → CountVectorizer → ตารางจำนวนคำ (features) → Multinomial Naive Bayes → ประเภทข้อความที่ทำนาย

## จากข้อความเป็นตัวเลข

### Bag-of-words: ใส่คำลงถุง แล้วนับแต่ละคำ

**Bag-of-words** แปลตรงตัวได้ว่า **“ถุงคำ” หรือ “กระเป๋าคำ”** ให้นึกว่านำคำในข้อความมาใส่รวมในถุง แล้วดูว่า **มีคำอะไร และแต่ละคำปรากฏกี่ครั้ง** โดยไม่เก็บว่าคำไหนมาก่อนหรือหลัง

เช่น กำหนดรายการคำเป็น `goal, team, wins` แล้วนับคำในแต่ละข้อความ:

| ข้อความ | goal | team | wins |
|---|---:|---:|---:|
| team team goal | 1 | 2 | 0 |
| goal team team | 1 | 2 | 0 |
| team wins | 0 | 1 | 1 |

**อ่านแถวแรก:** พบ goal 1 ครั้ง, team 2 ครั้ง และไม่พบ wins จึงได้ `[1, 2, 0]` ตามลำดับคอลัมน์ ตัวเลขเหล่านี้คือ **features ที่จะส่งให้โมเดล** ส่วน label เช่น sport คือประเภทข้อความที่ใช้กำกับตอนฝึก

แถวที่สองสลับลำดับคำ แต่จำนวนคำเท่าเดิม จึงได้ตัวเลขเหมือนแถวแรก **Bag-of-words แบบนับคำทีละคำในบทนี้จึงแยกสองข้อความนี้ไม่ได้**

### CountVectorizer: เครื่องมือสร้างตารางจำนวนคำ

**Vocabulary คือรายการคำไม่ซ้ำที่ตัวนับคำเก็บไว้จากชุดฝึก** แต่ละคำเป็นหนึ่งคอลัมน์ ส่วนตัวเลขในตารางคือจำนวนครั้งที่พบคำนั้นในแต่ละข้อความ เช่น แม้ team ปรากฏสองครั้ง ก็ยังใช้คอลัมน์ team เพียงคอลัมน์เดียว โดยใส่ค่า 2

- **fit:** ตรวจข้อความใน train แล้วกำหนดว่าจะมีคอลัมน์คำอะไรบ้าง
- **transform:** นับคำในแต่ละข้อความลงคอลัมน์ที่กำหนดไว้ ได้หนึ่งแถวต่อหนึ่งข้อความ

เมื่อแปลง test หรือข้อความใหม่ จะใช้คอลัมน์เดิม โดย **CountVectorizer ข้ามคำที่ไม่มีใน vocabulary ให้อัตโนมัติตอน transform** ใช้ `CountVectorizer()` ได้เลย **ไม่ต้องตั้ง `handle_unknown="ignore"` และ CountVectorizer ไม่มีพารามิเตอร์ `handle_unknown`**

**ข้ามเฉพาะคำที่ไม่รู้จัก ยังนับคำอื่นในข้อความต่อ:** ถ้า vocabulary เรียงเป็น `goal, team, wins` ข้อความ `team vegetables` จะได้ `[0, 1, 0]` เพราะนับ team ได้ 1 ครั้ง ส่วน vegetables ถูกข้าม หากทุกคำไม่มีใน vocabulary เช่น `recipe with vegetables` จะได้ `[0, 0, 0]`

**เทียบกับ OneHotEncoder ใน [EP6](../06-feature-engineering/README.md):**

| เครื่องมือและการตั้งค่า | เมื่อพบคำหรือหมวดหมู่ที่ไม่เคยพบตอน fit |
|---|---|
| `CountVectorizer()` ใน EP7 | ข้ามคำที่ไม่มีใน vocabulary อัตโนมัติ และนับคำที่รู้จักในข้อความนั้นตามปกติ |
| `OneHotEncoder(handle_unknown="ignore")` ใน EP6 | เมื่อพบบริการใหม่ เช่น same_day จะให้เฉพาะคอลัมน์ one-hot ของ service เป็น 0 ทั้งหมด ส่วนคอลัมน์ตัวเลขยังแปลงตามปกติ |

สำหรับ OneHotEncoder ถ้าไม่กำหนด `handle_unknown` ค่าเริ่มต้นคือ `"error"` จึงเกิดข้อผิดพลาดเมื่อ transform พบหมวดหมู่ใหม่ ส่วน CountVectorizer ข้ามคำใหม่ได้โดยไม่ต้องตั้งค่านี้

ทั้งสองกรณีใช้คอลัมน์เดิม **ไม่มีคอลัมน์ใหม่เพิ่มระหว่าง transform** เราจึงแบ่ง train/test ก่อน fit ตัวเตรียมข้อมูล เพื่อให้รายการคำหรือหมวดหมู่ที่ใช้สร้างคอลัมน์มาจาก train เท่านั้น

ค่าเริ่มต้นของ `CountVectorizer` เปลี่ยนตัวอักษรเป็นตัวเล็ก และเลือก token หรือหน่วยคำที่ยาวอย่างน้อย 2 อักขระ เช่น `Team` กลายเป็น `team` ส่วน `a` ไม่ถูกนับ ตัวอย่างหลักของบทนี้ใช้ภาษาอังกฤษ; ตัวอย่างภาษาไทยอยู่ในหัวข้อเสริมท้ายบท

## Naive Bayes จำแนกข้อความอย่างไร

**Naive Bayes** เป็นกลุ่มอัลกอริทึม **Supervised Learning สำหรับ Classification (การจำแนกประเภท)** ใช้กฎของ Bayes ประเมินว่าข้อความน่าจะอยู่ในคลาสใดจากคำที่พบ บทนี้ใช้ **Multinomial Naive Bayes** ผ่าน `MultinomialNB` ซึ่งเหมาะกับข้อมูลการนับ เช่น จำนวนครั้งของแต่ละคำ

### คำอ่านและที่มาของชื่อ Multinomial Naive Bayes

**คำอ่านโดยประมาณ:** **มัล-ที-โน-มี-อัล นา-อีฟ เบย์ซ** โดยเน้นเสียง **โน** ใน Multinomial และ **อีฟ** ใน Naive; Naive มีอีกเสียงว่า **ไน-อีฟ** ตามพจนานุกรม

| ส่วนของชื่อ | ความหมายและที่มา |
|---|---|
| **Multinomial** | ในชื่อโมเดลหมายถึง **การแจกแจงแบบพหุนาม (multinomial distribution)** ซึ่งใช้จำลองจำนวนครั้งของผลลัพธ์หลายประเภท บทนี้ใช้กับจำนวนครั้งของแต่ละคำ เช่น team 2 ครั้ง และ goal 1 ครั้ง |
| **Naive** | แปลตามคำได้ว่า “ไร้เดียงสา” ในชื่อโมเดลสื่อถึง **สมมติฐานที่ทำให้การคำนวณง่ายขึ้น**: เมื่อกำหนดคลาสแล้ว คิดหลักฐานจากแต่ละคำแยกกัน เช่น เมื่อสมมติว่าเป็น sport ใช้ความน่าจะเป็นของ goal กับ team แยกกัน แม้คำสองคำนี้อาจเกี่ยวข้องกันในภาษาจริง |
| **Bayes** | มาจากนามสกุล **Thomas Bayes — โทมัส เบย์ซ** โมเดลใช้ **กฎของเบย์ (Bayes' theorem)** ประเมินความน่าจะเป็นของคลาสจากคำที่พบในข้อความ |

จึงอธิบายชื่อในบริบทบทเรียนได้ว่า **“โมเดลจำแนกข้อความด้วยกฎของเบย์ โดยใช้จำนวนคำและสมมติให้หลักฐานจากคำเป็นอิสระกันเมื่อกำหนดคลาส”**

ฟังคำอ่านและดูที่มาของคำ: [Multinomial](https://www.merriam-webster.com/dictionary/multinomial) · [Naive](https://www.merriam-webster.com/dictionary/naive) · [Bayes' theorem](https://www.merriam-webster.com/dictionary/Bayes%27%20theorem) ส่วนการใช้ multinomial กับจำนวนคำอธิบายไว้ใน [เอกสาร scikit-learn](https://scikit-learn.org/1.7/modules/naive_bayes.html#multinomial-naive-bayes)

### จากตารางจำนวนคำไปเป็นคำทำนาย

โมเดลรับ **features เป็นตารางจำนวนคำ** คู่กับ **class labels เป็น sport/tech** แล้วเรียนรู้ว่าคำแต่ละคำพบมากน้อยเพียงใดในแต่ละประเภท จำนวนครั้งที่ป้อนให้โมเดลต้องไม่ติดลบ จึงส่งจำนวนคำเข้าโมเดลได้เลยโดยไม่ใช้ StandardScaler

![ฝึก CountVectorizer จากข้อความ train แล้วส่งจำนวนคำให้ Naive Bayes ฝึกกับประเภทข้อความ ตอนทำนายใช้ vocabulary และโมเดลเดิม](../../assets/figures/07-naive-bayes-text-workflow.png)

[เปิดภาพ SVG](../../assets/figures/07-naive-bayes-text-workflow.svg)

**ตอนฝึกโมเดลเรียนสองอย่าง:**

1. **Prior — อ่านว่า ไพร-เออร์:** ความน่าจะเป็นของแต่ละคลาสก่อนพิจารณาคำในข้อความที่จะทำนาย คำนวณจากจำนวนข้อความของคลาสนั้นหารด้วยจำนวนข้อความฝึกทั้งหมด
2. **ความน่าจะเป็นของคำภายในแต่ละคลาส:** ใช้จำนวนครั้งของคำแยกตาม sport/tech เพื่อดูว่ามีโอกาสพบแต่ละคำมากน้อยเพียงใด

**ตอนทำนาย** เริ่มคะแนนแต่ละคลาสด้วย prior แล้วคูณความน่าจะเป็นของคำที่พบ โดยคำซ้ำถูกนำมาคูณตามจำนวนครั้ง จากนั้นเลือกคลาสที่ได้คะแนนสูงกว่า

Pipeline เรียงขั้นให้: **CountVectorizer สร้างตารางจำนวนคำก่อน → Naive Bayes ฝึกจากตารางนั้นพร้อม labels** ตอนทำนายใช้ vocabulary และค่าความน่าจะเป็นที่เก็บไว้จากชุดฝึก

- **จุดเด่น:** ฝึกและทำนายได้เร็ว เหมาะเป็นโมเดลเริ่มต้นสำหรับงานจำแนกข้อความ
- **ข้อจำกัด:** การนับคำแบบนี้ไม่เก็บลำดับคำ และความน่าจะเป็นจากโมเดลอาจไม่ตรงกับอัตราที่ทายถูกจริง

## Smoothing: ทำไมต้องมี alpha

**Smoothing (สมูททิง)** เพิ่มจำนวนสมมติ `alpha` ให้ทุกคำในแต่ละคลาสก่อนคำนวณ เพื่อให้คำที่มีใน vocabulary แต่ยังไม่พบในคลาสหนึ่ง มีโอกาสมากกว่าศูนย์ เมื่อ `alpha=1` เรียกว่า **Laplace smoothing**

เช่น หาก goal ไม่เคยพบในข้อความ tech ความน่าจะเป็นของคำนั้นจะเป็นศูนย์และทำให้คะแนน tech ถูกคูณด้วยศูนย์เมื่อข้อความใหม่มี goal การทำ smoothing ช่วยให้โอกาสของคำนั้นมากกว่าศูนย์ โดยเพิ่มเฉพาะจำนวนในสูตร ไม่ได้เพิ่มข้อความฝึก

ส่วนคำที่ไม่มีใน vocabulary เลยจะถูก CountVectorizer ข้าม

## นับคำจากชุดฝึกก่อนแทนสูตร

ข้อมูล 24 ข้อความแบ่งเป็นชุดฝึก 18 ข้อความและชุดทดสอบ 6 ข้อความ ตารางต่อไปนี้แสดงข้อความฝึกทั้งหมดที่ใช้คำนวณมือ และใช้ชุดฝึกเดียวกันในส่วนโค้ด

**จำนวนคำ** นับทุกครั้งที่คำปรากฏ รวมคำซ้ำ ส่วนคอลัมน์ **team** และ **goal** แสดงจำนวนครั้งของสองคำที่จะนำไปใช้ในสูตร

**วิธีนับคำ:** `player scores a goal` นับได้ 3 คำ คือ `player`, `scores`, `goal` เพราะค่าเริ่มต้นของ CountVectorizer ไม่นับคำหนึ่งตัวอักษรอย่าง `a`

**sport — 9 ข้อความ**

| ข้อความฝึก | จำนวนคำ | team | goal |
| --- | --- | --- | --- |
| fans watch the final game | 5 | 0 | 0 |
| coach trains the team | 4 | 1 | 0 |
| goal keeper saves the team | 5 | 1 | 1 |
| training helps the team win | 5 | 1 | 0 |
| coach plans match tactics | 4 | 0 | 0 |
| player scores a goal | 3 | 0 | 1 |
| team wins the football match | 5 | 1 | 0 |
| the club signs a player | 4 | 0 | 0 |
| the match ends in a draw | 5 | 0 | 0 |
| **รวม** | **40** | **4** | **2** |

**tech — 9 ข้อความ**

| ข้อความฝึก | จำนวนคำ | team | goal |
| --- | --- | --- | --- |
| database stores application data | 4 | 0 | 0 |
| python code runs on computer | 5 | 0 | 0 |
| python library update released | 4 | 0 | 0 |
| new computer chip released | 4 | 0 | 0 |
| chip powers the new computer | 5 | 0 | 0 |
| software developer writes tests | 4 | 0 | 0 |
| database server runs software | 4 | 0 | 0 |
| the code has a bug | 4 | 0 | 0 |
| software update fixes a bug | 4 | 0 | 0 |
| **รวม** | **38** | **0** | **0** |

**ตารางสรุปแยกคลาส**

| ประเภท | จำนวนข้อความ | จำนวนคำ | team | goal |
| --- | --- | --- | --- | --- |
| sport | 9 | 40 | 4 | 2 |
| tech | 9 | 38 | 0 | 0 |

อ่านแถว sport: มี **9 ข้อความ** และคำปรากฏรวม **40 ครั้ง** โดยเป็น team **4 ครั้ง** และ goal **2 ครั้ง** ส่วน tech มีคำรวม **38 ครั้ง** และไม่พบทั้ง team กับ goal

**Prior มาจากจำนวนข้อความ:** ทั้งสองคลาสมีคลาสละ `9 / (9 + 9) = 9/18 = 0.5` ส่วนการหาความน่าจะเป็นของคำใช้จำนวนครั้งของคำนั้นและจำนวนคำรวมของคลาส

### Vocabulary: รวมคำไม่ซ้ำจากทั้งสองคลาส

คำปรากฏรวม `40 + 38 = 78` ครั้ง แต่เมื่อตัดคำซ้ำเหลือ **49 คำ** ตามรายการนี้:

```text
application, bug, chip, club, coach, code, computer
data, database, developer, draw, ends, fans, final
fixes, football, game, goal, has, helps, in
keeper, library, match, new, on, plans, player
powers, python, released, runs, saves, scores, server
signs, software, stores, tactics, team, tests, the
training, trains, update, watch, win, wins, writes
```

**เลข 49 ในสูตรมาจาก vocabulary นี้** ซึ่งรวมคำไม่ซ้ำจากทั้ง sport และ tech เมื่อใช้ `alpha=1` จึงเพิ่มจำนวนสมมติคำละ 1 ครบ 49 คำภายในแต่ละคลาส

### นับคำในข้อความใหม่ด้วย vocabulary เดิม

ใช้รายการคำ 49 คำที่รวบรวมไว้เป็นคอลัมน์ของข้อความใหม่ เช่น `team team goal` มี team 2 ครั้งและ goal 1 ครั้ง ตารางต่อไปนี้แสดงเฉพาะ 5 คำที่พบในสามข้อความตัวอย่าง ส่วนอีก 44 คอลัมน์เป็นศูนย์

![ข้อความใหม่ 3 ข้อความผ่าน vocabulary เดิม 49 คำ ตารางแสดงเฉพาะ 5 คำที่พบ โดย team มีจำนวน 2 และข้อความอาหารเป็นศูนย์ทุกคอลัมน์](../../assets/figures/07-naive-bayes-text-counts.png)

[เปิดภาพ SVG](../../assets/figures/07-naive-bayes-text-counts.svg)

### จากตารางนับคำไปสู่ Smoothing

![เปรียบเทียบ goal ใน tech: ไม่เพิ่ม alpha ได้ 0/38 = 0 ทำให้คะแนน tech เป็นศูนย์เมื่อข้อความมี goal; เพิ่ม alpha เท่ากับ 1 ได้ (0+1)/(38+49) = 1/87 ประมาณ 0.0115 ตัวส่วนเพิ่ม 49 เพราะ vocabulary มี 49 คำและเพิ่มคำละ 1](../../assets/figures/07-naive-bayes-text-smoothing.png)

[เปิดภาพ SVG](../../assets/figures/07-naive-bayes-text-smoothing.svg)

**ที่มาของ `(0+1)/(38+49)`:** แถว tech ในตารางสรุปมี goal 0 ครั้งและจำนวนคำรวม 38 ครั้ง ส่วน vocabulary มี 49 คำ และเลข 1 คือ `alpha` ที่กำหนดตอนสร้างโมเดล

**คำที่ไม่มีใน vocabulary เลย** เช่น `vegetables` จะถูก CountVectorizer ข้าม ส่วน smoothing ใช้กับคำที่มีอยู่ใน vocabulary

<details>
<summary>อ่านสูตรและการเลือก alpha เพิ่มเติม</summary>

เช่น goal อยู่ใน vocabulary แต่ไม่เคยพบในข้อความ tech หากไม่ทำ smoothing จะได้ความน่าจะเป็นของ goal ใน tech เป็นศูนย์ เมื่อข้อความใหม่มี goal คะแนนของ tech จึงถูกคูณด้วยศูนย์ แม้คำอื่นอาจสนับสนุน tech อยู่

**Additive smoothing** เพิ่มจำนวนสมมติ `alpha` ให้ทุกคำใน vocabulary ของแต่ละคลาส เมื่อ `alpha=1` จำนวน 0 จะกลายเป็น 1 ในสูตร และจำนวน 3 จะกลายเป็น 4 จำนวนสมมติเพิ่มเฉพาะในสูตร ไม่ได้เพิ่มข้อความฝึก

สูตรคือ **(จำนวนครั้งของคำนั้นในคลาส + alpha) ÷ (จำนวนคำทั้งหมดในคลาส + alpha × จำนวนคำไม่ซ้ำใน vocabulary)** โดยจำนวนคำทั้งหมดนับคำซ้ำด้วย ตัวส่วนต้องเพิ่มด้วย เพราะเราเพิ่มจำนวนสมมติให้ทุกคำ

**แทนค่าจากชุดฝึกของบทนี้:** นับคำในข้อความ tech รวมได้ 38 ครั้ง และ vocabulary จาก train ทั้งสองคลาสมีคำไม่ซ้ำ 49 คำ เมื่อเพิ่มคำละ 1 ตัวส่วนจึงเป็น `38 + (1 × 49) = 87` ส่วน goal ใน tech พบ 0 ครั้ง จึงได้ `(0 + 1) / 87 = 1/87 ≈ 0.0115`

ค่า 0.0115 เป็นความน่าจะเป็นของ **คำ goal ภายในคลาส tech** การหาความน่าจะเป็นว่า **ข้อความเป็น tech** ต้องนำ prior มาคูณกับความน่าจะเป็นของคำที่รู้จัก แล้วหารคะแนนแต่ละคลาสด้วยผลรวมคะแนนทุกคลาส

`alpha` ของบทนี้ควบคุม smoothing **คนละความหมายกับ alpha ของ Ridge ในบท 06** เมื่อเพิ่มมากขึ้น ความน่าจะเป็นของคำภายในแต่ละคลาสจะใกล้กันขึ้น บทนี้กำหนด 1.0 ไว้เรียนหลักการ หากจะเลือกค่าให้ใช้ validation/CV โดยเก็บ test ไว้ประเมินท้ายสุด

</details>

### ตัวอย่างแทนค่า: ใช้ Prior ทำนาย team team goal

อ่านตามลำดับ: **เตรียมค่าจากชุดฝึก → คูณ Prior กับความน่าจะเป็นของคำ → หารด้วยผลรวมคะแนน**

![ขั้นที่ 1: ชุดฝึกมี sport 9 และ tech 9 ข้อความ จึงได้ Prior คลาสละ 9/18 = 0.5; vocabulary 49 คำ ใช้ alpha เท่ากับ 1; sport มีคำรวม 40 ครั้ง พบ team 4 และ goal 2 จึงได้โอกาส 5/89 และ 3/89; tech มีคำรวม 38 ครั้ง ไม่พบทั้งสองคำ จึงได้โอกาสคำละ 1/87](../../assets/figures/07-naive-bayes-text-prediction-inputs.png)

[เปิดภาพที่มาของตัวเลข (SVG)](../../assets/figures/07-naive-bayes-text-prediction-inputs.svg)

![ขั้นที่ 2: team team goal มี team สองครั้งและ goal หนึ่งครั้ง คะแนน sport = 0.5 × (5/89) × (5/89) × (3/89) ประมาณ 0.00005319 ส่วน tech = 0.5 × (1/87) × (1/87) × (1/87) ประมาณ 0.0000007593; ขั้นที่ 3 หารแต่ละคะแนนด้วยผลรวมประมาณ 0.00005395 ได้ sport 0.986 และ tech 0.014](../../assets/figures/07-naive-bayes-text-prediction-scores.png)

[เปิดภาพคำนวณคำทำนาย (SVG)](../../assets/figures/07-naive-bayes-text-prediction-scores.svg)

**ผล:** `predict` เลือก **sport** ส่วน `predict_proba` ให้ sport ≈ **0.986** และ tech ≈ **0.014** แม้ Prior เริ่มเท่ากัน แต่คำที่พบสนับสนุน sport มากกว่า

<details>
<summary>อ่านวิธีแทนค่าทีละขั้นแบบข้อความ</summary>

**1. หาความน่าจะเป็นของคำด้วย smoothing**

จากข้อความใน train ของบทนี้ ฝั่ง sport นับคำรวมได้ 40 ครั้ง โดยพบ team 4 ครั้งและ goal 2 ครั้ง ส่วน tech นับคำรวมได้ 38 ครั้ง และไม่พบทั้ง team กับ goal ใช้ vocabulary 49 คำและ `alpha=1` จึงได้:

| คลาส | ความน่าจะเป็นของ team ภายในคลาส | ความน่าจะเป็นของ goal ภายในคลาส |
|---|---|---|
| sport | (4 + 1) / (40 + 49) = 5/89 | (2 + 1) / (40 + 49) = 3/89 |
| tech | (0 + 1) / (38 + 49) = 1/87 | (0 + 1) / (38 + 49) = 1/87 |

**2. นำ Prior มาคูณหลักฐานจากคำ**

ข้อความ `team team goal` มี team 2 ครั้งและ goal 1 ครั้ง จึงคูณความน่าจะเป็นของ team ซ้ำสองครั้ง โดยแต่ละคลาสใช้ค่าของตัวเอง:

```text
คะแนน sport = 0.5 × (5/89) × (5/89) × (3/89)
            ≈ 0.00005319

คะแนน tech  = 0.5 × (1/87) × (1/87) × (1/87)
            ≈ 0.0000007593
```

คะแนน sport สูงกว่า จึงเลือกคำทำนาย **sport** ได้แล้ว ตัวเลขสองค่านี้เป็น **คะแนนสำหรับเปรียบเทียบคลาส** ซึ่งยังไม่ได้ปรับให้รวมเป็น 1

**3. เปลี่ยนคะแนนเป็นความน่าจะเป็นของคลาส**

นำคะแนนทั้งสองมาบวกกัน แล้วหารคะแนนของแต่ละคลาสด้วยผลรวมเดียวกัน:

```text
ผลรวมคะแนน ≈ 0.00005319 + 0.0000007593 ≈ 0.00005395

ความน่าจะเป็นของ sport ≈ 0.00005319   / 0.00005395 ≈ 0.986
ความน่าจะเป็นของ tech  ≈ 0.0000007593 / 0.00005395 ≈ 0.014
```

นี่คือผลของ `predict_proba` สำหรับข้อความนี้เมื่อแสดงทศนิยม 3 ตำแหน่ง แม้ prior เริ่มเท่ากันที่ 0.5 แต่หลักฐานจากคำต่างกัน จึงได้ความน่าจะเป็นสุดท้ายต่างกัน ส่วน `predict` คืนชื่อคลาส **sport**

</details>

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรัน **Cell 1–3 ตามลำดับ** หรือเปิด [Notebook ของบทนี้](lesson.ipynb) ใช้โค้ดฝึกโมเดลและตรวจผลทำนายกับการคำนวณมือที่อธิบายไว้ก่อนหน้า

<!-- CODE:CELLS -->

<!-- CODE:START -->
### Cell 1 — เตรียมข้อความและประเภทที่กำกับไว้

```python
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts = [
    "team wins the football match", "player scores a goal", "coach trains the team",
    "football fans celebrate victory", "the match ends in a draw", "goal keeper saves the team",
    "basketball team wins tonight", "player joins the football club", "coach plans match tactics",
    "fans watch the final game", "the club signs a player", "training helps the team win",
    "software update fixes a bug", "python code runs on computer", "developer tests the software",
    "new computer chip released", "database stores application data", "the code has a bug",
    "software developer writes tests", "computer program processes data", "python library update released",
    "database server runs software", "chip powers the new computer", "developer fixes python code",
]
y = np.array(["sport"] * 12 + ["tech"] * 12)
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.25, stratify=y, random_state=42
)
print("Train:", len(X_train), "| Test:", len(X_test))
```

**อ่านโค้ด**

- `texts` เก็บข้อความดิบ 24 ข้อความ: 12 ข้อความแรกเป็นกีฬา และ 12 ข้อความหลังเป็นเทคโนโลยี ส่วน `y` เก็บประเภทข้อความ (class labels) เป็น sport และ tech ให้ตรงกับลำดับข้อความ
- `train_test_split` แบ่งข้อความกับประเภทที่กำกับไว้เป็นคู่ ๆ ก่อนสร้างรายการคำ (vocabulary): train 18 ข้อความ (คลาสละ 9) และ test 6 ข้อความ (คลาสละ 3) โดย `stratify=y` รักษาสัดส่วนคลาส
- `random_state=42` ทำให้รันซ้ำแล้วได้ชุดแบ่งเดิม ส่วน `display` ใช้แสดงตารางใน Jupyter แม้มีคำสั่งอื่นตามท้าย cell

**ผลที่ได้:** `Train: 18 | Test: 6`

### Cell 2 — ฝึก Pipeline และประเมินกับ test

```python
model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
display(pd.DataFrame({"text": X_test, "actual": y_test, "prediction": y_pred}))
print(classification_report(y_test, y_pred, zero_division=0))
```

**อ่านโค้ด**

- `make_pipeline(...)` ต่อ **CountVectorizer → MultinomialNB** ส่วน `alpha=1.0` เพิ่มจำนวนสมมติ 1 ให้ทุกคำในแต่ละคลาสตามวิธี smoothing
- `model.fit(X_train, y_train)` ให้ CountVectorizer สร้าง vocabulary และแปลงข้อความฝึกเป็นจำนวนคำก่อน จากนั้น Naive Bayes จึงเรียนจากจำนวนคำคู่กับ labels ใน `y_train`
- `model.predict(X_test)` ใช้ vocabulary และโมเดลที่ฝึกไว้ โดยไม่ fit ใหม่ ตาราง `actual` กับ `prediction` ให้เทียบประเภทจริงกับคำทำนายก่อนอ่าน `classification_report` ตามบท 04
- `zero_division=0` ให้ค่า 0 ในกรณีที่ตัวหารของ metric เป็นศูนย์

**ผลที่ได้:** test ทายถูกทั้ง 6 ข้อความ ได้ accuracy `6/6 = 1.00` คะแนนนี้มาจากข้อความที่แต่งขึ้นเพียง 6 ข้อความ จึงยังไม่ยืนยันผลกับข้อความจริงที่หลากหลาย

ต่อไปทดลองทำนายข้อความใหม่ แล้วเทียบผลกับค่าที่คำนวณมือไว้

### Cell 3 — ดูจำนวนคำและทำนายข้อความใหม่

```python
vectorizer = model.named_steps["countvectorizer"]
words = vectorizer.get_feature_names_out()

examples = ["team team goal", "developer fixes software", "recipe with vegetables"]
vectors = vectorizer.transform(examples)
print("Feature matrix shape:", vectors.shape)

used_columns = np.asarray(vectors.sum(axis=0)).ravel() > 0
counts = pd.DataFrame(
    vectors[:, used_columns].toarray(),
    index=examples,
    columns=words[used_columns],
)
display(counts)

results = pd.DataFrame({
    "text": examples,
    "prediction": model.predict(examples),
    "known_tokens": np.asarray(vectors.sum(axis=1)).ravel(),
})
probabilities = pd.DataFrame(
    model.predict_proba(examples),
    columns=model.classes_,
)
display(pd.concat([results, probabilities], axis=1).round(3))
```

**อ่านโค้ด**

- `model.named_steps["countvectorizer"]` ดึงตัวนับคำที่ฝึกแล้วจาก Pipeline ส่วน `get_feature_names_out()` คืนชื่อคำตามลำดับคอลัมน์
- `examples` มีข้อความกีฬา เทคโนโลยี และอาหาร เพื่อดูทั้งกรณีที่พบคำใน vocabulary และกรณีที่ไม่พบเลย
- `vectorizer.transform(examples)` นับคำลงคอลัมน์เดิม ได้ **Feature matrix หรือตาราง features** ขนาด `(3, 49)`: 3 ข้อความ × 49 คำ ใช้รูปแบบ sparse matrix เพื่อประหยัดพื้นที่สำหรับตารางที่ส่วนใหญ่เป็นศูนย์
- `used_columns` เลือกเฉพาะคำที่พบในตัวอย่างมาแสดง: `sum(axis=0)` รวมจำนวนตามคอลัมน์ และ `np.asarray(...).ravel() > 0` ทำเป็นรายการ True/False หนึ่งมิติ คำที่มีผลรวมมากกว่า 0 ได้ True
- `vectors[:, used_columns].toarray()` เลือก 5 คอลัมน์นั้นแล้วแปลงเป็น array สำหรับแสดงตาราง โมเดลยังใช้ features ครบ 49 คอลัมน์ ไม่ได้ฝึกใหม่ด้วย 5 คอลัมน์
- `known_tokens` มาจาก `vectors.sum(axis=1)` ซึ่งรวมจำนวนคำต่อข้อความ **นับคำซ้ำด้วย** จึงได้ `[3, 3, 0]` เช่น team สองครั้งกับ goal หนึ่งครั้งรวมเป็น 3; ค่า 0 หมายถึงไม่มีคำใดตรงกับ vocabulary
- `model.predict(examples)` รับข้อความดิบแล้วให้ Pipeline นับคำก่อนเลือกคลาส ส่วน `predict_proba` ให้ความน่าจะเป็นของแต่ละคลาส ใช้ `model.classes_` กำกับคอลัมน์ sport, tech ให้ตรงกับลำดับผลลัพธ์
- `pd.concat(..., axis=1)` ต่อข้อความ คำทำนาย และความน่าจะเป็นให้อยู่ในตารางเดียว ส่วน `round(3)` ปัดเลขเฉพาะการแสดงผล
<!-- CODE:END -->

## อ่านผล

**ตรวจผลจาก Cell 3 กับการคำนวณมือ:** ข้อความ `team team goal` ได้ sport ≈ **0.986** และ tech ≈ **0.014** ตรงกับค่าที่คำนวณไว้ ส่วนข้อความอื่นได้ผลดังตาราง (ปัดทศนิยม 3 ตำแหน่ง):

| ข้อความ | คำทำนาย | known_tokens | sport | tech |
|---|---|---:|---:|---:|
| team team goal | sport | 3 | 0.986 | 0.014 |
| developer fixes software | tech | 3 | 0.055 | 0.945 |
| recipe with vegetables | sport | 0 | 0.500 | 0.500 |

**ทำไมข้อความอาหารจึงถูกทายเป็น sport?** คำ recipe, with และ vegetables ไม่มีใน vocabulary ทั้งหมด จึงได้จำนวนคำเป็นศูนย์ทุกคอลัมน์และ `known_tokens=0` โมเดลเหลือเพียง prior ซึ่งเป็น 0.5 เท่ากัน ในกรณีคะแนนเสมอนี้ `predict` เลือก sport ซึ่งเป็นคลาสแรกใน `model.classes_`

โมเดลมีให้เลือกเพียง sport/tech และไม่ได้ปฏิเสธข้อความนอกเรื่องให้อัตโนมัติ ส่วนค่า tech 0.945 ในแถวที่สองเป็นความน่าจะเป็นตามโมเดล **ไม่ได้รับประกันว่าจะทายถูกจริง 94.5%**

## เสริม: ตรวจที่มาของจำนวนคำด้วยโค้ด

<details>
<summary>เปิดโค้ดตรวจข้อความฝึก ตารางนับคำ และ vocabulary</summary>

เลือกทำหลังรัน Cell 1–3: คัดลอกโค้ดนี้ลง Code cell ใหม่ เพื่อตรวจจำนวนที่ใช้คำนวณมือจากตัวนับคำที่ฝึกไว้

```python
vectorizer = model.named_steps["countvectorizer"]
words = vectorizer.get_feature_names_out()
train_vectors = vectorizer.transform(X_train)
train_word_counts = pd.DataFrame(train_vectors.toarray(), columns=words)

train_counts = pd.DataFrame({
    "text": X_train,
    "label": y_train,
    "token_count": train_word_counts.sum(axis=1),
    "team": train_word_counts["team"],
    "goal": train_word_counts["goal"],
})
for label in model.classes_:
    print(label)
    class_rows = train_counts.loc[
        train_counts["label"] == label,
        ["text", "token_count", "team", "goal"],
    ].reset_index(drop=True)
    class_rows.loc[len(class_rows)] = [
        "รวม", *class_rows[["token_count", "team", "goal"]].sum()
    ]
    display(class_rows)

class_summary = train_counts.groupby("label").agg(
    messages=("text", "size"),
    token_count=("token_count", "sum"),
    team=("team", "sum"),
    goal=("goal", "sum"),
)
class_summary["prior"] = class_summary["messages"] / len(X_train)
display(class_summary)

print("Total counted tokens:", train_counts["token_count"].sum())
print("Vocabulary size:", len(words))
for start in range(0, len(words), 7):
    print(", ".join(words[start:start + 7]))
```

**อ่านโค้ดสำคัญ**

- `transform(X_train)` นับคำด้วย vocabulary เดิม แล้วแสดงข้อความคู่กับจำนวนคำเพื่อให้ตรวจทีละแถวได้
- `token_count` คือจำนวนคำทั้งหมดในแต่ละข้อความ โดยนับคำซ้ำด้วย ส่วน `team` และ `goal` เป็นจำนวนครั้งของแต่ละคำ
- `groupby("label").agg(...)` สรุปจำนวนข้อความด้วย `size` และรวมจำนวนคำด้วย `sum`; `prior` คือจำนวนข้อความของคลาสนั้นหารด้วยจำนวนข้อความฝึกทั้งหมด
- ผลรวมของ `token_count` ได้ 78 ครั้ง ส่วน `len(words)` ได้คำไม่ซ้ำ 49 คำ ลูปท้ายแสดงรายการคำครั้งละ 7 คำ

**ผลที่ควรได้:** sport มี 9 ข้อความ คำรวม 40 ครั้ง พบ team 4 และ goal 2 ครั้ง; tech มี 9 ข้อความ คำรวม 38 ครั้ง ไม่พบทั้งสองคำ; Prior คลาสละ 0.5 ตรงกับตารางก่อนแทนสูตร

</details>

## เสริม: ถ้าจะใช้กับข้อความภาษาไทย

<details>
<summary>ดูตัวอย่างตัดคำและ character n-grams</summary>

ภาษาไทยมักเขียนคำติดกัน เช่น `ฉันชอบฟุตบอล` แต่ `CountVectorizer` ค่าเริ่มต้นไม่ได้แยกคำไทยตามความหมายให้ เราจึงเลือกวิธีเตรียมข้อความได้สองแนวทาง:

- **ตัดเป็นคำก่อนนับ:** แยก `ฉันชอบฟุตบอล` เป็น `ฉัน | ชอบ | ฟุตบอล` แล้วนับแต่ละคำเป็น feature ใช้เครื่องมือตัดคำ เช่น [PyThaiNLP](https://pythainlp.org/docs/5.1/api/tokenize.html#word-level) และตั้ง CountVectorizer ให้รับคำที่ตัดแล้ว
- **นับกลุ่มตัวอักษรที่ติดกัน หรือ character n-grams:** เช่น `แมว` เมื่อ n=2 ได้ `แม` กับ `มว`; เมื่อ n=3 ได้ `แมว` กลุ่มเหล่านี้เป็น features ได้แม้ไม่ใช่คำสมบูรณ์

![สองวิธีแปลงข้อความไทย: ตัด ฉันชอบฟุตบอล เป็นสามคำ หรือนับกลุ่มตัวอักษรจาก แมว โดย n=2 ได้ แม และ มว ส่วน n=3 ได้ แมว](../../assets/figures/07-naive-bayes-text-thai-features.png)

[เปิดภาพ SVG](../../assets/figures/07-naive-bayes-text-thai-features.svg)

ตัวนับกลุ่มตัวอักษรกำหนดได้ด้วย `CountVectorizer(analyzer="char", ngram_range=(2, 4))` โดยนับกลุ่มขนาด 2, 3 และ 4 ตัวรวมกัน ตัวอย่าง `แมว` ไม่มีกลุ่มขนาด 4 เพราะมีเพียง 3 ตัวอักษร

เมื่อลองกับภาษาไทย ให้ใช้ข้อความไทยคู่กับ class labels แบ่ง train/test แล้วฝึก Pipeline ใหม่ เพื่อสร้าง vocabulary และเรียนความสัมพันธ์กับประเภทข้อความตามวิธีนับที่เลือก

</details>

## แบบฝึกท้ายบท

1. เปลี่ยนข้อความแรกใน `examples` จาก `team team goal` เป็น `goal team team` แล้วรัน Cell 3 ใหม่ โดยไม่ฝึกโมเดลซ้ำ จำนวนคำและผลทำนายเปลี่ยนหรือไม่?
2. ข้อความ `recipe with vegetables` ได้คำทำนาย sport ทั้งที่ `known_tokens=0` เราควรสรุปว่าโมเดลพบคำเกี่ยวกับกีฬาหรือไม่ เพราะอะไร?

<details>
<summary>ดูแนวคำตอบ</summary>

1. จำนวนคำและผลทำนายเหมือนเดิม เพราะ Bag-of-words แบบนับคำทีละคำในบทนี้ไม่เก็บลำดับคำ
2. ไม่ควรสรุปเช่นนั้น เพราะไม่มีคำใดตรงกับ vocabulary โมเดลใช้ prior ที่เท่ากัน แล้วเลือกคลาสแรกเมื่อคะแนนเสมอ คำทำนายนี้จึงไม่ได้มาจากการพบคำเกี่ยวกับกีฬา

</details>

## อ้างอิง

- [scikit-learn: Naive Bayes — สมมติฐานและข้อจำกัดของความน่าจะเป็น](https://scikit-learn.org/1.7/modules/naive_bayes.html)
- [scikit-learn: MultinomialNB — alpha และ smoothing](https://scikit-learn.org/1.7/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [scikit-learn: CountVectorizer — vocabulary, analyzer และ ngram_range](https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [scikit-learn: OneHotEncoder — handle_unknown และหมวดหมู่ที่ไม่เคยพบ](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [PyThaiNLP: เครื่องมือตัดคำภาษาไทย](https://pythainlp.org/docs/5.1/api/tokenize.html#word-level)
- [scikit-learn: Text feature extraction — Bag-of-words](https://scikit-learn.org/1.7/modules/feature_extraction.html#text-feature-extraction)
- [Handbook: Naive Bayes — อ่านเพิ่มเติม](https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html)
