# พื้นฐานก่อนเข้าเรียน

[กลับหลักสูตร](../README.md)

หลักสูตรนี้เหมาะสำหรับผู้ที่เขียน Python เบื้องต้นและใช้ NumPy, pandas และ Matplotlib เป็นเครื่องมือในการทดลองได้แล้ว ส่วนความรู้ Machine Learning เริ่มใหม่ตั้งแต่การตั้งโจทย์ หากต้องการทบทวนเครื่องมือพื้นฐาน ใช้วิดีโอในช่อง **KOPE SOLUTION** ตามลิงก์ด้านล่าง

## ทักษะเดิมที่ใช้ต่อทันที

ตารางนี้เป็นทักษะที่ควรเตรียมสำหรับหลักสูตร ML ลิงก์วิดีโอใช้เป็นจุดเริ่มต้นในการเรียนหรือทบทวนแต่ละเครื่องมือ

| พื้นฐาน | สิ่งที่ควรทำได้ | จุดที่ใช้ใน ML |
|---|---|---|
| Python และ Jupyter notebook | ใช้ตัวแปร ฟังก์ชัน และ import; รัน notebook จาก kernel ใหม่ อ่าน traceback และค้น help | อ่านโค้ด ทำซ้ำการทดลอง และตรวจ error |
| [NumPy — วิดีโอในช่อง](https://www.youtube.com/watch?v=KghryuGZFMU&list=PLtB6tPOso1cYC_CP2LQSj7JlY2_SAAYO3) | indexing, shape, axis, masks, broadcasting และ vectorization | สร้าง X/y คำนวณ loss อ่าน shape ของ prediction และ PCA |
| [pandas — วิดีโอในช่อง](https://www.youtube.com/watch?v=Ejgb1q1A8uU&list=PLtB6tPOso1cY5OS1uyQjvrktkA-rHw2pU) | เลือกแถว/คอลัมน์ จัดการ missing, groupby, join/merge และ index | กำหนด feature ตรวจหน่วยตัวอย่าง ป้องกัน label เหลื่อมแถวและข้อมูลซ้ำ |
| [Matplotlib — วิดีโอในช่อง](https://www.youtube.com/watch?v=tUmJrKmaos8&list=PLea-bfI_jPIA) | ใช้ figure/axes, scatter, histogram, subplots, legend และ colorbar | อ่าน distribution, residual, learning curve และ embedding |

ไม่จำเป็นต้องจำทุก API เปิดเอกสารหรือวิดีโอทบทวนขณะทำโจทย์ได้ สิ่งสำคัญคืออ่านโค้ดจัดข้อมูลและกราฟในบทเรียนได้

## สิ่งที่เป็นเนื้อหาใหม่

- ML problem formulation: supervised/unsupervised, feature, target และ baseline
- Train/validation/test, cross-validation, leakage และการประเมินให้ตรงการใช้งาน
- scikit-learn estimator API: fit, transform, predict และ Pipeline
- Classification/regression metrics, model selection และ regularization
- Naive Bayes, SVM, trees/forests, PCA, manifold learning, clustering และ density estimation
- Loss/gradient descent, model artifact และการตรวจ input ตอนทำนาย

คณิตศาสตร์ที่ต้องใช้จะเชื่อมกับโค้ดทีละบท อ่าน [math primer](math-primer.md) เฉพาะจุดที่ต้องการ ไม่ต้องเรียน probability หรือ optimization ครบก่อนเริ่มหลักสูตร

## ตรวจความพร้อม 15–20 นาที

ทำก่อนเริ่มบท 01 หากต้องการทบทวน ไม่ใช่บทฝึกเครื่องมือพื้นฐานชุดใหม่:

1. เปิด notebook ที่ใช้ NumPy/pandas/Matplotlib แล้ว Restart Kernel → Run All โดยไม่มีตัวแปรค้างจากการรันก่อนหน้า
2. เลือกคอลัมน์ตัวเลขสี่คอลัมน์และคอลัมน์หมวดหมู่หนึ่งคอลัมน์จาก DataFrame เดียวกัน โดยรักษาแถวให้ตรงกัน อธิบายสิ่งที่หายไปเมื่อใช้ to_numpy()
3. ใช้ mask/groupby ตรวจค่าหายและจำนวนแถวต่อกลุ่ม พร้อมระบุว่า index แทนตัวอย่างใด
4. สร้าง scatter จากคอลัมน์ตัวเลขสองคอลัมน์ พร้อมสีของกลุ่ม ระบุหน่วยแกน และเขียนข้อสังเกตที่กราฟสนับสนุนได้หนึ่งข้อกับข้อสรุปที่กราฟยังยืนยันไม่ได้หนึ่งข้อ

<details>
<summary>เกณฑ์ตรวจคำตอบ</summary>

- Notebook ต้องรันได้จากสถานะใหม่ ไม่อาศัยลำดับการกด cell ที่ซ่อนอยู่
- ข้อมูลตัวเลขและหมวดหมู่ที่แยกออกมาต้องอ้างถึงรายการเดียวกันในลำดับเดียวกัน การแปลงเป็น array ทำให้ไม่มี index/ชื่อคอลัมน์ช่วยจับคู่
- ต้องแยกการนับค่าหายต่อคอลัมน์กับการนับแถว และรู้ว่าหลายแถวอาจเป็นคนหรือเหตุการณ์เดียวกัน
- กราฟช่วยตั้งสมมติฐานเรื่องการกระจายและความสัมพันธ์ แต่ยังไม่รับรองการทำนายบนข้อมูลใหม่หรือเหตุและผล

</details>

หากติดเฉพาะข้อใด ให้ทบทวนทักษะนั้นจากตารางด้านบน แล้วกลับมาลองตรวจความพร้อมอีกครั้ง

## เริ่มตรงไหน

1. เปิด notebook ที่ถนัดตาม [วิธีเริ่มเรียนจาก README](start-here.md) ใช้ Jupyter หรือ Colab ได้
2. อ่าน [บท 00](../lessons/00-setup/README.md) เพื่อรู้ภาพรวมหลักสูตรและวิธีเรียน ประมาณ 3–5 นาที
3. เริ่ม ML ที่ [บท 01](../lessons/01-ml-workflow/README.md) ตั้งแต่โจทย์ ข้อมูล Iris, sample, feature, target และ X/y จนถึง baseline แล้วใช้ [บท 02](../lessons/02-data-and-plots/README.md) ฝึก data audit ที่เกี่ยวกับการทำนาย
4. ต่อบท 03–17 ตามเส้นทางหลัก ส่วน Mojo เริ่มหลังเข้าใจ Pipeline และ gradient descent แล้ว

บท 02 ใช้ทักษะตาราง/กราฟเดิมกับคำถามใหม่: ข้อมูลใดมีให้ใช้จริงตอนทำนาย และข้อมูลใดทำให้การประเมินดูดีเกินจริง
