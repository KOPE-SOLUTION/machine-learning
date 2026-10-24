# พื้นฐานก่อนเข้าเรียน: ต่อจาก Handbook ส่วนที่ 1–4

[กลับหลักสูตร](../README.md)

หลักสูตรนี้เริ่มส่วนที่ 5: Machine Learning โดยถือว่าคุณเรียนส่วนที่ 1–4 ของ [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) มาแล้ว ใช้ NumPy, pandas และ Matplotlib เป็นเครื่องมือในการทดลองได้ ส่วนความรู้ ML เริ่มใหม่ตั้งแต่การตั้งโจทย์

## ทักษะเดิมที่ใช้ต่อทันที

| พื้นฐาน | สิ่งที่ควรทำได้ | จุดที่ใช้ใน ML |
|---|---|---|
| [ส่วนที่ 1: IPython](https://jakevdp.github.io/PythonDataScienceHandbook/01.00-ipython-beyond-normal-python.html) | รัน notebook จาก kernel ใหม่ อ่าน traceback ค้น help และใช้ %timeit | ทำซ้ำการทดลอง ตรวจ error และแยกเวลา fit/predict |
| [ส่วนที่ 2: NumPy](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html) | indexing, shape, axis, masks, broadcasting และ vectorization | สร้าง X/y คำนวณ loss อ่าน shape ของ prediction และ PCA |
| [ส่วนที่ 3: pandas](https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html) | เลือกแถว/คอลัมน์ จัดการ missing, groupby, join/merge และ index | กำหนด feature ตรวจหน่วยตัวอย่าง ป้องกัน label เหลื่อมแถวและข้อมูลซ้ำ |
| [ส่วนที่ 4: Matplotlib](https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html) | ใช้ figure/axes, scatter, histogram, subplots, legend และ colorbar | อ่าน distribution, residual, learning curve และ embedding |

ไม่จำเป็นต้องจำทุก API หรือหัวข้อเฉพาะทั้งหมดของส่วนที่ 1–4 เปิดเอกสารทบทวนขณะทำโจทย์ได้ สิ่งสำคัญคืออ่านโค้ดจัดข้อมูลและกราฟในบทเรียนได้ โดยไม่ต้องเริ่มสอนไวยากรณ์ของเครื่องมือเหล่านั้นใหม่

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
2. เลือก feature สี่คอลัมน์จาก DataFrame และแยก target โดยรักษาแถวให้ตรงกัน อธิบายสิ่งที่หายไปเมื่อใช้ to_numpy()
3. ใช้ mask/groupby ตรวจค่าหายและจำนวนแถวต่อกลุ่ม พร้อมระบุว่า index แทนตัวอย่างใด
4. สร้าง scatter สอง feature พร้อมสีของกลุ่ม ระบุหน่วยแกน และเขียนข้อสังเกตที่กราฟสนับสนุนได้หนึ่งข้อกับข้อสรุปที่กราฟยังยืนยันไม่ได้หนึ่งข้อ

<details>
<summary>เกณฑ์ตรวจคำตอบ</summary>

- Notebook ต้องรันได้จากสถานะใหม่ ไม่อาศัยลำดับการกด cell ที่ซ่อนอยู่
- X/y ต้องอ้างถึงตัวอย่างเดียวกันในลำดับเดียวกัน การแปลงเป็น array ทำให้ไม่มี index/ชื่อคอลัมน์ช่วยจับคู่
- ต้องแยกการนับค่าหายต่อคอลัมน์กับการนับแถว และรู้ว่าหลายแถวอาจเป็นคนหรือเหตุการณ์เดียวกัน
- กราฟช่วยตั้งสมมติฐานเรื่องการกระจายและความสัมพันธ์ แต่ยังไม่รับรองการทำนายบนข้อมูลใหม่หรือเหตุและผล

</details>

หากติดเฉพาะข้อใด ให้กลับไปส่วนที่เกี่ยวข้องในตารางอ้างอิง ไม่จำเป็นต้องเริ่มอ่านทั้งสี่ส่วนใหม่

## เริ่มตรงไหน

1. เตรียม environment ของ repo ตาม [setup](setup.md)
2. ใช้ [บท 00](../lessons/00-setup/README.md) ตรวจรุ่นแพ็กเกจและสัญญา input ของ scikit-learn ประมาณ 10–15 นาที
3. เริ่ม ML ที่ [บท 01](../lessons/01-ml-workflow/README.md) แล้วใช้ [บท 02](../lessons/02-data-and-plots/README.md) ฝึก data audit ที่เกี่ยวกับการทำนาย
4. ต่อบท 03–17 ตามเส้นทางหลัก ส่วน Mojo เริ่มหลังเข้าใจ Pipeline และ gradient descent แล้ว

บท 02 ใช้ทักษะตาราง/กราฟเดิมกับคำถามใหม่: ข้อมูลใดมีให้ใช้จริงตอนทำนาย และข้อมูลใดทำให้การประเมินดูดีเกินจริง
