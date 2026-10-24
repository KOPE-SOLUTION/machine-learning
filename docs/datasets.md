# ข้อมูลที่ใช้และขอบเขต

[กลับหน้าแรก](../README.md)

| ชุดข้อมูล | บท | ที่มาและหน่วยข้อมูล | ข้อจำกัด |
|---|---|---|---|
| Iris | 01–03, 10 | [load_iris](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_iris.html): 150 ดอก, 4 ค่าขนาด, 3 คลาส | เล็กและค่อนข้างสะอาด ไม่แทนข้อมูลภาคสนามทั้งหมด |
| Digits | 11–12, 16 | [load_digits](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_digits.html): 1,797 ภาพ 8×8, 10 คลาส, pixel 0–16 | ไม่ใช่ MNIST; split ในหลักสูตรไม่ได้แยกผู้เขียน |
| Binary classification | 04, 09 | [make_classification](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.make_classification.html), สร้างด้วย seed 42 | รูปแบบและ noise ถูกกำหนด ไม่ใช่ benchmark งานจริง |
| Regression / delivery | 05–06, 15 | NumPy random generator และสมการในโค้ด | หน่วยค่าบริการสมมติ ไม่ใช่ราคา/ข้อมูลลูกค้าจริง |
| Text | 07 | ข้อความอังกฤษแต่งขึ้น 24 ประโยค สองหัวข้อ | ข้อมูลน้อย มีคำเฉพาะชัด และไม่ครอบคลุมข้อความทั่วไป |
| Moons / blobs | 08, 13 | [generated datasets](https://scikit-learn.org/1.7/datasets/sample_generators.html) | ใช้แสดงรูปทรง ไม่ใช่กลุ่มผู้ใช้จริง |
| Mixture | 14 | NumPy สุ่ม Gaussian สอง component | รู้กระบวนการกำเนิด ต่างจากงานสำรวจทั่วไป |
| User groups / time series | 17 | สัญญาณประจำกลุ่มและอนุกรมเวลาจำลอง | สาธิต split และ one-step-ahead ไม่ใช่ forecasting benchmark |

Iris และ digits โหลดผ่าน API ของ scikit-learn โดยไม่คัดลอกไฟล์ข้อมูลเข้า repo ข้อมูลและเอกสารของแต่ละชุดมีที่มา/สิทธิ์ของตนเอง เปิดหน้า loader และส่วน References ก่อนนำไปเผยแพร่ต่อในรูปแบบอื่น

ทุกบทหลักรันได้โดยไม่ดาวน์โหลดชุดข้อมูลระหว่างรัน หลังติดตั้ง dependency แล้ว โมเดลและกราฟสร้างใน outputs/ ของผู้เรียนเอง ไม่มีข้อมูลส่วนบุคคลหรือภาพใบหน้าจริงอยู่ในเนื้อหาตัวอย่าง

บท 02 เพิ่มคอลัมน์ `sample_id` และ `species_name` เพื่อฝึกตรวจ feature: รายการแรกเป็นรหัสแถวในตัวอย่าง ส่วนรายการหลังสร้างจาก target โดยตรง ทั้งสองถูกตัดออกจาก feature ก่อนวิเคราะห์เชิงโมเดล

## ใช้ CSV ของตัวเอง

เริ่มจากระบุหนึ่งแถวคืออะไร เวลาที่แต่ละ feature เกิด และคำตอบที่ต้องการ ตรวจแถวซ้ำ ค่าหาย หน่วย กลุ่มบุคคล และลำดับเวลา จากนั้นกำหนด split ให้ตรงการใช้งานก่อนเลือกโมเดล อย่าสลับ dataset ในบทโดยคงวิธี split เดิมโดยไม่ตรวจข้อสมมติ

