# เอกสารอ้างอิงและแผนที่การอ่าน

[กลับหน้าแรก](../README.md)

ตรวจเอกสารประกอบหลักเมื่อ 24 ตุลาคม 2026 แหล่งต่อไปนี้เป็นหนังสือของผู้เขียนและเอกสารจากผู้พัฒนาเครื่องมือ เนื้อหาใน repo เขียนใหม่ให้เชื่อมกับตัวอย่างที่ทำซ้ำได้ ไม่จำเป็นต้องอ่านทั้งหมดก่อนเริ่ม

## จุดเริ่มต้นของผู้เรียน

ก่อนเริ่ม ML ควรใช้ Python/Jupyter, NumPy, pandas และ Matplotlib ได้แล้ว ทบทวนจากวิดีโอในช่อง KOPE SOLUTION: [NumPy](https://www.youtube.com/watch?v=KghryuGZFMU&list=PLtB6tPOso1cYC_CP2LQSj7JlY2_SAAYO3), [pandas](https://www.youtube.com/watch?v=Ejgb1q1A8uU&list=PLtB6tPOso1cY5OS1uyQjvrktkA-rHw2pU) และ [Matplotlib](https://www.youtube.com/watch?v=tUmJrKmaos8&list=PLea-bfI_jPIA)

ดู [รายการทักษะและจุดที่นำไปใช้](prerequisites.md) บท 00 เป็นการตรวจ environment; บท 02 ประยุกต์ทักษะจัดตารางและกราฟเดิมเพื่อป้องกัน leakage และกำหนด feature schema

## อ่านเพิ่มเติม: Python Data Science Handbook

[Jake VanderPlas — บท Machine Learning](https://jakevdp.github.io/PythonDataScienceHandbook/05.00-machine-learning.html) เหมาะสำหรับเชื่อมภาพรวมกับตัวอย่างหลาย algorithm หนังสือออนไลน์มี API จากช่วงเวลาที่เผยแพร่ ให้ใช้โค้ดใน repo และเอกสาร scikit-learn 1.7 เมื่อลองรันใน environment ที่ล็อกไว้

| หัวข้ออ่านคู่จาก Handbook | บทในหลักสูตรนี้ |
|---|---|
| [What Is Machine Learning?](https://jakevdp.github.io/PythonDataScienceHandbook/05.01-what-is-machine-learning.html) | 01 |
| [Introducing Scikit-Learn](https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html) | 03 |
| [Hyperparameters and Model Validation](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html) | 04, 10 |
| [Feature Engineering](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html) | 06, 07 |
| [Naive Bayes](https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html) | 07 |
| [Linear Regression](https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html) | 05, 15 |
| [Support Vector Machines](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html) | 08 |
| [Decision Trees and Random Forests](https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html) | 09 |
| [Principal Component Analysis](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html) | 11 |
| [Manifold Learning](https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html) | 12 |
| [k-Means Clustering](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html) | 13 |
| [Gaussian Mixture Models](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html) | 14 |
| [Kernel Density Estimation](https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html) | 14 |
| [A Face Detection Pipeline](https://jakevdp.github.io/PythonDataScienceHandbook/05.14-image-features.html) | 16: ประยุกต์ workflow กับ digits; ไม่ครอบคลุม face detection |

โครงงานท้ายหลักสูตรใช้ภาพตัวเลขที่โหลดออฟไลน์ได้แทนตัวอย่าง face detection ของหนังสือ จึงเป็นงานจำแนกภาพทั้งภาพ ไม่ใช่การค้นหาตำแหน่งใบหน้า บทเรียนเพิ่มเติมเรื่อง baseline, metrics, gradient descent, model artifact และ group/time split ช่วยให้ต่อเป็น workflow ได้ชัดเจน

## อ่าน API และแนวปฏิบัติ

| แหล่ง | ใช้เมื่อ |
|---|---|
| [scikit-learn 1.7 User Guide](https://scikit-learn.org/1.7/user_guide.html) | ตรวจโมเดล ข้อสมมติ และ API ให้ตรง environment |
| [Common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html) | ตรวจ leakage และ preprocessing |
| [Cross-validation](https://scikit-learn.org/1.7/modules/cross_validation.html) | เลือก random/group/time split |
| [Metrics](https://scikit-learn.org/1.7/modules/model_evaluation.html) | เลือกเกณฑ์ที่ตรงกับความผิดพลาด |
| [Pipelines](https://scikit-learn.org/1.7/modules/compose.html) | รวมขั้นตอนแปลงข้อมูลกับโมเดล |
| [Model persistence](https://scikit-learn.org/1.7/model_persistence.html) | บันทึก/โหลดโมเดลและข้อจำกัดของรูปแบบไฟล์ |
| [NumPy beginner guide](https://numpy.org/doc/stable/user/absolute_beginners.html) | ทบทวน array, shape, axis และ broadcasting |
| [pandas tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html) | อ่านตาราง เลือกแถว groupby และข้อมูลหาย |
| [Matplotlib quick start](https://matplotlib.org/stable/users/explain/quick_start.html) | สร้างและบันทึกกราฟ |

## เพิ่มความเข้าใจ

- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) — ทบทวน regression, classification, loss และระบบ ML ผ่านแบบฝึกและภาพประกอบ เลือกอ่านตามหัวข้อที่ติดขัด
- [An Introduction to Statistical Learning](https://www.statlearning.com/) — เว็บไซต์ของผู้เขียน มีฉบับ Python และแหล่งเรียนเพิ่มเติม เหมาะกับการลงลึกเรื่อง resampling, regularization และสมมติฐานทางสถิติ
- [scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/) — แบบฝึก workflow และการประเมินจากทีมที่ทำงานกับ scikit-learn

## Mojo

- [Install Mojo](https://mojolang.org/install/) และ [System requirements](https://mojolang.org/docs/requirements/) — ตรวจแพลตฟอร์มและวิธีติดตั้งก่อนเริ่ม
- [Quickstart](https://mojolang.org/docs/manual/quickstart/) — syntax ตัวแปร ชนิดข้อมูล และ control flow
- [Python interoperability](https://mojolang.org/docs/manual/python/) — ภาพรวมการเชื่อม Python
- [Calling Python from Mojo](https://mojolang.org/docs/manual/python/python-from-mojo/) — การ import และเลือก Python runtime

เอกสาร Mojo ที่ตรวจแสดงรุ่น 1.2.0 ตัวอย่าง Mojo ใน repo ยังไม่ได้ทดสอบด้วย compiler; อย่าใช้รายการอ้างอิงนี้แทนผลการทดสอบจริง ดู [สถานะเส้นทางเสริม](../mojo/README.md)

## สิทธิ์ของแหล่งข้อมูล

[Handbook ระบุสิทธิ์ข้อความและโค้ดแยกกัน](https://jakevdp.github.io/PythonDataScienceHandbook/) รายละเอียดวิธีใช้แหล่งอ้างอิงใน repo นี้อยู่ที่ [Third-party notices](../THIRD_PARTY_NOTICES.md)

