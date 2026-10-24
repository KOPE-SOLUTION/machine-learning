# Machine Learning ลงมือทำด้วย Python 🇹🇭

หลักสูตร Machine Learning ภาษาไทยสำหรับผู้ที่เขียน Python และใช้ Jupyter notebook, NumPy, pandas และ Matplotlib ได้แล้ว โดยเริ่มเรียน ML ได้จากพื้นฐาน เนื้อหาเริ่มจากการตั้งโจทย์และสร้างโมเดลแรก ไปจนถึงการประเมินผล เตรียม feature ลดมิติ จัดกลุ่ม และทำโครงงานที่บันทึกโมเดลกลับมาใช้ได้

**17 บท ML + 1 บทเตรียม environment + 2 บทเสริม Mojo** · ใช้ CPU · ไม่มีบริการเสียเงินหรือ API key · ใช้ข้อมูลใน scikit-learn และข้อมูลจำลอง

ทุกบทเริ่มด้วยพื้นฐานที่ใช้ต่อ อธิบายคำศัพท์ ML ใหม่และไล่ขั้นตอนก่อนรัน มีคำอธิบายสำหรับผู้เรียน โค้ดที่คัดลอกได้ครบชุด ไฟล์ Python, Jupyter notebook, แบบฝึกหัดพร้อมแนวคำตอบ และลิงก์อ่านต่อ เนื้อหาภาษาไทยและตัวอย่างใน repo นี้เขียนขึ้นใหม่

## พื้นฐานก่อนเข้าเรียน

ควรเขียน Python เบื้องต้นและใช้ Jupyter notebook ได้ พร้อมทักษะ NumPy, pandas และ Matplotlib ตามตาราง หากต้องการเรียนหรือทบทวน สามารถดูวิดีโอในช่อง **KOPE SOLUTION** แล้วนำทักษะเดิมมาลงมือทดลองตลอดหลักสูตร:

| เนื้อหาพื้นฐานในช่อง YouTube | ทักษะที่นำมาใช้กับ ML |
|---|---|
| [NumPy](https://www.youtube.com/watch?v=KghryuGZFMU&list=PLtB6tPOso1cYC_CP2LQSj7JlY2_SAAYO3) | indexing, shape, broadcasting, masks และ vectorized computation |
| [pandas](https://www.youtube.com/watch?v=Ejgb1q1A8uU&list=PLtB6tPOso1cY5OS1uyQjvrktkA-rHw2pU) | เลือก/รวมตาราง จัดการ missing values, groupby และรักษา index |
| [Matplotlib](https://www.youtube.com/watch?v=tUmJrKmaos8&list=PLea-bfI_jPIA) | สร้างและอ่านกราฟ distribution, scatter, subplots และ colorbar |

ควรรัน notebook จากต้นจนจบ อ่าน traceback และค้น help ได้ด้วย ส่วน scikit-learn, train/test split, metrics และ gradient descent จะเริ่มเรียนในหลักสูตรนี้ อ่าน [ทักษะพื้นฐานและแบบตรวจความพร้อม](docs/prerequisites.md) เพื่อเช็กว่าพร้อมเริ่มบทแรกหรือยัง

## เตรียม environment ของหลักสูตร

บน Windows แนะนำ **Ubuntu 24.04 / WSL 2** และเก็บ repo ในระบบไฟล์ Linux ตาม [คู่มือ WSL ตั้งแต่ติดตั้งจนถึง clone](docs/wsl.md) หากมี Ubuntu และ Git แล้ว เปิด Ubuntu terminal:

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/KOPE-SOLUTION/machine-learning.git
cd machine-learning
```

`uv` คือเครื่องมือจัดการ Python, environment และแพ็กเกจของแต่ละโปรเจกต์ อ่าน [คู่มือตั้ง environment ตั้งแต่ต้น](docs/setup.md) เพื่อติดตั้ง uv แล้วรัน:

```bash
uv python install 3.12
uv sync --frozen --python 3.12
uv run --frozen python lessons/00-setup/lesson.py
uv run --frozen jupyter lab
```

`uv sync` สร้าง `.venv` และติดตั้งแพ็กเกจตาม `uv.lock`; `uv run` เลือก environment ให้โดยไม่ต้อง activate เอง [บท 00](lessons/00-setup/README.md) อธิบาย environment, interpreter และ kernel พร้อมวิธีสร้าง **venv + pip ด้วยตัวเอง** หากไม่ใช้ uv เตรียมครั้งแรกประมาณ 30–45 นาทีบวกเวลาดาวน์โหลด; ถ้าติดตั้งแล้วใช้ 10–15 นาทีตรวจความพร้อม

ใน JupyterLab รันบท 00 แล้วเริ่ม `lessons/01-ml-workflow/lesson.ipynb` ทางเลือก Windows Python, macOS และ Colab อยู่ใน [คู่มือติดตั้ง](docs/setup.md) ทุกบทหลักและตัวอย่าง Mojo ใช้ CPU ได้ ไม่ต้องติดตั้ง CUDA

แพ็กเกจหลักล็อกรุ่นเพื่อทำซ้ำ: NumPy 2.2.6, pandas 2.3.2, scikit-learn 1.7.1 และ Matplotlib 3.10.3 รองรับ Python 3.11–3.13 โดยใช้ 3.12 เป็นเส้นทางเริ่มต้น `uv.lock` ล็อก dependency ทางอ้อมด้วย เอกสารออนไลน์รุ่น stable อาจใหม่กว่าหลักสูตร; ใช้ลิงก์ scikit-learn 1.7 ในแต่ละบทเมื่อเทียบ API

## เส้นทางการเรียน

```mermaid
flowchart LR
  P["พื้นฐาน: Python, NumPy, pandas, Matplotlib"] --> R["00: ตั้ง environment"]
  R --> A["01–03: โจทย์ ML ตรวจข้อมูล และโมเดลแรก"]
  A --> B["04–10: วัดผลและเลือกโมเดล"]
  B --> C["11–14: ลดมิติและเรียนรู้โครงสร้าง"]
  C --> D["15–17: กลไกการเรียนรู้และโครงงาน"]
  D --> E["M01–M02: Mojo เสริม"]
```

เมื่อ environment พร้อม ให้เริ่มบท 01 แล้วต่อบท 02 ซึ่งใช้ pandas/Matplotlib ตรวจความเหมาะสมของข้อมูลสำหรับ ML โดยเน้น target leakage, index alignment และ feature schema หากต้องการเห็นโครงงานเร็ว ให้เรียน 01–06 → 08 → 10 → 16 แล้วกลับมาเติมบทอื่น บท 17 ช่วยตรวจว่าการแบ่งข้อมูลตรงกับการใช้งานจริงหรือไม่

เวลาโดยประมาณรวม 15–22 ชั่วโมงเมื่ออ่าน รัน และทำแบบฝึกหัด; ไม่ใช่ความยาววิดีโอ วิดีโอประกอบยังไม่มีลิงก์เผยแพร่ใน repo ฉบับนี้ ใช้รหัสบทเทียบกับหัวข้อในชุดวิดีโอได้

## บทเรียน

| บท | เนื้อหา | ชิ้นงานที่ได้ |
|---|---|---|
| 00 | [ตั้ง environment และตรวจข้อมูลก่อนเริ่ม ML](lessons/00-setup/README.md) | ใช้ uv หรือสร้าง venv เอง แล้วตรวจ interpreter/kernel |
| 01 | [Machine Learning คืออะไร และทำไมต้องมี baseline](lessons/01-ml-workflow/README.md) | โจทย์ ML กับ baseline |
| 02 | [ตรวจข้อมูลสำหรับ ML: leakage, schema และ EDA](lessons/02-data-and-plots/README.md) | รายงาน audit และ feature ที่อนุญาตให้ใช้ |
| 03 | [โมเดลแรกด้วย scikit-learn และ KNN](lessons/03-first-classifier/README.md) | Pipeline จำแนกดอกไม้ |
| 04 | [วัดผลให้ตรงโจทย์: accuracy, precision, recall และ F1](lessons/04-validation-and-metrics/README.md) | รายงาน confusion matrix และ metrics |
| 05 | [Linear Regression และการอ่านความผิดพลาด](lessons/05-linear-regression/README.md) | เส้นทำนายและ residual plot |
| 06 | [เตรียมตัวเลข หมวดหมู่ และค่าที่หาย](lessons/06-feature-engineering/README.md) | Pipeline จัดการค่าที่หาย/หมวดหมู่ |
| 07 | [จำแนกข้อความด้วย Naive Bayes](lessons/07-naive-bayes-text/README.md) | ตัวจำแนกข้อความสั้น |
| 08 | [Support Vector Machines และขอบเขตไม่เป็นเส้นตรง](lessons/08-support-vector-machines/README.md) | ขอบเขต linear/RBF SVM |
| 09 | [Decision Trees, Random Forests และ overfitting](lessons/09-trees-and-forests/README.md) | ตาราง train/CV และ feature importance |
| 10 | [เลือก hyperparameter ด้วย CV และ learning curve](lessons/10-tuning-and-learning-curves/README.md) | Grid search และ learning curve |
| 11 | [PCA: ลดมิติโดยเก็บความแปรปรวน](lessons/11-principal-components/README.md) | PCA pipeline และกราฟ variance |
| 12 | [Manifold Learning และการอ่านภาพ t-SNE](lessons/12-manifold-learning/README.md) | ภาพเปรียบเทียบ PCA/t-SNE |
| 13 | [K-means: ค้นกลุ่มโดยไม่มี label](lessons/13-kmeans-clustering/README.md) | กลุ่มข้อมูลและการเลือก k |
| 14 | [Gaussian Mixtures และ Kernel Density Estimation](lessons/14-mixtures-and-density/README.md) | กราฟ GMM/KDE |
| 15 | [เขียน Gradient Descent ด้วย NumPy](lessons/15-gradient-descent/README.md) | Gradient descent จาก NumPy |
| 16 | [โครงงาน: จำแนกภาพตัวเลขและนำโมเดลกลับมาใช้](lessons/16-digits-project/README.md) | โมเดลตัวเลขพร้อมรายงานและไฟล์ทำนาย |
| 17 | [ข้อมูลคนเดิม ข้อมูลเวลา และการประเมินให้ตรงการใช้งาน](lessons/17-time-and-groups/README.md) | แผน split ตามกลุ่มและเวลา |
| M01 | [Gradient descent ด้วย Mojo](mojo/01-gradient-descent/README.md) | คำนวณ w,b ด้วย typed loop |
| M02 | [เรียก scikit-learn จาก Mojo](mojo/02-python-interop/README.md) | ใช้ Python pipeline จากโปรแกรม Mojo |

Mojo เป็นเส้นทางเสริม แยก environment จากบทหลัก ทั้งสองบททดสอบกับ Mojo 1.2.0 บน Ubuntu 24.04 ผ่าน WSL แล้ว อ่าน [สถานะและการติดตั้ง Mojo](mojo/README.md) ก่อนเริ่ม

## วิธีใช้แต่ละบท

1. อ่านโจทย์และแนวคิด แล้วลองคาดการณ์ผลก่อนรัน
2. รัน `lesson.py` หรือเปิด `lesson.ipynb` จากต้นจนจบ เลือกอย่างใดอย่างหนึ่ง
3. เปลี่ยนตัวแปรหนึ่งอย่าง บันทึกผลและอธิบายว่าทำไมจึงเปลี่ยน
4. ทำแบบฝึกหัดก่อนเปิดแนวคำตอบ และเทียบกับแหล่งอ่านต่อ

คัดลอกบล็อก `python` ทั้งชุดไป notebook ใหม่แล้วรันตามลำดับได้ บล็อก `bash` ใช้ใน terminal หลังติดตั้งแพ็กเกจตามคู่มือ โค้ดแต่ละบทเตรียมข้อมูลเอง ไม่ต้องรันบทก่อนหน้าเพื่อสร้างตัวแปร

ทุก `lesson.py` ทำงานได้อิสระ เมื่อรันจากโฟลเดอร์หลัก รูปและโมเดลจะถูกบันทึกใน `outputs/` ซึ่งไม่รวมเข้า Git เพื่อไม่เก็บ artifact ที่เกิดจากแต่ละการทดลอง

## สิ่งที่ควรจำตลอดหลักสูตร

- แยก test ก่อน preprocessing และการตัดสินใจเกี่ยวกับโมเดล
- Fit scaler, imputer, encoder และ PCA เฉพาะข้อมูลฝึกในแต่ละ fold ผ่าน Pipeline
- มี baseline และเลือก metric ตามโจทย์ ไม่เลือกจากตัวเลขที่ดูสูงที่สุด
- ใช้ validation/CV เลือกโมเดล แล้วใช้ test รายงานผลสุดท้าย
- random seed ช่วยทำซ้ำ แต่ไม่ได้รับรองว่าข้อมูลเป็นตัวแทนที่ดี
- ข้อมูลจำลองและชุดข้อมูลเล็กใช้ฝึกแนวคิด ไม่ใช่หลักฐานความพร้อมใช้งานจริง

## เอกสารเสริม

- [เริ่มต้น Ubuntu WSL, ย้ายโฟลเดอร์ และข้อควรรู้เรื่อง GPU](docs/wsl.md)
- [ตั้ง environment ด้วย uv หรือ venv + pip](docs/setup.md)
- [ทักษะพื้นฐานและแบบตรวจความพร้อม](docs/prerequisites.md)
- [คณิตศาสตร์ที่ใช้ในหลักสูตร](docs/math-primer.md)
- [ศัพท์ ML ไทย–อังกฤษ](docs/glossary.md)
- [ที่มาของข้อมูลและขอบเขตการใช้](docs/datasets.md)
- [Model card สำหรับโครงงานตัวเลข](docs/model-card.md)
- [เอกสารอ้างอิงและแหล่งอ่านเพิ่มเติม](docs/references.md)
- [ที่มาของเนื้อหา ตัวอย่าง และภาพในแต่ละบท](docs/content-origin.md)
- [วิธีตรวจซ้ำและสถานะการทดสอบ](docs/verification.md)

## ตรวจการทำงานในเครื่องตัวเอง

```bash
uv run --frozen python scripts/check_materials.py
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python scripts/verify_course.py
uv run --frozen python scripts/verify_course.py --mode readmes
uv run --frozen python scripts/verify_course.py --mode notebooks
```

โหมด `readmes` ดึงบล็อก Python จากหน้าแต่ละบทมารันใน process ใหม่ จึงตรวจเส้นทางที่คัดลอกไปวางโดยตรง ผลตรวจอยู่ใน `outputs/verification/` มี GitHub Actions สำหรับตรวจ Python 3.11/3.13 บน Windows/Linux เมื่อ repo ถูกเผยแพร่ สถานะในเครื่องนี้บันทึกแยกจากสถานะ CI เพื่อไม่สับสนว่าทดสอบแพลตฟอร์มใดแล้ว

## การใช้เนื้อหาและการอ้างอิง

โค้ดและข้อความต้นฉบับของ repo นี้ใช้ [MIT License](LICENSE) แหล่งอ้างอิงและข้อมูลภายนอกคงสิทธิ์ตามเจ้าของเดิม อ่าน [ที่มาและสิทธิ์ของแหล่งอ้างอิง](THIRD_PARTY_NOTICES.md) และ [แผนที่เนื้อหาที่เขียนขึ้นสำหรับหลักสูตร](docs/content-origin.md)

