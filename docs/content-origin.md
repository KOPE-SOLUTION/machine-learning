# ที่มาของเนื้อหาและการใช้แหล่งอ้างอิง

[กลับหน้าแรก](../README.md) · [แหล่งอ่านเพิ่มเติม](references.md) · [สิทธิ์ของงานภายนอก](../THIRD_PARTY_NOTICES.md)

คำอธิบายภาษาไทย ลำดับการทดลอง แบบฝึกหัด แนวคำตอบ และการประกอบตัวอย่างในหลักสูตรนี้เขียนขึ้นสำหรับ repo โดยตั้งต้นจากผู้เรียนที่ใช้ NumPy, pandas และ Matplotlib ได้แล้ว เนื้อหาสังเคราะห์แนวคิด ML และตรวจการใช้ API จากหลายแหล่ง การใช้ชื่อ algorithm สูตรมาตรฐาน หรือ API เดียวกันอาจทำให้โค้ดบางบรรทัดเหมือนตัวอย่างทั่วไปได้

## แหล่งที่ใช้ทำความเข้าใจและตรวจรายละเอียด

| แหล่ง | บทบาท |
|---|---|
| [scikit-learn User Guide 1.7](https://scikit-learn.org/1.7/user_guide.html) | ตรวจความหมายของ estimator, preprocessing, metrics และข้อจำกัดของ API ให้ตรงรุ่นที่ติดตั้ง |
| [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) | อ่านแนวคิด loss, classification, generalization และการประเมิน |
| [scikit-learn MOOC โดย Inria](https://inria.github.io/scikit-learn-mooc/) | อ่านความสัมพันธ์ของการเตรียมข้อมูล Pipeline และ cross-validation |
| [Python Data Science Handbook โดย Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/05.00-machine-learning.html) | แหล่งอ่านเสริมสำหรับภาพรวมและ algorithm ที่ผู้เรียนเลือกศึกษาเพิ่ม |
| [NumPy](https://numpy.org/doc/stable/), [pandas](https://pandas.pydata.org/docs/), [Matplotlib](https://matplotlib.org/stable/) | ตรวจการจัดการ array/ตารางและการสร้างกราฟ |
| [Mojo manual](https://mojolang.org/docs/manual/) | ตรวจภาษาและ Python interoperability ของบทเสริม |
| [Python venv](https://docs.python.org/3.12/library/venv.html), [uv](https://docs.astral.sh/uv/), [Microsoft WSL](https://learn.microsoft.com/en-us/windows/wsl/) | ตรวจขั้นตอนสร้าง environment และทำงานบน Ubuntu WSL |

ใช้เอกสารเพื่อทำความเข้าใจแนวคิดและข้อสัญญาของเครื่องมือ แล้วเขียนคำอธิบาย โจทย์ และโปรแกรมของหลักสูตร พร้อมรันตรวจผลของโปรแกรมนั้น เอกสารอ้างอิงเป็นทางให้ตรวจสอบและอ่านลึกขึ้น การตรวจว่าโค้ดรันผ่านช่วยยืนยันพฤติกรรม แต่ไม่ใช่การตรวจลิขสิทธิ์

## แผนที่ตัวอย่างที่เขียนสำหรับแต่ละบท

ตารางนี้บอกสิ่งที่ลงมือทำใน repo และลิงก์สำหรับตรวจแนวคิด/API ที่เกี่ยวข้อง ไม่ใช่รายการไฟล์ที่คัดลอกจากแหล่งเหล่านั้น

| บท | ตัวอย่างและการอธิบายใน repo | แหล่งตรวจแนวคิด/API |
|---|---|---|
| [00](../lessons/00-setup/README.md) | ตรวจ interpreter และอ่าน shape/index ของ Iris ก่อนเริ่มโมเดล | [Python venv](https://docs.python.org/3.12/library/venv.html), [uv](https://docs.astral.sh/uv/guides/projects/) |
| [01](../lessons/01-ml-workflow/README.md) | เปรียบเทียบคำทำนายคลาสส่วนใหญ่กับจำนวนดอกไม้ในชุดทดสอบ | [scikit-learn: DummyClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.dummy.DummyClassifier.html) |
| [02](../lessons/02-data-and-plots/README.md) | เพิ่ม sample_id และคอลัมน์ที่รั่วจาก target เพื่อฝึกตรวจ schema | [scikit-learn: Common pitfalls และ leakage](https://scikit-learn.org/1.7/common_pitfalls.html) |
| [03](../lessons/03-first-classifier/README.md) | ประกอบ scaler + KNN แล้วทำนายดอกไม้ใหม่ด้วยลำดับ feature เดิม | [scikit-learn: Nearest Neighbors](https://scikit-learn.org/1.7/modules/neighbors.html), [Common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html) |
| [04](../lessons/04-validation-and-metrics/README.md) | จำลองคลาสไม่สมดุล แยก train/validation/test และอ่านความผิดพลาดแต่ละช่อง | [Metrics and scoring](https://scikit-learn.org/1.7/modules/model_evaluation.html), [Cross-validation](https://scikit-learn.org/1.7/modules/cross_validation.html) |
| [05](../lessons/05-linear-regression/README.md) | สร้างสมการพร้อม noise เปรียบเทียบค่าเฉลี่ยกับ regression และอ่าน residual | [scikit-learn: Linear models](https://scikit-learn.org/1.7/modules/linear_model.html) |
| [06](../lessons/06-feature-engineering/README.md) | จำลองงานส่งสินค้า มีหมวดหมู่/ค่าหาย และเปรียบเทียบ Ridge กับ baseline | [Pipelines and composite estimators](https://scikit-learn.org/1.7/modules/compose.html), [Imputation](https://scikit-learn.org/1.7/modules/impute.html) |
| [07](../lessons/07-naive-bayes-text/README.md) | แต่งข้อความอังกฤษ 24 ประโยค สร้างคำศัพท์จาก train แล้วจำแนกสองหัวข้อ | [scikit-learn: Naive Bayes](https://scikit-learn.org/1.7/modules/naive_bayes.html), [Text feature extraction](https://scikit-learn.org/1.7/modules/feature_extraction.html#text-feature-extraction) |
| [08](../lessons/08-support-vector-machines/README.md) | เปรียบเทียบ linear/RBF บนข้อมูล moons และใช้ CV ก่อนรายงาน test | [scikit-learn: SVM](https://scikit-learn.org/1.7/modules/svm.html) |
| [09](../lessons/09-trees-and-forests/README.md) | เปรียบเทียบต้นไม้หลายความลึกกับ forest และอ่าน permutation importance | [scikit-learn: Ensembles](https://scikit-learn.org/1.7/modules/ensemble.html), [Permutation importance](https://scikit-learn.org/1.7/modules/permutation_importance.html) |
| [10](../lessons/10-tuning-and-learning-curves/README.md) | ค้นพารามิเตอร์หกชุดผ่าน Pipeline และวินิจฉัยด้วย learning curve | [Tuning hyperparameters](https://scikit-learn.org/1.7/modules/grid_search.html), [Validation and learning curves](https://scikit-learn.org/1.7/modules/learning_curve.html) |
| [11](../lessons/11-principal-components/README.md) | ใช้ PCA ใน Pipeline จำแนก digits และอธิบาย variance กับภาพสองมิติ | [scikit-learn: PCA](https://scikit-learn.org/1.7/modules/decomposition.html#pca) |
| [12](../lessons/12-manifold-learning/README.md) | เปรียบเทียบ PCA/t-SNE ด้วยสี label ที่ใช้แสดงผล พร้อมข้อจำกัดการตีความ | [scikit-learn: Manifold learning](https://scikit-learn.org/1.7/modules/manifold.html#t-sne) |
| [13](../lessons/13-kmeans-clustering/README.md) | สำรวจจำนวนกลุ่มด้วย silhouette และแยกหมายเลขกลุ่มจากความหมายจริง | [scikit-learn: Clustering](https://scikit-learn.org/1.7/modules/clustering.html) |
| [14](../lessons/14-mixtures-and-density/README.md) | สร้างส่วนผสม Gaussian หนึ่งมิติ เลือก GMM ด้วย BIC และ KDE ด้วย CV | [Gaussian mixtures](https://scikit-learn.org/1.7/modules/mixture.html), [Density estimation](https://scikit-learn.org/1.7/modules/density.html) |
| [15](../lessons/15-gradient-descent/README.md) | เขียน MSE/gradient ด้วย NumPy ตรวจ gradient และการลู่ออก | [Linear models](https://scikit-learn.org/1.7/modules/linear_model.html) |
| [16](../lessons/16-digits-project/README.md) | ฝึก digits พร้อม schema บันทึกโมเดล/รายงาน และโหลดมาทำนาย CSV ใหม่ | [scikit-learn: load_digits](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_digits.html), [Model persistence](https://scikit-learn.org/1.7/model_persistence.html) |
| [17](../lessons/17-time-and-groups/README.md) | จำลองคนซ้ำและข้อมูลเวลา เปรียบเทียบ random/group split และ split ตามเวลา | [Cross-validation: groups and time series](https://scikit-learn.org/1.7/modules/cross_validation.html), [Common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html) |
| [M01](../mojo/01-gradient-descent/README.md) | เขียน typed loop ของ gradient descent และเทียบ w, b, MSE กับ Python reference | [Mojo quickstart](https://mojolang.org/docs/manual/quickstart/) |
| [M02](../mojo/02-python-interop/README.md) | เรียก scikit-learn Pipeline จาก Mojo และเทียบคำทำนายกับ Python | [Calling Python from Mojo](https://mojolang.org/docs/manual/python/python-from-mojo/) |

## โค้ด ภาพ และข้อมูลมาจากไหน

- `lesson.py` เป็นไฟล์ต้นทางของตัวอย่างหลัก โค้ดใน README และ notebook สร้างให้ตรงกับไฟล์นี้ด้วย `scripts/sync_materials.py`
- กราฟใน `assets/figures/` และไฟล์ใน `outputs/` เกิดจากโปรแกรมของหลักสูตร ไม่ได้นำภาพประกอบหรือ screenshot ของหนังสือ/คอร์สอื่นมาแจก
- ข้อมูลข้อความ ค่าบริการ และอนุกรมเวลาจำลองกำหนดขึ้นในโค้ดเพื่ออธิบายประเด็นเฉพาะ ไม่ใช่ข้อมูลลูกค้าจริง
- Iris และ digits โหลดจาก scikit-learn; ชุดข้อมูลเหล่านี้มีผู้สร้างและสิทธิ์ของตนเอง แม้โปรแกรมวิเคราะห์และกราฟจะเขียนขึ้นใหม่ อ่าน [ที่มาของข้อมูล](datasets.md)
- repo ไม่รวมเนื้อหาบทหนังสือที่แปลตามต้นฉบับ notebook ต้นฉบับของแหล่งอื่น หรือภาพของแหล่งอื่นไว้เป็นบทเรียน

## เข้าใจขอบเขตสิทธิ์ก่อนนำไปใช้

ข้อความและโค้ดที่เขียนขึ้นสำหรับ repo ใช้ [MIT License](../LICENSE) ตามประกาศใน repo ส่วนเนื้อหา ข้อมูล และไลบรารีของแหล่งภายนอกใช้สิทธิ์ของเจ้าของเดิม การใส่ลิงก์อ้างอิงไม่ได้เพิ่มสิทธิ์คัดลอกภาพ ยกข้อความยาว หรือเผยแพร่คำแปลจากแหล่งนั้น

[WIPO อธิบายว่าลิขสิทธิ์คุ้มครองรูปแบบการแสดงออกของงาน และแยกจากแนวคิด วิธีการ หรือแนวคิดทางคณิตศาสตร์](https://www.wipo.int/en/web/copyright/faq-copyright) จึงเรียนแนวคิด ML จากหลายแหล่งแล้วอธิบายด้วยข้อความ ตัวอย่าง และภาพของตนเองได้ โดยยังต้องเคารพสิทธิ์ของงานที่นำมาใช้จริง

บันทึกนี้อธิบายการจัดทำและที่มาที่ระบุใน repo ไม่ใช่ผลตรวจความเหมือนกับทุกเว็บไซต์หรือคำรับรองทางกฎหมายว่าไม่มีข้อพิพาทใดเกิดขึ้นได้ หากนำข้อความ รูป หรือโค้ดเพิ่มเติมจากแหล่งอื่นเข้ามา ต้องตรวจเงื่อนไขและเก็บ attribution/license ของชิ้นนั้นตามต้นทาง
