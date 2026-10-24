# 16 · โครงงาน: จำแนกภาพตัวเลขและนำโมเดลกลับมาใช้

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

เวลาเรียนและทดลองโดยประมาณ: 90–150 นาที  
สิ่งที่จะทำได้: ทำ workflow ตั้งแต่โจทย์จนถึง model artifact พร้อมรายงานผล ข้อจำกัด และ input contract ได้

## ก่อนเริ่มบทนี้

เรียนบท 03–06, 08 และ 10 ก่อน เพราะโครงงานรวม Pipeline, metric, CV และ SVM ไว้ด้วยกัน แต่โค้ดโหลดข้อมูลและฝึกใหม่ทั้งหมด ไม่ต้องอาศัยตัวแปรจาก notebook บทก่อน

## ขอบเขตโครงงาน

ทำนายตัวเลข 0–9 จากภาพ grayscale 8×8 ในชุด digits ที่มากับ scikit-learn แต่ละภาพเป็น 64 pixel ค่า 0–16 ไม่ใช่ MNIST 28×28 และไม่ใช่ระบบอ่านเอกสารหรือภาพถ่ายทั่วไป

เกณฑ์หลักคือ macro F1 เพื่อให้แต่ละคลาสมีน้ำหนักเท่ากัน รายงาน accuracy ควบคู่ และใช้ DummyClassifier เป็น baseline กำหนด grid ก่อนดู test แบ่ง test 20% แบบ stratified แล้วเลือก C/gamma ของ SVC ด้วย 3-fold CV บน train

## สิ่งที่ส่งออก

โปรแกรมบันทึก Pipeline ทั้ง scaler และ SVC, metrics.json, index ของ train/test, input CSV ตัวอย่าง, confusion matrix และภาพข้อผิดพลาดถ้ามี จากนั้นโหลดโมเดลที่เพิ่งสร้างแล้วตรวจว่าทายเหมือนเดิม การเก็บ split index ช่วยตรวจว่าใช้ตัวอย่างใดประเมิน

Input contract กำหนด array สองมิติ มี 64 คอลัมน์อย่างน้อยหนึ่งแถว เป็น finite และอยู่ช่วง 0–16 ภาพใหม่ต้องผ่านขั้นตอนเตรียมภาพที่ตรงกับข้อมูลฝึกด้วย ไม่ใช่แค่ resize แล้วสรุปว่าใช้ได้

ไฟล์ joblib ใช้กลไก pickle ซึ่งเรียกโค้ดได้ระหว่างโหลด จึงโหลดเฉพาะ artifact ที่สร้างเองหรือเชื่อถือ และใช้รุ่นไลบรารีตาม environment เดิม อ่าน [model card](../../docs/model-card.md) เพื่อบันทึกขอบเขตการใช้งาน

## แบ่งโครงงานออกเป็นหกขั้น

| ขั้น | ทำอะไร | หลักฐานที่ตรวจได้ |
|---|---|---|
| 1 | โหลดและตรวจ 1,797 ภาพ แต่ละภาพ 64 pixel | `validate_pixels` ผ่าน shape/ช่วงค่า |
| 2 | กัน test 20% ก่อนเลือกโมเดล | train 1,437 และ test 360 ภาพ |
| 3 | สร้าง baseline และลอง C/gamma รวมสี่คู่ | 3-fold CV รวม 12 fits และ refit ผู้ชนะ |
| 4 | ใช้ macro F1 เลือก แล้วประเมิน test | baseline/model metric และ confusion matrix |
| 5 | บันทึก Pipeline ทั้งเส้นแล้วโหลดกลับ | คำทำนายก่อนและหลังบันทึกตรงกัน |
| 6 | อ่าน CSV สามแถวผ่าน schema เดิม | คำทำนายตัวอย่างพร้อมรัน CLI ต่อได้ |

**Macro F1** คือคำนวณ F1 แบบหนึ่งคลาสเทียบกับคลาสอื่นทีละคลาส แล้วเฉลี่ยสิบคลาสเท่ากัน ส่วน **artifact** คือไฟล์ผลลัพธ์ที่นำกลับมาใช้ เช่น `model.joblib` ไม่ได้เก็บแค่เลขคะแนน แต่เก็บ scaler ที่ fit แล้วและ SVC ที่เลือกไว้ด้วย

ฟังก์ชัน `main()` รวมขั้นตอนที่ต้องรันเมื่อเริ่มโปรแกรม ส่วน `if __name__ == "__main__"` ทำให้การ import ฟังก์ชันตรวจ input จาก `lesson.py` ใน CLI ไม่ฝึกโมเดลซ้ำ หากก๊อปโค้ดไป notebook เปล่า ให้รันทั้งบล็อก; เงื่อนไขนี้ทำงานใน notebook ได้เช่นกัน

## อ่านความผิดพลาดก่อนนำกลับไปใช้

confusion matrix มีเลข 0–9 บนทั้งสองแกน ช่องแนวทแยงคือทายถูก ส่วนช่องแถว 8 คอลัมน์ 3 คือเลขจริง 8 ที่ทายเป็น 3 รูป `errors.png` จะแสดงได้ไม่เกินหกตัวอย่างแรก จึงไม่ใช่ความผิดพลาดทั้งหมด

CSV ใช้ลำดับ pixel ไล่ซ้ายไปขวาทีละแถวจากบนลงล่าง (row-major) ค่า 0 เป็นพื้นหลังและค่าที่สูงกว่าบอกส่วนของเส้นตัวเลข โค้ดใช้ colormap `gray_r` ให้เส้นแสดงเป็นสีเข้ม ภาพที่กลับสีหรือมีช่วง 0–255 ต้องเตรียมให้ตรงความหมายเดิม ไม่ใช่เพียงทำให้ผ่าน shape

## คัดลอกโค้ดไปรัน

สร้างไฟล์ `.ipynb` ใหม่ คัดลอกโค้ดด้านล่างทั้งชุดลง Code cell แล้วกด **Shift+Enter** ใช้โปรแกรมเดิมที่เรียน Matplotlib ได้เลย ดู [วิธีเริ่มเรียน](../../docs/start-here.md) หากต้องการทำตามทีละขั้น

<!-- CODE:START -->
```python
"""โครงงานจำแนกภาพตัวเลข: เลือกโมเดล รายงานข้อผิดพลาด บันทึก Pipeline"""
# %% การตรวจ schema ใช้ร่วมกันทั้งตอนฝึกและทำนาย
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def validate_pixels(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 64 or len(array) == 0:
        raise ValueError("Expected shape (n_samples, 64) with at least one sample")
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 16)):
        raise ValueError("Pixels must be finite and in the interval [0, 16]")
    return array


def build_search():
    return GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("svc", SVC())]),
        {"svc__C": [1.0, 10.0], "svc__gamma": ["scale", 0.01]},
        scoring="f1_macro", cv=StratifiedKFold(3, shuffle=True, random_state=42),
        n_jobs=1,
    )


def main():
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    data = load_digits()
    X = validate_pixels(data.data)
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, stratify=data.target, random_state=42
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = data.target[train_idx], data.target[test_idx]
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    search = build_search().fit(X_train, y_train)
    pred = search.predict(X_test)
    report = {
        "dataset": "sklearn load_digits; 8x8 pixels; range 0..16",
        "split": "stratified random holdout 20%; seed 42; not a writer-group split",
        "train_rows": len(train_idx), "test_rows": len(test_idx),
        "baseline_accuracy": accuracy_score(y_test, baseline.predict(X_test)),
        "baseline_macro_f1": f1_score(y_test, baseline.predict(X_test), average="macro", zero_division=0),
        "best_params": search.best_params_, "cv_macro_f1": search.best_score_,
        "test_accuracy": accuracy_score(y_test, pred),
        "test_macro_f1": f1_score(y_test, pred, average="macro"),
        "classification_report": classification_report(y_test, pred, output_dict=True),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
    }
    print(json.dumps(report, indent=2))
    out = Path("outputs/16-digits-project")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez(out / "split-indices.npz", train=train_idx, test=test_idx)
    np.savetxt(out / "example-input.csv", X_test[:3], delimiter=",")
    restored = joblib.load(out / "model.joblib")  # โหลดเฉพาะไฟล์ที่เพิ่งสร้างเอง
    np.testing.assert_array_equal(pred, restored.predict(X_test))
    csv_pixels = validate_pixels(np.loadtxt(out / "example-input.csv", delimiter=",", ndmin=2))
    print("Predictions from saved CSV:", restored.predict(csv_pixels).tolist())
    print("True labels for these examples:", y_test[:3].tolist())
    errors = np.flatnonzero(pred != y_test)
    print("Misclassified held-out examples:", len(errors))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
    ax.set_title("Final holdout confusion matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion-matrix.png", dpi=140)
    plt.show()
    plt.close(fig)
    if len(errors):
        fig, axes = plt.subplots(1, min(6, len(errors)), figsize=(10, 2), squeeze=False)
        for ax, i in zip(axes[0], errors[:6]):
            ax.imshow(X_test[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
            ax.set_title(f"true={y_test[i]}, pred={pred[i]}")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "errors.png", dpi=140)
        plt.show()
        plt.close(fig)


# %% รันโครงงานตั้งแต่ต้น
if __name__ == "__main__":
    main()
```
<!-- CODE:END -->

## อ่านผลที่ได้

เกิด outputs/16-digits-project/model.joblib, metrics.json, split-indices.npz, example-input.csv และกราฟ ผลจากโมเดลที่โหลดกลับต้องตรงกับโมเดลก่อนบันทึก ค่าทศนิยมอาจต่างเล็กน้อยตามแพลตฟอร์ม

## ทำนายหลังฝึกเสร็จ

Python block ด้านบนสาธิตการบันทึก โหลดโมเดล และทำนายจาก CSV ครบแล้ว ถ้าใช้ Colab ให้ดาวน์โหลดไฟล์ใน `outputs/16-digits-project/` ที่ต้องการเก็บก่อนจบ session

การใช้ไฟล์ `predict.py` เพิ่มเติมอยู่ใน [คู่มือรันไฟล์จาก repo](../../docs/setup.md) ส่วนการทดลองในบทนี้รันครบใน notebook แล้ว

## ภาพตัวอย่างจากการรัน

![ผลการทดลอง 16-digits-project](../../assets/figures/16-digits-project-confusion-matrix.png)

ภาพนี้สร้างจากโค้ดและ seed ในบท เพื่อช่วยอ่านผลก่อนทดลองด้วยตัวเอง

## แบบฝึกหัด

ทำรายงานโครงงานหนึ่งหน้าให้มีโจทย์ baseline วิธี split metric หลัก ผลข้อผิดพลาด และข้อจำกัดอย่างน้อย 3 ข้อ โดยใช้ docs/model-card.md เป็นแนวทาง

<details>
<summary>แนวคำตอบ — ลองทำก่อนเปิดอ่าน</summary>

ข้อจำกัดสำคัญคือภาพขนาด/ช่วงค่าคงที่ การ split ไม่ได้แยกตามผู้เขียน และยังไม่มีการทดสอบภาพภายนอก ถ้านำ error analysis ไปปรับโมเดลต่อ ให้จัด validation ใหม่และกัน test ใหม่สำหรับข้อสรุปสุดท้าย ไม่รายงานผลจากการเลือกซ้ำบน test เดิมเป็น unbiased estimate

</details>

## อ่านต่อ

- [scikit-learn: load_digits](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_digits.html)
- [Model persistence](https://scikit-learn.org/1.7/model_persistence.html)
- [Handbook: Application pipeline](https://jakevdp.github.io/PythonDataScienceHandbook/05.14-image-features.html)

