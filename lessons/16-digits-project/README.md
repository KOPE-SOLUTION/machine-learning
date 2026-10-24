# 16 · โครงงาน: จำแนกภาพตัวเลขและนำโมเดลกลับมาใช้

[กลับสารบัญ](../../README.md) · [Notebook](lesson.ipynb) · [Python](lesson.py)

โครงงานนี้ทำนายเลข 0–9 จาก **digits ขนาด 8×8** แล้วบันทึกโมเดลกลับมาใช้

**ใช้สิ่งที่เรียนแล้ว:** Pipeline, metrics, SVM และ GridSearchCV จากบท 03–10 จะรวมเป็นงานเดียวโดยไม่ทวนอัลกอริทึม

## สิ่งที่เพิ่มในโครงงาน

| ขั้นตอน | สิ่งที่ตรวจ |
|---|---|
| ตรวจ input แล้วแบ่งข้อมูล | 64 pixel ต่อภาพ; train 1,437 / test 360 ภาพ |
| เลือก SVC จาก C/gamma สี่คู่ ด้วย 3-fold CV | ใช้ macro F1 ให้สิบคลาสมีน้ำหนักเท่ากัน; เทียบ baseline และรายงาน accuracy ด้วย |
| ประเมิน test และบันทึกผล | metrics, split index, confusion matrix และตัวอย่างที่ทายผิด |
| บันทึกและโหลด Pipeline | คำทำนายก่อน/หลังบันทึกต้องตรงกัน แล้วลองอ่าน CSV มาทำนาย |

**Artifact** คือไฟล์ผลลัพธ์ที่ใช้ต่อ เช่น `model.joblib` ต้องเก็บทั้ง scaler ที่ fit แล้วและโมเดล ไม่ใช่แค่คะแนน ไฟล์ joblib ใช้ pickle ซึ่งเรียกโค้ดขณะโหลดได้ จึงโหลดเฉพาะไฟล์ที่สร้างเองหรือเชื่อถือ และใช้รุ่นไลบรารีตรงกับตอนฝึก

**Input contract** กำหนด array `(n_samples, 64)` อย่างน้อยหนึ่งแถว ค่า finite อยู่ช่วง 0–16 เรียง pixel ซ้ายไปขวาทีละแถวจากบนลงล่าง 0 คือพื้นหลัง และค่าสูงคือเส้นตัวเลข

ผ่าน shape อย่างเดียวยังไม่พอ ภาพใหม่ต้องเตรียมให้ตรงความหมายและช่วงค่าด้วย digits นี้ไม่ใช่ MNIST และยังไม่ใช่ระบบอ่านภาพถ่ายทั่วไป

โค้ดรวมงานใน `main()`; เงื่อนไขท้ายไฟล์เรียกการทดลองเมื่อรัน แต่ไม่ฝึกซ้ำเมื่อไฟล์อื่น import ฟังก์ชันตรวจ input ไปใช้

## ลงมือทำ

สร้าง Notebook ใหม่แล้วรันโค้ดด้านล่างทั้งชุด หรือเปิด [Notebook ของบทนี้](lesson.ipynb) แล้วรันจากบนลงล่าง ใช้ kernel เดิมได้เลย

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

## อ่านผล

ไฟล์อยู่ใน `outputs/16-digits-project/`:
- `model.joblib`, `metrics.json` และ `split-indices.npz`
- `example-input.csv` และ `confusion-matrix.png`
- `errors.png` เมื่อมีตัวอย่างที่ทายผิด แสดงไม่เกินหกตัวอย่างแรก

ช่อง confusion matrix แถว 8 คอลัมน์ 3 คือเลขจริง 8 ที่ทายเป็น 3 โค้ดตรวจผลหลังโหลดโมเดลและทำนายจาก CSV ให้แล้ว ถ้าใช้ Colab ให้ดาวน์โหลดไฟล์ที่ต้องการเก็บก่อนจบ session; วิธีใช้ CLI อยู่ใน [คู่มือ repo](../../docs/setup.md)

![ผลการทดลอง 16-digits-project](../../assets/figures/16-digits-project-confusion-matrix.png)

<details>
<summary>ฝึกต่อพร้อมแนวคำตอบ</summary>

**ลองทำ:** เขียน [model card](../../docs/model-card.md) หนึ่งหน้า: โจทย์ baseline วิธีแบ่ง metric ผล และข้อจำกัด

**แนวคำตอบ:** ระบุอย่างน้อยว่าภาพมีขนาด/ช่วงค่าคงที่ การแบ่งยังไม่แยกผู้เขียน และยังไม่ทดสอบภาพภายนอก หากใช้ข้อผิดพลาดบน test ปรับโมเดลต่อ ต้องกันข้อมูลที่ยังไม่ใช้ตัดสินใจสำหรับประเมินสุดท้าย

</details>

## อ้างอิง

- [scikit-learn: load_digits](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_digits.html)
- [Model persistence](https://scikit-learn.org/1.7/model_persistence.html)
- [Handbook: Application pipeline](https://jakevdp.github.io/PythonDataScienceHandbook/05.14-image-features.html)
