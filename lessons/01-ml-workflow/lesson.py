# %% 1. นำเข้าเครื่องมือ
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# %% 2. อ่านข้อมูล Iris และแยก feature กับ target
data = load_iris(as_frame=True)
X = data.data
y = data.target

print(X.head(3).to_string(index=False))
print("First three labels:", y.head(3).to_list())
print("Classes:", dict(enumerate(data.target_names.tolist())))

# %% 3. ตรวจรูปร่างและลำดับแถวของ X/y
print("X shape:", X.shape)
print("y shape:", y.shape)
assert X.shape == (150, 4)
assert y.shape == (150,)
assert X.index.equals(y.index)
print("Shape and index checks passed")

# %% 4. แบ่งข้อมูลพร้อมคำตอบเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# %% 5. ตรวจจำนวนคลาสหลังแบ่งข้อมูล
class_counts = pd.DataFrame({
    "train": y_train.value_counts().sort_index(),
    "test": y_test.value_counts().sort_index(),
})
class_counts.index.name = "class"
print(class_counts.to_string())

# %% 6. ฝึก baseline จากชุด train
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
print("Baseline trained")

# %% 7. ทำนายโดยส่งเฉพาะ X_test
pred = baseline.predict(X_test)
print("Predictions:", pred)
print("Prediction shape:", pred.shape)

# %% 8. เปรียบเทียบคำทำนายกับคำตอบจริง
results = pd.DataFrame({"actual": y_test, "predicted": pred})
results["correct"] = results["actual"] == results["predicted"]
print(results.head(10).to_string())

# %% 9. วัดผลด้วย accuracy
accuracy = accuracy_score(y_test, pred)
print(f"Baseline test accuracy: {accuracy:.3f}")
print(f"Correct percentage: {accuracy:.1%}")

# %% 10. คำนวณ accuracy เองจากจำนวนที่ทายถูก
correct_count = int((pred == y_test).sum())
total_count = len(y_test)
manual_accuracy = correct_count / total_count
print("Correct test predictions:", correct_count, "of", total_count)
print(f"Manual accuracy: {manual_accuracy:.3f}")
print("Same result:", manual_accuracy == accuracy)
