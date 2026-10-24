"""ตั้งโจทย์ แยกข้อมูล และวัด baseline"""
# %% หนึ่งแถวคือดอกไม้หนึ่งดอก
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %% ทายคลาสที่พบมากที่สุดเสมอ
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
pred = baseline.predict(X_test)
print("Train rows:", len(X_train), "Test rows:", len(X_test))
print(f"Baseline test accuracy: {accuracy_score(y_test, pred):.3f}")
print("First five predictions:", pred[:5])
print("First five true labels:", y_test[:5])
print("Correct test predictions:", int((pred == y_test).sum()), "of", len(y_test))
