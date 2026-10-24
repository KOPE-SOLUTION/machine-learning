"""ตรวจ feature schema และความเสี่ยง leakage ก่อนสร้างโมเดล"""
# %% กำหนดหน่วยตัวอย่างและข้อมูลที่ใช้ได้ ณ เวลาทำนาย
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris(as_frame=True)
frame = data.frame.copy()
frame["sample_id"] = np.arange(len(frame))
label_names = dict(enumerate(data.target_names))
frame["species_name"] = frame["target"].map(label_names)  # สร้างจากคำตอบ: ใช้ทำนายไม่ได้

# ระบุ allowlist ตามความหมายข้อมูล ไม่เลือกทุกคอลัมน์นอกจาก target อัตโนมัติ
features = list(data.feature_names)
forbidden = {"target", "species_name", "sample_id"}
assert forbidden.isdisjoint(features)
X, y = frame[features], frame["target"]
assert X.index.equals(y.index)
print("Approved features:", features)
print("Excluded columns:", sorted(forbidden))

# %% กัน test ก่อนใช้กราฟและสถิติเพื่อตัดสินใจเกี่ยวกับโมเดล
train_idx, test_idx = train_test_split(
    frame.index, test_size=0.2, stratify=y, random_state=42
)
assert set(train_idx).isdisjoint(test_idx)
train = frame.loc[train_idx].copy()
print("Train rows:", len(train_idx), "Reserved test rows:", len(test_idx))
print("Training class counts:\n", train["target"].value_counts().sort_index())

# %% ตรวจคุณภาพและยืนยันว่า target ไม่ได้ปะปนใน feature
summary = train[features].agg(["min", "max", "mean", "std"]).T
summary["missing"] = train[features].isna().sum()
summary["dtype"] = train[features].dtypes.astype(str)
print("Training feature audit:\n", summary.round(3))
matrix = train[features].to_numpy()
assert np.isfinite(matrix).all(), "Handle non-finite values inside a training pipeline"
assert len(features) == matrix.shape[1]
print("Duplicate feature rows in train:", train[features].duplicated(keep=False).sum())
print("Constant features in train:", train[features].columns[train[features].nunique() <= 1].tolist())

# %% หนึ่งจุดคือหนึ่งดอก สีใช้ label เพื่อสำรวจ แต่ label ไม่เข้าสู่ X
fig, ax = plt.subplots(figsize=(6, 4))
for label, name in label_names.items():
    subset = train[train["target"] == label]
    ax.scatter(subset[features[2]], subset[features[3]], label=name, alpha=0.75)
ax.set(xlabel=features[2], ylabel=features[3], title="Approved features: training data only")
ax.legend()
fig.tight_layout()
out = Path("outputs/02-data-and-plots")
out.mkdir(parents=True, exist_ok=True)
summary.to_csv(out / "feature-audit.csv", index_label="feature")
fig.savefig(out / "iris.png", dpi=140)
plt.show()
plt.close(fig)
