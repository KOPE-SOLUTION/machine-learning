"""ทำนาย CSV ไม่มี header: หนึ่งแถวมี pixel 64 ค่าในช่วง 0..16"""
import argparse
from pathlib import Path

import joblib
import numpy as np

from lesson import validate_pixels

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv", type=Path)
parser.add_argument("--model", type=Path, default=Path("outputs/16-digits-project/model.joblib"))
args = parser.parse_args()
pixels = validate_pixels(np.loadtxt(args.csv, delimiter=",", ndmin=2))
# joblib ใช้ pickle: ใช้เฉพาะไฟล์ที่สร้างเองหรือเชื่อถือแหล่งที่มา
model = joblib.load(args.model)
print(model.predict(pixels).tolist())
