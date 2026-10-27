"""Render original EP02 teaching diagrams and the actual Cell 5 scatter plot."""
from pathlib import Path
from unittest.mock import patch
import contextlib
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures"
NAVY, BLUE, ORANGE, MUTED = "#142A43", "#1558D6", "#A95000", "#4C6078"


def thai_font():
    for name in ("Noto Sans Thai", "Tahoma", "Leelawadee UI", "Thonburi", "Loma"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        return name
    raise RuntimeError("Install a Thai font such as Noto Sans Thai before rendering.")


def label(ax, x, y, text, size=18, color=NAVY, weight="normal", ha="left"):
    return ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", linespacing=1.4)


def card(ax, x, y, w, h, color="white"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=color, edgecolor="#D9E2EF", linewidth=1.5,
    ))


def arrow(ax, start, end, color=BLUE):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=20, linewidth=2, color=color))


def canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor="#F6F8FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    label(ax, 0.05, 0.94, title, size=28, color=BLUE, weight="bold")
    label(ax, 0.05, 0.875, subtitle, size=17, color=MUTED)
    return fig, ax


def save(fig, suffix, dpi=150):
    stem = OUTPUT / ("02-data-and-plots-" + suffix)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for item in ax.texts:
            box = item.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0, item.get_text()
            assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, item.get_text()
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(
        encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    print(stem.with_suffix(".png"))


def feature_selection(ns):
    fig, ax = canvas(
        "เตรียมข้อมูล · แยก X กับ y แล้วแบ่งฝึก/ทดสอบ",
        "โจทย์: ทำนายสายพันธุ์ Iris จากขนาดกลีบที่วัดได้",
    )
    card(ax, 0.05, 0.61, 0.51, 0.20, "#EDF4FF")
    label(ax, 0.075, 0.765, "X · ขนาดกลีบ 150 ดอก × 4 ค่า", size=21, color=BLUE, weight="bold")
    for x, y, name in zip((0.075, 0.315, 0.075, 0.315),
                          (0.705, 0.705, 0.65, 0.65), ns["data"].feature_names):
        label(ax, x, y, name, size=16)
    card(ax, 0.62, 0.61, 0.33, 0.20, "#FFF5EA")
    label(ax, 0.645, 0.765, "y · คำตอบ 150 ดอก", size=22, color=ORANGE, weight="bold")
    label(ax, 0.645, 0.705, "target: 0, 1, 2", size=21, color=ORANGE)
    label(ax, 0.645, 0.65, "เก็บแยกจาก X", size=19)
    arrow(ax, (0.30, 0.60), (0.30, 0.565))
    arrow(ax, (0.78, 0.60), (0.78, 0.565), ORANGE)

    card(ax, 0.05, 0.43, 0.90, 0.12)
    label(ax, 0.5, 0.512, "train_test_split(X, y, test_size=0.2, ...)",
          size=24, ha="center", weight="bold")
    label(ax, 0.5, 0.46, "แบ่งแถวของ X และ y พร้อมกัน · ดอกไม้แต่ละดอกยังจับคู่กับคำตอบเดิม",
          size=18, ha="center", color=MUTED)
    arrow(ax, (0.265, 0.42), (0.265, 0.37))
    arrow(ax, (0.735, 0.42), (0.735, 0.37), ORANGE)

    card(ax, 0.05, 0.13, 0.43, 0.22, "#EDF4FF")
    label(ax, 0.075, 0.305, "ชุดฝึก · 120 ดอก", size=23, color=BLUE, weight="bold")
    label(ax, 0.075, 0.238, "X_train: 120 แถว × 4 คอลัมน์", size=20)
    label(ax, 0.075, 0.177, "y_train: 120 คำตอบ", size=20)
    card(ax, 0.52, 0.13, 0.43, 0.22, "#FFF5EA")
    label(ax, 0.545, 0.305, "ชุดทดสอบ · 30 ดอก", size=23, color=ORANGE, weight="bold")
    label(ax, 0.545, 0.238, "X_test: 30 แถว × 4 คอลัมน์", size=20)
    label(ax, 0.545, 0.177, "y_test: 30 คำตอบ", size=20)

    label(ax, 0.05, 0.059,
          "EP2 สำรวจเฉพาะ X_train · ใช้ y_train แยกสายพันธุ์ในกราฟ · เก็บชุดทดสอบไว้ก่อน",
          size=18, color=BLUE, weight="bold")
    save(fig, "feature-selection")


def target_leakage(ns):
    fig, ax = canvas(
        "Target leakage · เฉลยปะปนในข้อมูลเข้า",
        "โจทย์: ทำนายสายพันธุ์ของดอกใหม่ที่ยังไม่รู้คำตอบ",
    )
    card(ax, 0.05, 0.62, 0.43, 0.19, "#EDF4FF")
    label(ax, 0.075, 0.764, "X · ข้อมูลที่ใช้ทำนาย", size=23, color=BLUE, weight="bold")
    label(ax, 0.075, 0.701, "ขนาดกลีบ 4 ค่า (cm)", size=22)
    label(ax, 0.075, 0.651, "วัดได้ก่อนรู้สายพันธุ์", size=18, color=BLUE)
    card(ax, 0.52, 0.62, 0.43, 0.19, "#FFF5EA")
    label(ax, 0.545, 0.764, "y · คำตอบที่ต้องการทำนาย", size=22, color=ORANGE, weight="bold")
    label(ax, 0.545, 0.701, "target หรือชื่อสายพันธุ์", size=22)
    label(ax, 0.545, 0.651, "เก็บแยกไว้สอนตอนฝึกและตรวจผล", size=18, color=ORANGE)

    card(ax, 0.05, 0.415, 0.90, 0.13)
    label(ax, 0.5, 0.506, "ถ้าเพิ่มชื่อสายพันธุ์ลง X ก่อนแบ่งชุด",
          size=24, color=ORANGE, weight="bold", ha="center")
    label(ax, 0.5, 0.452,
          f'เช่น X มีทั้งขนาดกลีบและคำว่า "{ns["data"].target_names[0]}" ซึ่งบอกคำตอบอยู่แล้ว',
          size=19, ha="center")
    arrow(ax, (0.265, 0.405), (0.265, 0.35), ORANGE)
    arrow(ax, (0.735, 0.405), (0.735, 0.35), ORANGE)

    card(ax, 0.05, 0.15, 0.43, 0.18, "#FFF5EA")
    label(ax, 0.075, 0.28, "X_train มีเฉลยติดไปด้วย", size=22, color=ORANGE, weight="bold")
    label(ax, 0.075, 0.211, "โมเดลอาจเรียนจากชื่อสายพันธุ์", size=19)
    card(ax, 0.52, 0.15, 0.43, 0.18, "#FFF5EA")
    label(ax, 0.545, 0.28, "X_test ก็มีเฉลยให้ดู", size=22, color=ORANGE, weight="bold")
    label(ax, 0.545, 0.211, "คะแนนอาจดูดีเกินการใช้งานจริง", size=19)

    label(ax, 0.05, 0.065,
          "โค้ดบทนี้เลือกเฉพาะขนาดกลีบเข้า X · เก็บคำตอบแยกใน y",
          size=20, color=BLUE, weight="bold")
    save(fig, "target-leakage")


def schema(ns):
    cm = ns["X"].iloc[0].to_numpy()
    mm = cm * 10
    restored = mm / 10
    np.testing.assert_allclose(cm, restored)
    fig, ax = canvas(
        "Schema · ครบ 4 คอลัมน์ ยังต้องตรวจหน่วย",
        "ชื่อคอลัมน์  •  ลำดับ  •  ชนิดข้อมูล  •  หน่วย  ต้องสอดคล้องกับข้อมูลที่ใช้ฝึก",
    )
    xs = (0.38, 0.54, 0.70, 0.86)
    headings = ("sepal\nlength", "sepal\nwidth", "petal\nlength", "petal\nwidth")
    card(ax, 0.05, 0.61, 0.90, 0.20, "#EDF4FF")
    label(ax, 0.075, 0.742, "หน่วยที่ใช้ฝึก", size=20, color=BLUE, weight="bold")
    label(ax, 0.075, 0.667, "เซนติเมตร (cm)", size=20, color=BLUE)
    for x, heading, value in zip(xs, headings, cm):
        label(ax, x, 0.758, heading, size=17, weight="bold", ha="center")
        label(ax, x, 0.666, f"{value:.1f}", size=29, color=BLUE, ha="center", weight="bold")

    card(ax, 0.05, 0.385, 0.90, 0.155, "#FFF5EA")
    label(ax, 0.075, 0.485, "ถ้ารับค่าเป็น mm", size=21, color=ORANGE, weight="bold")
    label(ax, 0.075, 0.422, "ต้องแปลงก่อนใช้ในช่อง cm", size=16, color=ORANGE)
    for x, value in zip(xs, mm):
        label(ax, x, 0.464, f"{value:.0f}", size=29, color=ORANGE, ha="center", weight="bold")
    arrow(ax, (0.64, 0.376), (0.64, 0.329))
    label(ax, 0.67, 0.352, "ทุกค่า ÷ 10", size=18, color=BLUE)

    card(ax, 0.05, 0.165, 0.90, 0.155, "#EDF4FF")
    label(ax, 0.075, 0.265, "แปลงกลับเป็น cm", size=21, color=BLUE, weight="bold")
    label(ax, 0.075, 0.202, "51 mm = 5.1 cm", size=19, color=BLUE)
    for x, value in zip(xs, restored):
        label(ax, x, 0.244, f"{value:.1f}", size=29, color=BLUE, ha="center", weight="bold")
    label(ax, 0.05, 0.101, "ใส่ 51 ลงในช่อง cm จะกลายเป็น 51 cm  •  ใหญ่กว่าขนาดที่ตั้งใจ 10 เท่า",
          size=19, color=ORANGE, weight="bold")
    label(ax, 0.05, 0.047,
          "ตัวอย่างหน่วยโดยอิงค่าดอกแรกของ Iris; ย่อชื่อคอลัมน์ในภาพ  •  ข้อมูลในโค้ดมีหน่วย cm อยู่แล้ว",
          size=16, color=MUTED)
    save(fig, "schema")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "ml-ep02"})
    # Execute Cells 1–5 only. Optional Cell 6 writes to the learner's practice folder.
    source = (ROOT / "lessons/02-data-and-plots/lesson.py").read_text(encoding="utf-8")
    source = source.split("# %% Cell 6", 1)[0]
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, "show"):
        exec(compile(source, "ep02-cells-1-5", "exec"), ns)
    assert len(ns["X_train"]) == 120 and len(ns["X_test"]) == 30
    assert len(ns["fig"].axes[0].collections) == 3
    assert ns["fig"].axes[0].get_xlabel() == ns["data"].feature_names[2]
    assert ns["fig"].axes[0].get_ylabel() == ns["data"].feature_names[3]
    save(ns["fig"], "iris", dpi=140)
    plt.rcParams.update({"font.family": thai_font(), "mathtext.fontset": "dejavusans"})
    feature_selection(ns)
    schema(ns)
    target_leakage(ns)


if __name__ == "__main__":
    main()
