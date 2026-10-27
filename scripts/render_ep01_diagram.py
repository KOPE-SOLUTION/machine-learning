"""Render the original EP01 train/predict/evaluate diagram from lesson results."""
from pathlib import Path
import contextlib
import io
import runpy

from thai_figure_text import enable_thai_shaping

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
                   ha=ha, va="center", linespacing=1.45)


def card(ax, x, y, w, h, color="white"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=color, edgecolor="#D9E2EF", linewidth=1.5,
    ))


def arrow(ax, start, end, color=BLUE):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=20, linewidth=2, color=color))


def main():
    enable_thai_shaping()
    plt.rcParams.update({
        "font.family": thai_font(), "mathtext.fontset": "dejavusans",
        "svg.fonttype": "path", "svg.hashsalt": "ml-ep01",
    })
    with contextlib.redirect_stdout(io.StringIO()):
        ns = runpy.run_path(str(ROOT / "lessons/01-ml-workflow/lesson.py"))
    assert ns["X_train"].index.intersection(ns["X_test"].index).empty
    assert ns["X_train"].index.equals(ns["y_train"].index)
    assert ns["X_test"].index.equals(ns["y_test"].index)
    assert ns["pred"].shape == (len(ns["X_test"]),)
    assert set(ns["pred"]) == {0}
    train, test = len(ns["X_train"]), len(ns["X_test"])
    fig = plt.figure(figsize=(16, 9), facecolor="#F6F8FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    label(ax, 0.05, 0.94, "ML workflow · ฝึก → ทำนาย → ตรวจคำตอบ",
          size=28, color=BLUE, weight="bold")
    label(ax, 0.05, 0.875,
          "ตัวอย่าง Iris + DummyClassifier  •  แบ่งข้อมูลและคำตอบพร้อมกัน เพื่อรักษาคู่ของดอกไม้แต่ละดอก",
          size=17, color=MUTED)

    card(ax, 0.05, 0.32, 0.19, 0.45, "#EDF4FF")
    label(ax, 0.145, 0.714, f"Iris {len(ns['X'])} ดอก", size=24, weight="bold", ha="center")
    label(ax, 0.145, 0.642, "X: ขนาดกลีบ 4 ค่า", size=18, ha="center")
    label(ax, 0.145, 0.587, "y: รหัสสายพันธุ์", size=18, ha="center")
    label(ax, 0.145, 0.510, "แบ่ง train / test", size=18, ha="center", color=BLUE)
    label(ax, 0.145, 0.451, "80% / 20%", size=25, ha="center", color=BLUE, weight="bold")
    label(ax, 0.145, 0.376, "หนึ่งแถว = หนึ่งดอก", size=17, ha="center", color=MUTED)
    ax.plot([0.25, 0.28], [0.55, 0.55], color=BLUE, linewidth=2)
    ax.plot([0.28, 0.28], [0.24, 0.72], color=BLUE, linewidth=2)
    for y in (0.72, 0.48, 0.24):
        arrow(ax, (0.28, y), (0.31, y))

    card(ax, 0.32, 0.65, 0.25, 0.14, "#EDF4FF")
    label(ax, 0.445, 0.745, "X_train + y_train", size=22, color=BLUE, weight="bold", ha="center")
    label(ax, 0.445, 0.687, f"ชุดฝึก {train} ดอก · คลาสละ 40", size=17, ha="center")
    card(ax, 0.32, 0.41, 0.25, 0.14)
    label(ax, 0.445, 0.505, "X_test", size=25, color=BLUE, weight="bold", ha="center")
    label(ax, 0.445, 0.447, f"ข้อมูลขนาดกลีบ {test} ดอก", size=18, ha="center")
    card(ax, 0.32, 0.17, 0.25, 0.14, "#FFF5EA")
    label(ax, 0.445, 0.265, "y_test", size=25, color=ORANGE, weight="bold", ha="center")
    label(ax, 0.445, 0.207, f"คำตอบจริงของ {test} ดอกนั้น", size=18, ha="center")

    card(ax, 0.70, 0.65, 0.25, 0.14, "#EDF4FF")
    label(ax, 0.825, 0.745, "fit → baseline", size=23, weight="bold", color=BLUE, ha="center")
    label(ax, 0.825, 0.687, "เรียนความถี่คลาส → เลือก 0", size=17, ha="center")
    card(ax, 0.70, 0.41, 0.25, 0.14)
    label(ax, 0.825, 0.505, "predict → pred", size=23, weight="bold", color=BLUE, ha="center")
    label(ax, 0.825, 0.447, f"ตอบ 0 = setosa ทั้ง {test} ดอก", size=17, ha="center")
    card(ax, 0.70, 0.17, 0.25, 0.14, "#FFF5EA")
    label(ax, 0.825, 0.266, "accuracy_score", size=23, weight="bold", color=ORANGE, ha="center")
    label(ax, 0.825, 0.205,
          f"ถูก {ns['correct_count']}/{ns['total_count']} ≈ {ns['accuracy']:.1%}",
          size=23, color=ORANGE, weight="bold", ha="center")

    for y, caption, color in ((0.72, "ใช้ฝึก", BLUE), (0.48, "ส่งให้ทำนาย", BLUE), (0.24, "ใช้ตรวจผล", ORANGE)):
        arrow(ax, (0.58, y), (0.69, y), color)
        label(ax, 0.635, y + 0.037, caption, size=15, color=color, ha="center")
    arrow(ax, (0.825, 0.64), (0.825, 0.56))
    label(ax, 0.845, 0.598, "ใช้ตัวทำนายเดิม", size=14, color=MUTED)
    arrow(ax, (0.825, 0.40), (0.825, 0.32), ORANGE)
    label(ax, 0.845, 0.358, "นำ pred มาเทียบ", size=14, color=ORANGE)

    label(ax, 0.05, 0.105, "y_test ใช้เฉพาะขั้นวัดผล ไม่ส่งเข้า fit หรือ predict",
          size=22, color=ORANGE, weight="bold")
    label(ax, 0.05, 0.047,
          "ผลตรงกับ Cell 4–10  •  baseline นี้ไม่ใช้ขนาดกลีบในการตัดสิน  •  train/test ไม่มีดอกไม้แถวเดียวกัน",
          size=16, color=MUTED)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "01-ml-workflow-baseline"
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in ax.texts:
        box = item.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0, item.get_text()
        assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, item.get_text()
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(
        encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
