"""Render original EP06 preprocessing and Ridge teaching diagrams."""
from pathlib import Path

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


def canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor="#F6F8FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    label(ax, 0.05, 0.94, title, size=28, color=BLUE, weight="bold")
    label(ax, 0.05, 0.875, subtitle, size=17, color=MUTED)
    return fig, ax


def label(ax, x, y, text, size=18, color=NAVY, weight="normal", ha="left"):
    return ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", linespacing=1.45)


def card(ax, x, y, w, h, color="white", edge="#D9E2EF"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=color, edgecolor=edge, linewidth=1.5,
    ))


def arrow(ax, start, end, color=BLUE):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=20, linewidth=2, color=color,
    ))


def save(fig, suffix):
    stem = OUTPUT / ("06-feature-engineering-" + suffix)
    fig.canvas.draw()
    # Ensure all text lies within the canvas before exporting.
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for item in ax.texts:
            box = item.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0, item.get_text()
            assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, item.get_text()
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8", newline="\n",
    )
    plt.close(fig)
    print(stem.with_suffix(".png"))


def preprocessing():
    fig, ax = canvas(
        "ColumnTransformer · แยกเตรียม แล้วรวมคอลัมน์",
        "fit เรียนจาก train เท่านั้น  •  predict ใช้ค่าที่เรียนไว้กับ test และข้อมูลใหม่",
    )
    card(ax, 0.05, 0.65, 0.20, 0.15, "#EDF4FF")
    label(ax, 0.15, 0.747, "ตัวเลข", size=20, color=BLUE, weight="bold", ha="center")
    label(ax, 0.15, 0.686, "distance_km, weight_kg", size=15, ha="center")
    card(ax, 0.05, 0.435, 0.20, 0.15, "#FFF5EA")
    label(ax, 0.15, 0.532, "หมวดหมู่", size=20, color=ORANGE, weight="bold", ha="center")
    label(ax, 0.15, 0.471, "service", size=17, ha="center")

    card(ax, 0.30, 0.65, 0.29, 0.15)
    label(ax, 0.445, 0.751, "SimpleImputer: median", size=18, ha="center")
    label(ax, 0.445, 0.687, "→ StandardScaler", size=18, color=BLUE, ha="center")
    card(ax, 0.30, 0.435, 0.29, 0.15)
    label(ax, 0.445, 0.536, "SimpleImputer: most_frequent", size=15, ha="center")
    label(ax, 0.445, 0.471, "→ OneHotEncoder", size=18, color=ORANGE, ha="center")
    arrow(ax, (0.26, 0.725), (0.29, 0.725))
    arrow(ax, (0.26, 0.510), (0.29, 0.510), ORANGE)

    card(ax, 0.65, 0.49, 0.13, 0.26, "#EDF4FF")
    label(ax, 0.715, 0.694, "4 features", size=20, weight="bold", ha="center")
    label(ax, 0.715, 0.604, "ตัวเลข 2\none-hot 2", size=19, ha="center")
    arrow(ax, (0.60, 0.725), (0.64, 0.68))
    arrow(ax, (0.60, 0.510), (0.64, 0.55), ORANGE)
    card(ax, 0.84, 0.49, 0.11, 0.26)
    label(ax, 0.895, 0.681, "Ridge", size=24, color=BLUE, weight="bold", ha="center")
    label(ax, 0.895, 0.578, "↓\nค่าจัดส่ง", size=20, ha="center")
    arrow(ax, (0.79, 0.62), (0.83, 0.62))

    card(ax, 0.05, 0.105, 0.50, 0.265)
    label(ax, 0.075, 0.334, "One-hot: ลำดับคอลัมน์ตรงกับบทนี้", size=18, weight="bold")
    xs = (0.16, 0.345, 0.47)
    for x, text in zip(xs, ("service", "express", "standard")):
        label(ax, x, 0.276, text, size=16, ha="center", weight="bold")
    for y, row in zip((0.224, 0.177, 0.130),
                      (("express", "1", "0"), ("standard", "0", "1"), ("same_day*", "0", "0"))):
        for x, value in zip(xs, row):
            label(ax, x, y, value, size=18, ha="center",
                  color=ORANGE if row[0].startswith("same_day") else NAVY)
    card(ax, 0.59, 0.105, 0.36, 0.265)
    label(ax, 0.615, 0.334, "เติม median: ตัวอย่างสมมติ", size=18, weight="bold")
    label(ax, 0.615, 0.265, "train: [1, 3, 5, NaN] → median = 3", size=17)
    label(ax, 0.615, 0.205, "ค่าหายใน train หรือข้อมูลใหม่ → เติม 3", size=16)
    label(ax, 0.615, 0.145, "จากนั้นจึงปรับสเกลด้วยค่าจาก train", size=16, color=MUTED)
    label(ax, 0.05, 0.048,
          '*same_day ไม่เคยพบ: handle_unknown="ignore" ให้ [0, 0]  •  รันได้ ≠ เรียนรู้บริการใหม่นี้แล้ว',
          size=16, color=ORANGE)
    save(fig, "preprocessing")


def ridge():
    fig, ax = canvas(
        "Ridge · ลดความผิดพลาด พร้อมควบคุมขนาดน้ำหนัก",
        "Linear Regression + L2 regularization  •  ใช้ทำนายตัวเลข",
    )
    card(ax, 0.05, 0.62, 0.90, 0.20)
    label(ax, 0.50, 0.756,
          r"$\sum_{i=1}^{n}(y_i-\hat{y}_i)^2\quad+\quad\alpha\sum_{j=1}^{p}w_j^2$",
          size=29, ha="center")
    label(ax, 0.50, 0.648, "เลือก intercept และน้ำหนัก ให้ค่ารวมนี้ต่ำที่สุดบน train", size=17, ha="center", color=MUTED)

    card(ax, 0.05, 0.385, 0.43, 0.20, "#EDF4FF")
    label(ax, 0.075, 0.544, "ส่วนที่ 1 · ความผิดพลาด", size=22, color=BLUE, weight="bold")
    label(ax, 0.075, 0.478, "รวม (ค่าจริง − ค่าทำนาย)² ทุกแถวใน train", size=19)
    label(ax, 0.075, 0.421, "คือผลรวม residual² ที่ต่อยอดจาก EP05", size=17, color=MUTED)
    card(ax, 0.52, 0.385, 0.43, 0.20, "#FFF5EA")
    label(ax, 0.545, 0.544, "ส่วนที่ 2 · พจน์ลงโทษ", size=22, color=ORANGE, weight="bold")
    label(ax, 0.545, 0.492, "penalty term = alpha × ผลรวม weight²", size=17)
    label(ax, 0.545, 0.445, "เพิ่มในสูตร เพื่อควบคุมขนาดน้ำหนัก", size=17)
    label(ax, 0.545, 0.400, "alpha มาก → ควบคุมเข้มขึ้น", size=15, color=MUTED)

    card(ax, 0.05, 0.15, 0.90, 0.17)
    label(ax, 0.075, 0.272, "ตัวอย่างสมมติ: น้ำหนัก [2, −1] และ alpha = 1", size=21, weight="bold")
    label(ax, 0.075, 0.197, r"penalty $=1\times(2^2+(-1)^2)=5$", size=26, color=ORANGE)
    label(ax, 0.56, 0.197, "ถ้า residual² รวม 100 → ค่ารวม 105", size=18)
    label(ax, 0.05, 0.097,
          "n = จำนวนแถว train  •  p = จำนวน features หลังแปลง  •  w = น้ำหนัก  •  intercept ไม่รวมในพจน์ลงโทษ",
          size=15, color=MUTED)
    label(ax, 0.05, 0.044,
          "EP06 · สูตรสำหรับ fit ไม่ใช่ MAE ที่ใช้ประเมิน  •  ตัวอย่างนี้ไม่ใช่ค่าน้ำหนักที่ได้จาก Cell 2",
          size=15, color=MUTED)
    save(fig, "ridge")


def main():
    plt.rcParams.update({
        "font.family": thai_font(), "mathtext.fontset": "dejavusans",
        "svg.fonttype": "path", "svg.hashsalt": "ml-ep06",
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    preprocessing()
    ridge()


if __name__ == "__main__":
    main()
