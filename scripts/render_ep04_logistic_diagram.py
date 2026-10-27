"""Render the original EP04 Logistic Regression concept diagram."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures"
BLUE, NAVY, ORANGE, MUTED = "#1558D6", "#142A43", "#A95000", "#4C6078"


def thai_font():
    for name in ("Noto Sans Thai", "Tahoma", "Leelawadee UI", "Thonburi", "Loma"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        return name
    raise RuntimeError("Install a Thai font before rendering.")


def text(ax, x, y, value, size=18, color=NAVY, weight="normal", ha="left"):
    return ax.text(x, y, value, ha=ha, va="center", fontsize=size,
                   color=color, fontweight=weight, linespacing=1.45)


def card(ax, x, y, width, height, color):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=color, edgecolor="#D9E2EF", linewidth=1.5,
    ))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": thai_font(), "mathtext.fontset": "dejavusans",
                         "svg.fonttype": "path", "svg.hashsalt": "ml-ep04-logistic"})
    fig = plt.figure(figsize=(16, 9), facecolor="#F6F8FC")
    page = fig.add_axes([0, 0, 1, 1])
    page.set(xlim=(0, 1), ylim=(0, 1))
    page.axis("off")
    text(page, 0.05, 0.94, "Logistic Regression · จากข้อมูลเข้า สู่คำทำนายคลาส",
         size=27, color=BLUE, weight="bold")
    text(page, 0.05, 0.875, "ตัวอย่างสมมติของโมเดลสองคลาส · ตัวเลขนี้ไม่ใช่ผล test ในบท",
         size=18, color=MUTED)
    card(page, 0.05, 0.738, 0.90, 0.085, "white")
    text(page, 0.5, 0.78,
         r"$z = b + w_1x_1 + \cdots + w_px_p$"
         "    →    "
         r"$P(y=1\mid X) = \frac{1}{1+e^{-z}}$",
         size=24, ha="center")

    ax = fig.add_axes([0.09, 0.22, 0.49, 0.43], facecolor="white")
    z = np.linspace(-6, 6, 300)
    p = 1 / (1 + np.exp(-z))
    ax.plot(z, p, color=BLUE, linewidth=3)
    ax.axhline(0.5, color=ORANGE, linestyle="--", linewidth=1.4, label="threshold = 0.5")
    ax.axvline(0, color="#B7C5D6", linestyle=":", linewidth=1)
    examples = np.array([-2.0, 2.0])
    probabilities = 1 / (1 + np.exp(-examples))
    assert np.isclose(probabilities[1], 0.8807970779778823)
    ax.scatter(examples, probabilities, color=ORANGE, s=65, zorder=3)
    ax.annotate("z = -2\np ≈ 0.119", xy=(-2, probabilities[0]), xytext=(-5.3, 0.28),
                color=NAVY, fontsize=14, arrowprops={"arrowstyle": "->", "color": ORANGE})
    ax.annotate("z = 2\np ≈ 0.881", xy=(2, probabilities[1]), xytext=(3.0, 0.67),
                color=NAVY, fontsize=14, arrowprops={"arrowstyle": "->", "color": ORANGE})
    ax.set(xlim=(-6, 6), ylim=(-0.04, 1.04), xlabel="คะแนน z", ylabel="ความน่าจะเป็นคลาส 1")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", fontsize=12)
    text(page, 0.335, 0.69, "Sigmoid แปลงคะแนนเป็น probability", size=20,
         weight="bold", ha="center")

    card(page, 0.65, 0.45, 0.30, 0.21, "#EDF4FF")
    text(page, 0.675, 0.615, "เมื่อคะแนน z = 2", size=22, color=BLUE, weight="bold")
    text(page, 0.675, 0.553, "โอกาสคลาส 1 ≈ 0.881", size=21)
    text(page, 0.675, 0.492, "เกณฑ์ 0.5 → ทายคลาส 1", size=20, color=BLUE)

    card(page, 0.65, 0.21, 0.30, 0.18, "#FFF5EA")
    text(page, 0.675, 0.345, "Probability กับคำทำนาย", size=21, color=ORANGE, weight="bold")
    text(page, 0.675, 0.287, "predict_proba → โอกาสแต่ละคลาส", size=16)
    text(page, 0.675, 0.241, "predict → คลาสที่เลือก", size=17)

    text(page, 0.05, 0.105,
         "fit เรียนน้ำหนักและค่าคงที่จาก train · probability สูงไม่ได้รับประกันว่าคำตอบจะถูก",
         size=18, color=MUTED)
    text(page, 0.05, 0.047,
         "ถ้าปรับ threshold ให้เลือกด้วย validation แล้วค่อยประเมิน test",
         size=20, color=BLUE, weight="bold")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in page.texts:
        box = item.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0, item.get_text()
        assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, item.get_text()
    stem = OUTPUT / "04-validation-and-metrics-logistic"
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(
        encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
