"""Render an original EP05 conceptual figure (not the fitted experiment result)."""
from pathlib import Path

from thai_figure_text import enable_thai_shaping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures/05-linear-regression-residual-explained"
INTERCEPT, SLOPE, X_EXAMPLE, Y_EXAMPLE = 12, 3.5, 2, 21


def thai_font():
    for family in ("Noto Sans Thai", "Tahoma", "Leelawadee UI", "Thonburi", "Loma"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return family
    raise RuntimeError("Install a Thai font such as Noto Sans Thai before rendering.")


def main():
    enable_thai_shaping()
    plt.rcParams.update({
        "font.family": thai_font(), "font.size": 14,
        "svg.fonttype": "path", "svg.hashsalt": "ml-ep05-residual",
    })
    navy, blue, orange, muted = "#142A43", "#1558D6", "#B75700", "#4C6078"
    predicted = INTERCEPT + SLOPE * X_EXAMPLE
    residual = Y_EXAMPLE - predicted
    fig = plt.figure(figsize=(16, 8.5), facecolor="#F6F8FC")
    fig.text(0.05, 0.93, "Linear Regression: เส้นทำนายและ Residual",
             fontsize=27, fontweight="bold", color=navy)
    fig.text(0.05, 0.875, "ตัวอย่างสมมติ: กำหนดเส้นทำนายไว้เพื่อฝึกอ่านภาพ ไม่ใช่ผล fit จาก Cell 2",
             fontsize=16, color=muted)
    ax = fig.add_axes([0.075, 0.20, 0.47, 0.58], facecolor="white")
    x = np.linspace(0, 4, 100)
    ax.plot(x, INTERCEPT + SLOPE * x, color=blue, lw=3, label="เส้นทำนาย")
    ax.vlines(X_EXAMPLE, 10, predicted, colors="#A8B5C5", linestyles="--", lw=1.5)
    ax.plot([X_EXAMPLE, X_EXAMPLE], [predicted, Y_EXAMPLE],
            color=orange, lw=4, zorder=3)
    ax.scatter([X_EXAMPLE], [predicted], marker="D", s=130,
               color=blue, edgecolor="white", zorder=5, label="ค่าทำนาย")
    ax.scatter([X_EXAMPLE], [Y_EXAMPLE], marker="o", s=155,
               color=orange, edgecolor="white", zorder=5, label="ค่าจริง")
    ax.scatter([0], [INTERCEPT], s=65, color=blue, zorder=4)
    ax.annotate("b = 12", xy=(0, INTERCEPT), xytext=(0.15, 10.8),
                fontsize=16, color=blue, arrowprops={"arrowstyle": "-", "color": blue})
    ax.annotate("y = 21", xy=(X_EXAMPLE, Y_EXAMPLE), xytext=(0.9, 22.2),
                fontsize=17, color=orange, arrowprops={"arrowstyle": "-", "color": orange})
    ax.annotate("ŷ = 19", xy=(X_EXAMPLE, predicted), xytext=(2.6, 17.7),
                fontsize=17, color=blue, arrowprops={"arrowstyle": "-", "color": blue})
    ax.annotate("Residual = +2", xy=(X_EXAMPLE, (predicted + Y_EXAMPLE) / 2),
                xytext=(2.65, 20.2), fontsize=16, color=orange,
                arrowprops={"arrowstyle": "->", "color": orange})
    ax.set(xlim=(-0.25, 4.25), ylim=(10, 28), xlabel="Input x", ylabel="Target y")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([12, 16, 20, 24, 28])
    ax.tick_params(colors=muted)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, fontsize=13)
    for spine in ax.spines.values():
        spine.set_color("#BAC8DB")

    def text(x, y, value, size=18, color=navy, weight="normal"):
        fig.text(x, y, value, fontsize=size, color=color, fontweight=weight, va="center")

    def card(y, height, border):
        fig.add_artist(FancyBboxPatch(
            (0.60, y), 0.35, height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            transform=fig.transFigure, facecolor="white", edgecolor=border, linewidth=1.5,
        ))

    card(0.585, 0.19, "#C7D9FC")
    text(0.625, 0.735, "ŷ = b + wx", size=25, color=blue, weight="bold")
    text(0.625, 0.680, "เมื่อ x = 2", size=18)
    text(0.625, 0.625, f"ŷ = 12 + 3.5 × 2 = {predicted:g}", size=22, weight="bold")
    card(0.355, 0.18, "#F2D6B7")
    text(0.625, 0.492, "Residual = ค่าจริง − ค่าทำนาย", size=19, weight="bold")
    text(0.625, 0.438, f"{Y_EXAMPLE} − {predicted:g} = +{residual:g}", size=26, color=orange, weight="bold")
    text(0.625, 0.385, "ค่าจริงอยู่เหนือเส้น → ทายต่ำไป 2", size=18)
    card(0.17, 0.13, "#D9E2EF")
    text(0.625, 0.257, "Residual บวก → ทายต่ำกว่าค่าจริง", size=17)
    text(0.625, 0.210, "Residual ลบ → ทายสูงกว่าค่าจริง", size=17)
    fig.text(0.075, 0.115, "w = 3.5: เมื่อ x เพิ่ม 1 ค่าทำนายเพิ่ม 3.5", fontsize=17, color=navy)
    fig.text(0.05, 0.045, "EP05 · Residual วัดระยะแนวดิ่งที่ x เดียวกัน และมีหน่วยเดียวกับ y",
             fontsize=15, color=muted)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=150)
    svg = OUTPUT.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8", newline="\n",
    )
    plt.close(fig)
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".svg"))


if __name__ == "__main__":
    main()
