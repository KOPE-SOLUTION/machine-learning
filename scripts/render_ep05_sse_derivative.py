"""Original EP05 figure: a zero derivative needs a minimum condition."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from render_ep05_diagram import thai_font
from render_ep05_sensor_case import (
    X, Y, NAVY, BLUE, ORANGE, MUTED, PALE, BORDER, base, text, box, save,
)
from thai_figure_text import enable_thai_shaping


def sse_at_b(b):
    """Hold w=1; return SSE on the three synthetic sensor/reference pairs."""
    b = np.asarray(b, dtype=float)
    return np.sum((Y - (X + b[..., None])) ** 2, axis=-1)


def slope_at_b(b):
    return 6 * np.asarray(b, dtype=float) - 12


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(color=BORDER, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=NAVY, labelsize=15)


def main():
    enable_thai_shaping()
    plt.rcParams.update({
        "font.family": thai_font(), "font.size": 16,
        "svg.fonttype": "path", "svg.hashsalt": "ml-ep05-sse-derivative",
    })
    b_values = np.linspace(0, 4, 321)
    np.testing.assert_allclose(sse_at_b(b_values), 3 * (b_values - 2) ** 2 + 6)
    np.testing.assert_allclose(sse_at_b([1, 2, 3]), [9, 6, 9])
    np.testing.assert_allclose(slope_at_b([1, 2, 3]), [-6, 0, 6])
    step = 1e-5
    numerical_slope = (sse_at_b(b_values + step) - sse_at_b(b_values - step)) / (2 * step)
    np.testing.assert_allclose(numerical_slope, slope_at_b(b_values), atol=1e-8)
    np.testing.assert_allclose(sse_at_b([1.9, 2, 2.1]), [6.03, 6, 6.03])

    fig = base(
        "อนุพันธ์เป็น 0 ยังต้องดูรูปกราฟ",
        "ซ้าย: SSE ของข้อมูลสามคู่เมื่อคง w = 1 • ขวา: ฟังก์ชันสมมติสำหรับเปรียบเทียบ",
    )
    text(fig, .33, .795, "SSE ของโมเดลเส้นตรง", 22, BLUE, "bold", ha="center")
    text(fig, .825, .795, "กราฟคว่ำ: ไม่ใช่ SSE", 22, ORANGE, "bold", ha="center")
    text(fig, .825, .750, "g(b) = 12 − 3(b − 2)²", 17, MUTED, ha="center")

    ax = fig.add_axes([.075, .325, .51, .405])
    style_axes(ax)
    ax.plot(b_values, sse_at_b(b_values), color=BLUE, lw=3)
    ax.set(xlim=(-.08, 4.08), ylim=(0, 20), xticks=[0, 1, 2, 3, 4],
           yticks=[0, 6, 9, 12, 18], xlabel="b (µg/m³)", ylabel="SSE")
    for chosen in [1, 2, 3]:
        value, slope = float(sse_at_b(chosen)), float(slope_at_b(chosen))
        segment = np.array([chosen - .4, chosen + .4])
        ax.plot(segment, value + slope * (segment - chosen), color=ORANGE,
                lw=2.4, ls="--", zorder=3)
        ax.scatter([chosen], [value], s=72, color=BLUE, edgecolor="white", zorder=4)
    ax.annotate("จุดต่ำสุด", xy=(2, 6), xytext=(2, 2.1), ha="center", va="center",
                fontsize=17, color=BLUE, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.3})
    for x, chosen, value, slope in [
        (.15, 1, 9, "−6"), (.33, 2, 6, "0"), (.51, 3, 9, "+6")
    ]:
        text(fig, x, .235, f"b = {chosen} • SSE = {value}", 17, NAVY, "bold", ha="center")
        text(fig, x, .193, f"ความชัน = {slope}", 17, ORANGE, ha="center")
    text(fig, .33, .755, "เส้นประสีส้ม = เส้นสัมผัส ณ จุดที่เลือก", 15, MUTED, ha="center")

    comparison = fig.add_axes([.70, .39, .25, .315])
    style_axes(comparison)
    comparison.plot(b_values, 12 - 3 * (b_values - 2) ** 2, color=MUTED, lw=3)
    comparison.plot([1.3, 2.7], [12, 12], color=ORANGE, lw=2.4, ls="--")
    comparison.scatter([2], [12], s=75, color=ORANGE, edgecolor="white", zorder=4)
    comparison.set(xlim=(-.08, 4.08), ylim=(-1, 14), xticks=[0, 2, 4],
                   yticks=[0, 6, 12], xlabel="b", ylabel="g(b)")
    comparison.annotate("จุดสูงสุด\nความชัน = 0", xy=(2, 12), xytext=(2, 5.8),
                        ha="center", va="center", fontsize=17, color=ORANGE,
                        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.3})
    text(fig, .825, .285, "เส้นสัมผัสราบเหมือนกัน", 19, NAVY, ha="center")
    text(fig, .825, .242, "แต่ไม่ได้เป็นจุดต่ำสุด", 19, ORANGE, "bold", ha="center")

    box(fig, .05, .105, .90, .055, fill="#EAF2FF")
    text(fig, .50, .132, "อนุพันธ์เป็น 0 บอกว่าเส้นสัมผัสราบ • ไม่ได้บอกว่า SSE = 0",
         21, BLUE, "bold", ha="center")
    text(fig, .05, .047,
         "OLS: SSE เป็นฟังก์ชันนูน จึงหาจุดต่ำสุดจากอนุพันธ์ของ w และ b ที่เป็น 0 พร้อมกัน",
         17, MUTED)
    save(fig, "05-linear-regression-sse-derivative")
    print("Validated: b=1,2,3 -> SSE=9,6,9; slopes=-6,0,6; b=1.9,2.1 -> SSE=6.03")


if __name__ == "__main__":
    main()
