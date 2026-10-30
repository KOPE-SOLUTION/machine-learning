"""Original anonymous PM2.5 case diagrams, using synthetic data only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from sklearn.linear_model import LinearRegression

from render_ep05_diagram import thai_font
from thai_figure_text import enable_thai_shaping

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/figures"
X = np.array([20.0, 40.0, 60.0])
Y = np.array([23.0, 40.0, 63.0])
NAVY, BLUE, ORANGE = "#142A43", "#1558D6", "#B75700"
MUTED, PALE, BORDER = "#4C6078", "#F6F8FC", "#D3DFEE"


def text(fig, x, y, value, size=18, color=NAVY, weight="normal", ha="left"):
    return fig.text(x, y, value, fontsize=size, color=color,
                    fontweight=weight, ha=ha, va="center")


def box(fig, x, y, w, h, edge=BORDER, fill="white"):
    fig.add_artist(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=fig.transFigure, facecolor=fill, edgecolor=edge, linewidth=1.5,
        zorder=0,
    ))


def base(title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor=PALE)
    text(fig, .05, .935, title, 27, weight="bold")
    text(fig, .05, .875, subtitle, 17, MUTED)
    return fig


def device(fig, left, value, color):
    # A generic schematic, with no manufacturer-specific shape or labels.
    box(fig, left, .43, .16, .18, edge="#889AAE", fill="#E9EEF5")
    fig.add_artist(Rectangle((left + .023, .50), .114, .073,
                            transform=fig.transFigure, facecolor="#F9FCFF",
                            edgecolor="#B7C8DA", linewidth=1))
    text(fig, left + .08, .535, str(value), 26, color, "bold", ha="center")
    text(fig, left + .08, .468, "PM2.5", 13, MUTED, ha="center")


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (stem + ".png"), dpi=150)
    svg = OUT / (stem + ".svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in
                   svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")
    plt.close(fig)
    print(stem)


def render_pairs():
    fig = base("กรณีศึกษา: ค่าฝุ่นจากเครื่องวัดกับเครื่องอ้างอิง",
               "ข้อมูลสมมติทั้งหมด • ภาพเครื่องแบบทั่วไป • ไม่ใช่ผลวัดจากหน้างาน")
    for left in [.05, .36, .67]:
        box(fig, left, .24, .28, .54)
    text(fig, .19, .725, "เครื่องวัดที่ศึกษา", 21, weight="bold", ha="center")
    text(fig, .19, .663, "x: ข้อมูลนำเข้า", 18, BLUE, ha="center")
    device(fig, .11, 20, BLUE)
    text(fig, .19, .358, "x = [20, 40, 60]", 20, BLUE, "bold", ha="center")
    text(fig, .19, .296, "ค่าที่เครื่องอ่านได้", 17, MUTED, ha="center")

    text(fig, .50, .725, "เครื่องอ้างอิง", 21, weight="bold", ha="center")
    text(fig, .50, .663, "y: ค่าที่ใช้เทียบ", 18, ORANGE, ha="center")
    device(fig, .42, 23, ORANGE)
    text(fig, .50, .358, "y = [23, 40, 63]", 20, ORANGE, "bold", ha="center")
    text(fig, .50, .296, "วัดในช่วงเวลาเดียวกัน", 17, MUTED, ha="center")

    text(fig, .81, .725, "คู่ข้อมูลสำหรับฝึก", 21, weight="bold", ha="center")
    text(fig, .81, .665, "หนึ่งแถว = หนึ่งคู่เวลา", 17, MUTED, ha="center")
    table_x, top, width, height = .695, .61, .23, .075
    rows = [("x", "y"), *zip(X.astype(int), Y.astype(int))]
    for row, values in enumerate(rows):
        y = top - (row + 1) * height
        for col, value in enumerate(values):
            x = table_x + col * width / 2
            fig.add_artist(Rectangle(
                (x, y), width / 2, height, transform=fig.transFigure,
                facecolor="#E7F0FE" if row == 0 else "white",
                edgecolor=BORDER, linewidth=1,
            ))
            text(fig, x + width / 4, y + height / 2, str(value), 21,
                 BLUE if col == 0 else ORANGE,
                 "bold" if row == 0 else "normal", ha="center")
    text(fig, .81, .273, "หน่วย: µg/m³", 17, MUTED, ha="center")
    text(fig, .50, .186, "จับคู่จุดวัด ช่วงเวลาเฉลี่ย และหน่วยให้ตรงกัน", 19,
         ha="center")
    box(fig, .12, .062, .76, .074, edge="#BFD3F5", fill="#EAF2FF")
    text(fig, .50, .10, "ใช้ x ประมาณค่าอ้างอิงด้วยสมการ ŷ = b + wx", 22,
         BLUE, "bold", ha="center")
    save(fig, "05-linear-regression-sensor-pairs")


def render_least_squares():
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    fitted = model.predict(X.reshape(-1, 1))
    residual = Y - fitted
    raw_sse = float(np.sum((Y - X) ** 2))
    fitted_sse = float(np.sum(residual ** 2))
    np.testing.assert_allclose([model.intercept_, model.coef_[0]], [2, 1])
    np.testing.assert_allclose(fitted, [22, 42, 62])
    np.testing.assert_allclose(residual, [1, -2, 1])
    np.testing.assert_allclose([raw_sse, fitted_sse], [18, 6])
    fig = base("Least Squares: โมเดลหาเส้นที่เหมาะกับข้อมูลอย่างไร",
               "ตัวอย่างชุดฝึกสมมติ 3 คู่ • ค่าอ้างอิง y = [23, 40, 63]")
    ax = fig.add_axes([.075, .235, .425, .545], facecolor="white")
    grid = np.linspace(15, 65, 150)
    ax.scatter(X, Y, s=105, color=ORANGE, edgecolor="white", zorder=5,
               label="ค่าอ้างอิง y")
    ax.plot(grid, grid, "--", color="#8796A8", lw=2, label="ใช้ค่าเดิม: ŷ = x")
    ax.plot(grid, model.predict(grid[:, None]), color=BLUE, lw=2.6,
            label="OLS: ŷ = 2 + x")
    ax.vlines(X, fitted, Y, color=ORANGE, linewidth=4, zorder=4)
    ax.scatter(X, fitted, marker="D", s=55, color=BLUE, zorder=4)
    ax.annotate("residual = −2", xy=(40, 41), xytext=(44, 29),
                color=ORANGE, fontsize=15,
                arrowprops={"arrowstyle": "->", "color": ORANGE})
    ax.set(xlabel="ค่าจากเครื่อง x (µg/m³)", ylabel="ค่าอ้างอิง y (µg/m³)",
           xlim=(15, 65), ylim=(10, 74))
    ax.set_xticks([20, 40, 60])
    ax.set_yticks([20, 40, 60])
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    ax.legend(loc="upper left", fontsize=13, frameon=True)
    for spine in ax.spines.values():
        spine.set_color(BORDER)

    box(fig, .57, .545, .38, .235)
    text(fig, .595, .739, "ใช้ค่าจากเครื่องโดยตรง", 21, weight="bold")
    text(fig, .595, .681, "ŷ = x", 24, MUTED, "bold")
    text(fig, .595, .627, "residual: 3, 0, 3", 18)
    text(fig, .595, .576, "SSE = 3² + 0² + 3² = 18", 20, weight="bold")

    box(fig, .57, .245, .38, .235, edge="#A9C6F4", fill="#EEF5FF")
    text(fig, .595, .440, "เส้นที่ OLS fit ได้", 21, weight="bold")
    text(fig, .595, .383, "ŷ = 2 + 1x", 24, BLUE, "bold")
    text(fig, .595, .329, "residual: 1, −2, 1", 18)
    text(fig, .595, .277, "SSE = 1² + (−2)² + 1² = 6", 20, BLUE, "bold")

    text(fig, .075, .155, "residual วัดแนวดิ่งที่ x เดียวกัน", 17, ORANGE)
    text(fig, .595, .155, "SSE = ผลรวมความผิดพลาดกำลังสอง", 17, MUTED)
    text(fig, .05, .068,
         "b = 2 และ w = 1 เรียนจากตัวอย่างนี้ • SSE มีหน่วย (µg/m³)² • ยังไม่ใช่ผลบน test",
         17, MUTED)
    save(fig, "05-linear-regression-sensor-least-squares")
    print(f"OLS checks: intercept={model.intercept_:g}, slope={model.coef_[0]:g}, "
          f"raw SSE={raw_sse:g}, fitted SSE={fitted_sse:g}")


def main():
    enable_thai_shaping()
    plt.rcParams.update({
        "font.family": thai_font(), "font.size": 15,
        "svg.fonttype": "path", "svg.hashsalt": "ml-ep05-synthetic-sensor",
    })
    render_pairs()
    render_least_squares()


if __name__ == "__main__":
    main()
