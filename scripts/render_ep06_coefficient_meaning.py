"""Illustrate EP06 coefficients for numeric and one-hot features.

Render after installing scripts/requirements-figures.txt and a Thai font.
All examples use the same illustrative coefficients; learner code is unchanged.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thai_figure_text import enable_thai_shaping
from render_ep06_diagrams import BLUE, ORANGE, MUTED, card, label, thai_font

ROOT = Path(__file__).resolve().parents[1]
STEM = "06-feature-engineering-coefficients-numeric-onehot"


def predict(distance_z, weight_z, express, standard):
    """One fixed illustrative linear model, in synthetic cost units."""
    return 50 + 10 * distance_z + 4 * weight_z + 5 * express - 5 * standard


def render(output_dir):
    enable_thai_shaping()
    plt.rcParams.update({
        "font.family": thai_font(),
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "path",
        "svg.hashsalt": STEM,
    })
    fig = plt.figure(figsize=(16, 10), facecolor="#F6F8FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    # Draw cards first; exact shaped Thai and mathematical text follow.
    card(ax, .04, .079, .445, .670, "#EDF4FF")
    card(ax, .515, .079, .445, .670, "#FFF5EA")
    for y in (.482, .387):
        card(ax, .535, y, .405, .073, "white")
    card(ax, .545, .176, .385, .112, "white")

    label(ax, .04, .946, "น้ำหนัก w · อ่านตามชนิดของ feature", size=29, color=BLUE, weight="bold")
    label(ax, .04, .890, "ตัวอย่างสมมติ: b = 50, w₁ = 10, w₂ = 4, w₃ = 5, w₄ = −5",
          size=19, color=MUTED)
    label(ax, .50, .827,
          r"$\hat y = 50 + 10z_{\mathrm{distance}} + 4z_{\mathrm{weight}}"
          r" + 5I_{\mathrm{express}} - 5I_{\mathrm{standard}}$",
          size=26, ha="center")

    label(ax, .2625, .705, "ตัวเลข: ระยะทางหลังปรับสเกล",
          size=21, color=BLUE, weight="bold", ha="center")
    label(ax, .2625, .654, "คง z_weight = 0 และ service = express", size=16, ha="center")
    label(ax, .2625, .600, r"$\hat y = 50 + 10z_{\mathrm{distance}} + 5"
          r" = 55 + 10z_{\mathrm{distance}}$", size=19, ha="center")

    plot = fig.add_axes([.105, .247, .321, .285], facecolor="white")
    x = [0, 1]
    y = [predict(v, 0, 1, 0) for v in x]
    plot.plot(x, y, color=BLUE, lw=3)
    plot.scatter(x, y, color=BLUE, s=65, zorder=4)
    plot.plot([0, 1, 1], [55, 55, 65], "--", color=ORANGE, lw=1.8)
    plot.annotate("+1", (.50, 55), xytext=(0, -19), textcoords="offset points",
                  ha="center", fontsize=17, color=ORANGE)
    plot.annotate("+10", (1, 60), xytext=(9, 0), textcoords="offset points",
                  va="center", fontsize=17, color=ORANGE)
    plot.annotate("(0, 55)", (0, 55), xytext=(-3, 14), textcoords="offset points",
                  fontsize=14, color=BLUE)
    plot.annotate("(1, 65)", (1, 65), xytext=(-62, 13), textcoords="offset points",
                  fontsize=14, color=BLUE)
    plot.set(xlim=(-.15, 1.29), ylim=(52, 68), xticks=[0, 1], yticks=[55, 60, 65],
             xlabel=r"$z_{\mathrm{distance}}$", ylabel="ค่าทำนาย")
    plot.tick_params(labelsize=13)
    plot.xaxis.label.set_size(17)
    plot.yaxis.label.set_size(16)
    plot.spines[["top", "right"]].set_visible(False)
    label(ax, .2625, .168, "w₁ = 10 คือความชันต่อ z_distance",
          size=18, color=BLUE, weight="bold", ha="center")
    label(ax, .2625, .115, "z เพิ่ม 1 = ระยะทางดิบเพิ่ม\n1 ส่วนเบี่ยงเบนมาตรฐานของ train",
          size=15, ha="center")

    label(ax, .7375, .705, "One-hot: เปลี่ยนประเภทบริการ",
          size=21, color=ORANGE, weight="bold", ha="center")
    label(ax, .7375, .654, "คง z_distance = z_weight = 0", size=17, ha="center")
    columns = [.591, .710, .813, .905]
    for cx, value in zip(columns, ("service", "I_express", "I_standard", "ŷ")):
        label(ax, cx, .590, value, size=14, color=MUTED, weight="bold", ha="center")
    rows = [("express", 1, 0, predict(0, 0, 1, 0)),
            ("standard", 0, 1, predict(0, 0, 0, 1))]
    for cy, row in zip((.519, .424), rows):
        for cx, value in zip(columns, row):
            result_column = cx == columns[-1]
            value_label = (f"50 + 5 = {value}" if row[0] == "express" else f"50 − 5 = {value}") if result_column else str(value)
            label(ax, cx, cy, value_label, size=14 if result_column else 19,
                  color=ORANGE if result_column else "#142A43",
                  weight="bold" if result_column else "normal", ha="center")
    label(ax, .7375, .337, "express → standard: 55 → 45", size=19, ha="center")
    label(ax, .7375, .253, "ค่าทำนายเปลี่ยน = w₄ − w₃",
          size=20, color=ORANGE, weight="bold", ha="center")
    label(ax, .7375, .203, "−5 − 5 = −10 หน่วย", size=21, color=ORANGE, ha="center")
    label(ax, .7375, .115, "I_express: 1 → 0  •  I_standard: 0 → 1",
          size=17, ha="center")
    label(ax, .50, .035, "ใช้ w และ b ชุดเดิมตลอดการเปรียบเทียบ",
          size=16, color=MUTED, ha="center")

    # Validate displayed values and text bounds before writing either format.
    assert y == [55, 65]
    assert rows[0][-1] == 55 and rows[1][-1] == 45
    assert rows[1][-1] - rows[0][-1] == -5 - 5 == -10
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for axes in fig.axes:
        for artist in axes.texts:
            bounds = artist.get_window_extent(renderer)
            assert bounds.x0 >= 0 and bounds.y0 >= 0, artist.get_text()
            assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, artist.get_text()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{STEM}.png", dpi=150)
    svg = output_dir / f"{STEM}.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf8").splitlines()) + "\n",
                   encoding="utf8", newline="\n")
    plt.close(fig)
    return {"stem": STEM, "png_size": [2400, 1500], "numeric_checks": "passed", "text_bounds": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/figures")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = render(args.output_dir)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
