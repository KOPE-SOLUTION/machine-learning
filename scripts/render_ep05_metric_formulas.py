"""Render original EP05 metric formula cards from one shared three-row example."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures"
Y_TRUE = np.array([10.0, 20.0, 30.0])
Y_PRED = np.array([9.0, 21.0, 26.0])


def thai_font():
    for name in ("Noto Sans Thai", "Tahoma", "Leelawadee UI", "Thonburi", "Loma"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        return name
    raise RuntimeError("Install a Thai font such as Noto Sans Thai before rendering.")


def example_metrics():
    residual = Y_TRUE - Y_PRED
    return {
        "residual": residual,
        "mean": Y_TRUE.mean(),
        "sse": np.sum(residual ** 2),
        "sst": np.sum((Y_TRUE - Y_TRUE.mean()) ** 2),
        "mae": np.mean(np.abs(residual)),
        "rmse": np.sqrt(np.mean(residual ** 2)),
        "r2": 1 - np.sum(residual ** 2) / np.sum((Y_TRUE - Y_TRUE.mean()) ** 2),
    }


def render(metric, values):
    colors = {"mae": "#1558D6", "rmse": "#7343AD", "r2": "#A95000"}
    titles = {
        "mae": "MAE · Mean Absolute Error",
        "rmse": "RMSE · Root Mean Squared Error",
        "r2": "R² · Coefficient of Determination",
    }
    subtitles = {
        "mae": "นำขนาดความผิดพลาดมาเฉลี่ย โดยไม่ให้เครื่องหมายบวกและลบหักล้างกัน",
        "rmse": "ยกกำลังสองความผิดพลาด → หาค่าเฉลี่ย → ถอดราก",
        "r2": "เทียบความผิดพลาดกำลังสองของโมเดล กับการทายค่าเฉลี่ยของชุดประเมิน",
    }
    accent, navy, muted = colors[metric], "#142A43", "#4C6078"
    fig = plt.figure(figsize=(16, 9), facecolor="#F6F8FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def text(x, y, value, size=18, color=navy, weight="normal", ha="left"):
        ax.text(x, y, value, fontsize=size, color=color,
                fontweight=weight, ha=ha, va="center")

    def card(x, y, w, h, edge="#D9E2EF"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.012",
            facecolor="white", edgecolor=edge, linewidth=1.5,
        ))

    text(0.05, 0.94, titles[metric], size=29, color=accent, weight="bold")
    text(0.05, 0.88, subtitles[metric], size=17, color=muted)
    card(0.05, 0.265, 0.31, 0.52)
    card(0.40, 0.265, 0.55, 0.52)
    text(0.205, 0.746, "ข้อมูลตัวอย่าง 3 แถว", size=19, weight="bold", ha="center")
    columns = (0.103, 0.205, 0.307)
    for x, label in zip(columns, ("ค่าจริง y", "ค่าทำนาย ŷ", "Residual")):
        text(x, 0.680, label, size=15, weight="bold", ha="center")
    ax.plot([0.07, 0.34], [0.645, 0.645], color="#D9E2EF", lw=1.5)
    for row, (actual, predicted, residual) in enumerate(zip(Y_TRUE, Y_PRED, values["residual"])):
        y = 0.599 - row * 0.088
        text(columns[0], y, f"{actual:g}", size=25, ha="center")
        text(columns[1], y, f"{predicted:g}", size=25, ha="center")
        text(columns[2], y, f"{residual:+g}", size=25, color=accent, weight="bold", ha="center")
    text(0.205, 0.346, "Residual = y − ŷ", size=16, color=muted, ha="center")
    text(0.205, 0.298, "n = 3 · สมมติหน่วยเป็นนาที", size=15, color=muted, ha="center")

    if metric == "mae":
        text(0.435, 0.689,
             r"$\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|$",
             size=31, color=accent)
        text(0.435, 0.563, "แทนค่า residual จากทั้ง 3 แถว", size=18)
        text(0.435, 0.456, r"$\frac{|1|+|-1|+|4|}{3}=\frac{6}{3}=2$", size=34)
        text(0.435, 0.325, "|...| คือขนาดของค่า เช่น |−1| = 1", size=17, color=muted)
        footer = f"คลาดเคลื่อนโดยเฉลี่ย {values['mae']:g} นาที"
        detail = "ยิ่งต่ำยิ่งดี · มีหน่วยเดียวกับคำตอบ y"
    elif metric == "rmse":
        text(0.435, 0.689,
             r"$\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}$",
             size=29, color=accent)
        text(0.435, 0.563, "แทนค่า → เฉลี่ยกำลังสอง → ถอดราก", size=18)
        text(0.435, 0.452,
             r"$\sqrt{\frac{1^2+(-1)^2+4^2}{3}}=\sqrt{\frac{18}{3}}\approx2.45$",
             size=28)
        text(0.435, 0.325, "ความผิดพลาด 4 หน่วย กลายเป็น 16 ก่อนเฉลี่ย", size=17, color=muted)
        footer = f"RMSE ≈ {values['rmse']:.2f} นาที"
        detail = "ยิ่งต่ำยิ่งดี · ไวต่อความผิดพลาดขนาดใหญ่มากกว่า MAE"
    else:
        text(0.435, 0.692,
             r"$R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}$",
             size=28, color=accent)
        text(0.435, 0.565, "ค่าเฉลี่ยชุดนี้: ȳ = (10 + 20 + 30) / 3 = 20", size=17)
        text(0.435, 0.502, r"โมเดล: $1^2+(-1)^2+4^2=18$", size=19)
        text(0.435, 0.435, r"ทายค่าเฉลี่ย: $(10-20)^2+(20-20)^2+(30-20)^2=200$", size=17)
        text(0.435, 0.333, r"$R^2=1-\frac{18}{200}=0.91$", size=30, color=accent)
        footer = f"R² = {values['r2']:.2f} ไม่ได้หมายถึงทายถูก 91%"
        detail = "1 = ทายตรงทั้งหมด · 0 = เท่าการทายค่าเฉลี่ย · ติดลบ = แย่กว่า"

    card(0.05, 0.092, 0.90, 0.123)
    text(0.075, 0.179, footer, size=22, color=accent, weight="bold")
    text(0.075, 0.124, detail, size=17)
    text(0.05, 0.041, "EP05 · ตัวอย่างสมมติสำหรับอ่านสูตร ไม่ใช่ผล test จาก Cell 2", size=14, color=muted)
    filename = OUTPUT / f"05-linear-regression-{metric}-formula"
    fig.savefig(filename.with_suffix(".png"), dpi=150)
    svg = filename.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8", newline="\n",
    )
    plt.close(fig)
    print(filename.with_suffix(".png"))


def main():
    plt.rcParams.update({
        "font.family": thai_font(), "mathtext.fontset": "dejavusans",
        "svg.fonttype": "path", "svg.hashsalt": "ml-ep05-metrics",
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    values = example_metrics()
    for metric in ("mae", "rmse", "r2"):
        render(metric, values)
    print({key: float(values[key]) for key in ("mean", "sse", "sst", "mae", "rmse", "r2")})


if __name__ == "__main__":
    main()
