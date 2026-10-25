"""Render the EP04 teaching diagram; this example is not a model's test output."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "figures" / "04-validation-and-metrics-confusion-matrix"
TN, FP, FN, TP = 80, 10, 5, 5


def thai_font():
    for family in ("Noto Sans Thai", "Tahoma", "Leelawadee UI", "Thonburi", "Loma"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return family
    raise RuntimeError("Install a Thai font such as Noto Sans Thai before rendering.")


def main():
    plt.rcParams.update({
        "font.family": thai_font(),
        "svg.fonttype": "path",
        "svg.hashsalt": "ml-ep04-confusion-matrix",
    })
    navy, muted = "#142A43", "#4C6078"
    blue, purple = "#1558D6", "#7343AD"
    fig = plt.figure(figsize=(16, 10), facecolor="#F6F8FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def text(x, y, value, size=16, color=navy, weight="normal", ha="left"):
        return ax.text(x, y, value, fontsize=size, color=color,
                       fontweight=weight, ha=ha, va="center")

    def card(x, y, w, h, color="white", border="#D9E2EF"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=color, edgecolor=border, linewidth=1.3,
        ))

    text(0.055, 0.947, "อ่าน Confusion Matrix", size=30, weight="bold")
    text(0.055, 0.890,
         "ตัวอย่างสมมติ 100 ข้อความ  ·  คลาส 1 = สแปม  ·  คลาส 0 = ข้อความปกติ",
         size=17, color=muted)
    text(0.055, 0.848, "Positive = คลาสที่เราสนใจตรวจหา ในตัวอย่างนี้คือสแปม",
         size=15, color=muted)
    text(0.400, 0.790, "คำทำนายของโมเดล (Predicted)", size=18, weight="bold", ha="center")
    text(0.290, 0.744, "ทาย 0 · ปกติ", size=16, ha="center")
    text(0.510, 0.744, "ทาย 1 · สแปม", size=16, ha="center")

    text(0.155, 0.672, "ค่าจริง", size=15, weight="bold", ha="right")
    text(0.155, 0.642, "(Actual)", size=13, color=muted, ha="right")
    text(0.155, 0.588, "จริง 0", size=16, weight="bold", ha="right")
    text(0.155, 0.553, "ปกติ", size=15, ha="right")
    text(0.155, 0.403, "จริง 1", size=16, weight="bold", ha="right")
    text(0.155, 0.368, "สแปม", size=15, ha="right")

    cells = [
        (0.18, 0.49, "TN", "True Negative", TN, "ทายปกติถูก", "#E9F0FF", blue),
        (0.40, 0.49, "FP", "False Positive", FP, "ปกติ แต่ทายเป็นสแปม", "#FFF0DE", "#A64F00"),
        (0.18, 0.27, "FN", "False Negative", FN, "สแปม แต่ทายเป็นปกติ", "#FFF0DE", "#A64F00"),
        (0.40, 0.27, "TP", "True Positive", TP, "ทายสแปมถูก", "#E9F0FF", blue),
    ]
    for x, y, acronym, name, count, meaning, fill, ink in cells:
        ax.add_patch(Rectangle((x, y), 0.22, 0.22,
                               facecolor=fill, edgecolor="white", linewidth=3))
        text(x + 0.11, y + 0.177, acronym + " · " + name, size=13, weight="bold",
             color=ink, ha="center")
        text(x + 0.11, y + 0.108, str(count), size=40, weight="bold",
             color=ink, ha="center")
        text(x + 0.11, y + 0.041, meaning, size=14, ha="center")
    text(0.40, 0.229, "ตัวเลขในแต่ละช่อง = จำนวนข้อความ", size=14, color=muted, ha="center")

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * TP / (2 * TP + FP + FN)
    card(0.665, 0.495, 0.28, 0.21, border="#C7D9FC")
    text(0.682, 0.674, "Precision", size=22, color=blue, weight="bold")
    text(0.682, 0.632, "ดูคอลัมน์ทาย 1: ทายสแปมถูกกี่ส่วน?", size=14)
    text(0.682, 0.583, "TP / (TP + FP)", size=19, color=blue)
    text(0.682, 0.533, f"{TP} / ({TP} + {FP}) ≈ {precision:.3f}", size=20, weight="bold")

    card(0.665, 0.265, 0.28, 0.20, border="#DECFEF")
    text(0.682, 0.436, "Recall", size=22, color=purple, weight="bold")
    text(0.682, 0.394, "ดูแถวจริง 1: สแปมจริง เจอกี่ส่วน?", size=14)
    text(0.682, 0.345, "TP / (TP + FN)", size=19, color=purple)
    text(0.682, 0.297, f"{TP} / ({TP} + {FN}) = {recall:.2f}", size=20, weight="bold")

    card(0.18, 0.092, 0.44, 0.099)
    text(0.198, 0.158, f"Accuracy = {accuracy:.2f}", size=20, weight="bold")
    text(0.198, 0.115, f"ทายถูกทั้งหมด: ({TN} + {TP}) / 100", size=16)
    card(0.665, 0.092, 0.28, 0.099)
    text(0.682, 0.158, f"F1 = {f1:.2f}", size=20, weight="bold")
    text(0.682, 0.115, "2TP / (2TP + FP + FN)", size=16)

    text(0.055, 0.038, "EP04 · ตัวอย่างสำหรับอ่านตารางและคำนวณคะแนน ไม่ใช่ผล test จาก Cell 3",
         size=13, color=muted)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=150)
    svg_path = OUTPUT.with_suffix(".svg")
    fig.savefig(svg_path, metadata={"Date": None})
    # Matplotlib emits trailing spaces in SVG paths; normalize without changing geometry.
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8", newline="\n",
    )
    plt.close(fig)
    print(f"Created {OUTPUT.with_suffix('.png')}")
    print(f"Created {OUTPUT.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
