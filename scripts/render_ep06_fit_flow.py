"""Render the EP06 training/prediction flow with precise Thai text."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from thai_figure_text import enable_thai_shaping
from render_ep06_diagrams import BLUE, ORANGE, MUTED, card, label, arrow, save, thai_font


def reuse_arrow(ax, x, start, end):
    ax.add_patch(FancyArrowPatch(
        (x, start), (x, end), arrowstyle="-|>", mutation_scale=19,
        linewidth=2, linestyle=(0, (4, 3)), color=MUTED,
    ))


def main():
    enable_thai_shaping()
    plt.rcParams.update({
        "font.family": thai_font(),
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "path",
        "svg.hashsalt": "ml-ep06-fit-transform-flow",
    })
    fig = plt.figure(figsize=(16, 10), facecolor="#F6F8FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    # Draw backgrounds first, then exact labels with Thai shaping.
    for y, tint in ((0.63, "#EDF4FF"), (0.29, "#FFF5EA")):
        card(ax, 0.04, y, 0.13, 0.19, tint)
        card(ax, 0.215, y, 0.32, 0.19)
        card(ax, 0.58, y, 0.14, 0.19, tint)
        card(ax, 0.765, y, 0.195, 0.19, tint)
    card(ax, 0.80, 0.852, 0.125, 0.035, "#EDF4FF")
    card(ax, 0.765, 0.225, 0.195, 0.04, "#FFF5EA")
    for x, width in ((0.04, 0.28), (0.35, 0.29), (0.67, 0.29)):
        card(ax, x, 0.047, width, 0.128)

    for y, color in ((0.725, BLUE), (0.385, ORANGE)):
        arrow(ax, (0.181, y), (0.204, y), color)
        arrow(ax, (0.547, y), (0.568, y), color)
        arrow(ax, (0.732, y), (0.754, y), color)
    arrow(ax, (0.8625, 0.842), (0.8625, 0.832), BLUE)
    reuse_arrow(ax, 0.375, 0.615, 0.495)
    reuse_arrow(ax, 0.8625, 0.615, 0.495)
    arrow(ax, (0.8625, 0.279), (0.8625, 0.275), ORANGE)

    label(ax, 0.04, 0.946, "Pipeline · เตรียมข้อมูลก่อนฝึก Ridge",
          size=29, color=BLUE, weight="bold")
    label(ax, 0.04, 0.898, "แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบก่อน", size=17, color=MUTED)

    label(ax, 0.04, 0.856, "ตอนฝึก", size=23, color=BLUE, weight="bold")
    label(ax, 0.105, 0.750, "X_train", size=23, color=BLUE, weight="bold", ha="center")
    label(ax, 0.105, 0.695, "ข้อมูลชุดฝึก", size=17, ha="center")
    label(ax, 0.375, 0.785, "ตัวเตรียมข้อมูล", size=20, weight="bold", ha="center")
    label(ax, 0.375, 0.738, "fit → transform", size=25, color=BLUE, weight="bold", ha="center")
    label(ax, 0.375, 0.694, "หาค่าที่ใช้ → แปลงข้อมูล", size=19, ha="center")
    label(ax, 0.375, 0.651, "เติมค่า • ปรับสเกล • one-hot", size=16, color=MUTED, ha="center")
    label(ax, 0.65, 0.725, "features\nที่แปลงแล้ว", size=20, ha="center")
    label(ax, 0.8625, 0.870, "y_train", size=18, color=BLUE, weight="bold", ha="center")
    label(ax, 0.8625, 0.776, "Ridge · fit", size=23, color=BLUE, weight="bold", ha="center")
    label(ax, 0.8625, 0.700, "หาน้ำหนัก w\nและค่าคงที่ b", size=20, ha="center")

    label(ax, 0.395, 0.555, "ค่าเตรียมข้อมูลเดิม", size=16, color=MUTED)
    label(ax, 0.84, 0.555, "w และ b เดิม", size=16, color=MUTED, ha="right")

    label(ax, 0.04, 0.530, "ตอนทำนาย", size=23, color=ORANGE, weight="bold")
    label(ax, 0.105, 0.413, "X_test", size=23, color=ORANGE, weight="bold", ha="center")
    label(ax, 0.105, 0.355, "หรือข้อมูลใหม่", size=16, ha="center")
    label(ax, 0.375, 0.444, "ตัวเตรียมข้อมูล", size=20, weight="bold", ha="center")
    label(ax, 0.375, 0.397, "transform", size=25, color=ORANGE, weight="bold", ha="center")
    label(ax, 0.375, 0.353, "แปลงด้วยค่าที่เก็บไว้", size=19, ha="center")
    label(ax, 0.375, 0.310, "เติมค่า • ปรับสเกล • one-hot", size=16, color=MUTED, ha="center")
    label(ax, 0.65, 0.385, "features\nที่แปลงแล้ว", size=20, ha="center")
    label(ax, 0.8625, 0.434, "Ridge · predict", size=21, color=ORANGE, weight="bold", ha="center")
    label(ax, 0.8625, 0.360, "ใช้ w และ b\nที่ฝึกไว้", size=20, ha="center")
    label(ax, 0.8625, 0.245, "ค่าบริการที่ทำนาย", size=17, color=ORANGE, weight="bold", ha="center")

    label(ax, 0.04, 0.205, "ตัวอย่าง: เติมค่าหายด้วย median", size=18, weight="bold")
    label(ax, 0.18, 0.140, "fit: หาและเก็บมัธยฐาน", size=18, color=BLUE, weight="bold", ha="center")
    label(ax, 0.18, 0.086, "[2, 4, NaN] → 3", size=22, ha="center")
    label(ax, 0.495, 0.140, "transform ชุดฝึก", size=18, color=BLUE, weight="bold", ha="center")
    label(ax, 0.495, 0.086, "[2, 4, NaN] → [2, 4, 3]", size=19, ha="center")
    label(ax, 0.815, 0.140, "transform ชุดทดสอบ", size=18, color=ORANGE, weight="bold", ha="center")
    label(ax, 0.815, 0.086, "[100, NaN] → [100, 3]", size=19, ha="center")
    save(fig, "fit-transform-flow")


if __name__ == "__main__":
    main()
