"""Render the EP08 StandardScaler/SVC training and prediction flow.

Run with scripts/requirements-figures.txt; no lesson code is executed.
"""
from pathlib import Path
import argparse

from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch

import render_model_diagrams as art


def reuse_arrow(ax, x, start, end):
    ax.add_patch(FancyArrowPatch(
        (x, start), (x, end), arrowstyle="-|>", mutation_scale=18,
        linewidth=1.8, linestyle=(0, (4, 3)), color=art.MUTED,
    ))


def pipeline():
    fig = art.plt.figure(figsize=(16, 10), facecolor=art.BG)
    page = fig.add_axes([0, 0, 1, 1])
    page.set(xlim=(0, 1), ylim=(0, 1))
    page.axis("off")

    # Draw backgrounds first, then shaped Thai and Latin labels.
    for y, tint in ((.60, "#EDF4FF"), (.25, "#FFF5EA")):
        for x, width in ((.045, .14), (.23, .32), (.595, .14), (.78, .18)):
            art.box(page, x, y, width, .18, color="white" if x == .23 else tint)
    art.box(page, .78, .805, .18, .035, color="#EDF4FF")
    art.box(page, .78, .174, .18, .038, color="#FFF5EA")
    art.box(page, .045, .047, .915, .085, color="white")

    for y, color in ((.69, art.BLUE), (.34, art.ORANGE)):
        for left, right in ((.193, .222), (.558, .587), (.743, .772)):
            art.arrow(page, (left, y), (right, y), color)
    art.arrow(page, (.87, .798), (.87, .787), art.BLUE)
    art.arrow(page, (.87, .242), (.87, .219), art.ORANGE)
    reuse_arrow(page, .39, .59, .44)
    reuse_arrow(page, .87, .59, .44)

    art.label(page, .045, .946, "Pipeline · StandardScaler → SVM", 29, art.BLUE, True)
    art.label(page, .045, .895,
              "แบ่งชุดฝึกและชุดทดสอบก่อน · ใช้ลำดับเดียวกันทั้ง Linear และ RBF", 19, art.MUTED)
    art.label(page, .045, .833, "1 · ตอนฝึก", 23, art.BLUE, True)
    art.label(page, .475, .833, "model.fit(X_train, y_train)", 19, art.BLUE)

    art.label(page, .115, .716, "X_train", 23, art.BLUE, True, ha="center")
    art.label(page, .115, .660, "features ชุดฝึก", 17, ha="center")
    art.label(page, .39, .750, "StandardScaler", 22, bold=True, ha="center")
    art.label(page, .39, .710, "fit → transform", 23, art.BLUE, True, ha="center")
    art.label(page, .39, .650,
              "หาค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน\nแล้วปรับสเกลแยกทีละคอลัมน์", 17, ha="center")
    art.label(page, .665, .690, "features\nที่ปรับสเกลแล้ว", 18, ha="center")
    art.label(page, .87, .823, "y_train · labels", 17, art.BLUE, True, ha="center")
    art.label(page, .87, .743, "SVC", 23, art.BLUE, True, ha="center")
    art.label(page, .87, .699, "fit", 22, art.BLUE, True, ha="center")
    art.label(page, .87, .645, "เรียนรู้แนวแบ่งคลาส", 17, ha="center")

    art.label(page, .41, .523, "เก็บสถิติไว้ใช้ต่อ", 17, art.MUTED)
    art.label(page, .85, .523, "เก็บโมเดลที่ฝึกไว้", 17, art.MUTED, ha="right")
    art.label(page, .045, .478, "2 · ตอนทำนาย", 23, art.ORANGE, True)
    art.label(page, .475, .478, "model.predict(X_test)", 19, art.ORANGE)

    art.label(page, .115, .366, "X_test", 23, art.ORANGE, True, ha="center")
    art.label(page, .115, .310, "หรือข้อมูลใหม่", 17, ha="center")
    art.label(page, .39, .400, "StandardScaler", 22, bold=True, ha="center")
    art.label(page, .39, .360, "transform", 23, art.ORANGE, True, ha="center")
    art.label(page, .39, .300,
              "ใช้ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน\nที่เก็บไว้จากชุดฝึก", 17, ha="center")
    art.label(page, .665, .340, "features\nที่ปรับสเกลแล้ว", 18, ha="center")
    art.label(page, .87, .393, "SVC", 23, art.ORANGE, True, ha="center")
    art.label(page, .87, .349, "predict", 22, art.ORANGE, True, ha="center")
    art.label(page, .87, .295, "ใช้โมเดลที่ฝึกไว้", 17, ha="center")
    art.label(page, .87, .193, "ทำนายคลาส 0 หรือ 1", 17, art.ORANGE, True, ha="center")

    art.label(page, .5025, .105,
              "StandardScaler แปลงเฉพาะ features · labels ส่งให้ SVC โดยตรง", 18, ha="center")
    art.label(page, .5025, .071,
              "ตอนทำนาย: ไม่หาสถิติใหม่จาก test และไม่ฝึก SVC ใหม่", 18, art.BLUE, True, ha="center")
    art.save(fig, page, "08-support-vector-machines-pipeline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=art.OUT)
    args = parser.parse_args()
    for filename in ("tahoma.ttf", "tahomabd.ttf"):
        path = Path("/mnt/c/Windows/Fonts") / filename
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    art.configure()
    art.plt.rcParams["svg.hashsalt"] = "ml-ep08-pipeline"
    art.OUT = args.output_dir
    pipeline()


if __name__ == "__main__":
    main()
