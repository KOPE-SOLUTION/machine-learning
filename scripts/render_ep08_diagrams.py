"""Original EP08 illustrations: margin, support vectors and RBF parameters.

Run with the figure-authoring dependencies in scripts/requirements-figures.txt.
"""
from pathlib import Path
import argparse

import numpy as np
from matplotlib import font_manager
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import render_model_diagrams as art
import render_learning_workflows as workflow

ROOT = Path(__file__).resolve().parents[1]


def margin():
    fig, page = art.page(
        "SVM · แนวแบ่งอยู่ตรงกลางแถบ Margin",
        "เส้นทึบแบ่งคลาส · เส้นประเป็นขอบแถบ · วงแหวนระบุ support vectors",
    )
    X = np.array([[-2, -1], [-2, 1], [-1, 0], [1, 0], [2, -1], [2, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = SVC(kernel="linear", C=100).fit(X, y)
    assert np.allclose(model.coef_, [[1, 0]])
    assert np.allclose(model.intercept_, [0])
    assert set(model.support_) == {2, 3}

    ax = fig.add_axes([.07, .23, .52, .52])
    ax.axvspan(-2.8, 0, color="#EDF4FF")
    ax.axvspan(0, 2.8, color="#FFF5EA")
    ax.axvspan(-1, 1, color=art.PURPLE, alpha=.10)
    ax.axvline(0, color=art.NAVY, linewidth=2.5)
    for x in (-1, 1):
        ax.axvline(x, color=art.PURPLE, linestyle="--", linewidth=1.8)
    for cls, color in [(0, art.BLUE), (1, art.ORANGE)]:
        points = X[y == cls]
        ax.scatter(*points.T, color=color, s=125, zorder=3)
    ax.scatter(*X[model.support_].T, s=370, facecolors="none",
               edgecolors=art.NAVY, linewidths=1.7, zorder=4)
    ax.annotate("", xy=(1, 1.5), xytext=(-1, 1.5),
                arrowprops={"arrowstyle": "<->", "color": art.PURPLE, "lw": 2})
    ax.text(0, 1.73, "Margin", ha="center", color=art.PURPLE, fontsize=20,
            bbox={"facecolor": "#F0EDF9", "edgecolor": "none", "pad": 2})
    ax.text(-1.9, -1.6, "ทายคลาส 0", ha="center", color=art.BLUE, fontsize=17)
    ax.text(1.9, -1.6, "ทายคลาส 1", ha="center", color=art.ORANGE, fontsize=17)
    ax.set(xlim=(-2.8, 2.8), ylim=(-2, 2.25), xticks=[], yticks=[])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    art.box(page, .64, .61, .30, .14, "แนวแบ่งคลาส\nจุดใหม่อยู่ฝั่งไหน → ทายคลาสนั้น", size=18)
    art.box(page, .64, .42, .30, .14, "Margin\nแถบระหว่างเส้นประทั้งสองฝั่ง", size=18)
    art.box(page, .64, .23, .30, .14, "Support vectors\nจุดที่ใช้กำหนดแนวแบ่ง", size=18)
    art.footer(page,
        "SVM พยายามให้แถบกว้าง พร้อมลดจุดที่อยู่ในแถบหรือถูกแบ่งผิดฝั่ง",
        "เมื่อข้อมูลซ้อนกัน support vectors อาจอยู่ในแถบหรือผิดฝั่งได้ด้วย")
    art.save(fig, page, "08-support-vector-machines-margin")


def parameters():
    fig, page = art.page(
        "RBF · เปลี่ยน C กับ gamma ทีละค่า",
        "จุดฝึกชุดเดียวกันทุกช่อง · สีพื้นคือคลาสที่ทำนาย · เส้นเข้มคือแนวแบ่ง",
    )
    X, y = make_moons(n_samples=400, noise=.25, random_state=42)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=42
    )
    scaled = StandardScaler().fit_transform(X_train)
    xx, yy = np.meshgrid(np.linspace(-2.8, 2.8, 250),
                         np.linspace(-2.8, 2.8, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]

    art.label(page, .05, .79, "เปลี่ยน C · คง gamma = 0.5", 20, art.BLUE, True)
    art.label(page, .05, .42, "เปลี่ยน gamma · คง C = 1", 20, art.ORANGE, True)
    configs = [
        (.49, [(0.1, .5), (1, .5), (100, .5)], "C"),
        (.12, [(1, .02), (1, .5), (1, 20)], "gamma"),
    ]
    for y0, configs_row, varying in configs:
        for x0, (c, gamma) in zip((.06, .38, .70), configs_row):
            ax = fig.add_axes([x0, y0, .26, .235])
            model = SVC(C=c, gamma=gamma).fit(scaled, y_train)
            values = model.decision_function(grid).reshape(xx.shape)
            ax.contourf(xx, yy, values, levels=[-1e9, 0, 1e9],
                        colors=["#DFEAFF", "#FFE6CA"])
            ax.contour(xx, yy, values, levels=[0], colors=[art.NAVY], linewidths=1.5)
            for cls, color in [(0, art.BLUE), (1, art.ORANGE)]:
                a = scaled[y_train == cls]
                ax.scatter(*a.T, c=color, s=8, alpha=.75)
            value = c if varying == "C" else gamma
            ax.set(title=f"{varying} = {value:g}", xlim=(-2.8, 2.8),
                   ylim=(-2.8, 2.8), xticks=[], yticks=[])
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    art.label(page, .05, .05,
        "แนวแบ่งที่ตามจุดฝึกมากขึ้น อาจทายข้อมูลใหม่ได้แย่ลง → ใช้ CV ช่วยเลือก",
        19, art.BLUE, True)
    art.save(fig, page, "08-support-vector-machines-parameters")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/figures")
    args = parser.parse_args()
    # Make the same installed Windows Thai fonts available when authoring in WSL.
    for font in ("tahoma.ttf", "tahomabd.ttf"):
        path = Path("/mnt/c/Windows/Fonts") / font
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    art.configure()
    art.OUT = args.output_dir
    workflow.OUT = args.output_dir
    margin()
    parameters()
    workflow.ep08()


if __name__ == "__main__":
    main()
