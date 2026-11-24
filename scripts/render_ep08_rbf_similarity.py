"""Draw a concrete RBF distance example before the EP08 SVM comparison."""
from pathlib import Path
import argparse

import numpy as np
from matplotlib import font_manager
from sklearn.metrics.pairwise import rbf_kernel

import render_model_diagrams as art


def similarity_example():
    reference = np.array([[2., 1.]])
    examples = np.array([[1., 1.], [3., 1.], [5., 1.]])
    distances = np.linalg.norm(examples - reference, axis=1)
    similarities = rbf_kernel(examples, reference, gamma=.5).ravel()
    assert np.allclose(distances, [1, 1, 3])
    assert np.isclose(similarities[0], similarities[1])
    assert similarities[2] < similarities[0]

    fig, page = art.page(
        "RBF · เปลี่ยนระยะห่างเป็นค่าความคล้าย",
        "ใช้ดาว P เป็นจุดอ้างอิง · เปรียบเทียบแต่ละจุดกับ P ด้วยการตั้งค่า RBF เดียวกัน",
    )
    # Draw cards first; all Thai labels use the shared shaping helper.
    art.box(page, .045, .375, .55, .425, color="white")
    art.box(page, .625, .375, .33, .425, color="white")

    graph = fig.add_axes([.09, .445, .47, .275])
    graph.set_facecolor("white")
    graph.set(xlim=(0, 6), ylim=(0, 1.9), xticks=[1, 2, 3, 5], yticks=[0, 1],
              xlabel="Feature 1", ylabel="Feature 2")
    graph.spines[["top", "right"]].set_visible(False)
    graph.spines[["left", "bottom"]].set_color(art.EDGE)
    graph.tick_params(labelsize=12, colors=art.MUTED)
    graph.axhline(1, color=art.EDGE, lw=1, linestyle="--", zorder=1)
    graph.scatter(*examples.T, s=125, color=art.BLUE, edgecolors="white", zorder=4)
    graph.scatter(*reference.T, marker="*", s=380, color=art.PURPLE,
                  edgecolors="white", linewidths=1, zorder=5)
    for x, name, color in [(1, "A", art.BLUE), (2, "P", art.PURPLE),
                            (3, "B", art.BLUE), (5, "จุดไกล", art.BLUE)]:
        graph.text(x, 1.54, name, ha="center", va="center", fontsize=19,
                   color=color, fontweight="bold")
        graph.text(x, 1.26, f"({x}, 1)", ha="center", va="center", fontsize=14, color=art.MUTED)
    for start, end, y, label, color in [(1, 2, .69, "1 หน่วย", art.PURPLE),
                                       (2, 3, .69, "1 หน่วย", art.PURPLE),
                                       (2, 5, .19, "3 หน่วย", art.NAVY)]:
        graph.annotate("", xy=(end, y), xytext=(start, y),
                       arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.8,
                                   "shrinkA": 0, "shrinkB": 0})
        graph.text((start + end) / 2, y - .15, label, ha="center", va="center",
                   fontsize=15, color=color, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    art.label(page, .07, .765, "ทุกจุดมี Feature 2 = 1 จึงวัดระยะตามแนวนอนได้", 15, art.MUTED)
    art.label(page, .65, .755, "อ่านระยะ แล้วเทียบความคล้าย", 19, art.BLUE, True)
    art.label(page, .65, .68, "A และ B ห่าง P เท่ากัน: 1 หน่วย", 17)
    art.label(page, .65, .625, "→ คล้าย P เท่ากัน", 20, art.PURPLE, True)
    art.label(page, .65, .535, "จุดไกลห่าง P มากกว่า: 3 หน่วย", 17)
    art.label(page, .65, .48, "→ คล้าย P น้อยกว่า A และ B", 19, art.BLUE, True)
    art.label(page, .65, .413, "อยู่คนละทิศก็ได้ · ใช้ระยะห่างเป็นตัวกำหนด", 14, art.MUTED)

    art.label(page, .05, .32, "เมื่อนำไปทำนายจุดใหม่", 20, art.BLUE, True)
    art.box(page, .05, .17, .18, .105, "features\nของจุดใหม่", size=18)
    art.box(page, .29, .17, .29, .105, "RBF: เทียบความคล้าย\nกับ support vectors แต่ละจุด", size=17)
    art.box(page, .64, .17, .31, .105, "SVM: รวมด้วยน้ำหนักที่เรียนไว้\nและค่าคงที่ เพื่อเลือกคลาส", size=17)
    art.arrow(page, (.23, .2225), (.29, .2225))
    art.arrow(page, (.58, .2225), (.64, .2225))
    art.label(page, .5, .08, "ค่าความคล้ายยังไม่ใช่คำทำนายคลาส · SVM ต้องนำมาคำนวณคะแนนก่อน", 18,
              art.MUTED, ha="center")
    art.save(fig, page, "08-support-vector-machines-rbf-similarity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=art.OUT)
    args = parser.parse_args()
    for filename in ("tahoma.ttf", "tahomabd.ttf"):
        path = Path("/mnt/c/Windows/Fonts") / filename
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    art.configure()
    art.plt.rcParams["svg.hashsalt"] = "ml-ep08-rbf-similarity"
    art.OUT = args.output_dir
    similarity_example()


if __name__ == "__main__":
    main()
