"""Original EP08 illustrations: SVM, RBF parameters and cross-validation.

Run with the figure-authoring dependencies in scripts/requirements-figures.txt.
"""
from pathlib import Path
import argparse

import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import render_model_diagrams as art
import render_learning_workflows as workflow

ROOT = Path(__file__).resolve().parents[1]


def moons_data():
    fig, page = art.page(
        "make_moons · จุดหลายจุดเรียงเป็นแนวโค้งสองกลุ่ม",
        "หนึ่งจุดแทนข้อมูลหนึ่งแถว · พิกัดแนวนอนและแนวตั้งคือ features สองคอลัมน์",
    )
    clean_X, clean_y = make_moons(n_samples=40, noise=0, shuffle=False)
    X, y = make_moons(n_samples=400, noise=.25, random_state=42)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=42
    )
    assert X_train.shape == (300, 2)
    assert np.bincount(y_train).tolist() == [150, 150]
    for left, data, labels, title in [
        (.07, clean_X, clean_y, "เริ่มจากจุดที่เรียงเป็นโค้ง · noise = 0"),
        (.55, X_train, y_train, "ชุดฝึก 300 จุดในบท · noise = 0.25"),
    ]:
        ax = fig.add_axes([left, .28, .37, .48])
        for cls, color, marker in [(0, art.BLUE, "o"), (1, art.ORANGE, "s")]:
            points = data[labels == cls]
            ax.scatter(*points.T, color=color, marker=marker,
                       s=65 if left == .07 else 20, alpha=.9,
                       edgecolors="white", linewidths=.4,
                       label=f"Class {cls}", zorder=3)
        art.plot_style(ax, xlabel="Feature 1", ylabel="Feature 2")
        ax.set(xlim=(-1.8, 2.8), ylim=(-1.4, 1.9), title=title)
        if left == .07:
            for caption, point, label_pos, color in [
                ("โค้งด้านบน: คลาส 0", (0, 1), (-1.55, 1.55), art.BLUE),
                ("โค้งด้านล่าง: คลาส 1", (1, -.5), (.75, -1.05), art.ORANGE),
                ("1 จุด = 1 แถว\nfeatures = (-1, 0)\nlabel = คลาส 0", (-1, 0), (-1.65, -1.05), art.NAVY),
            ]:
                ax.annotate(caption, xy=point, xytext=label_pos,
                            fontsize=15, color=color,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 3},
                            arrowprops={"arrowstyle": "-|>", "color": color,
                                        "lw": 1.7, "shrinkB": 7}, zorder=4)
        else:
            ax.legend(loc="upper right", fontsize=13)
    art.label(page, .05, .175,
              "วงกลมน้ำเงิน = คลาส 0 · สี่เหลี่ยมส้ม = คลาส 1 · สีและรูปจุดบอก label จริง", 18)
    art.footer(page,
        "เมื่อเพิ่ม noise จุดจะกระจายออกจากแนวโค้งและอาจปะปนกัน",
        "โจทย์ของ SVM: ใช้พิกัดสองค่าทำนายว่าจุดใหม่เป็นคลาส 0 หรือ 1")
    art.save(fig, page, "08-support-vector-machines-moons-data")


def linear_margin_example():
    """Fit the six-point example shared by the margin illustrations."""
    X = np.array([[-2, -1], [-2, 1], [-1, 0], [1, 0], [2, -1], [2, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = SVC(kernel="linear", C=100).fit(X, y)
    assert np.allclose(model.coef_, [[1, 0]])
    assert np.allclose(model.intercept_, [0])
    assert set(model.support_) == {2, 3}
    assert np.array_equal(model.predict(X), y)
    assert np.allclose(model.decision_function([[-1, 0], [0, 0], [1, 0]]), [-1, 0, 1])
    assert np.isclose(2 / np.linalg.norm(model.coef_[0]), 2)
    return X, y, model


def boundary_choice():
    """Compare an illustrative separating line with the actual fitted optimum."""
    X, y, fitted = linear_margin_example()
    fitted_w, fitted_b = fitted.coef_[0], fitted.intercept_[0]
    candidate_w, candidate_b = np.array([2.5, 0.]), -1.5
    green = "#18785C"

    # Check every constraint used in the collapsed hand calculation, including
    # nonzero Feature 2 values. The two central points alone do not constrain w2.
    signed_y = 2 * y - 1
    for w2 in (-1., -.5, 0., .5, 1.):
        assert np.all(signed_y * (X @ np.array([1., w2])) >= 1)
    for w2 in (-1.1, 1.1):
        assert not np.all(signed_y * (X @ np.array([1., w2])) >= 1)
    assert np.allclose(fitted.decision_function(X), [-2, -2, -1, 1, 2, 2])

    fig, page = art.page(
        "SVM · ทายถูกเหมือนกัน แต่เว้นระยะต่างกัน",
        "จุดฝึกหกจุดชุดเดียวกัน · ซ้าย: เส้นที่สมมติไว้เปรียบเทียบ · ขวา: ผลจากการฝึก Linear SVC",
    )
    art.label(page, .07, .80, "สีจุด = label จริง · น้ำเงิน 0 / ส้ม 1", 17)
    art.label(page, .56, .80, "สีพื้น = คลาสที่แนวแบ่งนั้นทำนาย", 17)
    for left, w, b, title, nearest in [
        (.07, candidate_w, candidate_b, "แนวแบ่งที่ลอง: พิกัด 0.6", .4),
        (.56, fitted_w, fitted_b, "แนวแบ่งที่ฝึกได้: พิกัด 0", 1.),
    ]:
        boundary = -b / w[0]
        scores = X @ w + b
        distances = np.abs(scores) / np.linalg.norm(w)
        assert np.array_equal((scores > 0).astype(int), y)
        assert np.all(signed_y * scores >= 1 - 1e-9)
        assert np.isclose(distances.min(), nearest)
        assert np.isclose(abs(-1 - boundary), distances[2])
        assert np.isclose(abs(1 - boundary), distances[3])

        ax = fig.add_axes([left, .30, .37, .43])
        ax.axvspan(-2.7, boundary, color="#EDF4FF")
        ax.axvspan(boundary, 2.7, color="#FFF5EA")
        ax.axvline(boundary, color=art.NAVY, linewidth=2.7)
        for cls, color in [(0, art.BLUE), (1, art.ORANGE)]:
            ax.scatter(*X[y == cls].T, color=color, s=125, zorder=5)
        for point, label_y, distance in [(-1, .30, distances[2]), (1, -.47, distances[3])]:
            ax.annotate("", xy=(point, 0), xytext=(boundary, 0),
                        arrowprops={"arrowstyle": "<->", "color": green, "lw": 2.5,
                                    "shrinkA": 3, "shrinkB": 9}, zorder=4)
            ax.text((point + boundary) / 2, label_y, f"{distance:g} หน่วย",
                    ha="center", fontsize=18, color=green,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 3}, zorder=6)
        ax.set(xlim=(-2.7, 2.7), ylim=(-1.65, 1.65),
               xticks=[-2, -1, 0, 1, 2], yticks=[-1, 0, 1],
               xlabel="Feature 1 (พิกัด)", ylabel="Feature 2", title=title)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        art.label(page, left + .185, .19, f"ระยะถึงจุดฝึกที่ใกล้ที่สุด = {nearest:g} หน่วย",
                  18, art.NAVY, True, ha="center")
    art.footer(page,
        "เส้นทึบ = แนวแบ่งคลาส · ลูกศรเขียว = ระยะตั้งฉากจากแนวแบ่งถึงจุดฝึก",
        "ทั้งสองเส้นทายถูกครบ 6 จุด · แต่ระยะถึงจุดฝึกที่ใกล้ที่สุดเพิ่มจาก 0.4 → 1 หน่วย")
    art.save(fig, page, "08-support-vector-machines-boundary-choice")


def margin():
    fig, page = art.page(
        "Margin · ระยะถึงจุดฝึก กับแถบคะแนน",
        "ลูกศรเขียว: จากแนวแบ่งถึงจุดฝึก · ลูกศรม่วง: ระหว่างเส้นคะแนน −1 กับ +1",
    )
    art.label(page, .07, .81, "สีจุด = label จริง: น้ำเงิน 0 · ส้ม 1", 18)
    art.label(page, .64, .81, "สีพื้น = คลาสที่โมเดลทำนาย", 18)
    X, y, model = linear_margin_example()
    green = "#18785C"

    ax = fig.add_axes([.07, .23, .52, .52])
    ax.axvspan(-2.8, 0, color="#EDF4FF")
    ax.axvspan(0, 2.8, color="#FFF5EA")
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
    ax.text(0, 1.73, "แถบคะแนนกว้าง 2 หน่วย", ha="center", color=art.PURPLE, fontsize=18,
            bbox={"facecolor": "#F0EDF9", "edgecolor": "none", "pad": 2})
    for side in (-1, 1):
        ax.annotate("", xy=(side, 0), xytext=(0, 0),
                    arrowprops={"arrowstyle": "<->", "color": green, "lw": 2.5,
                                "shrinkA": 2, "shrinkB": 9}, zorder=5)
        ax.text(side * .5, .25, "1 หน่วย", ha="center", color=green, fontsize=16,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=6)
    ax.text(-1.65, -1.6, "พื้นน้ำเงิน: ทาย 0", ha="center", color=art.BLUE, fontsize=17)
    ax.text(1.65, -1.6, "พื้นส้ม: ทาย 1", ha="center", color=art.ORANGE, fontsize=17)
    ax.set(xlim=(-2.8, 2.8), ylim=(-2, 2.25), xticks=[-1, 0, 1], yticks=[-1, 0, 1],
           xlabel="Feature 1 (พิกัด)", ylabel="Feature 2")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    art.box(page, .64, .61, .30, .14, "Decision boundary\nแนวแบ่งคลาส · คะแนน = 0", size=19)
    art.box(page, .64, .42, .30, .14, "Support vectors\nจุดที่ใช้กำหนดแนวแบ่ง", size=19)
    art.box(page, .64, .23, .30, .14, "เส้นประ = ขอบแถบคะแนน\nคะแนน = −1 และ +1", size=18)
    for start, end, color in [((.64, .68), (0, 1.05), art.NAVY),
                              ((.64, .49), (1, 0), art.NAVY),
                              ((.64, .30), (1, -1.15), art.PURPLE)]:
        ax.annotate("", xy=end, xycoords="data", xytext=start,
                    textcoords=fig.transFigure,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2,
                                "shrinkA": 6, "shrinkB": 7})
    art.footer(page,
        "โมเดลที่ฝึกได้: f(x) = 1×Feature 1 + 0×Feature 2 + 0 = Feature 1",
        "ระยะถึงจุดฝึก = 1 หน่วยต่อฝั่ง · ความกว้างแถบคะแนน = 2 หน่วยทั้งแถบ")
    art.save(fig, page, "08-support-vector-machines-margin")


def margin_scaling():
    """Compare geometric distances with fixed-score contours after rescaling."""
    X, y, model = linear_margin_example()
    w, b = model.coef_[0], model.intercept_[0]
    green = "#18785C"
    fig, page = art.page(
        "คูณคะแนนด้วย 2 · ระยะถึงจุดฝึกยังเท่าเดิม",
        "จุดฝึก แนวแบ่ง และสเกลพิกัดเหมือนเดิมทั้งสองกราฟ · เปลี่ยนเฉพาะสเกลคะแนน ไม่ได้ฝึกใหม่",
    )
    for left, factor, title in [(.07, 1., "คะแนนเดิม: f(x) = Feature 1"),
                                 (.56, 2., "คูณ 2: g(x) = 2 × Feature 1")]:
        scaled_w, scaled_b = factor * w, factor * b
        scores = X @ scaled_w + scaled_b
        norm = np.linalg.norm(scaled_w)
        # Scaling both numerator and denominator preserves geometric distance.
        assert np.array_equal((scores > 0).astype(int), y)
        assert np.allclose(np.abs(scores) / norm, np.abs(model.decision_function(X)))
        assert np.isclose(np.min(np.abs(scores) / norm), 1)
        # Fixed scores -1,+1 instead locate different parallel lines.
        edges = (np.array([-1., 1.]) - scaled_b) / scaled_w[0]
        assert np.allclose(edges, np.array([-1., 1.]) / factor)
        assert np.isclose(edges[1] - edges[0], 2 / norm)
        assert np.allclose(np.array([-1., 1.]) * scaled_w[0] + scaled_b,
                           np.array([-1., 1.]) * factor)

        art.label(page, left + .185, .79, title, 21, art.NAVY, True, ha="center")
        ax = fig.add_axes([left, .31, .37, .42])
        ax.axvspan(-2.7, 0, color="#EDF4FF")
        ax.axvspan(0, 2.7, color="#FFF5EA")
        ax.axvline(0, color=art.NAVY, linewidth=2.5)
        for edge in edges:
            ax.axvline(edge, color=art.PURPLE, linestyle="--", linewidth=1.8)
        for cls, color in [(0, art.BLUE), (1, art.ORANGE)]:
            ax.scatter(*X[y == cls].T, color=color, s=115, zorder=3)
        # Mark the same nearest points; no refit and no new support-vector selection.
        ax.scatter(*X[model.support_].T, s=340, facecolors="none",
                   edgecolors=art.NAVY, linewidths=1.7, zorder=4)
        ax.annotate("", xy=(edges[1], 1.45), xytext=(edges[0], 1.45),
                    arrowprops={"arrowstyle": "<->", "color": art.PURPLE, "lw": 2.5})
        ax.text(0, 1.7, f"แถบคะแนนกว้าง {2 / factor:g} หน่วย",
                ha="center", fontsize=17, color=art.PURPLE,
                bbox={"facecolor": "#F0EDF9", "edgecolor": "none", "pad": 2})
        for side in (-1, 1):
            ax.annotate("", xy=(side, 0), xytext=(0, 0),
                        arrowprops={"arrowstyle": "<->", "color": green, "lw": 2.5,
                                    "shrinkA": 2, "shrinkB": 8}, zorder=5)
            ax.text(side * .5, .27, "1 หน่วย", ha="center", fontsize=15, color=green,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=6)
            score_text = f"−{factor:g}" if side == -1 else f"+{factor:g}"
            ax.text(side, -.45, f"คะแนน {score_text}", ha="center", fontsize=15,
                    color=art.NAVY,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=6)
        ax.set(xlim=(-2.7, 2.7), ylim=(-1.5, 2.05),
               xticks=[-2, -1, 0, 1, 2], yticks=[-1, 0, 1],
               xlabel="Feature 1 (พิกัด)", ylabel="Feature 2")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        positions = "−1 และ +1" if factor == 1 else "−0.5 และ +0.5"
        art.label(page, left + .185, .20, f"เส้นคะแนน −1, +1 อยู่ที่พิกัด {positions}",
                  16, art.PURPLE, ha="center")
    art.footer(page,
        "เขียว = ระยะถึงจุดฝึก · ม่วง = ความกว้างแถบคะแนน −1 ถึง +1 · เส้นทึบอยู่ที่พิกัด 0 ทั้งคู่",
        "ถ้าใช้ g(x) วาดขอบที่พิกัดเดิม −1, +1 → ต้องใช้คะแนน −2, +2 จึงกว้าง 2 หน่วยเท่าเดิม")
    art.save(fig, page, "08-support-vector-machines-margin-scaling")


def c_effect():
    fig, page = art.page(
        "C · ดูจุดเดิม แล้วดูว่าแนวแบ่งเปลี่ยนอย่างไร",
        "Linear SVM · ใช้จุดฝึก 7 จุดเดิมทั้งสองภาพ · เปลี่ยนเฉพาะ C",
    )
    X = np.array([[-2, -1], [-2, 1], [-1, 0], [1, 0],
                  [2, -1], [2, 1], [.7, .7]])
    y = np.array([0, 0, 0, 1, 1, 1, 0])
    xx, yy = np.meshgrid(np.linspace(-2.8, 2.8, 250),
                         np.linspace(-2.1, 2.3, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    for left, c, title, caption in [
        (.07, .1, "C น้อย = 0.1", "ยอมให้จุดนี้ผิดฝั่ง · Margin กว้าง"),
        (.55, 10, "C มาก = 10", "ปรับแนวแบ่งให้จุดนี้ถูก · Margin แคบลง"),
    ]:
        ax = fig.add_axes([left, .29, .37, .44])
        model = SVC(kernel="linear", C=c).fit(X, y)
        score = model.decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, score, levels=[-100, 0, 100],
                    colors=["#DCE9FF", "#FFE6CA"])
        ax.contour(xx, yy, score, levels=[0], colors=[art.NAVY], linewidths=2.5)
        ax.contour(xx, yy, score, levels=[-1, 1], colors=[art.PURPLE],
                    linestyles="--", linewidths=2)
        for cls, color in [(0, art.BLUE), (1, art.ORANGE)]:
            pts = X[y == cls]
            ax.scatter(*pts.T, s=125, color=color, edgecolors="white", zorder=4)
        ax.scatter(*X[-1], s=430, facecolors="none", edgecolors=art.NAVY,
                   linewidths=2, zorder=5)
        predicted = model.predict(X[-1:])[0]
        assert predicted == (1 if c == .1 else 0)
        ax.annotate("จุดนี้ทายผิด" if c == .1 else "จุดเดิมทายถูก",
                    xy=X[-1], xytext=(-2.45, 1.8), fontsize=18,
                    color=art.NAVY,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 4},
                    arrowprops={"arrowstyle": "-|>", "color": art.NAVY,
                                "lw": 2, "shrinkB": 14}, zorder=6)
        # Endpoints are on the learned decision scores -1 and +1.
        w, b = model.coef_[0], model.intercept_[0]
        center = np.array([0., -1.45]) if c == .1 else np.array([-.7, -1.2])
        center -= (w @ center + b) * w / (w @ w)
        offset = w / (w @ w)
        ax.annotate("", xy=center + offset, xytext=center - offset,
                    arrowprops={"arrowstyle": "<->", "color": art.PURPLE, "lw": 2.5})
        ax.text(0 if c == .1 else -.7, -1.95,
                "Margin กว้าง" if c == .1 else "Margin แคบลง",
                fontsize=17, color=art.PURPLE, ha="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
        ax.set(xlim=(-2.8, 2.8), ylim=(-2.1, 2.3), xticks=[], yticks=[], title=title)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        art.label(page, left + .185, .22, caption, 17, art.NAVY, ha="center")
    art.footer(page,
        "สีจุด = คลาสจริง · สีพื้น = คลาสที่ทำนาย · เส้นทึบ = แนวแบ่ง · เส้นประ = ขอบ Margin",
        "C มาก → เน้นให้จุดฝึกอยู่ถูกฝั่งและพ้น Margin มากขึ้น")
    art.save(fig, page, "08-support-vector-machines-c-effect")


def gamma_effect():
    fig, page = art.page(
        "gamma · จุดห่างเท่ากัน แต่ความคล้ายลดลงต่างกัน",
        "RBF วัดความคล้ายกับจุดกลาง · สีเข้ม = คล้ายมาก · สีอ่อน = คล้ายน้อย",
    )
    xx, yy = np.meshgrid(np.linspace(-2.3, 2.3, 300),
                         np.linspace(-2, 2, 300))
    cmap = LinearSegmentedColormap.from_list("rbf_similarity", ["#FFFFFF", art.BLUE])
    for left, gamma, title, caption in [
        (.07, .2, "gamma น้อย = 0.2", "บริเวณสีเข้มแผ่กว้าง · ความคล้ายลดช้า"),
        (.55, 4, "gamma มาก = 4", "บริเวณสีเข้มแคบลง · ความคล้ายลดเร็ว"),
    ]:
        ax = fig.add_axes([left, .29, .37, .44])
        similarity = np.exp(-gamma * (xx ** 2 + yy ** 2))
        ax.imshow(similarity, origin="lower", extent=[-2.3, 2.3, -2, 2],
                   cmap=cmap, vmin=0, vmax=1, aspect="equal")
        ax.scatter(0, 0, marker="*", s=250, color="white", edgecolors=art.NAVY, zorder=4)
        ax.scatter(1, 0, s=110, color=art.ORANGE, edgecolors="white", zorder=4)
        ax.plot([0, 1], [0, 0], color=art.NAVY, linestyle="--", lw=1.5)
        ax.annotate("จุดกลาง", xy=(0, 0), xytext=(-1.9, -.9), fontsize=17,
                    color=art.NAVY,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 3},
                    arrowprops={"arrowstyle": "-|>", "color": art.NAVY, "lw": 2})
        ax.annotate("จุดนี้ยังคล้ายมาก" if gamma == .2 else "จุดเดิมคล้ายน้อยลงมาก",
                    xy=(1, 0), xytext=(0, 1.35), fontsize=17, ha="center",
                    color=art.NAVY,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 4},
                    arrowprops={"arrowstyle": "-|>", "color": art.NAVY,
                                "lw": 2, "shrinkB": 8})
        ax.set(xticks=[], yticks=[], title=title)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        art.label(page, left + .185, .22, caption, 17, art.NAVY, ha="center")
    art.footer(page,
        "ดาวและจุดสีส้มอยู่ตำแหน่งเดิมทั้งสองภาพ · สีพื้นแสดงความคล้าย ไม่ใช่คลาสที่ทำนาย",
        "gamma มาก → แต่ละจุดมีผลในบริเวณแคบ → แนวแบ่งอาจตามรายละเอียดเฉพาะจุดมากขึ้น")
    art.save(fig, page, "08-support-vector-machines-gamma-effect")


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
            if varying == "C" and c in (.1, 100):
                point = scaled[187]
                prediction = model.predict(point[None])[0]
                assert prediction != y_train[187] if c == .1 else prediction == y_train[187]
                ax.scatter(*point, s=95, facecolors="none", edgecolors=art.NAVY, lw=1.5)
                ax.annotate("จุดนี้ทายผิด" if c == .1 else "จุดเดิมทายถูก",
                            xy=point, xytext=(-.35, -2.25), fontsize=14,
                            color=art.NAVY,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 3},
                            arrowprops={"arrowstyle": "-|>", "color": art.NAVY,
                                        "lw": 1.8, "shrinkB": 6})
            elif varying == "gamma" and gamma in (.02, 20):
                target = (0, -.26) if gamma == .02 else (-1.45, -2.1)
                text_pos = (-1.2, 2.0) if gamma == .02 else (-.2, -2.4)
                ax.annotate("แนวแบ่งเปลี่ยนช้า" if gamma == .02 else "แนวแบ่งล้อมจุดเดียว",
                            xy=target, xytext=text_pos, fontsize=14, color=art.NAVY,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 3},
                            arrowprops={"arrowstyle": "-|>", "color": art.NAVY, "lw": 1.8})
            value = c if varying == "C" else gamma
            ax.set(title=f"{varying} = {value:g}", xlim=(-2.8, 2.8),
                   ylim=(-2.8, 2.8), xticks=[], yticks=[])
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    art.label(page, .05, .05,
        "แนวแบ่งที่ตามจุดฝึกมากขึ้น อาจทายข้อมูลใหม่ได้แย่ลง → ใช้ CV ช่วยเลือก",
        19, art.BLUE, True)
    art.save(fig, page, "08-support-vector-machines-parameters")


def cv_scores():
    X, y = make_moons(n_samples=400, noise=.25, random_state=42)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(X_train, y_train))
    scores = {}
    for name in ("linear", "rbf"):
        candidate = make_pipeline(StandardScaler(), SVC(kernel=name, C=1, gamma="scale"))
        scores[name] = cross_val_score(candidate, X_train, y_train,
                                      cv=folds, scoring="accuracy")
    correct = {name: np.rint(values * 60).astype(int) for name, values in scores.items()}
    assert correct["linear"].tolist() == [43, 48, 51, 50, 49]
    assert correct["rbf"].tolist() == [54, 50, 57, 57, 54]

    fig, page = art.page(
        "Cross-validation · คะแนนแต่ละรอบมาจากไหน",
        "แต่ละรอบฝึก Pipeline ใหม่บน 240 แถว · ตรวจผลบน validation 60 แถวชุดเดียวกันทั้งสอง kernel",
    )
    art.box(page, .05, .31, .56, .50, color="white")
    for x, title in [(.11, "Fold"), (.30, "Linear"), (.50, "RBF")]:
        art.label(page, x, .765, title, 21, art.BLUE, True, ha="center")
    for i in range(5):
        ypos = .685 - i * .075
        art.label(page, .11, ypos, str(i + 1), 21, ha="center")
        for x, name in [(.30, "linear"), (.50, "rbf")]:
            art.label(page, x, ypos,
                      f"{correct[name][i]}/60 ≈ {scores[name][i]:.3f}",
                      19, art.NAVY, ha="center")
    art.box(page, .05, .15, .56, .10,
            f"ค่าเฉลี่ย  Linear ≈ {scores['linear'].mean():.3f}  |  RBF ≈ {scores['rbf'].mean():.3f}",
            size=21)
    art.box(page, .66, .59, .29, .20,
            "รอบที่ 1 ของ RBF\nฝึก 240 → ทำนายอีก 60\nทายถูก 54 → 54/60 = 0.900", size=19)
    art.arrow(page, (.615, .685), (.655, .685), art.BLUE)
    art.box(page, .66, .30, .29, .24,
            "เฉลี่ย RBF ทั้ง 5 รอบ\n(0.900 + 0.833 + 0.950\n+ 0.950 + 0.900) ÷ 5\n≈ 0.907", size=19)
    art.footer(page,
        "เลือกการตั้งค่า RBF เพราะ CV accuracy เฉลี่ยสูงกว่า · ไม่เลือกเฉพาะรอบที่ดีที่สุด",
        "จากนั้นฝึกใหม่ด้วย train 300 แถว → ประเมินบน test 100 แถวที่กันไว้")
    art.save(fig, page, "08-support-vector-machines-cv-scores")


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
    moons_data()
    boundary_choice()
    margin()
    margin_scaling()
    c_effect()
    gamma_effect()
    art.svm()
    parameters()
    workflow.ep08()
    cv_scores()


if __name__ == "__main__":
    main()
