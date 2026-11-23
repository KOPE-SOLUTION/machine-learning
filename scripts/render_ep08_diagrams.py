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

    art.box(page, .64, .61, .30, .14, "Decision boundary\nเส้นทึบ = แนวแบ่งคลาส", size=19)
    art.box(page, .64, .42, .30, .14, "Support vectors\nจุดที่ใช้กำหนดแนวแบ่ง", size=19)
    art.box(page, .64, .23, .30, .14, "ขอบ Margin\nเส้นประสองฝั่งของแนวแบ่ง", size=19)
    for start, end, color in [((.64, .68), (0, 1.05), art.NAVY),
                              ((.64, .49), (1, 0), art.NAVY),
                              ((.64, .30), (1, -1.15), art.PURPLE)]:
        ax.annotate("", xy=end, xycoords="data", xytext=start,
                    textcoords=fig.transFigure,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2,
                                "shrinkA": 6, "shrinkB": 7})
    art.footer(page,
        "SVM พยายามให้ Margin กว้าง พร้อมให้จุดฝึกอยู่ถูกฝั่งและพ้นแถบ",
        "Margin ไม่ใช่คลาสที่สาม · จุดใหม่ยังทายตามฝั่งของเส้นทึบ")
    art.save(fig, page, "08-support-vector-machines-margin")


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
    margin()
    c_effect()
    gamma_effect()
    art.svm()
    parameters()
    workflow.ep08()
    cv_scores()


if __name__ == "__main__":
    main()
