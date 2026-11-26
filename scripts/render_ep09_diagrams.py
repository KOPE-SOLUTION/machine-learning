"""Original EP09 illustrations with counted examples and reproducible CV scores."""
from pathlib import Path
import argparse

import numpy as np
from matplotlib import font_manager
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

import render_model_diagrams as art
import render_learning_workflows as workflow


def tree():
    fig, p = art.page("Decision Tree · ตรวจเงื่อนไขจากรากถึงใบ",
                      "กฎสมมติ · ข้อมูลใหม่: feature_3 = 0.8 และ feature_1 = 0.7")
    art.label(p, .5, .805, "Root · เงื่อนไขแรก", 19, art.BLUE, True, ha="center")
    art.box(p, .35, .65, .30, .12, "feature_3 ≤ 0.5 ?", size=24)
    art.box(p, .15, .40, .22, .12, "Leaf · ใบ\nทายคลาส 0", size=20)
    art.box(p, .57, .40, .30, .12, "feature_1 ≤ 1.0 ?", color="#FFF0DF", size=24)
    art.box(p, .46, .19, .20, .11, "Leaf · ใบ\nทายคลาส 1", color="#FFF0DF", size=20, ink=art.ORANGE)
    art.box(p, .75, .19, .20, .11, "Leaf · ใบ\nทายคลาส 0", size=20)
    for start, end, color in [((.43,.65),(.26,.52),art.MUTED),
                              ((.57,.65),(.72,.52),art.ORANGE),
                              ((.65,.40),(.56,.30),art.ORANGE),
                              ((.79,.40),(.85,.30),art.MUTED)]:
        art.arrow(p, start, end, color, 3)
    art.label(p, .24, .59, "จริง", 18, art.MUTED)
    art.label(p, .68, .60, "ไม่จริง: 0.8 > 0.5", 17, art.ORANGE)
    art.label(p, .43, .355, "จริง: 0.7 ≤ 1.0", 17, art.ORANGE)
    art.label(p, .87, .355, "ไม่จริง", 17, art.MUTED)
    for y, depth in ((.71,0),(.46,1),(.245,2)):
        art.label(p, .045, y, f"ลึก {depth}", 17, art.MUTED)
    art.footer(p, "เส้นสีส้มเน้นเส้นทางของแถวใหม่ · รากลึก 0 · ลงหนึ่งขั้นเพิ่มความลึก 1",
               "fit เรียนคอลัมน์และค่าตัดจาก train · predict ใช้เงื่อนไขที่เรียนไว้")
    art.save(fig, p, "09-trees-and-forests-tree")


def gini():
    x = np.arange(1, 7).reshape(-1, 1)
    labels = np.array([0, 0, 0, 1, 1, 1])
    model = DecisionTreeClassifier(max_depth=1, random_state=42).fit(x, labels)
    assert model.tree_.threshold[0] == 3.5
    assert np.isclose(model.tree_.impurity[0], .5)
    impurity_right = 1 - (2/5)**2 - (3/5)**2
    assert np.isclose(impurity_right, .48) and np.isclose(5/6*impurity_right, .4)
    fig, p = art.page("Gini · เปรียบเทียบเงื่อนไขก่อนแบ่งข้อมูล",
                      "ตัวอย่างหกแถว · อ่าน feature กับ label เป็นคู่แนวตั้ง คู่ละหนึ่งแถว")
    art.box(p, .05, .68, .90, .14, color="white")
    art.label(p, .08, .777, "feature", 20, bold=True)
    art.label(p, .08, .715, "label", 20, bold=True)
    for j, label in enumerate(labels):
        xpos = .30 + j * .115
        art.label(p, xpos, .777, str(j+1), 22, ha="center")
        art.label(p, xpos, .715, str(label), 22,
                  art.BLUE if label == 0 else art.ORANGE, True, ha="center")
    art.label(p, .5, .636,
              "ก่อนแบ่ง: คลาสละ 3/6 → Gini = 1 − (3/6)² − (3/6)² = 0.50",
              19, art.MUTED, ha="center")
    for left, title, groups, impurities, weighted, gain, tint in [
        (.05, "ลอง feature ≤ 1.5", "ซ้าย [0]  |  ขวา [0, 0, 1, 1, 1]",
         "ซ้าย: Gini = 0\nขวา: 1 − (2/5)² − (3/5)² = 0.48",
         "(1/6 × 0) + (5/6 × 0.48) = 0.40", "ลดลง 0.50 − 0.40 = 0.10", "white"),
        (.52, "ลอง feature ≤ 3.5", "ซ้าย [0, 0, 0]  |  ขวา [1, 1, 1]",
         "คลาสไม่ปนกัน: Gini ซ้าย = ขวา = 0",
         "(3/6 × 0) + (3/6 × 0) = 0", "ลดลง 0.50 − 0 = 0.50", "#EDF4FF"),
    ]:
        art.box(p, left, .24, .43, .35, color=tint)
        center = left + .215
        art.label(p, center, .548, title, 23, art.BLUE, True, ha="center")
        art.label(p, center, .485, groups, 17, ha="center")
        art.label(p, center, .411, impurities, 17, ha="center")
        art.label(p, center, .338, weighted, 18, ha="center")
        art.label(p, center, .281, gain, 21, art.BLUE, True, ha="center")
    art.label(p, .5, .162, "เลือก 3.5: ลดความปะปนได้มากกว่าในตัวอย่างนี้",
              25, art.BLUE, True, ha="center")
    art.label(p, .5, .07, "Gini หลังแบ่งต้องถ่วงตามจำนวนแถวของแต่ละฝั่ง", 20, art.MUTED, ha="center")
    art.save(fig, p, "09-trees-and-forests-gini")


def forest():
    fig = art.plt.figure(figsize=(16, 10), facecolor=art.BG)
    p = fig.add_axes([0, 0, 1, 1])
    p.set(xlim=(0, 1), ylim=(0, 1)); p.axis("off")
    art.label(p, .05, .945, "Random Forest · นับในใบ แล้วเฉลี่ยข้ามต้นไม้", 27, art.BLUE, True)
    art.label(p, .05, .89, "ตัวอย่างสมมติสองต้น · แต่ละต้นรับข้อมูลใหม่แถวเดียวกัน", 19, art.MUTED)
    art.box(p, .34, .76, .32, .07, "ข้อมูล train ชุดเดียวกัน", size=22)
    art.label(p, .05, .80, "ตอน fit", 22, art.BLUE, True)
    for left, name, count in ((.05, "ต้นที่ 1", 4), (.55, "ต้นที่ 2", 2)):
        center = left + .20
        art.arrow(p, (.5, .75), (center, .70), art.MUTED)
        art.box(p, left, .55, .40, .14, color="#EDF4FF")
        art.label(p, center, .66, name, 22, art.BLUE, True, ha="center")
        art.label(p, center, .595, "สุ่มแถวแบบคืนกลับ\nสุ่มคอลัมน์ที่พิจารณาในแต่ละจุดแบ่ง", 18, ha="center")
        art.arrow(p, (center, .54), (center, .49))
        art.box(p, left, .27, .40, .21, color="white")
        art.label(p, center, .455, "ใบที่แถวใหม่ไปถึง", 21, bold=True, ha="center")
        art.label(p, center, .409, f"คลาส 0: {5-count} รายการ  |  คลาส 1: {count} รายการ", 19, ha="center")
        art.label(p, center, .359, f"P(คลาส 0) = {5-count}/5 = {(5-count)/5:.1f}", 22, art.BLUE, True, ha="center")
        art.label(p, center, .309, f"P(คลาส 1) = {count}/5 = {count/5:.1f}", 22, art.ORANGE, True, ha="center")
        art.arrow(p, (center, .26), (.5, .235), art.ORANGE)
    art.label(p, .5, .505, "ตอนทำนาย", 21, art.ORANGE, True, ha="center")
    art.box(p, .05, .12, .90, .105, color="#FFF0DF")
    art.label(p, .5, .198, "คลาส 0: (0.2 + 0.6) / 2 = 0.4", 22, art.BLUE, True, ha="center")
    art.label(p, .5, .153, "คลาส 1: (0.8 + 0.4) / 2 = 0.6", 22, art.ORANGE, True, ha="center")
    art.label(p, .5, .078, "เลือกคลาส 1 เพราะ 0.6 มากกว่า 0.4", 23, art.ORANGE, True, ha="center")
    art.label(p, .5, .028, "จำนวนรายการในใบรวมจำนวนครั้งที่แถวฝึกถูกสุ่มซ้ำ", 17, art.MUTED, ha="center")
    assert np.isclose(np.mean([4/5, 2/5]), .6)
    art.save(fig, p, "09-trees-and-forests-ensemble")


def validation_flow():
    fig, p = art.page("เปรียบเทียบโมเดล · Train / CV / Test",
                      "ทั้งสามโมเดลใช้ fold เดียวกัน · แต่ละรอบเริ่มฝึกสำเนาโมเดลใหม่")
    art.box(p, .34, .735, .32, .08, "ข้อมูลทั้งหมด 800 แถว", size=23)
    art.arrow(p, (.40, .725), (.33, .675), art.BLUE)
    art.arrow(p, (.60, .725), (.835, .675), art.ORANGE)
    art.box(p, .05, .50, .56, .165, color="white")
    art.label(p, .33, .628, "Train 600 แถว → แบ่งเป็น 5 fold", 22, art.BLUE, True, ha="center")
    for i in range(5):
        art.box(p, .079+i*.10, .525, .085, .055, "120", color="#EDF4FF" if i<4 else "#FFF0DF", size=19)
    art.box(p, .72, .50, .23, .165, "Test 200 แถว\nเก็บไว้ประเมินท้ายสุด", color="#FFF0DF", size=21)
    art.arrow(p, (.25,.49), (.175,.425), art.BLUE)
    art.arrow(p, (.52,.49), (.485,.425), art.ORANGE)
    art.box(p, .05, .28, .25, .135, "4 fold → ฝึก 480 แถว\nวัด Train accuracy", size=20)
    art.box(p, .36, .28, .25, .135, "1 fold → ตรวจ 120 แถว\nวัด Validation accuracy", color="#FFF0DF", size=18)
    art.arrow(p, (.31,.348), (.35,.348), art.BLUE)
    art.arrow(p, (.485,.27), (.485,.235), art.ORANGE)
    art.box(p, .05, .075, .56, .15,
            "เฉลี่ย CV ทั้ง 5 รอบ → เลือกโมเดล\nฝึกใหม่ด้วย Train ครบ 600 แถว", size=22)
    art.box(p, .72, .075, .23, .15, "ทำนาย Test 200 แถว\nวัด Test accuracy", color="#FFF0DF", size=20)
    art.arrow(p, (.62,.15), (.71,.15), art.BLUE)
    art.arrow(p, (.755,.49), (.755,.235), art.ORANGE)
    art.label(p, .85, .35, "ยังไม่ใช้เลือกโมเดล", 15, art.ORANGE, ha="center")
    art.save(fig, p, "09-trees-and-forests-validation-flow")


def cv_comparison():
    X, y = make_classification(n_samples=800, n_features=10, n_informative=4,
                              n_redundant=2, flip_y=.08, random_state=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X_train, y_train))
    models = [DecisionTreeClassifier(max_depth=3, random_state=42),
              DecisionTreeClassifier(random_state=42),
              RandomForestClassifier(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=1)]
    scores = [cross_validate(model, X_train, y_train, cv=folds, scoring="accuracy",
                             return_train_score=True) for model in models]
    train = np.array([r['train_score'].mean() for r in scores])
    valid = np.array([r['test_score'].mean() for r in scores])
    assert np.allclose(np.round(valid, 3), [.775, .737, .815])
    fig, p = art.page("Overfitting · เทียบคะแนน Train กับ CV",
                      "ผลจากข้อมูลจำลองในบท · แต่ละแท่งเป็น Accuracy เฉลี่ยจาก 5 รอบ")
    ax = fig.add_axes([.075, .24, .57, .55])
    x = np.arange(3)
    bars1 = ax.bar(x-.18, train, .34, color=art.BLUE, label="Train")
    bars2 = ax.bar(x+.18, valid, .34, color=art.ORANGE, label="CV (validation)")
    ax.bar_label(bars1, fmt="%.3f", padding=4, fontsize=16)
    ax.bar_label(bars2, fmt="%.3f", padding=4, fontsize=16)
    ax.set(xticks=x, xticklabels=["shallow_tree", "deep_tree", "forest"], ylim=(0,1.14), ylabel="Accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=14)
    ax.grid(axis="y", alpha=.12); ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=15)
    ax.annotate("", xy=(1.48, train[1]), xytext=(1.48, valid[1]),
                arrowprops={"arrowstyle":"<->", "color":art.PURPLE, "lw":2.2})
    ax.plot([.82, 1.48], [train[1], train[1]], color=art.PURPLE, ls=":", lw=1.4)
    ax.plot([1.18, 1.48], [valid[1], valid[1]], color=art.PURPLE, ls=":", lw=1.4)
    art.box(p, .70, .53, .25, .25, "ต้นไม้ลึก\nTrain 1.000 · CV 0.737\nช่องว่างประมาณ 0.263\nสัญญาณ overfitting", color="#F0EAF8", size=18)
    art.box(p, .70, .245, .25, .22, "forest\nCV สูงสุด = 0.815\nเลือกไปฝึก Train ทั้งหมด\nแล้วประเมิน Test", size=18)
    art.footer(p, "ลูกศรม่วงชี้ส่วนต่าง Train − CV ของต้นไม้ลึก · ไม่ได้เลือกจาก Train สูงที่สุด",
               "forest ยังมีช่องว่าง แต่ CV สูงที่สุดในสามตัวเลือก")
    art.save(fig, p, "09-trees-and-forests-cv-comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*")
    parser.add_argument("--output-dir", type=Path, default=art.OUT)
    args = parser.parse_args()
    for name in ("tahoma.ttf", "tahomabd.ttf"):
        font = Path("/mnt/c/Windows/Fonts") / name
        if font.exists():
            font_manager.fontManager.addfont(str(font))
    art.configure()
    art.OUT = workflow.OUT = args.output_dir
    art.plt.rcParams["svg.hashsalt"] = "ml-ep09-concepts"
    renderers = {"tree": tree, "gini": gini, "forest": forest,
                 "validation": validation_flow, "comparison": cv_comparison,
                 "permutation": workflow.ep09}
    if set(args.figures) - set(renderers):
        parser.error("Unknown figure name")
    for name in args.figures or renderers:
        renderers[name]()


if __name__ == "__main__":
    main()
