"""Render original EP07 diagrams using the lesson's actual training data."""
from pathlib import Path
import contextlib
import io
import runpy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures"
NAVY, BLUE, ORANGE, MUTED = "#142A43", "#1558D6", "#A95000", "#4C6078"


def thai_font():
    for name in ("Noto Sans Thai", "Tahoma", "Leelawadee UI", "Thonburi", "Loma"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        return name
    raise RuntimeError("Install a Thai font such as Noto Sans Thai before rendering.")


def label(ax, x, y, text, size=18, color=NAVY, weight="normal", ha="left"):
    return ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", linespacing=1.45)


def canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor="#F6F8FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    label(ax, 0.05, 0.94, title, size=28, color=BLUE, weight="bold")
    label(ax, 0.05, 0.875, subtitle, size=17, color=MUTED)
    return fig, ax


def card(ax, x, y, w, h, color="white"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=color, edgecolor="#D9E2EF", linewidth=1.5,
    ))


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=20, linewidth=2, color=BLUE))


def save(fig, suffix):
    stem = OUTPUT / ("07-naive-bayes-text-" + suffix)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for item in ax.texts:
            box = item.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0, item.get_text()
            assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, item.get_text()
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(
        encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    print(stem.with_suffix(".png"))


def counts_diagram(ns):
    fig, ax = canvas(
        "Bag-of-words · เปลี่ยนข้อความเป็นจำนวนคำ",
        "CountVectorizer เรียน vocabulary จาก train  •  ข้อความใหม่ใช้คอลัมน์เดิม",
    )
    for x, w, title, detail in (
        (0.05, 0.24, "ข้อความใหม่ 3 ข้อความ", "ยังเป็นข้อความดิบ"),
        (0.37, 0.26, "CountVectorizer", f"vocabulary {len(ns['words'])} คำ"),
        (0.71, 0.24, "ตารางตัวเลข 3 × 49", "หนึ่งคอลัมน์ต่อหนึ่งคำ"),
    ):
        card(ax, x, 0.69, w, 0.12, "#EDF4FF")
        label(ax, x + w/2, 0.768, title, size=20, ha="center", weight="bold")
        label(ax, x + w/2, 0.715, detail, size=16, ha="center", color=MUTED)
    arrow(ax, (0.30, 0.75), (0.36, 0.75))
    arrow(ax, (0.64, 0.75), (0.70, 0.75))

    card(ax, 0.05, 0.32, 0.90, 0.32)
    table = ns["counts"]
    xs = (0.49, 0.60, 0.69, 0.80, 0.91)
    label(ax, 0.075, 0.595, "ข้อความ / คำที่นับ", size=18, weight="bold")
    for x, word in zip(xs, table.columns):
        label(ax, x, 0.595, word, size=17, weight="bold", ha="center")
    for y, (text, row) in zip((0.52, 0.445, 0.37), table.iterrows()):
        color = ORANGE if row.sum() == 0 else NAVY
        label(ax, 0.075, y, text, size=18, color=color)
        for x, count in zip(xs, row):
            label(ax, x, y, str(count), size=23, color=color, ha="center",
                  weight="bold" if count else "normal")

    card(ax, 0.05, 0.11, 0.43, 0.15, "#EDF4FF")
    label(ax, 0.075, 0.216, "team team goal → รู้จัก 3 คำ", size=21, weight="bold", color=BLUE)
    label(ax, 0.075, 0.151, "team นับ 2 ครั้ง + goal นับ 1 ครั้ง", size=18)
    card(ax, 0.52, 0.11, 0.43, 0.15, "#FFF5EA")
    label(ax, 0.545, 0.216, "recipe with vegetables → รู้จัก 0 คำ", size=19, weight="bold", color=ORANGE)
    label(ax, 0.545, 0.151, "ไม่มีใน vocabulary → ทุกคอลัมน์เป็น 0", size=18)
    label(ax, 0.05, 0.049,
          "ผลจาก Cell 1–3  •  แสดง 5 คอลัมน์ที่มีค่าในตัวอย่าง; อีก 44 คอลัมน์เป็น 0  •  โมเดลยังใช้ 49 features",
          size=15, color=MUTED)
    save(fig, "counts")


def smoothing_diagram(ns):
    model = ns["model"].named_steps["multinomialnb"]
    words = list(ns["words"])
    goal = model.feature_count_[:, words.index("goal")].astype(int)
    team = model.feature_count_[:, words.index("team")].astype(int)
    totals = model.feature_count_.sum(axis=1).astype(int)
    vocabulary = len(words)
    alpha = model.alpha
    denominator = (totals + alpha * vocabulary).astype(int)
    goal_numerator = (goal + alpha).astype(int)
    team_numerator = (team + alpha).astype(int)
    prior = np.exp(model.class_log_prior_)
    scores = prior * (team_numerator / denominator)**2 * (goal_numerator / denominator)
    probs = scores / scores.sum()
    np.testing.assert_allclose(probs, ns["model"].predict_proba(["team team goal"])[0])

    fig, ax = canvas(
        "Naive Bayes · จากความถี่คำสู่คำทำนาย",
        "Smoothing เพิ่มจำนวนสมมติให้ทุกคำในแต่ละคลาส  •  alpha = 1 เรียกว่า Laplace smoothing",
    )
    card(ax, 0.05, 0.665, 0.90, 0.155)
    label(ax, 0.50, 0.757,
          r"$P(w\mid c)=\frac{N_{cw}+\alpha}{N_c+\alpha V}$", size=31, ha="center")
    label(ax, 0.50, 0.691,
          "(จำนวนครั้งของคำ w ในคลาส c + alpha) ÷ (จำนวนคำรวมในคลาส c + alpha × V)",
          size=16, ha="center", color=MUTED)

    for i, x in enumerate((0.05, 0.52)):
        color = BLUE if i == 0 else ORANGE
        card(ax, x, 0.395, 0.43, 0.215, "#EDF4FF" if i == 0 else "#FFF5EA")
        label(ax, x + 0.025, 0.565,
              f"{model.classes_[i]} · prior = 9/18 = {prior[i]:.1f}",
              size=22, color=color, weight="bold")
        label(ax, x + 0.025, 0.515,
              f"คำรวม {totals[i]} ครั้ง  |  vocabulary V = {vocabulary}", size=18)
        label(ax, x + 0.025, 0.466,
              f"goal: ({goal[i]}+1)/({totals[i]}+{vocabulary})"
              f" = {goal_numerator[i]}/{denominator[i]}", size=18, color=color)
        label(ax, x + 0.025, 0.418,
              f"team: ({team[i]}+1)/({totals[i]}+{vocabulary})"
              f" = {team_numerator[i]}/{denominator[i]}", size=18, color=color)

    card(ax, 0.05, 0.105, 0.90, 0.235)
    label(ax, 0.075, 0.297,
          'ข้อความ "team team goal" → prior × P(team | คลาส)² × P(goal | คลาส)',
          size=20, weight="bold")
    for i, y in enumerate((0.225, 0.153)):
        color = BLUE if i == 0 else ORANGE
        equation = (rf"$0.5\times({team_numerator[i]}/{denominator[i]})^2"
                    rf"\times({goal_numerator[i]}/{denominator[i]})$")
        label(ax, 0.075, y, model.classes_[i], size=20, color=color, weight="bold")
        label(ax, 0.17, y, equation, size=22, color=color)
        label(ax, 0.53, y, f"คะแนน ≈ {scores[i]:.3g}", size=18, color=color)
        label(ax, 0.755, y, f"→ P = {probs[i]:.3f}", size=23, color=color, weight="bold")
    label(ax, 0.05, 0.049,
          "ใช้ค่าจริงจาก train  •  P ด้านขวา = คะแนนคลาส ÷ ผลรวมคะแนนสองคลาส  •  คำซ้ำจึงมีผลต่อคะแนน",
          size=15, color=MUTED)
    save(fig, "smoothing")


def main():
    plt.rcParams.update({
        "font.family": thai_font(), "mathtext.fontset": "dejavusans",
        "svg.fonttype": "path", "svg.hashsalt": "ml-ep07",
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ns = runpy.run_path(str(ROOT / "lessons/07-naive-bayes-text/lesson.py"))
    counts_diagram(ns)
    smoothing_diagram(ns)


if __name__ == "__main__":
    main()
