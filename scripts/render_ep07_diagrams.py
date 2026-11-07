"""Render original EP07 diagrams using the lesson's actual training data."""
from pathlib import Path
import contextlib
import io
import runpy

import numpy as np
from thai_figure_text import enable_thai_shaping

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
    svg_buffer = io.StringIO()
    fig.savefig(svg_buffer, format="svg", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg_buffer.getvalue().splitlines())
                   + "\n", encoding="utf-8", newline="\n")
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
          "ผลจาก Cell 1–5  •  แสดง 5 คอลัมน์ที่มีค่าในตัวอย่าง; อีก 44 คอลัมน์เป็น 0  •  โมเดลยังใช้ 49 features",
          size=15, color=MUTED)
    save(fig, "counts")


def prediction_values(ns):
    """Read every displayed count from the fitted lesson pipeline."""
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
    class_counts = model.class_count_.astype(int)
    prior = class_counts / class_counts.sum()
    np.testing.assert_allclose(prior, np.exp(model.class_log_prior_))
    scores = prior * (team_numerator / denominator)**2 * (goal_numerator / denominator)
    probs = scores / scores.sum()
    np.testing.assert_allclose(probs, ns["model"].predict_proba(["team team goal"])[0])
    np.testing.assert_allclose(
        team_numerator / denominator,
        np.exp(model.feature_log_prob_[:, words.index("team")]),
    )
    np.testing.assert_allclose(
        goal_numerator / denominator,
        np.exp(model.feature_log_prob_[:, words.index("goal")]),
    )
    return dict(
        classes=model.classes_, class_counts=class_counts, prior=prior,
        team=team, goal=goal, totals=totals, vocabulary=vocabulary,
        alpha=alpha, denominator=denominator,
        team_numerator=team_numerator, goal_numerator=goal_numerator,
        scores=scores, probs=probs,
    )


def smoothing_diagram(ns):
    data = prediction_values(ns)
    tech = list(data["classes"]).index("tech")
    goal, total = data["goal"][tech], data["totals"][tech]
    vocabulary = data["vocabulary"]
    numerator, denominator = data["goal_numerator"][tech], data["denominator"][tech]

    fig, ax = canvas(
        "Smoothing · ทำไมต้องเพิ่ม alpha",
        "goal อยู่ใน vocabulary แต่ไม่เคยพบในข้อความ tech  •  ลองเทียบก่อนและหลังเพิ่ม alpha",
    )
    card(ax, 0.05, 0.435, 0.40, 0.365, "#FFF5EA")
    label(ax, 0.075, 0.746, "ไม่เพิ่ม alpha", size=25, color=ORANGE, weight="bold")
    label(ax, 0.075, 0.678, f"goal พบ {goal} ครั้ง / คำรวม {total} ครั้ง", size=21)
    label(ax, 0.25, 0.575, rf"$\frac{{{goal}}}{{{total}}}=0$",
          size=38, color=ORANGE, ha="center")
    label(ax, 0.25, 0.474, "เมื่อมี goal → คะแนน tech × 0 = 0", size=18, ha="center")

    arrow(ax, (0.465, 0.615), (0.535, 0.615))
    card(ax, 0.55, 0.435, 0.40, 0.365, "#EDF4FF")
    label(ax, 0.575, 0.746, "เพิ่ม alpha = 1", size=25, color=BLUE, weight="bold")
    label(ax, 0.575, 0.678, "เพิ่มจำนวนสมมติ 1 ให้ทุกคำในคลาส", size=20)
    label(ax, 0.75, 0.575,
          rf"$\frac{{{goal}+1}}{{{total}+{vocabulary}}}=\frac{{{numerator}}}{{{denominator}}}$",
          size=35, color=BLUE, ha="center")
    label(ax, 0.75, 0.474, f"โอกาสของ goal ใน tech ≈ {numerator / denominator:.4f}",
          size=18, ha="center")

    card(ax, 0.05, 0.10, 0.52, 0.265)
    label(ax, 0.075, 0.311, f"ทำไมตัวส่วนเพิ่ม {vocabulary}?", size=23, weight="bold")
    label(ax, 0.075, 0.245, f"vocabulary มี {vocabulary} คำ → เพิ่มคำละ 1", size=21)
    label(ax, 0.075, 0.160, f"{total} + (1 × {vocabulary}) = {denominator}",
          size=30, color=BLUE)
    card(ax, 0.61, 0.10, 0.34, 0.265)
    label(ax, 0.635, 0.311, "เพิ่มให้ทุกคำในสูตร", size=23, weight="bold")
    label(ax, 0.635, 0.230, "พบ 0 ครั้ง  →  ใช้ 0 + 1", size=23, color=BLUE)
    label(ax, 0.635, 0.155, "พบ 3 ครั้ง  →  ใช้ 3 + 1", size=23, color=BLUE)
    label(ax, 0.05, 0.045,
          "alpha = 1 เรียกว่า Laplace smoothing  •  เพิ่มเฉพาะจำนวนในสูตร ไม่ได้เพิ่มข้อความฝึก",
          size=17, color=MUTED)
    save(fig, "smoothing")


def prediction_inputs_diagram(ns):
    data = prediction_values(ns)
    total_messages = data["class_counts"].sum()
    fig, ax = canvas(
        "1 · เตรียมค่าจากชุดฝึก",
        "ข้อความที่จะทำนาย: team team goal  •  มี team 2 ครั้ง และ goal 1 ครั้ง",
    )
    card(ax, 0.05, 0.685, 0.90, 0.125)
    label(ax, 0.50, 0.760,
          f"ชุดฝึก {total_messages} ข้อความ  •  sport {data['class_counts'][0]}"
          f" + tech {data['class_counts'][1]} ข้อความ",
          size=23, weight="bold", ha="center")
    label(ax, 0.50, 0.710,
          f"vocabulary รวม {data['vocabulary']} คำไม่ซ้ำ  •  alpha = {data['alpha']:g}",
          size=20, ha="center", color=MUTED)

    for i, x in enumerate((0.05, 0.52)):
        color = BLUE if i == 0 else ORANGE
        card(ax, x, 0.125, 0.43, 0.505, "#EDF4FF" if i == 0 else "#FFF5EA")
        label(ax, x + 0.025, 0.577,
              f"{data['classes'][i]} · Prior = {data['class_counts'][i]}/{total_messages}"
              f" = {data['prior'][i]:.1f}", size=23, color=color, weight="bold")
        label(ax, x + 0.025, 0.517,
              f"คำรวม {data['totals'][i]} ครั้ง · team {data['team'][i]} · goal {data['goal'][i]}",
              size=20)
        for word, y in (("team", 0.440), ("goal", 0.270)):
            label(ax, x + 0.025, y, f"โอกาสพบ {word} ใน {data['classes'][i]}",
                  size=20, color=MUTED)
            equation = (rf"$\frac{{{data[word][i]}+1}}{{{data['totals'][i]}+{data['vocabulary']}}}"
                        rf"=\frac{{{data[word + '_numerator'][i]}}}{{{data['denominator'][i]}}}$")
            label(ax, x + 0.215, y - 0.076, equation, size=32, color=color, ha="center")
    label(ax, 0.05, 0.052,
          "Prior ใช้จำนวนข้อความ  •  โอกาสพบคำใช้จำนวนครั้งของคำ โดยนับคำซ้ำด้วย",
          size=18, color=MUTED)
    save(fig, "prediction-inputs")


def prediction_scores_diagram(ns):
    data = prediction_values(ns)
    scores, probs = data["scores"], data["probs"]
    score_strings = [f"{value:.8f}" if value >= 1e-5 else f"{value:.10f}" for value in scores]
    total_string = f"{scores.sum():.8f}"
    fig, ax = canvas(
        "2–3 · จาก Prior ไปถึงคำทำนาย",
        "team team goal → คูณโอกาสพบ team สองครั้ง และ goal หนึ่งครั้ง",
    )
    for i, x in enumerate((0.05, 0.52)):
        color = BLUE if i == 0 else ORANGE
        card(ax, x, 0.470, 0.43, 0.345, "#EDF4FF" if i == 0 else "#FFF5EA")
        label(ax, x + 0.025, 0.765, f"2 · คำนวณคะแนน {data['classes'][i]}",
              size=23, color=color, weight="bold")
        label(ax, x + 0.215, 0.701, "Prior  ×  team  ×  team  ×  goal",
              size=21, ha="center", color=MUTED)
        team = f"{data['team_numerator'][i]}/{data['denominator'][i]}"
        goal = f"{data['goal_numerator'][i]}/{data['denominator'][i]}"
        equation = f"{data['prior'][i]:.1f} × ({team}) × ({team}) × ({goal})"
        label(ax, x + 0.215, 0.625, equation, size=22, color=color, ha="center")
        label(ax, x + 0.215, 0.536, f"คะแนน ≈ {score_strings[i]}",
              size=25, color=color, weight="bold", ha="center")
        arrow(ax, (x + 0.215, 0.459), (x + 0.215, 0.414))

    card(ax, 0.05, 0.325, 0.90, 0.080)
    label(ax, 0.50, 0.365,
          f"ผลรวมคะแนน ≈ {score_strings[0]} + {score_strings[1]} ≈ {total_string}",
          size=21, ha="center", weight="bold")

    for i, x in enumerate((0.05, 0.52)):
        color = BLUE if i == 0 else ORANGE
        arrow(ax, (x + 0.215, 0.316), (x + 0.215, 0.285))
        card(ax, x, 0.085, 0.43, 0.19)
        label(ax, x + 0.215, 0.240, f"3 · {data['classes'][i]}: คะแนน ÷ ผลรวม",
              size=21, ha="center", color=MUTED)
        equation = rf"$\frac{{{score_strings[i]}}}{{{total_string}}}\approx{probs[i]:.3f}$"
        label(ax, x + 0.215, 0.156, equation, size=30, color=color, ha="center")
    winner = data["classes"][scores.argmax()]
    label(ax, 0.50, 0.035, f"{winner} ได้คะแนนสูงกว่า → predict เลือก {winner}",
          size=20, color=BLUE, weight="bold", ha="center")
    save(fig, "prediction-scores")


def main():
    enable_thai_shaping()
    plt.rcParams.update({
        "font.family": thai_font(), "mathtext.fontset": "dejavusans",
        "svg.fonttype": "path", "svg.hashsalt": "ml-ep07",
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ns = runpy.run_path(str(ROOT / "lessons/07-naive-bayes-text/lesson.py"))
    counts_diagram(ns)
    smoothing_diagram(ns)
    prediction_inputs_diagram(ns)
    prediction_scores_diagram(ns)


if __name__ == "__main__":
    main()
