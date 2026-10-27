"""Regression checks for stacked Thai marks and PNG/SVG drawing."""
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_agg import RendererAgg

from thai_figure_text import enable_thai_shaping, shaped_path
from render_model_diagrams import configure


def main():
    configure()
    for weight in ("normal", "bold"):
        filename = font_manager.findfont(font_manager.FontProperties(
            family=plt.rcParams["font.family"], weight=weight))
        for base, marked in [("ที", "ที่"), ("นี", "นี่")]:
            base_path, width, _ = shaped_path(filename, base)
            marked_path, marked_width, _ = shaped_path(filename, marked)
            # A tone mark must be above the upper vowel, not hidden behind it.
            assert marked_path.get_extents().ymax > base_path.get_extents().ymax
            assert abs(width - marked_width) < 1

    renderer = RendererAgg(800, 400, 100)
    prop = font_manager.FontProperties(size=20)
    latin = renderer.get_text_width_height_descent("KNN 123", prop, False)
    original_method = RendererAgg.draw_text
    enable_thai_shaping()
    assert RendererAgg.draw_text is original_method  # idempotent
    assert renderer.get_text_width_height_descent("KNN 123", prop, False) == latin

    fig, ax = plt.subplots(figsize=(10, 4))
    sample = "ที่ นี่ ค่าเฉลี่ย เพื่อนบ้าน หนึ่ง"
    ax.text(.1, .75, sample, fontsize=25)
    ax.text(.1, .5, sample, fontsize=25, weight="bold")
    ax.text(.1, .25, "KNN ใช้ค่าที่ปรับสเกล → หาเพื่อนบ้าน", fontsize=20)
    ax.set_ylabel("ค่าที่ใช้", fontsize=18)
    ax.set_xlabel(r"$z=(x-\mu)/\sigma$")
    fig.tight_layout()
    fig.canvas.draw()
    with TemporaryDirectory() as folder:
        png, svg = Path(folder)/"check.png", Path(folder)/"check.svg"
        fig.savefig(png)
        fig.savefig(svg)
        assert png.stat().st_size > 10000
        text = svg.read_text(encoding="utf-8")
        assert sample in text and "<path" in text and "<text" not in text
    plt.close(fig)
    print("Thai stacked marks, bold, mixed text, rotation, math, PNG/SVG: passed")


if __name__ == "__main__":
    main()
