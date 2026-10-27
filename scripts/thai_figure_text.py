"""Shape Thai text before drawing it in Matplotlib PNG/SVG figures.

Only Thai-containing plain text is intercepted. Latin-only labels and mathtext
keep Matplotlib's normal renderer. HarfBuzz positions stacked marks, then the
same glyph outlines are drawn in raster and vector outputs.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from matplotlib import font_manager
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont

try:
    import uharfbuzz as hb
except ImportError as exc:
    raise ImportError(
        "Rendering Thai course figures requires uharfbuzz. "
        "Install scripts/requirements-figures.txt in the figure-authoring environment."
    ) from exc


def contains_thai(text):
    return any("\u0e00" <= c <= "\u0e7f" for c in text)


class _OutlinePen(BasePen):
    def __init__(self, glyph_set):
        super().__init__(glyph_set)
        self.vertices = []
        self.codes = []

    def _moveTo(self, p):
        self.vertices.append(p)
        self.codes.append(MplPath.MOVETO)

    def _lineTo(self, p):
        self.vertices.append(p)
        self.codes.append(MplPath.LINETO)

    def _curveToOne(self, p1, p2, p3):
        self.vertices.extend([p1, p2, p3])
        self.codes.extend([MplPath.CURVE4] * 3)

    def _qCurveToOne(self, p1, p2):
        self.vertices.extend([p1, p2])
        self.codes.extend([MplPath.CURVE3] * 2)

    def _closePath(self):
        self.vertices.append((0, 0))
        self.codes.append(MplPath.CLOSEPOLY)


@lru_cache(maxsize=16)
def _font(filename):
    data = Path(filename).read_bytes()
    face = hb.Face(data)
    font = hb.Font(face)
    font.scale = (face.upem, face.upem)
    hb.ot_font_set_funcs(font)
    tt = TTFont(filename)
    return font, face.upem, tt.getGlyphSet(), tt.getGlyphOrder()


@lru_cache(maxsize=2048)
def _glyph(filename, glyph_id):
    _, _, glyph_set, names = _font(filename)
    pen = _OutlinePen(glyph_set)
    glyph_set[names[glyph_id]].draw(pen)
    return np.asarray(pen.vertices, dtype=float).reshape((-1, 2)), pen.codes


@lru_cache(maxsize=4096)
def shaped_path(filename, text):
    font, upem, _, _ = _font(filename)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction = "ltr"
    buf.script = "Thai"
    buf.language = "th"
    hb.shape(font, buf)
    vertices, codes = [], []
    x = y = 0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        if info.codepoint == 0:
            raise ValueError(f"Missing glyph in {filename}: {text!r}")
        v, c = _glyph(filename, info.codepoint)
        if len(v):
            vertices.extend(v + [x + pos.x_offset, y + pos.y_offset])
            codes.extend(c)
        x += pos.x_advance
        y += pos.y_advance
    path = MplPath(np.asarray(vertices).reshape((-1, 2)), codes)
    return path, x, upem


def _metrics(renderer, text, prop):
    path, advance, upem = shaped_path(font_manager.findfont(prop), text)
    factor = renderer.points_to_pixels(prop.get_size_in_points()) / upem
    bounds = path.get_extents()
    return advance * factor, bounds.height * factor, max(0, -bounds.y0) * factor


def enable_thai_shaping():
    """Enable correct Thai glyph positioning for this rendering process."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.backends.backend_svg import RendererSVG
    for renderer_type in [RendererAgg, RendererSVG]:
        if getattr(renderer_type, "_course_thai_shaping", False):
            continue
        original_draw = renderer_type.draw_text
        original_metrics = renderer_type.get_text_width_height_descent

        def draw(self, gc, x, y, s, prop, angle, ismath=False, mtext=None,
                 _original=original_draw):
            if ismath or not contains_thai(s):
                return _original(self, gc, x, y, s, prop, angle, ismath, mtext)
            path, _, upem = shaped_path(font_manager.findfont(prop), s)
            factor = self.points_to_pixels(prop.get_size_in_points()) / upem
            transform = Affine2D().scale(factor).rotate_deg(angle).translate(x, self.height - y)
            glyph_gc = self.new_gc()
            glyph_gc.copy_properties(gc)
            glyph_gc.set_linewidth(0)
            if hasattr(self, "writer"):
                self.writer.comment(s.replace("--", "—"))
            self.draw_path(glyph_gc, path, transform, rgbFace=gc.get_rgb())
            glyph_gc.restore()

        def metrics(self, s, prop, ismath, _original=original_metrics):
            if ismath or not contains_thai(s):
                return _original(self, s, prop, ismath)
            return _metrics(self, s, prop)

        renderer_type.draw_text = draw
        renderer_type.get_text_width_height_descent = metrics
        renderer_type._course_thai_shaping = True
