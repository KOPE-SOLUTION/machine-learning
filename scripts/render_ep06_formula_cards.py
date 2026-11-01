"""Render EP06 formula images: draw backgrounds first, then precise text.

Thai labels use HarfBuzz shaping; mathematical symbols use Matplotlib mathtext.
The formulas remain editable here. PNG/SVG assets replace Markdown math blocks.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from thai_figure_text import enable_thai_shaping
from render_ep06_diagrams import thai_font

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/figures"
NAVY, BLUE, MUTED = "#142A43", "#1558D6", "#4C6078"
PREFIX = "06-feature-engineering-formula-"

FORMULAS = [
    dict(blocks=[1], slug="prediction", title="Ridge · คูณแต่ละ feature ด้วยน้ำหนัก แล้วบวก intercept",
         rows=[r"\hat y=b+w_1z_{\mathrm{distance}}+w_2z_{\mathrm{weight}}+w_3I_{\mathrm{express}}+w_4I_{\mathrm{standard}}"],
         note="z คือค่าหลังปรับสเกล  •  I คือคอลัมน์ 0/1  •  w และ b เรียนจาก train",
         alt="ค่าทำนายเท่ากับ b บวก w1 คูณ z ของระยะทาง บวก w2 คูณ z ของน้ำหนัก บวก w3 คูณ I express บวก w4 คูณ I standard",
         size=31),
    dict(blocks=[3,14], slug="weights", title="Ridge · สูตรหาน้ำหนักและ intercept เมื่อมี feature เดียว",
         size=32, positions=[.54, .28],
         rows=[r"w=\frac{\sum_{i=1}^{n}(x_i-\bar x)(y_i-\bar y)}{\sum_{i=1}^{n}(x_i-\bar x)^2+\alpha}",
               r"b=\bar y-w\bar x"],
         note="x̄ และ ȳ คือค่าเฉลี่ยจาก train  •  alpha อยู่ในตัวหารของ w  •  ไม่ลงโทษ b",
         alt="w เท่ากับผลรวม x ลบค่าเฉลี่ย x คูณ y ลบค่าเฉลี่ย y หารด้วยผลรวมกำลังสองของ x ลบค่าเฉลี่ย x บวก alpha ส่วน b เท่ากับค่าเฉลี่ย y ลบ w คูณค่าเฉลี่ย x"),
    dict(blocks=[4], slug="proof-objective", title="พิสูจน์ · เขียนค่ารวมที่ต้องลดให้ครบ",
         rows=[r"J(b,w)=\sum_{i=1}^{n}(y_i-b-wx_i)^2+\alpha w^2"],
         note="ส่วนแรกคือ SSE  •  ส่วนที่สองคือพจน์ลงโทษ  •  กรณีนี้มีน้ำหนัก w เพียงค่าเดียว",
         alt="J ของ b และ w เท่ากับผลรวมกำลังสองของ y ลบ b ลบ wx บวก alpha คูณ w กำลังสอง"),
    dict(blocks=[5], slug="proof-intercept-derivative", title="พิสูจน์ · หาอนุพันธ์เทียบกับ b โดยตรึง w ไว้",
         rows=[r"\frac{\partial J}{\partial b}=-2\sum_{i=1}^{n}(y_i-b-wx_i)=0"],
         note="อนุพันธ์ของ residual เทียบกับ b คือ −1  •  พจน์ alpha × w² ไม่มี b",
         alt="อนุพันธ์ย่อยของ J เทียบกับ b เท่ากับลบสองคูณผลรวม y ลบ b ลบ wx แล้วตั้งเท่ากับศูนย์"),
    dict(blocks=[6], slug="proof-intercept-isolate", title="พิสูจน์ · หารด้วย −2 แล้วรวบพจน์ที่มี b",
         rows=[r"\sum_i y_i-nb-w\sum_i x_i=0",
               r"nb=\sum_i y_i-w\sum_i x_i"],
         note="b ซ้ำกัน n แถว จึงรวมเป็น nb  •  ย้ายพจน์เพื่อเตรียมหาค่า b",
         alt="ผลรวม y ลบ nb ลบ w คูณผลรวม x เท่ากับศูนย์ จัดรูปได้ nb เท่ากับผลรวม y ลบ w คูณผลรวม x"),
    dict(blocks=[7], slug="proof-intercept-mean", title="พิสูจน์ · หารด้วยจำนวนแถว n เพื่อให้เหลือ b",
         rows=[r"b=\bar y-w\bar x"],
         note="ผลรวมหารด้วยจำนวนแถวคือค่าเฉลี่ย  •  x̄ อ่านว่าเอ็กซ์บาร์  •  ȳ อ่านว่าวายบาร์",
         alt="b เท่ากับค่าเฉลี่ย y ลบ w คูณค่าเฉลี่ย x"),
    dict(blocks=[8], slug="proof-weight-derivative", title="พิสูจน์ · หาอนุพันธ์เทียบกับ w โดยตรึง b ไว้",
         rows=[r"\frac{\partial J}{\partial w}=-2\sum_i x_i(y_i-b-wx_i)+2\alpha w=0"],
         note="อนุพันธ์ของ residual เทียบกับ w คือ −xᵢ  •  อนุพันธ์ของ alpha × w² คือ 2alpha × w",
         alt="อนุพันธ์ย่อยของ J เทียบกับ w เท่ากับลบสองคูณผลรวม x คูณ residual บวกสอง alpha w แล้วตั้งเท่ากับศูนย์"),
    dict(blocks=[9], slug="proof-weight-balance", title="พิสูจน์ · ย้ายพจน์ผลรวม แล้วหารด้วย 2",
         rows=[r"\sum_i x_i(y_i-b-wx_i)=\alpha w"],
         note="ลดตัวคูณร่วมก่อนแทนค่า b  •  สมการยังเทียบเท่ากับขั้นก่อนหน้า",
         alt="ผลรวม x คูณ y ลบ b ลบ wx เท่ากับ alpha คูณ w"),
    dict(blocks=[10], slug="proof-centered-residual", title="พิสูจน์ · แทน b เพื่อให้เหลือตัวแปรที่ต้องหาเพียง w",
         rows=[r"\sum_i x_i(y_i-\bar y)-w\sum_i x_i(x_i-\bar x)=\alpha w"],
         note="แทน b = ȳ − wx̄ แล้วกระจายผลรวม  •  residual คือ (yᵢ − ȳ) − w(xᵢ − x̄)",
         alt="หลังแทน b ได้ผลรวม x คูณ y ลบค่าเฉลี่ย y ลบ w คูณผลรวม x คูณ x ลบค่าเฉลี่ย x เท่ากับ alpha w"),
    dict(blocks=[11], slug="proof-centered-xy", title="พิสูจน์ · เขียนผลรวมแรกด้วยค่าที่ลบค่าเฉลี่ยแล้ว",
         rows=[r"\sum_i x_i(y_i-\bar y)=\sum_i(x_i-\bar x)(y_i-\bar y)"],
         note="ผลรวม (yᵢ − ȳ) เป็นศูนย์ พจน์ที่คูณ x̄ จึงหายไป",
         alt="ผลรวม x คูณ y ลบค่าเฉลี่ย y เท่ากับผลรวม x ลบค่าเฉลี่ย x คูณ y ลบค่าเฉลี่ย y"),
    dict(blocks=[12], slug="proof-centered-xx", title="พิสูจน์ · เขียนผลรวมที่สองเป็นกำลังสอง",
         rows=[r"\sum_i x_i(x_i-\bar x)=\sum_i(x_i-\bar x)^2"],
         note="ผลรวม (xᵢ − x̄) เป็นศูนย์ จึงเหลือผลรวมกำลังสองของ (xᵢ − x̄)",
         alt="ผลรวม x คูณ x ลบค่าเฉลี่ย x เท่ากับผลรวมกำลังสองของ x ลบค่าเฉลี่ย x"),
    dict(blocks=[13], slug="proof-factor-weight", title="พิสูจน์ · รวบพจน์ที่มี w แล้วดึง w ออกเป็นตัวคูณร่วม",
         rows=[r"\sum_i(x_i-\bar x)(y_i-\bar y)=w\sum_i(x_i-\bar x)^2+\alpha w",
               r"\sum_i(x_i-\bar x)(y_i-\bar y)=w\left[\sum_i(x_i-\bar x)^2+\alpha\right]"],
         note="ขั้นถัดไปหารทั้งสองข้างด้วยวงเล็บทางขวา เพื่อให้เหลือ w",
         alt="ผลรวมผลคูณของค่าที่ลบค่าเฉลี่ยแล้ว เท่ากับ w คูณผลรวมกำลังสองของ x ลบค่าเฉลี่ย x บวก alpha w ดึง w ออกได้ w คูณวงเล็บผลรวมกำลังสองบวก alpha"),
    dict(blocks=[15], slug="proof-minimum", title="พิสูจน์ · ตรวจว่าคำตอบให้ค่าต่ำสุดจริง",
         rows=[r"J=(-1-u+v)^2+(-u)^2+(1-u-v)^2+2(1+v)^2",
               r"J=4+3u^2+4v^2\geq4"],
         note="ให้ b = 3 + u และ w = 1 + v  •  ค่ารวมเท่ากับ 4 เมื่อ u = v = 0 เท่านั้น",
         alt="เมื่อขยับ b เป็น 3 บวก u และ w เป็น 1 บวก v จะได้ J เท่ากับ 4 บวก 3u กำลังสอง บวก 4v กำลังสอง จึงไม่น้อยกว่า 4"),
    dict(blocks=[16], slug="fitted-prediction", title="แทนค่าน้ำหนักที่เรียนได้ · จาก features ไปเป็นค่าบริการ",
         rows=[r"\hat y\approx168.432+56.380(-1.042254)+11.288(0.019219)",
               r"\qquad+20.473(0)-20.473(0)",
               r"\hat y\approx109.89"],
         note="ตัวเลขในภาพปัดเพื่อให้อ่านง่าย  •  โค้ดคำนวณด้วยค่าจริงก่อนปัดแสดงผล",
         alt="แทนค่า 168.432 บวก 56.380 คูณลบ 1.042254 บวก 11.288 คูณ 0.019219 บวก 20.473 คูณศูนย์ ลบ 20.473 คูณศูนย์ ได้ค่าบริการประมาณ 109.89",
         size=31),
]


def background(height):
    """Layer 1: build the background and card before placing any text."""
    fig = plt.figure(figsize=(16, height), facecolor="#F6F8FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (.035, .17), .93, .59, boxstyle="round,pad=0.008,rounding_size=.02",
        facecolor="white", edgecolor="#D9E2EF", linewidth=1.4, zorder=0,
    ))
    ax.add_patch(Rectangle((.035, .84), .007, .09, facecolor=BLUE, edgecolor="none", zorder=0))
    return fig, ax


def render(item):
    count = len(item["rows"])
    height = {1: 3.6, 2: 5.0, 3: 5.8}[count]
    fig, ax = background(height)
    # Layer 2: exact Thai text and mathematical glyphs over the background.
    texts = [ax.text(.055, .885, item["title"], fontsize=24, color=BLUE,
                     weight="bold", va="center", zorder=2)]
    positions = item.get("positions", {1: [.47], 2: [.59, .32], 3: [.65, .46, .27]}[count])
    for row, y in zip(item["rows"], positions):
        texts.append(ax.text(.5, y, "$"+row+"$", fontsize=item.get("size", 34),
                             color=NAVY if y != positions[-1] else BLUE,
                             ha="center", va="center", zorder=2))
    texts.append(ax.text(.05, .075, item["note"], fontsize=16, color=MUTED,
                         va="center", zorder=2))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in texts:
        while text.get_window_extent(renderer).width > fig.bbox.width*.90:
            minimum = 21 if text in texts[1:-1] else 14
            assert text.get_fontsize() > minimum
            text.set_fontsize(text.get_fontsize()-1)
            fig.canvas.draw()
        box=text.get_window_extent(renderer)
        assert box.x0>=0 and box.x1<=fig.bbox.width, item["slug"]
        assert box.y0>=0 and box.y1<=fig.bbox.height, item["slug"]
    # Keep every equation inside the white card, with a small inset.
    for text in texts[1:-1]:
        box=text.get_window_extent(renderer)
        assert box.y0 >= fig.bbox.height*.18, item["slug"]
        assert box.y1 <= fig.bbox.height*.75, item["slug"]
    # Check all foreground elements for unintended overlap.
    boxes=[t.get_window_extent(renderer) for t in texts]
    for i,a in enumerate(boxes):
        for b in boxes[i+1:]:
            assert not a.overlaps(b), f"Text overlap: {item['slug']}"
    stem=OUTPUT/(PREFIX+item["slug"])
    fig.savefig(stem.with_suffix(".png"),dpi=150)
    svg=stem.with_suffix(".svg")
    fig.savefig(svg,metadata={"Date":None})
    svg.write_text("\n".join(s.rstrip() for s in svg.read_text(encoding="utf-8").splitlines())+"\n",
                   encoding="utf-8",newline="\n")
    plt.close(fig)
    print(stem.name)


def main():
    enable_thai_shaping()
    plt.rcParams.update({"font.family":thai_font(),"mathtext.fontset":"dejavusans",
                         "svg.fonttype":"path","svg.hashsalt":"ml-ep06-formulas"})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for item in FORMULAS:
        render(item)


if __name__=="__main__":
    main()
