"""Original, reproducible model illustrations for the learner-facing course.

These explanatory examples are deliberately separate from lesson experiments.
Run from any directory with Python; PNG/SVG assets go into assets/figures.
"""
from pathlib import Path
import argparse

from thai_figure_text import enable_thai_shaping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/figures"
BLUE, ORANGE, PURPLE = "#1558D6", "#BD5A00", "#7743B7"
NAVY, MUTED, BG, EDGE = "#142A43", "#52667D", "#F6F8FC", "#CCD9EA"


def configure():
    enable_thai_shaping()
    for name in ("Noto Sans Thai", "Tahoma", "Leelawadee UI", "Thonburi", "Loma"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            break
        except ValueError:
            pass
    else:
        raise RuntimeError("Install a Thai font before rendering.")
    plt.rcParams.update({"font.family": name, "font.size": 14,
                         "axes.labelsize": 15, "axes.titlesize": 18,
                         "mathtext.fontset": "dejavusans", "svg.fonttype": "path",
                         "svg.hashsalt": "ml-model-concepts-v1"})


def label(ax, x, y, s, size=18, color=NAVY, bold=False, ha="left"):
    return ax.text(x, y, s, fontsize=size, color=color,
                   fontweight="bold" if bold else "normal",
                   va="center", ha=ha, linespacing=1.45)


def box(ax, x, y, w, h, s="", color="#EDF4FF", size=18, ink=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.008,rounding_size=0.012",
                 facecolor=color, edgecolor=EDGE, linewidth=1.5))
    if s:
        return label(ax, x+w/2, y+h/2, s, size, ink, ha="center")


def arrow(ax, start, end, color=MUTED, lw=2):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw,
                            "shrinkA": 4, "shrinkB": 4})


def page(title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    label(ax, .05, .94, title, 27, BLUE, True)
    label(ax, .05, .875, subtitle, 17, MUTED)
    return fig, ax


def footer(ax, first, second):
    label(ax, .05, .105, first, 18, MUTED)
    label(ax, .05, .045, second, 19, BLUE, True)


def plot_style(ax, xlabel="feature 1", ylabel="feature 2"):
    ax.set_facecolor("white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.tick_params(labelsize=12)


def save(fig, page_ax, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in page_ax.texts:
        bounds = item.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0, item.get_text()
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, item.get_text()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / stem
    fig.savefig(path.with_suffix(".png"), dpi=150)
    svg = path.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(
        encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    print(stem)


def knn():
    fig, p = page("KNN · ใช้เพื่อนบ้าน 5 จุด ทายดอกใหม่",
                  "ตัวอย่างสมมติสองมิติหลังปรับสเกล · โค้ด Iris ในบทใช้ feature ทั้งสี่ค่า")
    X = np.array([[-.4, .3], [.45, .2], [-.15, -.5], [.75, -.5], [-.6, -.7],
                  [2, 1.6], [-2, 1.5], [1.8, -1.6], [-2, -1.8]])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0])
    query = np.zeros((1, 2))
    model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    distances, indices = model.kneighbors(query)
    assert np.bincount(y[indices[0]], minlength=2).tolist() == [3, 2]
    assert model.predict(query)[0] == 0
    ax = fig.add_axes([.08, .235, .46, .53])
    plot_style(ax, "feature 1 หลังปรับสเกล", "feature 2 หลังปรับสเกล")
    ax.set_aspect("equal", adjustable="box")
    for cls, color, marker, name in [(0, BLUE, "o", "versicolor"), (1, ORANGE, "s", "virginica")]:
        a = X[y == cls]
        ax.scatter(a[:, 0], a[:, 1], s=95, c=color, marker=marker, label=name, zorder=3)
    for rank, idx in enumerate(indices[0], 1):
        ax.plot([0, X[idx, 0]], [0, X[idx, 1]], color=MUTED, alpha=.65, linewidth=1)
        ax.annotate(str(rank), X[idx], xytext=(7, 8), textcoords="offset points",
                    color=NAVY, fontsize=13, fontweight="bold")
    ax.add_patch(Circle((0, 0), distances[0, -1]+.04, fill=False,
                        color=PURPLE, linewidth=2, linestyle="--"))
    ax.scatter([0], [0], c=PURPLE, s=280, marker="*", label="ดอกใหม่", zorder=4)
    ax.set(xlim=(-2.6, 2.6), ylim=(-2.3, 2.3))
    ax.legend(loc="upper left", fontsize=12, framealpha=.95)
    box(p, .60, .63, .34, .12, "1  วัดระยะ → เลือก 5 จุดใกล้สุด", size=18)
    box(p, .60, .45, .34, .13, "2  นับเสียงเท่ากันทุกจุด\nversicolor 3  ·  virginica 2", size=19)
    box(p, .60, .26, .34, .13, "3  ทำนาย: versicolor", color="#F0EAF8", size=22, ink=PURPLE)
    arrow(p, (.77, .63), (.77, .58))
    arrow(p, (.77, .45), (.77, .39))
    footer(p, "ดาวคือดอกใหม่ · ตัวเลข 1–5 เรียงเพื่อนบ้านตามระยะ · จุดที่ไกลกว่านี้ไม่ร่วมโหวต",
           "fit เก็บตัวอย่างพร้อมคำตอบ · predict หาเพื่อนบ้าน · k คือจำนวนเพื่อนบ้าน ไม่ใช่จำนวนคลาส")
    save(fig, p, "03-first-classifier-knn")


def svm():
    fig, p = page("SVM · แนวแบ่งคลาส และ support vectors",
                  "ตัวอย่างจำลองชุดเดียวกัน · fit หลังปรับสเกล · ตรึง C=1 เพื่อเปรียบเทียบ kernel")
    X, y = make_moons(n_samples=110, noise=.13, random_state=31)
    X = StandardScaler().fit_transform(X)
    xx, yy = np.meshgrid(np.linspace(-2.3, 2.3, 240), np.linspace(-2.4, 2.4, 240))
    grid = np.c_[xx.ravel(), yy.ravel()]
    for pos, kernel, title in [(.075, "linear", "Linear · แนวแบ่งเส้นตรง"),
                                (.555, "rbf", "RBF · แนวแบ่งโค้ง")]:
        ax = fig.add_axes([pos, .25, .37, .49])
        model = SVC(kernel=kernel, C=1, gamma="scale").fit(X, y)
        score = model.decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, score, levels=[-100, 0, 100], colors=["#DCE9FF", "#FFE6CA"], alpha=.7)
        ax.contour(xx, yy, score, levels=[0], colors=[NAVY], linewidths=2)
        if kernel == "linear":
            ax.contour(xx, yy, score, levels=[-1, 1], colors=[MUTED], linestyles="--", linewidths=1.5)
        for cls, c, marker in [(0, BLUE, "o"), (1, ORANGE, "s")]:
            a = X[y == cls]
            ax.scatter(a[:, 0], a[:, 1], c=c, s=34, marker=marker, zorder=3)
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=100, facecolors="none",
                   edgecolors=NAVY, linewidths=1, zorder=4)
        plot_style(ax)
        ax.set(xlim=(-2.3, 2.3), ylim=(-2.4, 2.4), title=title)
    footer(p, "เส้นทึบ = แนวแบ่ง · เส้นประฝั่ง Linear = ขอบ margin · วงแหวน = support vectors",
           "RBF ใช้ความคล้ายตามระยะเพื่อสร้างขอบเขตโค้ง · ภาพนี้อธิบายหลักการ ไม่ใช่ผล test ในบท")
    save(fig, p, "08-support-vector-machines-model")


def tree():
    fig, p = page("Decision Tree · เดินตามเงื่อนไขจนถึงใบ",
                  "กฎสมมติเพื่ออธิบาย · ข้อมูลใหม่: feature_3 = 0.8 และ feature_1 = 0.7")
    box(p, .35, .65, .30, .12, "feature_3 ≤ 0.5 ?", size=24)
    box(p, .12, .40, .20, .12, "ใบ: ทายคลาส 0", size=21)
    box(p, .56, .40, .30, .12, "feature_1 ≤ 1.0 ?", color="#FFF0DF", size=24)
    box(p, .43, .19, .22, .11, "ใบ: ทายคลาส 1", color="#FFF0DF", size=22, ink=ORANGE)
    box(p, .74, .19, .22, .11, "ใบ: ทายคลาส 0", size=21)
    for start, end, c in [((.43,.65),(.22,.52),MUTED), ((.57,.65),(.71,.52),ORANGE),
                          ((.65,.40),(.54,.30),ORANGE), ((.77,.40),(.85,.30),MUTED)]:
        arrow(p, start, end, c, 3)
    label(p, .29, .585, "จริง", 18, MUTED)
    label(p, .66, .585, "ไม่จริง: 0.8 > 0.5", 17, ORANGE)
    label(p, .45, .355, "จริง: 0.7 ≤ 1.0", 17, ORANGE)
    label(p, .86, .355, "ไม่จริง", 18, MUTED)
    assert .8 > .5 and .7 <= 1
    footer(p, "เส้นสีส้มคือเส้นทางของข้อมูลแถวนี้ · ทุกแถวเริ่มที่ราก แล้วไปจบที่ใบหนึ่งใบ",
           "ตอน fit โมเดลเรียน feature และค่าตัดจาก train · ตอน predict ใช้กฎที่เรียนไว้แล้ว")
    save(fig, p, "09-trees-and-forests-tree")


def forest():
    fig, p = page("Random Forest · รวมคำตอบจากต้นไม้หลายต้น",
                  "ภาพย่อด้วยสองต้น · RandomForestClassifier ในโค้ดใช้ 120 ต้น")
    box(p, .35, .71, .30, .10, "ข้อมูล train ชุดเดียวกัน", size=23)
    for x, name, prob in [(.08, "ต้นที่ 1", .8), (.58, "ต้นที่ 2", .4)]:
        arrow(p, (.5, .71), (x+.17, .61))
        box(p, x, .52, .34, .09, "สุ่มแถวแบบคืนกลับ", color="white", size=21)
        arrow(p, (x+.17, .52), (x+.17, .45))
        box(p, x, .30, .34, .15, name+"\nสุ่ม feature ที่พิจารณาในแต่ละจุดแบ่ง", size=17)
        label(p, x+.17, .245, f"แถวใหม่เดียวกัน → P(คลาส 1) = {prob}", 18, ha="center")
        arrow(p, (x+.17, .22), (.5, .17))
    box(p, .20, .115, .60, .07, "เฉลี่ย (0.8 + 0.4) / 2 = 0.6 → ทายคลาส 1",
        color="#FFF0DF", size=22, ink=ORANGE)
    assert np.isclose(np.mean([.8,.4]), .6)
    label(p, .05, .045, "scikit-learn เฉลี่ย probability ของต้นไม้ · ไม่ใช่นับแค่คลาสที่แต่ละต้นเลือก",
          20, BLUE, True)
    save(fig, p, "09-trees-and-forests-ensemble")


def pca():
    fig, p = page("PCA · เปลี่ยนแกน แล้วเก็บพิกัดให้น้อยลง",
                  "ตัวอย่างจำลองจาก 2 feature เหลือ 1 component · ไม่ได้ใช้ label กำหนดแกน")
    rng = np.random.default_rng(42)
    X = rng.multivariate_normal([0, 0], [[2.0, 1.7],[1.7, 2.0]], size=22)
    model = PCA(n_components=1).fit(X)
    z = model.transform(X)
    projected = model.inverse_transform(z)
    assert z.shape == (22,1)
    a = fig.add_axes([.08,.26,.38,.48])
    plot_style(a)
    a.scatter(X[:,0],X[:,1], c=BLUE, s=50, zorder=3)
    for original, target in zip(X, projected):
        a.plot([original[0], target[0]],[original[1], target[1]],c=MUTED,alpha=.45,lw=1)
    pc = model.components_[0]
    line = model.mean_ + np.array([[-4],[4]]) * pc
    a.plot(line[:,0],line[:,1],c=ORANGE,lw=2.5,label="แกน PC1")
    a.scatter(projected[:,0],projected[:,1],s=27,c=ORANGE,zorder=4)
    a.set(xlim=(-3.5,3.5),ylim=(-3.5,3.5),title="เดิม: จุดหนึ่งมีพิกัดสองค่า")
    a.legend(fontsize=13)
    a.set_aspect("equal", adjustable="box")
    arrow(p,(.48,.50),(.55,.50),BLUE,3)
    b=fig.add_axes([.60,.35,.33,.25])
    plot_style(b,"พิกัดบน PC1","")
    b.axhline(0,color=ORANGE,lw=2)
    b.scatter(z[:,0],np.zeros(len(z)),s=45,c=ORANGE,zorder=3)
    b.set(ylim=(-.5,.5),yticks=[],title="ใหม่: เหลือพิกัดค่าเดียว")
    box(p,.60,.69,.33,.10,"ฉายจุดลงแกนที่เก็บ variance มาก",size=17)
    footer(p,"เส้นเชื่อมแสดงส่วนที่หายไปเมื่อเหลือแกนเดียว · PCA ผสม feature ไม่ใช่เลือกคอลัมน์เดิม",
           "ในบทใช้ 64 feature และเก็บ variance 90% · ตัวเลขนี้ไม่ใช่ accuracy 90%")
    save(fig,p,"11-principal-components-projection")


def multiclass():
    fig,p=page("PCA → Logistic Regression · ลดมิติแล้วทายเลข",
               "ภาพรวม Pipeline ในบท 11 · probability ด้านล่างเป็นตัวอย่างสมมติ")
    steps=[("64 pixel","ภาพ 8×8"),("StandardScaler","ใช้สถิติจาก train"),
           ("PCA","เก็บ variance 90%"),("Logistic Regression","คะแนน → softmax")]
    for i,(title,desc) in enumerate(steps):
        x=.05+i*.24
        box(p,x,.65,.20,.14,title+"\n"+desc,size=17)
        if i<3: arrow(p,(x+.20,.72),(x+.24,.72),BLUE)
    prob=np.array([.06,.05,.08,.10,.07,.09,.06,.30,.11,.08])
    assert np.isclose(prob.sum(),1) and prob.argmax()==7
    ax=fig.add_axes([.09,.26,.54,.29])
    ax.bar(np.arange(10),prob,color=[ORANGE if i==7 else BLUE for i in range(10)])
    plot_style(ax,"คลาสตัวเลข","probability")
    ax.set(xticks=np.arange(10),ylim=(0,.35))
    ax.annotate("0.30",(7,.30),xytext=(0,8),textcoords="offset points",ha="center",color=ORANGE)
    box(p,.70,.31,.25,.22,"เลือกค่าสูงสุด\nทายเลข 7\nแม้ 0.30 < 0.5",color="#FFF0DF",size=22,ink=ORANGE)
    footer(p,"PCA เรียนแกนจาก X · Logistic Regression เรียนจากพิกัดใหม่พร้อม y",
           "สิบคลาส → probability สิบค่ารวมเป็น 1 → เลือกคลาสสูงสุด ไม่ใช้เกณฑ์ 0.5 แบบสองคลาส")
    save(fig,p,"11-principal-components-classifier")


def tsne():
    fig,p=page("t-SNE · จัดตำแหน่งใหม่จากความสัมพันธ์ของเพื่อนบ้าน",
               "แผนภาพแนวคิด · ตำแหน่ง A–F วาดเพื่ออธิบาย ไม่ใช่ผลรัน t-SNE")
    box(p,.055,.69,.37,.10,"ภาพ 8×8 → 64 feature ต่อภาพ",size=21)
    box(p,.575,.69,.37,.10,"พิกัดใหม่ 2 ค่า → จุดบนกราฟ",size=21)
    arrow(p,(.44,.74),(.56,.74),BLUE,3)
    left=np.array([[.15,.38],[.25,.56],[.36,.40],[.64,.39],[.76,.57],[.87,.37]])
    right=np.array([[.15,.45],[.24,.58],[.33,.42],[.70,.39],[.78,.54],[.89,.42]])
    for pos,points in [(.07,left),(.57,right)]:
        ax=fig.add_axes([pos,.22,.36,.42])
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
        for group in [[0,1,2],[3,4,5]]:
            for i,j in [(group[0],group[1]),(group[1],group[2]),(group[0],group[2])]:
                ax.plot(points[[i,j],0],points[[i,j],1],c=MUTED,alpha=.5)
        for i,(x,y) in enumerate(points):
            ax.scatter(x,y,s=550,c=BLUE if i<3 else ORANGE,zorder=3)
            ax.text(x,y,chr(65+i),ha="center",va="center",color="white",fontweight="bold",zorder=4)
    label(p,.25,.28,"A, B, C คล้ายกัน · D, E, F คล้ายกัน",16,ha="center")
    label(p,.75,.28,"พยายามรักษาความเป็นเพื่อนบ้าน",17,ha="center")
    label(p,.25,.61,"ความสัมพันธ์ในข้อมูลเดิม",19,bold=True,ha="center")
    label(p,.75,.61,"ตำแหน่งใหม่ในภาพสองมิติ",19,bold=True,ha="center")
    footer(p,"สีใช้ติดตามจุดในภาพนี้ · t-SNE เรียนจาก X โดยไม่ใช้ y และไม่ได้ทำนายคลาส",
           "อย่าตีความระยะห่างระหว่างเกาะเป็นระยะจริง · ภาพแยกชัดไม่ได้ยืนยัน accuracy")
    save(fig,p,"12-manifold-learning-neighbors")


def kmeans():
    fig,p=page("K-means · จัดเข้ากลุ่ม แล้วขยับศูนย์กลาง",
               "ตัวอย่างสมมติ k=2 · ใช้ X อย่างเดียว ไม่มี label ให้คำตอบ")
    X=np.array([[0,0],[0,2],[2,0],[6,4],[6,6],[8,4]],dtype=float)
    old=np.array([[0,3],[8,6]],dtype=float)
    labels=((X[:,None,:]-old[None,:,:])**2).sum(axis=2).argmin(axis=1)
    new=np.vstack([X[labels==i].mean(axis=0) for i in range(2)])
    assert labels.tolist()==[0,0,0,1,1,1]
    assert np.allclose(new,[[2/3,2/3],[20/3,14/3]])
    inertia_old=((X-old[labels])**2).sum()
    inertia_new=((X-new[labels])**2).sum()
    assert inertia_new < inertia_old
    for pos,update,title in [(.07,False,"1  เลือกศูนย์กลางที่ใกล้ที่สุด"),
                              (.56,True,"2  ย้ายไปค่าเฉลี่ยของสมาชิก")]:
        ax=fig.add_axes([pos,.27,.38,.47])
        plot_style(ax)
        for i,c in enumerate([BLUE,ORANGE]):
            a=X[labels==i]
            ax.scatter(a[:,0],a[:,1],c=c,s=65,marker="o" if i==0 else "s",zorder=3)
            for x in a:
                center=new[i] if update else old[i]
                ax.plot([x[0],center[0]],[x[1],center[1]],c=c,alpha=.3)
            ax.scatter(*old[i],s=250,c=c,marker="X",alpha=.3 if update else 1,zorder=4)
            if update:
                ax.scatter(*new[i],s=260,c=c,marker="X",edgecolors=NAVY,zorder=4)
                arrow(ax,old[i],new[i],c)
        ax.set(xlim=(-1,9),ylim=(-1,7),title=title)
    footer(p,"เครื่องหมาย X คือ centroid · ขวา: สีจางคือตำแหน่งเดิม ลูกศรชี้ไปค่าเฉลี่ยใหม่",
           "ทำสองขั้นตอนนี้ซ้ำ · k ของ K-means คือจำนวนกลุ่ม ส่วน k ของ KNN คือจำนวนเพื่อนบ้าน")
    save(fig,p,"13-kmeans-clustering-steps")


def density():
    fig,p=page("GMM และ KDE · สร้างเส้นความหนาแน่นอย่างไร",
               "ตัวอย่างหนึ่งมิติสมมติ · เส้นสีเข้มคือผลรวมของเส้นย่อย ไม่ใช่คำทำนายคลาส")
    x=np.linspace(-6,6,1200)
    pdf=lambda mu,sigma: np.exp(-.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
    weights=np.array([.6,.4])
    components=np.vstack([weights[0]*pdf(-1.6,.75),weights[1]*pdf(1.7,1.0)])
    ax=fig.add_axes([.075,.28,.38,.45])
    plot_style(ax,"ค่า x","density")
    for v,c,n in zip(components,[BLUE,ORANGE],["0.6 × Gaussian 1","0.4 × Gaussian 2"]):
        ax.plot(x,v,c=c,ls="--",lw=2,label=n)
    ax.plot(x,components.sum(axis=0),c=NAVY,lw=3,label="รวม GMM")
    ax.set(title="GMM · Gaussian สององค์ประกอบ",ylim=(0,.4))
    ax.legend(fontsize=12,loc="upper right")
    data=np.array([-2.2,-1.6,-.9,1.0,1.8,2.4])
    h=.6
    kernels=np.vstack([pdf(v,h)/len(data) for v in data])
    kde=KernelDensity(bandwidth=h).fit(data[:,None])
    assert np.allclose(kernels.sum(axis=0),np.exp(kde.score_samples(x[:,None])))
    assert np.isclose(np.trapezoid(components.sum(axis=0),x),1,atol=1e-4)
    assert np.isclose(np.trapezoid(kernels.sum(axis=0),x),1,atol=1e-4)
    ax=fig.add_axes([.565,.28,.38,.45])
    plot_style(ax,"ค่า x","density")
    for row in kernels: ax.plot(x,row,c=ORANGE,alpha=.65,ls="--",lw=1.5)
    ax.plot(x,kernels.sum(axis=0),c=NAVY,lw=3,label="รวม KDE")
    ax.scatter(data,np.zeros(len(data)),marker="|",s=150,c=BLUE,label="ข้อมูล 6 จุด")
    ax.set(title="KDE · kernel รอบข้อมูลทุกจุด",ylim=(-.015,.4))
    ax.legend(fontsize=12,loc="upper right")
    label(p,.265,.20,"น้ำหนักรวมเป็น 1 · เรียนจุดกลางและความกว้าง",16,ha="center")
    label(p,.755,.20,"เส้นย่อยแต่ละเส้นหารด้วย 6 · bandwidth = 0.6",16,ha="center")
    footer(p,"GMM เรียนองค์ประกอบผ่าน EM · KDE ใช้ทุกจุดเป็นศูนย์กลางของ kernel",
           "ความสูงคือ density · พื้นที่ใต้เส้นในช่วงที่สนใจจึงเป็น probability · พื้นที่รวมเท่ากับ 1")
    save(fig,p,"14-mixtures-and-density-models")


def gradient():
    fig,p=page("Linear Regression · Gradient Descent ปรับเส้นอย่างไร",
               "ตัวอย่างหนึ่งก้าวจาก README · x=1, y=5, learning rate=0.1 · ไม่ใช่ผลฝึกทั้งชุด")
    x0,y0,lr=1.,5.,.1
    w,b=0.,0.
    error=w*x0+b-y0
    dw,db=2*error*x0,2*error
    w2,b2=w-lr*dw,b-lr*db
    assert (w2,b2)==(1.,1.)
    assert (w2*x0+b2-y0)**2==9
    ax=fig.add_axes([.075,.27,.40,.48])
    xx=np.linspace(0,2,30)
    ax.plot(xx,w*xx+b,c=MUTED,lw=2,ls="--",label="ก่อน: w=0, b=0")
    ax.plot(xx,w2*xx+b2,c=BLUE,lw=2.5,label="หลัง: w=1, b=1")
    ax.scatter([1],[5],c=ORANGE,s=110,zorder=4,label="ค่าจริง (1, 5)")
    ax.plot([1,1],[2,5],c=ORANGE,lw=2)
    ax.scatter([1],[2],c=BLUE,s=70,zorder=4)
    ax.annotate("ทายใหม่ = 2",(1,2),xytext=(12,-18),textcoords="offset points",color=BLUE)
    plot_style(ax,"x","y")
    ax.set(xlim=(0,2),ylim=(-.4,6),title="โมเดลเดิม: y-hat = wx + b")
    ax.legend(fontsize=12,loc="upper left")
    box(p,.55,.61,.40,.15,"1  error = 0 − 5 = −5\n2  dw = −10  และ  db = −10",size=22)
    box(p,.55,.39,.40,.15,"3  w ใหม่ = 0 − 0.1 × (−10) = 1\n    b ใหม่ = 0 − 0.1 × (−10) = 1",size=19)
    box(p,.55,.21,.40,.11,"squared error: 25 → 9",color="#FFF0DF",size=24,ink=ORANGE)
    arrow(p,(.75,.61),(.75,.54),BLUE)
    arrow(p,(.75,.39),(.75,.32),BLUE)
    footer(p,"คำนวณ dw และ db จากค่ารอบเดิมให้ครบก่อนอัปเดต · Gradient Descent คือวิธีฝึกโมเดล",
           "จุดเดียวนี้ยังบอก w กับ b ที่เหมาะสมแยกกันไม่ได้ · โค้ดจริงใช้ทุกตัวอย่างใน train")
    save(fig,p,"15-gradient-descent-update")


def groups():
    fig,p=page("KNN และ Linear Regression · ใช้กับโจทย์ต่างกัน",
               "ภาพรวมบท 17 · การแบ่งข้อมูลเปลี่ยนวิธีประเมิน แต่ไม่เปลี่ยนหลักการของโมเดล")
    label(p,.05,.77,"ทำนายคนใหม่",24,BLUE,True)
    box(p,.05,.60,.25,.11,"กันคน test ออกทั้งกลุ่ม",size=20)
    box(p,.38,.60,.25,.11,"KNN: k = 3\nโหวตจากเพื่อนบ้านใน train",size=17)
    box(p,.71,.60,.24,.11,"คืนคลาส 0 หรือ 1",size=21)
    arrow(p,(.30,.655),(.38,.655),BLUE)
    arrow(p,(.63,.655),(.71,.655),BLUE)
    label(p,.05,.52,"groups ใช้แบ่งข้อมูล · ไม่ส่งรหัสคนเป็น feature ให้ KNN",18,MUTED)
    label(p,.05,.43,"ทำนายค่าถัดไป",24,ORANGE,True)
    box(p,.05,.25,.25,.11,"ค่าจริงก่อนหน้า = 12",color="#FFF0DF",size=21)
    box(p,.38,.25,.25,.11,"Linear Regression\nทาย b + w × 12",color="#FFF0DF",size=20)
    box(p,.71,.25,.24,.11,"คืนคำตอบตัวเลข",color="#FFF0DF",size=21)
    arrow(p,(.30,.305),(.38,.305),ORANGE)
    arrow(p,(.63,.305),(.71,.305),ORANGE)
    label(p,.05,.18,"เทียบกับ persistence baseline: ทาย 12 ตามค่าล่าสุด โดยไม่เรียน w/b",18,MUTED)
    footer(p,"KNN เรียนจากคนใน train · Linear Regression เรียน w/b จากช่วงอดีต",
           "กรณีเวลาเป็น one-step-ahead: รู้ค่าจริงก่อนหน้าของแต่ละจุดแล้ว ไม่ใช่ทายหลายก้าวล่วงหน้า")
    save(fig,p,"17-time-and-groups-models")


RENDERERS = {"knn":knn,"svm":svm,"tree":tree,"forest":forest,"pca":pca,
             "multiclass":multiclass,"tsne":tsne,"kmeans":kmeans,"density":density,
             "gradient":gradient,"groups":groups}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures",nargs="*",metavar="FIGURE")
    args=parser.parse_args()
    unknown=set(args.figures)-set(RENDERERS)
    if unknown:
        parser.error("Unknown figures: " + ", ".join(sorted(unknown)))
    configure()
    for name in args.figures or RENDERERS:
        RENDERERS[name]()
