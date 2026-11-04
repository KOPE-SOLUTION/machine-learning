"""Original EP07–EP17 workflow illustrations; exact Thai text over vector shapes."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from thai_figure_text import enable_thai_shaping
from render_ep06_diagrams import thai_font

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/figures"
BLUE, ORANGE, NAVY, MUTED, GREEN = "#1558D6", "#A95000", "#142A43", "#4C6078", "#19735E"
PALE, WARM, BG = "#EDF4FF", "#FFF5EA", "#F6F8FC"
OWNERS = []
RESULTS = []

def text(ax, x, y, value, size=19, color=NAVY, weight="normal", ha="left"):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", linespacing=1.45)

def rect(ax, x, y, w, h, fill="white", edge="#D9E2EF"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.006,rounding_size=0.012",
                 facecolor=fill,edgecolor=edge,linewidth=1.5,zorder=0))

def box(ax,x,y,w,h,title,body="",accent=BLUE,fill="white",ts=20,bs=18):
    rect(ax,x,y,w,h,fill)
    a=text(ax,x+w/2,y+h-.045,title,ts,accent,"bold","center")
    OWNERS.append((a,(x+.009,y+.009,x+w-.009,y+h-.009)))
    if body:
        a=text(ax,x+w/2,y+h*.36,body,bs,ha="center")
        OWNERS.append((a,(x+.009,y+.009,x+w-.009,y+h-.009)))

def arrow(ax,start,end,color=BLUE,dashed=False,rad=0):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=18,
        linewidth=2,color=color,linestyle=(0,(4,3)) if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",zorder=1))

def canvas(title,subtitle=""):
    OWNERS.clear()
    fig=plt.figure(figsize=(16,9),facecolor=BG)
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    text(ax,.04,.944,title,28,BLUE,"bold")
    if subtitle: text(ax,.04,.886,subtitle,17,MUTED)
    return fig,ax

def flow(ax,nodes,y=.59,h=.20,color=BLUE):
    n=len(nodes); gap=.035; w=(.92-gap*(n-1))/n
    centers=[]
    for i,(title,body) in enumerate(nodes):
        x=.04+i*(w+gap)
        box(ax,x,y,w,h,title,body,accent=color,fill=PALE if color==BLUE else WARM,ts=19,bs=18)
        centers.append(x+w/2)
        if i<n-1: arrow(ax,(x+w+.007,y+h/2),(x+w+gap-.008,y+h/2),color)
    return centers

def finish(fig,slug):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for artist,bounds in OWNERS:
        x0,y0,x1,y1=artist.axes.transData.transform([[bounds[0],bounds[1]],[bounds[2],bounds[3]]]).ravel()
        b=artist.get_window_extent(renderer)
        while (b.x0<x0 or b.x1>x1 or b.y0<y0 or b.y1>y1) and artist.get_fontsize()>15:
            artist.set_fontsize(artist.get_fontsize()-1)
            fig.canvas.draw(); b=artist.get_window_extent(fig.canvas.get_renderer())
        assert b.x0>=x0-1 and b.x1<=x1+1 and b.y0>=y0-1 and b.y1<=y1+1,(slug,artist.get_text(),bounds)
    fig.canvas.draw()
    for ax in fig.axes:
        for artist in ax.texts:
            b=artist.get_window_extent(fig.canvas.get_renderer())
            assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(slug,artist.get_text())
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f"{slug}.png",dpi=150)
    svg=OUT/f"{slug}.svg"
    fig.savefig(svg,metadata={"Date":None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8",newline="\n")
    RESULTS.append({"slug":slug,"png_size":[2400,1350],"text_bounds":"passed"})
    plt.close(fig)
    print(slug)

def dual(slug,title,train,test,links,notes):
    fig,ax=canvas(title,"แบ่ง train/test ก่อน • ตอนทำนายใช้ค่าที่เก็บไว้จากชุดฝึก")
    text(ax,.04,.829,"ตอนฝึก",22,BLUE,"bold")
    centers=flow(ax,train,.62,.17)
    text(ax,.04,.500,"ตอนทำนาย",22,ORANGE,"bold")
    flow(ax,test,.29,.17,ORANGE)
    for index,caption in links:
        x=centers[index]
        arrow(ax,(x,.607),(x,.475),MUTED,True)
        text(ax,x+.018 if index<3 else x-.018,.547,caption,15,MUTED,
             ha="left" if index<3 else "right")
    for i,(title,body) in enumerate(notes):
        box(ax,.04+i*.475,.06,.445,.15,title,body,ts=18,bs=17)
    finish(fig,slug)

def ep07():
    dual("07-naive-bayes-text-workflow","ข้อความ → จำนวนคำ → Naive Bayes",
         [("X_train","18 ข้อความ"),("CountVectorizer","fit → transform\nสร้าง vocabulary"),("จำนวนคำ","18 แถว\n49 คอลัมน์"),("MultinomialNB","fit พร้อม y_train\nเรียน prior และโอกาสของคำ")],
         [("X_test / ข้อความใหม่","ข้อความดิบ"),("CountVectorizer","transform\nใช้ vocabulary เดิม"),("จำนวนคำ","49 คอลัมน์เดิม"),("MultinomialNB","predict\nเลือก sport หรือ tech")],
         [(1,"vocabulary เดิม"),(3,"ค่าของโมเดลเดิม")],
         [("prior: โอกาสของคลาสก่อนอ่านข้อความ","sport = tech = 9/18 = 0.5"),
          ("ตอนทำนาย","prior × หลักฐานจากคำ → เลือกคะแนนสูงสุด")])
    fig,ax=canvas("ข้อความภาษาไทย · จะนับอะไรเป็น feature","เลือกนับคำที่ตัดแล้ว หรือนับกลุ่มตัวอักษรที่ติดกัน")
    box(ax,.04,.16,.44,.65,"1. ตัดคำแล้วนับคำ",accent=BLUE)
    box(ax,.52,.16,.44,.65,"2. Character n-grams",accent=ORANGE)
    text(ax,.26,.67,"ฉันชอบฟุตบอล",29,BLUE,"bold","center")
    arrow(ax,(.26,.62),(.26,.55))
    for x,word in zip((.075,.21,.345),("ฉัน","ชอบ","ฟุตบอล")):
        box(ax,x,.39,.10,.13,word,"1 ครั้ง",ts=22,bs=17)
    text(ax,.26,.28,"แต่ละคำเป็นหนึ่ง feature",20,ha="center")
    text(ax,.26,.21,"ใช้เครื่องมือตัดคำ เช่น PyThaiNLP",16,MUTED,ha="center")
    text(ax,.74,.67,"แมว",31,ORANGE,"bold","center")
    text(ax,.74,.585,"เลื่อนกลุ่มครั้งละ 1 ตัวอักษร",18,ha="center")
    box(ax,.56,.41,.36,.12,"n = 2","แม   |   มว",accent=ORANGE,ts=18,bs=24)
    box(ax,.56,.24,.36,.12,"n = 3","แมว",accent=ORANGE,ts=18,bs=24)
    text(ax,.50,.075,"ทั้งสองวิธีนำจำนวนครั้งมาเป็น features • กลุ่มตัวอักษรไม่จำเป็นต้องเป็นคำสมบูรณ์",18,MUTED,ha="center")
    finish(fig,"07-naive-bayes-text-thai-features")

def ep08():
    fig,ax=canvas("Cross-validation · เลือก kernel ก่อนเปิด test","ข้อมูล 400 แถว → train 300 แถว / test 100 แถว")
    text(ax,.04,.813,"train 300 แถว → 5 fold",21,BLUE,"bold")
    for r in range(5):
        y=.707-r*.073
        text(ax,.045,y+.021,f"รอบ {r+1}",17)
        for c in range(5):
            x=.14+c*.081
            ax.add_patch(Rectangle((x,y),.072,.045,facecolor=WARM if c==r else PALE,edgecolor=ORANGE if c==r else BLUE))
            text(ax,x+.036,y+.022,"60",16,ORANGE if c==r else BLUE,ha="center")
    text(ax,.14,.352,"น้ำเงิน: ฝึก 240     ส้ม: validation 60",16,MUTED)
    box(ax,.61,.55,.35,.25,"แต่ละรอบฝึก Pipeline ใหม่","240 แถว: scaler → SVM\n60 แถว: transform → predict",ts=21,bs=20)
    box(ax,.61,.36,.35,.14,"test 100 แถว","เก็บไว้ประเมินท้ายสุด",accent=ORANGE,fill=WARM,ts=20,bs=18)
    flow(ax,[("เทียบ linear กับ RBF","CV accuracy เฉลี่ย 5 รอบ"),
             ("เลือก kernel แล้ว refit","ใช้ train ทั้ง 300 แถว"),
             ("ประเมิน test","accuracy บน 100 แถว")],.075,.20)
    finish(fig,"08-support-vector-machines-cross-validation")

def ep09():
    fig,ax=canvas("Permutation importance · ลองสลับคอลัมน์","ตัวอย่างคะแนน • ใช้โมเดลที่ฝึกไว้ตัวเดิมทั้งสองครั้ง")
    text(ax,.04,.817,"ก่อนสลับ",21,BLUE,"bold")
    flow(ax,[("test เดิม","คอลัมน์อยู่ตามแถวเดิม"),("โมเดลที่ฝึกแล้ว","predict โดยไม่ fit ซ้ำ"),("Accuracy","0.90")],.61,.17)
    text(ax,.04,.496,"หลังสลับ",21,ORANGE,"bold")
    flow(ax,[("สลับ 1 คอลัมน์","คอลัมน์อื่นและ y คงเดิม"),("โมเดลตัวเดิม","predict อีกครั้ง"),("Accuracy","0.85")],.29,.17,ORANGE)
    arrow(ax,(.19,.595),(.19,.475),MUTED,True)
    box(ax,.04,.06,.92,.15,"คะแนนลดลง = 0.90 − 0.85 = 0.05","ลดลง 5 จุดเปอร์เซ็นต์ → โมเดลพึ่งข้อมูลในคอลัมน์นี้",ts=24,bs=20)
    finish(fig,"09-trees-and-forests-permutation-importance")

def ep10():
    fig,ax=canvas("GridSearchCV · ลอง 6 ชุด แล้วเลือกด้วย CV","KNN: k = 3, 5, 9 × weights = uniform, distance")
    text(ax,.12,.802,"uniform",18,BLUE,"bold","center")
    text(ax,.34,.802,"distance",18,BLUE,"bold","center")
    for r,k in enumerate((3,5,9)):
        for c in range(2):
            box(ax,.04+c*.22,.637-r*.115,.19,.09,f"k = {k}",accent=BLUE,fill=PALE,ts=20)
    arrow(ax,(.455,.56),(.51,.56))
    box(ax,.525,.44,.195,.28,"5-fold CV","6 ชุด × 5 รอบ\n= 30 ครั้ง",ts=23,bs=22)
    arrow(ax,(.735,.58),(.77,.58))
    box(ax,.785,.44,.175,.28,"เลือกชุดค่า","accuracy เฉลี่ย\nสูงที่สุด",ts=21,bs=20)
    text(ax,.04,.328,"ทุก fold: scaler และ KNN fit จากส่วนฝึกของรอบนั้น",18,MUTED)
    flow(ax,[("ชุดค่าที่เลือก","ค่า k และ weights"),("refit","ฝึกบน train ทั้ง 120 แถว"),("test","ตรวจ 30 แถวท้ายสุด")],.075,.19)
    finish(fig,"10-tuning-and-learning-curves-grid-search")


def ep11():
    dual("11-principal-components-workflow","PCA ลดมิติ · Logistic Regression จำแนกภาพ",
         [("X_train","ภาพละ 64 features"),("StandardScaler","fit → transform\nปรับสเกล"),("PCA","fit → transform\nเก็บ variance > 90%"),("LogisticRegression","fit ด้วยพิกัดใหม่\nพร้อม y_train")],
         [("X_test / ภาพใหม่","ภาพละ 64 features"),("StandardScaler","transform\nใช้สถิติเดิม"),("PCA","transform\nใช้แกนเดิม"),("LogisticRegression","predict\nเลข 0–9")],
         [(1,"สถิติเดิม"),(2,"แกนเดิม"),(3,"โมเดลเดิม")],
         [("Variance","วัดการกระจายของข้อมูลที่ PCA เก็บได้"),
          ("Accuracy","เทียบเลขที่ทายกับเลขจริงบน test")])

def ep12():
    fig,ax=canvas("PCA กับ t-SNE · เปลี่ยนภาพเป็นพิกัดบนกราฟ","ใช้ X สร้างพิกัด • ใช้ y กำหนดสีของจุด")
    box(ax,.04,.42,.22,.30,"X: 600 ภาพ","ภาพละ 64 features\nขนาด (600, 64)",ts=23,bs=20)
    box(ax,.34,.65,.25,.17,"PCA","fit_transform(X)",ts=25,bs=21)
    box(ax,.34,.37,.25,.17,"t-SNE","fit_transform(X)",accent=ORANGE,ts=25,bs=21)
    arrow(ax,(.272,.65),(.328,.735))
    arrow(ax,(.272,.49),(.328,.455),ORANGE)
    box(ax,.67,.65,.29,.17,"กราฟ PCA","600 จุด × 2 พิกัด\nสีจาก y จริง",ts=23,bs=19)
    box(ax,.67,.37,.29,.17,"กราฟ t-SNE","600 จุด × 2 พิกัด\nสีจาก y จริง",accent=ORANGE,ts=23,bs=19)
    arrow(ax,(.604,.735),(.658,.735))
    arrow(ax,(.604,.455),(.658,.455),ORANGE)
    box(ax,.04,.08,.44,.19,"ผลลัพธ์เป็นตำแหน่งจุด","ไม่ใช่คำทำนายเลข 0–9",ts=22,bs=20)
    box(ax,.52,.08,.44,.19,"Trustworthiness","ยิ่งใกล้ 1 → เพื่อนบ้านใหม่\nสอดคล้องกับข้อมูลเดิมมาก",ts=22,bs=19)
    finish(fig,"12-manifold-learning-workflow")

def ep13():
    fig,ax=canvas("K-means · เลือกกลุ่มก่อนตรวจด้วย label","X มี 450 จุด • truth เก็บไว้ตรวจ ARI หลังเลือก k")
    flow(ax,[("X","450 จุด × 2 features"),
             ("StandardScaler","fit_transform(X)"),
             ("ลอง K-means","k = 2, 3, 4, 5, 6"),
             ("เลือก k","silhouette สูงที่สุด")],.61,.20)
    box(ax,.04,.34,.44,.17,"Inertia","ดูความกระชับรอบ centroid",ts=22,bs=20)
    box(ax,.52,.34,.44,.17,"ARI หลังเลือก k","เทียบกลุ่มที่ได้กับ truth",accent=ORANGE,fill=WARM,ts=22,bs=20)
    arrow(ax,(.855,.596),(.855,.522),ORANGE)
    text(ax,.04,.263,"เมื่อมีจุดใหม่",21,BLUE,"bold")
    flow(ax,[("จุดใหม่","ใช้ features แบบเดิม"),("scaler.transform","ใช้สถิติที่เก็บไว้"),("K-means.predict","เลือก centroid ที่ใกล้ที่สุด")],.055,.17)
    finish(fig,"13-kmeans-clustering-selection")

def ep14():
    fig,ax=canvas("GMM กับ KDE · เลือกวิธี แล้วตรวจบน test","ข้อมูล 600 แถว → train 450 แถว / test 150 แถว")
    text(ax,.04,.826,"GMM: เลือกจำนวนองค์ประกอบ",21,BLUE,"bold")
    flow(ax,[("fit ด้วย train","จำนวน component 1–4"),
             ("เปรียบเทียบ BIC","บน train ชุดเดียวกัน"),
             ("เก็บ GMM ที่เลือก","BIC ต่ำที่สุด")],.62,.17)
    text(ax,.04,.507,"KDE: เลือก bandwidth",21,ORANGE,"bold")
    flow(ax,[("ลอง 4 ค่า","0.15, 0.3, 0.6, 1.0"),
             ("5-fold CV","เลือกคะแนน CV สูงสุด"),
             ("refit KDE","ใช้ train ทั้ง 450 แถว")],.30,.17,ORANGE)
    box(ax,.04,.06,.92,.15,"ประเมินทั้งสองโมเดลด้วย test 150 แถว","เปรียบเทียบ mean log density: ค่ายิ่งสูงยิ่งให้ความหนาแน่นกับจุด test มาก",ts=22,bs=18)
    finish(fig,"14-mixtures-and-density-selection")

    fig,ax=canvas("Density คือความสูง · Probability คือพื้นที่","ตัวอย่าง: กระจายสม่ำเสมอในช่วง 0 ถึง 0.5")
    plot=fig.add_axes([.09,.30,.45,.48])
    plot.plot([-.03,0,0,.5,.5,.54],[0,0,2,2,0,0],color=BLUE,lw=3)
    plot.fill_between([0,.1],[2,2],color="#F6B66B",alpha=.8)
    plot.set(xlim=(-.03,.55),ylim=(0,2.6),xticks=[0,.1,.5],yticks=[0,1,2],xlabel="x",ylabel="Density")
    plot.tick_params(labelsize=15); plot.xaxis.label.set_size(17);plot.yaxis.label.set_size(17)
    plot.spines[["top","right"]].set_visible(False)
    plot.annotate("20%",xy=(.05,.9),ha="center",fontsize=20,color=ORANGE)
    plot.annotate("2",xy=(.29,2.14),ha="center",fontsize=22,color=BLUE)
    box(ax,.64,.57,.32,.22,"ความสูง = 2","density มากกว่า 1 ได้",ts=24,bs=21)
    box(ax,.64,.29,.32,.22,"พื้นที่สีส้ม","0.1 × 2 = 0.2\nโอกาส = 20%",accent=ORANGE,fill=WARM,ts=24,bs=22)
    box(ax,.04,.065,.92,.15,"พื้นที่ทั้งหมด = 0.5 × 2 = 1","หา probability ของช่วงที่สนใจจากพื้นที่ใต้กราฟ",ts=24,bs=20)
    finish(fig,"14-mixtures-and-density-area")


def ep15():
    fig,ax=canvas("Gradient Descent · วนปรับ w และ b จาก train","เริ่ม w = 0, b = 0 • learning rate = 0.1")
    centers=flow(ax,[("1. ทำนายแล้วหา error","error = wx + b − y"),
                     ("2. คำนวณ gradient","dw = 2 × mean(error × x)\ndb = 2 × mean(error)"),
                     ("3. ปรับพารามิเตอร์","w ← w − 0.1 × dw\nb ← b − 0.1 × db")],.57,.23)
    ax.plot([centers[2],centers[2],centers[0],centers[0]],[.558,.426,.426,.535],color=BLUE,lw=2)
    arrow(ax,(centers[0],.515),(centers[0],.558))
    rect(ax,.31,.395,.38,.064,BG,BG)
    text(ax,.50,.475,"ทำซ้ำ 1,000 ก้าวด้วยชุดฝึก",20,BLUE,"bold","center")
    text(ax,.04,.331,"ประเมินด้วยข้อมูลที่กันไว้",21,ORANGE,"bold")
    flow(ax,[("x_test","ข้อมูลทดสอบ"),
             ("ใช้ w และ b สุดท้าย","ŷ = wx_test + b"),
             ("เทียบกับ y_test","ประเมินด้วย MAE")],.10,.19,ORANGE)
    finish(fig,"15-gradient-descent-training-loop")

def ep16():
    fig,ax=canvas("โครงงาน SVM · ฝึก ประเมิน บันทึก และใช้ซ้ำ","บันทึก Pipeline ทั้ง StandardScaler และ SVC ไว้ด้วยกัน")
    centers=flow(ax,[("1. ตรวจและแบ่งข้อมูล","64 pixel ต่อภาพ\ntrain 1,437 / test 360"),
                     ("2. เลือก Pipeline","4 คู่ C/gamma • CV 3 fold\nmacro F1 → เลือกแล้ว refit"),
                     ("3. ประเมิน test","accuracy / macro F1\nconfusion matrix")],.55,.25)
    ax.plot([centers[2],centers[2],centers[0]],[.535,.454,.454],color=BLUE,lw=2)
    arrow(ax,(centers[0],.454),(centers[0],.382))
    text(ax,.50,.494,"นำ Pipeline ที่เลือกไปบันทึก",18,MUTED,ha="center")
    flow(ax,[("4. บันทึก","joblib.dump\nmodel.joblib"),
             ("5. โหลดกลับมา","joblib.load\nได้ scaler + SVC เดิม"),
             ("6. รับ CSV แล้วทำนาย","ตรวจ input → predict\nได้เลข 0–9")],.12,.25,ORANGE)
    text(ax,.50,.045,"ตรวจว่าคำทำนายก่อนและหลังบันทึกตรงกัน",19,MUTED,ha="center")
    finish(fig,"16-digits-project-workflow")

def ep17():
    fig,ax=canvas("Group split · คนเดียวกันต้องอยู่ฝั่งเดียว","ตัวอย่าง: คน ก มีข้อมูล 5 แถว")
    box(ax,.04,.30,.44,.50,"สุ่มแถว",accent=ORANGE)
    box(ax,.52,.30,.44,.50,"แบ่งตามคน",accent=BLUE)
    text(ax,.16,.660,"train",20,BLUE,"bold","center")
    text(ax,.37,.660,"test",20,ORANGE,"bold","center")
    for i in range(3):
        box(ax,.075+i*.060,.52,.050,.085,f"ก{i+1}",accent=BLUE,fill=PALE,ts=17)
    for i in range(2):
        box(ax,.315+i*.060,.52,.050,.085,f"ก{i+4}",accent=ORANGE,fill=WARM,ts=17)
    text(ax,.26,.405,"คน ก อยู่ทั้ง train และ test",20,ORANGE,"bold","center")
    text(ax,.74,.660,"train: คนอื่น",20,BLUE,"bold","center")
    text(ax,.74,.570,"test: คน ก ทั้ง 5 แถว",19,ORANGE,"bold","center")
    for i in range(5):
        box(ax,.560+i*.073,.425,.057,.085,f"ก{i+1}",accent=ORANGE,fill=WARM,ts=17)
    text(ax,.74,.352,"คน ก อยู่ใน test ฝั่งเดียว",19,BLUE,ha="center")
    text(ax,.04,.235,"ข้อมูลในบท: 60 คน × 5 แถว = 300 แถว",20,weight="bold")
    box(ax,.04,.065,.44,.125,"train: 48 คน","240 แถว",fill=PALE,ts=21,bs=19)
    box(ax,.52,.065,.44,.125,"test: 12 คน","60 แถว",accent=ORANGE,fill=WARM,ts=21,bs=19)
    finish(fig,"17-time-and-groups-group-split")

    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit
    splits=list(TimeSeriesSplit(n_splits=4,gap=1).split(np.zeros((179,1))))
    assert [len(tr) for tr,va in splits]==[38,73,108,143]
    assert all(len(va)==35 for tr,va in splits)
    fig,ax=canvas("TimeSeriesSplit · ฝึกอดีต ตรวจช่วงถัดไป","4 รอบ • gap = 1 แถว • ชุดข้อมูล lag/target มี 179 คู่")
    text(ax,.18,.800,"train: อดีต",18,BLUE,"bold")
    text(ax,.45,.800,"gap: เว้น 1 แถว",18,ORANGE,"bold")
    text(ax,.71,.800,"validation: ช่วงถัดไป",18,GREEN,"bold")
    unit=.78/179
    for i,(tr,va) in enumerate(splits):
        y=.690-i*.090
        text(ax,.04,y+.026,f"รอบ {i+1}",18)
        ax.add_patch(Rectangle((.18,y),.78,.053,facecolor="#E9EEF5",edgecolor="none"))
        ax.add_patch(Rectangle((.18,y),len(tr)*unit,.053,facecolor=PALE,edgecolor=BLUE))
        ax.add_patch(Rectangle((.18+(tr[-1]+1)*unit,y),unit,.053,facecolor=ORANGE,edgecolor=ORANGE))
        ax.add_patch(Rectangle((.18+va[0]*unit,y),len(va)*unit,.053,facecolor="#E2F2ED",edgecolor=GREEN))
        text(ax,.18+len(tr)*unit/2,y+.026,f"train {len(tr)}",16,BLUE,ha="center")
        text(ax,.18+(va[0]+len(va)/2)*unit,y+.026,"ตรวจ 35",16,GREEN,ha="center")
    arrow(ax,(.18,.365),(.96,.365),MUTED)
    text(ax,.18,.337,"อดีต",16,MUTED)
    text(ax,.96,.337,"อนาคต",16,MUTED,ha="right")
    box(ax,.04,.075,.44,.19,"Lag → Target","10 → 12     12 → 11     11 → 15",ts=23,bs=20)
    box(ax,.52,.075,.44,.19,"One-step-ahead","ใช้ค่าจริงล่าสุด → ทายจุดถัดไป\nแล้วรอค่าจริงก่อนทายจุดต่อไป",ts=23,bs=18)
    finish(fig,"17-time-and-groups-time-split")

RENDERERS={7:ep07,8:ep08,9:ep09,10:ep10,11:ep11,12:ep12,13:ep13,14:ep14,15:ep15,16:ep16,17:ep17}

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lessons",type=int,nargs="+",default=list(RENDERERS))
    parser.add_argument("--output-dir",type=Path,default=OUT)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    OUT=args.output_dir
    enable_thai_shaping()
    plt.rcParams.update({"font.family":thai_font(),"mathtext.fontset":"dejavusans",
                         "svg.fonttype":"path","svg.hashsalt":"ml-learning-workflows"})
    for number in args.lessons:
        RENDERERS[number]()
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(RESULTS,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
