import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from wc_style import RED, LIME, NAVY, BERRY, GOLD, GREY, INK, draw_trophy, add_trophy

fig, ax = plt.subplots(figsize=(12, 6.9), dpi=200)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

# brand strip
hb = fig.add_axes([0,0.90,1,0.10]); hb.axis("off"); hb.set_xlim(0,1); hb.set_ylim(0,1)
for x,w,c in [(0,0.34,BERRY),(0.34,0.30,RED),(0.64,0.20,LIME),(0.84,0.16,NAVY)]:
    hb.add_patch(Rectangle((x,0),w,1,facecolor=c))

ax.text(0.2, 6.35, "How the model predicts the World Cup", fontsize=21, fontweight="bold", color=INK)
ax.text(0.2, 5.92, "From raw football data to a champion probability", fontsize=12.5, color=GREY)

stages = [
    ("1  DATA", ["Elo & FIFA rankings","Recent form & results","Goals for / against","Venue, H2H, rest days","Squad quality"], RED),
    ("2  FEATURES", ["Pre-game features only","No outcome leakage","Per-match modeling table"], BERRY),
    ("3  MODEL", ["Baseline -> XGBoost","Outputs win probability","Tuned on prob. quality"], NAVY),
    ("4  SIMULATE", ["Monte Carlo, thousands","of tournament runs","Count every outcome"], BERRY),
    ("5  RESULT", ["Group advancement %","Bracket odds","Champion probability"], "#3a7d00"),
]
n=len(stages); x0=0.35; w=2.05; gap=(12 - 2*x0 - n*w)/(n-1); y=1.9; h=2.95
centers=[]
for i,(title,items,accent) in enumerate(stages):
    x=x0+i*(w+gap)
    box=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.14",
                       linewidth=0, facecolor=accent, zorder=2)
    ax.add_patch(box)
    # accent top bar in lime for the result, gold elsewhere
    ax.add_patch(Rectangle((x+0.12, y+h-0.18), w-0.24, 0.08, facecolor=LIME if accent!="#3a7d00" else GOLD, zorder=3))
    ax.text(x+w/2, y+h-0.5, title, ha="center", va="center", color="white",
            fontsize=12.5, fontweight="bold")
    for j,it in enumerate(items):
        ax.text(x+w/2, y+h-1.02-j*0.44, it, ha="center", va="center",
                color="#f0f2f6", fontsize=8.6)
    centers.append((x+w, x))

for i in range(n-1):
    ar=FancyArrowPatch((centers[i][0]+0.03,y+h/2),(centers[i+1][1]-0.03,y+h/2),
                       arrowstyle="-|>", mutation_scale=18, lw=2.4, color=GREY, zorder=3)
    ax.add_patch(ar)

ax.text(6, 1.15, "Built in Python with pandas, scikit-learn, and XGBoost",
        ha="center", fontsize=10.5, color=GREY, style="italic")

add_trophy(fig, 12, 6.9, size_in=0.8, pad_right_in=0.5, top_in=0.12)
plt.savefig("post-assets/pipeline_diagram.png", facecolor="white", bbox_inches="tight")
print("saved pipeline_diagram.png")
