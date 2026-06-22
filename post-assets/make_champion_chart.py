import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from wc_style import RED, LIME, NAVY, BERRY, GOLD, GREY, INK, draw_trophy, add_trophy

# Dixon-Coles champion odds, 2026 forward run (as-of 2026-04-23, cutoff 2026-06-11)
teams = ["Spain","Argentina","France","England","Brazil","Colombia","Portugal","Ecuador","Netherlands","Belgium"]
odds  = [17.2, 15.0, 7.1, 6.8, 5.2, 4.5, 4.1, 3.6, 3.5, 3.0]
teams, odds = teams[::-1], odds[::-1]

fig, ax = plt.subplots(figsize=(10, 6.8), dpi=200)
fig.patch.set_facecolor("white")
fig.subplots_adjust(top=0.80, left=0.17, right=0.95, bottom=0.10)
ax.set_facecolor("white")

hb = fig.add_axes([0,0.90,1,0.10]); hb.axis("off"); hb.set_xlim(0,1); hb.set_ylim(0,1)
for x,w,c in [(0,0.34,BERRY),(0.34,0.30,RED),(0.64,0.20,LIME),(0.84,0.16,NAVY)]:
    hb.add_patch(Rectangle((x,0),w,1,facecolor=c))

fig.text(0.17, 0.845, "Who wins the 2026 World Cup? My model's odds",
         fontsize=17, fontweight="bold", color=INK)
fig.text(0.17, 0.815, "20,000 tournament simulations",
         fontsize=10.5, color=GREY)

bars = ax.barh(teams, odds, color=NAVY, height=0.66, zorder=3)
ax.set_xlim(0, 20)
for b, v in zip(bars, odds):
    ax.text(v + 0.4, b.get_y()+b.get_height()/2, f"{v:.1f}%", va="center", ha="left",
            color=INK, fontsize=11, fontweight="bold", zorder=4)

ax.grid(axis="x", color="#e5e7eb", zorder=0)
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#d1d5db")
ax.tick_params(axis="y", length=0, labelsize=12)
ax.tick_params(axis="x", colors=GREY, labelsize=9)
for lbl in ax.get_yticklabels(): lbl.set_color("#374151")

fig.text(0.17, 0.025, "Champion probability. Trained on international results through 11 Jun 2026.",
         fontsize=8.5, color="#9ca3af")
add_trophy(fig, 10, 6.8, size_in=0.78, pad_right_in=0.45, top_in=0.12)
plt.savefig("champion_odds_v2.png", facecolor="white", bbox_inches="tight")
print("saved champion_odds_v2.png")
