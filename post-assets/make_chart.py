import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from wc_style import RED, LIME, NAVY, BERRY, GOLD, GREY, INK, draw_trophy, add_trophy

teams = ["Spain","Argentina","France","England","Brazil","Portugal","Colombia","Netherlands","Ecuador","Germany"]
elo   = [2155, 2114, 2063, 2021, 1991, 1986, 1982, 1948, 1938, 1932]
teams = teams[::-1]; elo = elo[::-1]

colors = [NAVY for t in teams]

fig, ax = plt.subplots(figsize=(10, 6.6), dpi=200)
fig.patch.set_facecolor("white")
fig.subplots_adjust(top=0.80, left=0.16, right=0.96, bottom=0.10)
ax.set_facecolor("white")

# top brand strip
hb = fig.add_axes([0,0.90,1,0.10]); hb.axis("off"); hb.set_xlim(0,1); hb.set_ylim(0,1)
for x,w,c in [(0,0.34,BERRY),(0.34,0.30,RED),(0.64,0.20,LIME),(0.84,0.16,NAVY)]:
    hb.add_patch(Rectangle((x,0),w,1,facecolor=c))

fig.text(0.16, 0.845, "World Cup 2026 contenders by Elo rating",
         fontsize=18, fontweight="bold", color=INK)
fig.text(0.16, 0.815, "World Football Elo Ratings, as of June 2026",
         fontsize=10.5, color=GREY)

bars = ax.barh(teams, elo, color=colors, height=0.66, zorder=3)
ax.set_xlim(1800, 2200)
for b, v, t in zip(bars, elo, teams):
    tc = "white"
    ax.text(v - 8, b.get_y()+b.get_height()/2, f"{v}", va="center", ha="right",
            color=tc, fontsize=11, fontweight="bold", zorder=4)

ax.grid(axis="x", color="#e5e7eb", zorder=0)
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#d1d5db")
ax.tick_params(axis="y", length=0, labelsize=12)
ax.tick_params(axis="x", colors=GREY, labelsize=9)
for lbl in ax.get_yticklabels():
    lbl.set_color("#374151")

fig.text(0.16, 0.025, "Source: eloratings.net", fontsize=8.5, color="#9ca3af")
add_trophy(fig, 10, 6.6, size_in=0.78, pad_right_in=0.45, top_in=0.12)
plt.savefig("post-assets/elo_chart.png", facecolor="white", bbox_inches="tight")
print("saved elo_chart.png")
