import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from wc_style import RED, LIME, NAVY, BERRY, GOLD, GREY, INK, add_trophy

w  = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
ll = [1.950, 2.005, 2.033, 2.052, 2.091, 2.159, 2.202, 2.267, 2.364]

fig, ax = plt.subplots(figsize=(10, 6.4), dpi=200)
fig.patch.set_facecolor("white")
fig.subplots_adjust(top=0.78, left=0.11, right=0.95, bottom=0.13)
ax.set_facecolor("white")

hb = fig.add_axes([0,0.90,1,0.10]); hb.axis("off"); hb.set_xlim(0,1); hb.set_ylim(0,1)
for x,wd,c in [(0,0.34,BERRY),(0.34,0.30,RED),(0.64,0.20,LIME),(0.84,0.16,NAVY)]:
    hb.add_patch(Rectangle((x,0),wd,1,facecolor=c))

fig.text(0.11, 0.845, "Does blending the two models help? I tested it.",
         fontsize=16.5, fontweight="bold", color=INK)
fig.text(0.11, 0.815, "Mean error predicting the last 4 World Cup champions (lower = better)",
         fontsize=10.5, color=GREY)

ax.plot(w, ll, color=NAVY, lw=2.6, zorder=3, marker="o", markersize=6, markerfacecolor=NAVY)
# highlight the best point (pure Elo, w=0)
ax.scatter([0.0], [1.950], s=240, color=LIME, edgecolor=NAVY, linewidth=1.5, zorder=5)
ax.annotate("Best: pure Elo model\n(blending only made it worse)",
            xy=(0.0, 1.950), xytext=(0.18, 1.99),
            fontsize=10.5, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5))

ax.set_xlim(-0.04, 1.04); ax.set_ylim(1.88, 2.42)
ax.set_xlabel("Weight on the Dixon-Coles goal model  (0 = pure Elo,  1 = pure goal model)",
              fontsize=10.5, color="#374151")
ax.set_ylabel("Champion prediction error", fontsize=10.5, color="#374151")
ax.grid(axis="both", color="#e8eaed", zorder=0)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color("#d1d5db")
ax.tick_params(colors=GREY, labelsize=9)

fig.text(0.11, 0.03, "Validated on the 2010, 2014, 2018 and 2022 World Cups. Lower error = better champion prediction.",
         fontsize=8.5, color="#9ca3af")
add_trophy(fig, 10, 6.4, size_in=0.74, pad_right_in=0.4, top_in=0.12)
plt.savefig("model_weight_test.png", facecolor="white", bbox_inches="tight")
print("saved model_weight_test.png")
