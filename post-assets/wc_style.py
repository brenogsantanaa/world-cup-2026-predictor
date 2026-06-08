# Shared World Cup 2026 styling helpers
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, FancyBboxPatch, Rectangle

# WC 2026-inspired palette
RED    = "#E8112D"
LIME   = "#B6E300"
NAVY   = "#16235A"
BERRY  = "#8A1538"
GOLD   = "#E9C46A"
GOLD_D = "#C9982F"
INK    = "#16235A"
GREY   = "#5b6472"

def draw_trophy(ax, cx, cy, s):
    """Draw a simple generic gold trophy centered at (cx,cy), scale s (axis units)."""
    g = GOLD; gd = GOLD_D
    # bowl (cup)
    bowl = Polygon([(cx-0.55*s, cy+0.9*s),(cx+0.55*s, cy+0.9*s),
                    (cx+0.40*s, cy+0.25*s),(cx-0.40*s, cy+0.25*s)],
                   closed=True, facecolor=g, edgecolor=gd, lw=1.2, zorder=6)
    ax.add_patch(bowl)
    # rim
    ax.add_patch(Rectangle((cx-0.6*s, cy+0.88*s), 1.2*s, 0.12*s, facecolor=gd, zorder=7))
    # handles
    for sgn in (-1,1):
        h = Ellipse((cx+sgn*0.62*s, cy+0.62*s), 0.34*s, 0.6*s, angle=0,
                    facecolor="none", edgecolor=gd, lw=3.2, zorder=5)
        ax.add_patch(h)
    # stem
    ax.add_patch(Rectangle((cx-0.07*s, cy-0.15*s), 0.14*s, 0.42*s, facecolor=g, edgecolor=gd, lw=0.8, zorder=6))
    # base
    base = Polygon([(cx-0.42*s, cy-0.45*s),(cx+0.42*s, cy-0.45*s),
                    (cx+0.30*s, cy-0.15*s),(cx-0.30*s, cy-0.15*s)],
                   closed=True, facecolor=gd, edgecolor=gd, lw=1, zorder=6)
    ax.add_patch(base)
    ax.add_patch(Rectangle((cx-0.5*s, cy-0.58*s), 1.0*s, 0.14*s, facecolor=NAVY, zorder=6))

def color_header(fig, height=0.16):
    """Diagonal-ish multicolor brand strip across the top of the figure."""
    ax = fig.add_axes([0,1-height,1,height]); ax.axis("off")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    blocks = [(0.00,0.34,BERRY),(0.34,0.30,RED),(0.64,0.20,LIME),(0.84,0.16,NAVY)]
    for x,w,c in blocks:
        ax.add_patch(Rectangle((x,0),w,1,facecolor=c,zorder=1))
    return ax

def add_trophy(fig, fig_w_in, fig_h_in, size_in=0.7, pad_right_in=0.35, top_in=0.18):
    """Add a square axes near top-right and draw a clean trophy in it."""
    w = size_in / fig_w_in
    h = size_in / fig_h_in
    left = 1 - (pad_right_in + size_in) / fig_w_in
    bottom = 1 - (top_in + size_in) / fig_h_in
    tax = fig.add_axes([left, bottom, w, h]); tax.axis("off")
    tax.set_xlim(0,1); tax.set_ylim(0,1); tax.set_aspect("equal")
    draw_trophy(tax, 0.5, 0.42, 0.46)
    return tax
