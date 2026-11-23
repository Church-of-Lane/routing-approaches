import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath

from machine_learning import FEATURE_COUNT, FEATURE_LAYERS, HEAD_LAYERS, HISTORY, TEXT_VECTOR_SIZE, TOKENS_PER_MESSAGE

"""
Draws the routing model architecture (text branch + features branch) for given dense layer sizes.

Usage: python architecture_diagram.py OUT.png "FEATURE_LAYERS" "HEAD_LAYERS" [title] [--no-text] [--no-title]
    e.g. python architecture_diagram.py models/v4/architecture_3_layers.png "64" "32,16" "v4 routing model, 3 hidden layers"
    --no-text  draws the features-only model (USE_TEXT = False), as a single column
    --no-title leaves the title off (an empty title does the same)
"""

WIDTH, BOX_W, BOX_H, STEP = 1018, 386, 83, 126
NARROW_WIDTH              = 528
TITLE_SIZE                = 21
LEFT_X, RIGHT_X, MID_X    = 240, 779, 509

WHITE, BLUE, ORANGE, GREY = "#ffffff", "#dce8fa", "#fcebcc", "#f0efeb"
EDGE, TEXT, SUBTEXT       = "#9a9a8f", "#2a2a2a", "#6b6b6b"


def draw(path: str, feature_layers: tuple[int, ...], head_layers: tuple[int, ...], title: str, text_branch: bool = True) -> None:
    v = TEXT_VECTOR_SIZE
    text = [
        ("text input",       f"(B, {HISTORY}, {TOKENS_PER_MESSAGE})",       WHITE),
        ("Embedding",        f"(B, {HISTORY}, {TOKENS_PER_MESSAGE}, {v})",  BLUE),
        ("MessagePooling",   f"(B, {HISTORY}, {v})",                        BLUE),
        ("Dropout 0.5",      f"(B, {HISTORY}, {v})",                        BLUE),
        (f"LSTM {v}",        f"(B, {v})",                                   BLUE),
        ("Dropout 0.5",      f"(B, {v})",                                   BLUE),
    ]
    features = [("features input", f"(B, {FEATURE_COUNT})", WHITE)]
    features += [(f"Dense {u}, relu", f"(B, {u})", ORANGE) for u in feature_layers]

    joined = v + (feature_layers[-1] if feature_layers else FEATURE_COUNT)
    head   = [("Concatenate", f"(B, {joined})", GREY)]
    head  += [(f"Dense {u}, relu", f"(B, {u})", GREY) for u in head_layers]
    head  += [("Dense 1, sigmoid", "P(person responds)", GREY)]

    top = 90 if title else 27

    if text_branch:
        width   = WIDTH
        merge_y = top + (len(text) - 1) * STEP + BOX_H + 40
        head_y  = merge_y + 27
    else:
        # no concatenate, the features go straight into the head
        title_w = TextPath((0, 0), title, size=TITLE_SIZE, prop=FontProperties(weight="bold")).get_extents().width * 100 / 72 if title else 0
        width   = max(NARROW_WIDTH, int(title_w) + 40)  # wide enough for the title
        head    = features + head[1:]
        head_y  = top

    height = head_y + (len(head) - 1) * STEP + BOX_H + 27

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax  = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    if title:
        ax.text(width / 2, 30, title, ha="center", va="center", fontsize=TITLE_SIZE, fontweight="bold", color=TEXT)

    def box(cx: float, y: float, name: str, shape: str, color: str) -> None:
        ax.add_patch(FancyBboxPatch((cx - BOX_W / 2, y), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=12",
                                    facecolor=color, edgecolor=EDGE, linewidth=1.8))
        ax.text(cx, y + 29, name,  ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT)
        ax.text(cx, y + 59, shape, ha="center", va="center", fontsize=13, color=SUBTEXT)

    def arrow(x: float, y0: float, y1: float) -> None:
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.5", color=EDGE, lw=1.8, shrinkA=0, shrinkB=0))

    def column(cx: float, y: float, boxes: list) -> float:
        for i, (name, shape, color) in enumerate(boxes):
            if i:
                arrow(cx, y - STEP + BOX_H, y)
            box(cx, y, name, shape, color)
            y += STEP
        return y - STEP + BOX_H  # bottom of last box

    if not text_branch:
        column(width / 2, head_y, head)
    else:
        left_end  = column(LEFT_X,  top, text)
        right_end = column(RIGHT_X, top, features)

        line = dict(color=EDGE, lw=1.8, solid_capstyle="butt")
        ax.plot([LEFT_X,  LEFT_X],  [left_end,  merge_y], **line)
        ax.plot([RIGHT_X, RIGHT_X], [right_end, merge_y], **line)
        ax.plot([LEFT_X,  RIGHT_X], [merge_y,   merge_y], **line)
        arrow(MID_X, merge_y, head_y)

        column(MID_X, head_y, head)

    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    from sys import argv

    text_branch = "--no-text"  not in argv
    has_title   = "--no-title" not in argv
    argv        = [a for a in argv if a not in ("--no-text", "--no-title")]

    sizes = lambda s: tuple(int(u) for u in s.split(",") if u.strip())

    draw(
        argv[1] if len(argv) > 1 else "architecture.png",
        sizes(argv[2]) if len(argv) > 2 else FEATURE_LAYERS,
        sizes(argv[3]) if len(argv) > 3 else HEAD_LAYERS,
        (argv[4] if len(argv) > 4 else "v4 routing model") if has_title else "",
        text_branch,
    )
