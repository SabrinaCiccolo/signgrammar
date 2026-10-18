"""Figures from data/ and results/ -> figures/*.png."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "figures"
L = list(range(13))
INK, MUTED, GRID, NULL = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"
# entity -> (label, colour, marker); same colour for the same readout in every figure
SERIES = {
    "direct": ("probe on the layer (all clips)", "#2a78d6", "o"),
    "matched": ("probe on the layer, clips matched in length", "#eb6834", "s"),
    "decoded": ("head pitch decoded from the layer (no question labels)", "#1baf7a", "^"),
    "face_only": ("face stream only, clips matched in length", "#eda100", "D"),
    "residual": ("probe after removing the measured face and head gesture", "#e87ba4", "v"),
}
REFS = {"dino_face": ("face crop features (DINOv2)", (0, (4, 2))),
        "dino_hands": ("hand crop features (DINOv2)", (0, (4, 1, 1, 1))),
        "pitch_mean": ("measured head pitch alone", (0, (1, 1.5)))}
TASKS = {"yn": "yes/no question", "wh": "wh-question", "neg": "negation", "cond": "conditional", "topic": "topic",
         "fs": "fingerspelling (manual control)"}
SIGNERS = ["Cory", "Jonathan", "Rachel"]
LAYER_X = "SHuBERT layer (in = fused input, before the 12 transformer layers)"
AUROC_Y = "AUROC (0.5 = chance, 1 = perfect)"


def load():
    return (pd.read_csv(RES / f) for f in ["layer_probes.csv", "layer_probes_only_face.csv", "residual_probes.csv"])


def curve(d, task, variant, signer, feat=lambda i: f"shubert_L{i}"):
    d = d[(d.task == task) & (d.variant == variant) & (d.signer == signer)].set_index("feature")
    keys = [feat(i) for i in L]
    if not set(keys) <= set(d.index):
        return None
    return d.loc[keys, ["auroc", "lo", "hi"]].to_numpy()


def decoded(probes, signer):
    return curve(probes, "yn", "untrained, z within signer", signer, lambda i: f"decoded_shubert_L{i}:pitch")


def ref(probes, task, feat, variant="centred"):
    if feat == "pitch_mean":
        variant = "untrained, z within signer"
    d = probes[(probes.task == task) & (probes.feature == feat) & (probes.variant == variant) & (probes.signer == "pooled")]
    return float(d.auroc.iloc[0]) if len(d) else None


def null_hi(probes, r, task):
    return max(probes[probes.task == task].null_hi.max(), r[r.task == task].null_hi.max() if (r.task == task).any() else 0)


def axes_style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def layer_style(ax, nul, ylim=(0.4, 1.0), every=2):
    ax.axhspan(ylim[0], nul, color=NULL, lw=0, zorder=0)
    ax.axhline(0.5, color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(-0.4, 12.4)
    ax.set_ylim(*ylim)
    ticks = list(range(0, 13, every))
    ax.set_xticks(ticks, ["in" if i == 0 else str(i) for i in ticks])
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    axes_style(ax)


def draw(ax, ser, bands=True, ms=4):
    for k, v in ser.items():
        if v is None:
            continue
        name, col, mk = SERIES[k]
        if bands:
            ax.fill_between(L, v[:, 1], v[:, 2], color=col, alpha=0.12, lw=0)
        ax.plot(L, v[:, 0], color=col, lw=2, marker=mk, ms=ms, mec="white", mew=0.8, label=name, zorder=3)


def draw_refs(ax, probes, task, feats, variant="centred", label=True):
    for f in feats:
        y = ref(probes, task, f, variant)
        if y is None:
            continue
        name, ls = REFS[f]
        ax.axhline(y, color=MUTED, lw=1, ls=ls, zorder=1)
        if label:
            ax.text(12.4, y + 0.005, f"{name}: {y:.2f}", color=MUTED, fontsize=7, ha="right", va="bottom")


def title(fig, ax, text, sub):
    ax.set_title(text, color=INK, fontsize=10.5, loc="left", pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=8, va="bottom")


def yn_figure(probes, face, r):
    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.3, 1], hspace=0.5, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    nul = null_hi(probes, r, "yn")
    layer_style(ax, nul, every=1)
    draw_refs(ax, probes, "yn", ["dino_face", "pitch_mean"])
    ser = lambda s: {"direct": curve(probes, "yn", "centred", s), "decoded": decoded(probes, s),
                     "residual": curve(r, "yn", "residual", s)}
    draw(ax, ser("pooled"))
    ax.set_ylabel(AUROC_Y, color=INK, fontsize=9)
    ax.set_xlabel(LAYER_X, color=INK, fontsize=8.5)
    title(fig, ax, "Every SHuBERT layer tells yes/no questions apart, through head pitch",
          "yes/no question vs any other utterance; 1,788 utterances, 231 questions; trained on two signers, tested on "
          "the third")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.3), fontsize=7.5, frameon=False, labelcolor=INK)
    for j, s in enumerate(SIGNERS):
        a = fig.add_subplot(gs[1, j])
        layer_style(a, nul, every=4)
        draw(a, ser(s), bands=False, ms=3)
        a.set_title(f"tested on {s}", color=INK, fontsize=9, loc="left")
        if j:
            a.tick_params(labelleft=False)
    fig.text(0.5, 0.02, "Bands: 95% bootstrap intervals. Bottom: the same three readouts, scored on each held-out "
             "signer alone.", color=MUTED, fontsize=7, ha="center")
    fig.savefig(OUT / "yn_layers.png", dpi=150, bbox_inches="tight", facecolor="white")


def qmark_figure():
    d = pd.read_csv(RES / "qmark.csv").set_index("condition")
    conds = {"AC": "full video", "NF": "face hidden", "NH": "hands hidden", "NHB": "only the face visible"}
    n = d.loc["AC"]
    groups = [("rate_manual", f"questions with a manual question sign, QMwg or YOU (n = {n.n_manual})", "#2a78d6"),
              ("rate_face_only", f"questions marked only by face and head (n = {n.n - n.n_manual})", "#eb6834"),
              ("rate_decl", f"declaratives, not questions (n = {n.n_decl})", "#9a9893")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.26
    for j, (c, name, col) in enumerate(groups):
        x = [i + (j - 1) * (w + 0.02) for i in range(len(conds))]
        y = d.loc[list(conds), c] * 100
        ax.bar(x, y, w, color=col, label=name, zorder=2)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.8, f"{yi:.0f}" if yi >= 9.5 else f"{yi:.1f}", ha="center", va="bottom", fontsize=7,
                    color=INK)
    ax.set_xticks(range(len(conds)), list(conds.values()), color=INK, fontsize=8.5)
    ax.set_ylabel('translations ending in "?" (%)', color=INK, fontsize=9)
    ax.set_ylim(0, 52)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    axes_style(ax)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK)
    title(fig, ax, 'The translator writes "?" only when the hands sign the question',
          "Published SHuBERT + ByT5 outputs (Karabüklü et al.) on the benchmark clips of Cory, Jonathan, Rachel; "
          "regions greyed out at test time.")
    fig.savefig(OUT / "qmark.png", dpi=150, bbox_inches="tight", facecolor="white")


def main():
    OUT.mkdir(exist_ok=True)
    probes, face, r = load()
    yn_figure(probes, face, r)
    qmark_figure()
    for f in sorted(OUT.glob("*.png")):
        print(f, f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
