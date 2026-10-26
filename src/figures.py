"""Figures from data/ and results/ -> figures/*.png."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def null_note(ax, nul, x=12.4, y=0.41):
    ax.text(x, y, f"grey band: also reached with shuffled labels (up to {nul:.2f}), so not significant",
            color=MUTED, fontsize=7, ha="right", va="bottom")


def dataset_figure():
    u = pd.read_csv(ROOT / "data/utterances.csv")
    cols = {"yn": "yes/no\nquestion", "wh": "wh-\nquestion", "neg": "negation", "cond": "conditional",
            "topic": "topic", "plain": "no grammatical\ntier"}
    signers = ["Cory", "Jonathan", "Rachel", "Ben"]
    tot = u.groupby("signer")[list(cols)].sum().loc[signers]
    bench = u[u.in_benchmark == 1].groupby("signer")[list(cols)].sum().reindex(signers).fillna(0)
    n = u.groupby("signer").size().loc[signers]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    v = tot.to_numpy() / n.to_numpy()[:, None]
    ax.imshow(v, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#ffffff", "#2a78d6"]),
              vmin=0, vmax=0.7, aspect="auto")
    for i in range(len(signers)):
        for j in range(len(cols)):
            c = "white" if v[i, j] > 0.45 else INK
            ax.text(j, i - 0.12, f"{tot.iat[i, j]}", ha="center", va="center", fontsize=9.5, color=c)
            ax.text(j, i + 0.24, f"ASL-MTP: {int(bench.iat[i, j])}", ha="center", va="center", fontsize=6.5,
                    color=c if c == "white" else MUTED)
    ax.set_xticks(range(len(cols)), list(cols.values()), fontsize=8, color=INK)
    ax.set_yticks(range(len(signers)), [f"{s}  ({n[s]} utt.)" for s in signers], fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(signers)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    fig.subplots_adjust(top=0.7)
    fig.text(0.02, 1.0, "The data: 2,127 annotated ASLLRP utterances from four signers", color=INK, fontsize=10.5,
             va="bottom")
    fig.text(0.02, 0.88, "Utterances with each nonmanual tier (colour = share of the signer's utterances; one utterance "
             "can carry several).\nASL-MTP: clips of the translation benchmark (970 in total). "
             "Ben signs lectures: no yes/no questions.", color=MUTED, fontsize=7.5, va="bottom")
    fig.savefig(OUT / "dataset.png", dpi=150, bbox_inches="tight", facecolor="white")


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
    null_note(ax, nul)
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


def yn_controls_figure(probes, face, r):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    nul = null_hi(probes, r, "yn")
    layer_style(ax, nul, every=1)
    v = curve(probes, "yn", "centred", "pooled")
    ax.plot(L, v[:, 0], color=MUTED, lw=1.2, ls=(0, (3, 2)), label="probe on the layer (all clips), as before", zorder=2)
    draw(ax, {"matched": curve(probes, "yn", "length-matched", "pooled"),
              "face_only": curve(face, "yn", "length-matched", "pooled")})
    null_note(ax, nul)
    ax.set_ylabel(AUROC_Y, color=INK, fontsize=9)
    ax.set_xlabel(LAYER_X, color=INK, fontsize=8.5)
    title(fig, ax, "Controls: not clip length, and the face stream alone is enough",
          "Questions are shorter; matching lengths lowers the deep layers (0.80 to 0.72). "
          "Feeding SHuBERT only the face keeps 0.81.")
    ax.legend(loc="lower left", fontsize=7.5, frameon=False, labelcolor=INK, bbox_to_anchor=(0, 0.12))
    fig.savefig(OUT / "yn_controls.png", dpi=150, bbox_inches="tight", facecolor="white")


def functions_figure(probes, face, r):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2), sharey=True)
    for ax, (t, name) in zip(axes.flat, TASKS.items()):
        nul = null_hi(probes, r, t)
        layer_style(ax, nul, ylim=(0.35, 1.0), every=4)
        draw_refs(ax, probes, t, ["dino_face", "dino_hands"], variant="length-matched", label=False)
        draw(ax, {"matched": curve(probes, t, "length-matched", "pooled"),
                  "face_only": curve(face, t, "length-matched", "pooled"),
                  "residual": curve(r, t, "residual", "pooled")}, ms=3)
        n = probes[(probes.task == t) & (probes.signer == "pooled")].n_pos.iloc[0]
        ax.set_title(f"{name} ({n})", color=INK, fontsize=9, loc="left")
    handles = [plt.Line2D([], [], color=SERIES[k][1], lw=2, marker=SERIES[k][2], ms=4, mec="white")
               for k in ["matched", "face_only", "residual"]]
    labels = [SERIES[k][0] for k in ["matched", "face_only", "residual"]]
    handles += [plt.Line2D([], [], color=MUTED, lw=1, ls=REFS[f][1]) for f in ["dino_face", "dino_hands"]]
    labels += [f"input reference: {REFS[f][0]}" for f in ["dino_face", "dino_hands"]]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=NULL))
    labels.append("also reached with shuffled labels (not significant)")
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5, frameon=False, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.1))
    fig.supylabel(AUROC_Y, color=INK, fontsize=9)
    fig.supxlabel(LAYER_X, color=INK, fontsize=8.5, y=0.0)
    fig.suptitle("Six functions per layer: yes/no is strong and facial; negation and fingerspelling need hands and body",
                 color=INK, fontsize=10.5, x=0.02, ha="left")
    fig.text(0.02, 0.905, "Each function vs every other utterance (number of positives in brackets), 2,127 utterances, "
             "trained on all but one signer.\nBands: 95% bootstrap intervals.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "functions_layers.png", dpi=150, bbox_inches="tight", facecolor="white")


CLIP = {"measured": ("measured (MediaPipe)", INK, "-", 2.2), "L0": ("decoded from the input", "#1baf7a", "-", 1.4),
        "L6": ("from layer 6", "#eda100", (0, (4, 2)), 1.4), "L12": ("from layer 12", "#2a78d6", (0, (1, 1.2)), 1.6)}


def clips_figure():
    d = pd.read_csv(RES / "clip_pitch.csv")
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.8), sharey=True)
    for ax, (s, c) in zip(axes, d.groupby("signer")):
        dur = c.n.iloc[0] / 29.97
        ax.axvspan(c.start.iloc[0] * dur, c.end.iloc[0] * dur, color=NULL, lw=0, zorder=0)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        for k, (name, col, ls, lw) in CLIP.items():
            ax.plot(c.time, c[k], color=col, ls=ls, lw=lw, label=name, zorder=3 if k == "measured" else 2)
        ax.set_xlim(0, dur)
        ax.set_title(f"{s}, yes/no clip {c.dai_id.iloc[0]}", color=INK, fontsize=9, loc="left")
        axes_style(ax)
    h, lab = axes[0].get_legend_handles_labels()
    h.append(plt.Rectangle((0, 0), 1, 1, color=NULL))
    lab.append("question part (annotation, approximate)")
    fig.legend(h, lab, loc="upper left", ncol=3, fontsize=7.5, frameon=False, labelcolor=INK,
               bbox_to_anchor=(0.06, 0.975))
    axes[-1].set_xlabel("time (s)", color=INK, fontsize=9)
    fig.supylabel("head pitch (z-score within signer; up = head tilted forward)", color=INK, fontsize=9)
    fig.suptitle("Head pitch read out of SHuBERT follows the real head, frame by frame", color=INK, fontsize=10.5,
                 x=0.06, ha="left", y=1.0)
    fig.text(0.5, -0.01, "Decoders: linear (ridge) on single frames of a layer, trained on the other two signers, "
             "never told which clips are questions.\nShaded span: y/n question tier, mapped from gloss columns to "
             "frames proportionally.", color=MUTED, fontsize=7, ha="center", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(OUT / "yn_clips.png", dpi=150, bbox_inches="tight", facecolor="white")


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


def repair_figure():
    rep = pd.read_csv(RES / "repair.csv").set_index(["row", "signer"])
    lr = pd.read_csv(RES / "lowres.csv")
    lr = lr[(lr.readout == "pitch_detector") & (lr.row == "low")].rename(columns={"value": "balanced"}).set_index("signer")
    rows = [("translator (published scores)", rep.xs("published")),
            ("translator, threshold tuned", rep.xs("published_offset")),
            ("probe on SHuBERT input layers", rep.xs("detector")), ("measured head pitch (one number)", rep.xs("pitch_detector")),
            ("head pitch, video at 1/3 resolution", lr)]
    signers = {"Cory": "o", "Jonathan": "s", "Rachel": "^"}
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, (name, d) in enumerate(rows):
        y = len(rows) - 1 - i
        col = MUTED if i < 2 else "#2a78d6"
        p = d.loc["pooled"]
        ax.plot([p.lo, p.hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(p.balanced, y, "o", color=col, ms=9, mec="white", mew=1.5, zorder=4)
        for s, mk in signers.items():
            ax.plot(d.loc[s, "balanced"], y + 0.28, mk, color=col, ms=4, mfc="none", mew=1, zorder=3)
        ax.text(1.02, y, f"{p.balanced + 1e-9:.2f}", color=INK, fontsize=8, va="center", ha="left",
                transform=ax.get_yaxis_transform())
        ax.text(1.2, y, f"{p.acc_marked:.2f} / {p.acc_unmarked:.2f}", color=MUTED, fontsize=8, va="center",
                ha="left", transform=ax.get_yaxis_transform())
    ax.text(1.02, len(rows) - 0.55, "all", color=MUTED, fontsize=7, transform=ax.get_yaxis_transform())
    ax.text(1.2, len(rows) - 0.55, "right on questions / declaratives", color=MUTED, fontsize=7,
            transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=GRID, lw=1, zorder=0)
    ax.text(0.505, -0.45, "chance", color=MUTED, fontsize=7, ha="left", va="bottom")
    ax.axhline(2.5, xmin=-0.5, color=GRID, lw=0.8, clip_on=False)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows][::-1], color=INK, fontsize=8.5)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.set_xlabel("balanced accuracy (mean of the two columns on the right)", color=INK, fontsize=8.5)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    axes_style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], color=MUTED, marker="o", ls="-", ms=6, mec="white")]
    handles += [plt.Line2D([], [], color=MUTED, marker=m, ls="", ms=4, mfc="none") for m in signers.values()]
    fig.legend(handles, ["all signers, 95% interval"] + [f"tested on {s}" for s in signers], loc="lower left",
               fontsize=7, frameon=False, labelcolor=MUTED, ncol=4, bbox_to_anchor=(0.0, 0.0))
    fig.suptitle("Head pitch repairs the yes/no choice the translator gets wrong", color=INK, fontsize=10.5,
                 x=0.02, ha="left")
    fig.text(0.02, 0.89, "For each benchmark clip: question or declarative translation? 150 questions, 135 "
             "declaratives. Detectors alone; fused with the translator they score the same.", color=MUTED, fontsize=7)
    fig.subplots_adjust(left=0.33, right=0.68, top=0.82, bottom=0.24)
    fig.savefig(OUT / "repair.png", dpi=150, bbox_inches="tight", facecolor="white")


def main():
    OUT.mkdir(exist_ok=True)
    probes, face, r = load()
    dataset_figure()
    yn_figure(probes, face, r)
    yn_controls_figure(probes, face, r)
    functions_figure(probes, face, r)
    clips_figure()
    qmark_figure()
    repair_figure()
    for f in sorted(OUT.glob("*.png")):
        print(f, f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
