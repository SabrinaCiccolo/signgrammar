"""Per-frame head pitch, measured and decoded from SHuBERT layers 0, 6 and 12 (ridge, trained on the other signers), on
one yes/no clip per signer -> results/clip_pitch.csv. Clip: the longest yes/no clip without QMwg whose single question
span covers 30-70% of the gloss line; span columns are mapped proportionally to frames."""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from common import CACHE, CORPUS, FEATS, ROOT, YN_SIGNERS, center_by, utterances

LAYERS = [0, 6, 12]


def pick(u):
    j = {r["dai_id"]: r for r in map(json.loads, open(CORPUS))}
    rows = []
    for r in u[(u.yn == 1) & (u.qm == 0) & u.signer.isin(YN_SIGNERS)].itertuples():
        t, g = j[r.dai_id]["tiers"], j[r.dai_id]
        width = max([len(g.get(k) or "") for k in ("dh", "ndh")] + [e + 1 for v in t.values() for _, e, _ in v])
        sp = t["y/n question"]
        if len(sp) == 1 and 0.3 <= (sp[0][1] - sp[0][0] + 1) / width <= 0.7:
            rows.append((r.signer, r.dai_id, sp[0][0] / width, (sp[0][1] + 1) / width,
                         len(np.load(FEATS / "head_pose" / f"{r.dai_id}.npz")["pose"])))
    d = pd.DataFrame(rows, columns=["signer", "dai_id", "start", "end", "n"])
    return d.loc[d.groupby("signer").n.idxmax()]


def main():
    u = utterances()
    u = u[u.dai_id.isin(np.load(CACHE)["dai_id"])].reset_index(drop=True)
    clips = pick(u)
    F, t, sig, cid, fidx = {k: [] for k in LAYERS}, [], [], [], []
    for r in u.itertuples():
        h = np.load(FEATS / "shubert" / f"{r.dai_id}.npy")
        g = np.load(FEATS / "head_pose" / f"{r.dai_id}.npz")["pose"][::2, 0]
        sel = np.isfinite(g)
        for k in LAYERS:
            F[k].append(h[k][sel].astype(np.float32))
        t.append(g[sel])
        sig += [r.signer] * sel.sum()
        cid += [r.dai_id] * sel.sum()
        fidx.append(np.nonzero(sel)[0])
    t, sig, cid, fidx = np.concatenate(t), np.array(sig), np.array(cid), np.concatenate(fidx)
    sd = pd.Series(t).groupby(sig).transform("std").to_numpy()
    tc = center_by(t[:, None], sig)[:, 0]
    out = pd.DataFrame({"signer": sig, "dai_id": cid, "frame": fidx, "measured": tc / sd})
    keep = np.isin(cid, clips.dai_id)
    out = out[keep].copy()
    for k in LAYERS:
        X = center_by(np.concatenate(F[k]), sig)
        for s in clips.signer:
            tr, te = sig != s, keep & (sig == s)
            m = RidgeCV(alphas=np.logspace(-1, 5, 7)).fit(X[tr], tc[tr])
            out.loc[te[keep], f"L{k}"] = m.predict(X[te]) / sd[te]
        print(f"L{k} done", flush=True)
    out = out.merge(clips[["dai_id", "start", "end", "n"]], on="dai_id")
    out["time"] = out.frame * 2 / 29.97
    out.round(4).to_csv(ROOT / "results/clip_pitch.csv", index=False)
    print(clips.to_string(index=False))


if __name__ == "__main__":
    main()
