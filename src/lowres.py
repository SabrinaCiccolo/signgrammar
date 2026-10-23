"""Measured head pitch on frames downscaled 3x: face detection, per-frame agreement with full resolution, and the
pitch detector of src/repair.py at both resolutions -> results/lowres.csv."""
import numpy as np
import pandas as pd

from common import FEATS, ROOT, YN_SIGNERS, setup, zsig
from repair import bacc, best_b, boot, items


def frames(folder, i):
    return np.load(FEATS / folder / f"{i}.npz")["pose"][:, 0]


def main():
    _, u, sig, _ = setup()
    high = [frames("head_pose", i) for i in u.dai_id]
    low = [frames("head_pose_lowres", i) for i in u.dai_id]
    rows = []
    for s in np.unique(sig):
        k = np.flatnonzero(sig == s)
        h, lo = np.concatenate([high[j] for j in k]), np.concatenate([low[j] for j in k])
        both = np.isfinite(h) & np.isfinite(lo)
        rows += [dict(readout="detection", signer=s, row="high", value=np.isfinite(h).mean()),
                 dict(readout="detection", signer=s, row="low", value=np.isfinite(lo).mean()),
                 dict(readout="agreement", signer=s, row="r", value=np.corrcoef(h[both], lo[both])[0, 1])]
    with np.errstate(all="ignore"):
        pz = {"high": zsig(np.array([np.nanmean(v) for v in high]), sig),
              "low": zsig(np.array([np.nanmean(v) for v in low]), sig)}
    pos = {s: {d: k for k, d in enumerate(u.dai_id[sig == s])} for s in np.unique(sig)}
    val = lambda p, s, ids: np.array([p[sig == s][pos[s][d]] for d in ids])
    it = items()
    tes = []
    for T in YN_SIGNERS:
        tu, te = it[it.signer.isin([t for t in YN_SIGNERS if t != T])], it[it.signer == T].copy()
        for name, p in pz.items():
            b = best_b(np.array([p[sig == r.signer][pos[r.signer][r.dai_id]] for r in tu.itertuples()]),
                       tu.marked.to_numpy())[0]
            te[name] = val(p, T, te.dai_id) > b
        tes.append(te)
    for T, te in [*zip(YN_SIGNERS, tes), ("pooled", pd.concat(tes))]:
        for col in pz:
            m, pred = te.marked.to_numpy(), te[col].to_numpy()
            lo, hi = boot(te, col)
            rows.append(dict(readout="pitch_detector", signer=T, row=col, value=bacc(m, pred), lo=lo, hi=hi,
                             acc_marked=pred[m == 1].mean(), acc_unmarked=(~pred[m == 0]).mean()))
    out = pd.DataFrame(rows).round(3)
    out.to_csv(ROOT / "results/lowres.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
