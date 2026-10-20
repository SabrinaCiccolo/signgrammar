"""Yes/no vs declarative choice on the ASL-MTP clips: published SHuBERT + ByT5 preference, a detector read from SHuBERT
layers, and their fusion. Each signer is tested once with layer, weight and threshold tuned on the
other two -> results/repair.csv."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import ITEMS, LAYERS, PUBLISHED, ROOT, YN_SIGNERS, center_by, setup  # noqa: E402

LAMBDAS = np.concatenate([[0], np.logspace(-3, 1, 17)])
ROWS = ["published", "published_offset", "detector", "fused"]


def items():
    """yes/no (marked) and yes/no-reversed (declarative) items with the published preference p for the marked one."""
    bench = pd.read_csv(ITEMS)
    out = []
    for phen, marked in [("yes/no", 1), ("yes/no-reversed", 0)]:
        j = pd.read_json(PUBLISHED / f"asllrp_{phen.replace('/', '-')}_AC_0_AC.json", lines=True)
        it = bench[bench.phenomenon == phen].merge(j[["utterance_id", "nll_diff"]], on="utterance_id", validate="1:1")
        it["marked"] = marked
        it["p"] = it.nll_diff if marked else -it.nll_diff
        out.append(it)
    return pd.concat(out, ignore_index=True)


def detector(X, y, sig, scene, train):
    """Logistic probe trained on signers `train`; z-scored scores for every utterance of each signer."""
    tr = np.flatnonzero(np.isin(sig, train))
    if len(train) > 1:
        inner = [(np.flatnonzero(sig[tr] != t), np.flatnonzero(sig[tr] == t)) for t in train]
    else:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y[tr], scene[tr])
        inner = [(a, b) for a, b in cv if len(np.unique(y[tr][b])) == 2]
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=inner, scoring="roc_auc",
                                                               max_iter=5000)).fit(X[tr], y[tr])
    out = {}
    for s in np.unique(sig):
        te = sig == s
        sc = clf.decision_function(X[te])
        out[s] = (sc - sc.mean()) / (sc.std() + 1e-12)
    return out


def bacc(m, pred):
    return (pred[m == 1].mean() + (~pred[m == 0]).mean()) / 2


def best_b(f, m):
    """Threshold on f maximising balanced accuracy; ties -> median of the tied thresholds."""
    v = np.unique(f)
    cand = np.concatenate([[v[0] - 1], (v[1:] + v[:-1]) / 2, [v[-1] + 1]])
    ba = np.array([bacc(m, f > t) for t in cand])
    return np.median(cand[ba == ba.max()]), ba.max()


def tune(p, s, m):
    best = (-1, 0, 0)
    for lam in LAMBDAS:
        b, ba = best_b(p + lam * s, m)
        if ba > best[0] + 1e-12:
            best = (ba, lam, b)
    return best[1], best[2]


def rotation(T, tuning, scores, y, sig, it, pos):
    """Layer, lambda and b from the tuning signers only (nested scores); predictions for T's items."""
    keep = np.isin(sig, tuning)
    auc = []
    for L in range(LAYERS):
        sc = np.zeros(len(y))
        for S in tuning:
            sc[sig == S] = scores[(tuple(t for t in tuning if t != S), L)][S]
        auc.append(roc_auc_score(y[keep], sc[keep]))
    L = int(np.argmax(np.round(auc, 6)))
    tu = it[it.signer.isin(tuning)]
    s_tu = np.array([scores[(tuple(t for t in tuning if t != r.signer), L)][r.signer][pos[r.signer][r.dai_id]]
                     for r in tu.itertuples()])
    m, p = tu.marked.to_numpy(), tu.p.to_numpy()
    lam, b = tune(p, s_tu, m)
    te = it[it.signer == T].copy()
    s = np.array([scores[(tuple(tuning), L)][T][pos[T][d]] for d in te.dai_id])
    te["published"] = te.p > 0
    te["published_offset"] = te.p > best_b(p, m)[0]
    te["detector"] = s > best_b(s_tu, m)[0]
    te["fused"] = te.p + lam * s > b
    return te


def boot(te, col, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m, pred = te.marked.to_numpy(), te[col].to_numpy()
    pos, neg = np.flatnonzero(m == 1), np.flatnonzero(m == 0)
    st = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        st.append(bacc(m[i], pred[i]))
    return np.percentile(st, [2.5, 97.5])


def report(te, group, cols):
    rows = []
    for col in cols:
        m, pred = te.marked.to_numpy(), te[col].to_numpy()
        lo, hi = boot(te, col)
        acc_m, acc_u = pred[m == 1].mean(), (~pred[m == 0]).mean()
        rows.append(dict(signer=group, row=col, n_marked=int((m == 1).sum()), n_unmarked=int((m == 0).sum()),
                         acc_marked=acc_m, acc_unmarked=acc_u, balanced=(acc_m + acc_u) / 2, lo=lo, hi=hi))
    return rows


def main():
    z, u, sig, feats = setup()
    scene = u.scene.to_numpy()
    pos = {s: {d: k for k, d in enumerate(u.dai_id[sig == s])} for s in np.unique(sig)}
    X = [center_by(feats[f"shubert_L{L}"], sig) for L in range(LAYERS)]
    y = u.yn.to_numpy()
    it = items()
    rots = [(T, [t for t in YN_SIGNERS if t != T]) for T in YN_SIGNERS]
    sets = {tuple(tun) for _, tun in rots} | {tuple(t for t in tun if t != S) for _, tun in rots for S in tun}
    jobs = [(tr, L) for tr in sorted(sets) for L in range(LAYERS)]
    print(len(jobs), "detectors", flush=True)
    scores = dict(zip(jobs, Parallel(n_jobs=-1)(delayed(detector)(X[L], y, sig, scene, list(tr)) for tr, L in jobs)))
    tes = [rotation(T, tun, scores, y, sig, it, pos) for T, tun in rots]
    rows = [r for (T, _), te in zip(rots, tes) for r in report(te, T, ROWS)]
    rows += report(pd.concat(tes), "pooled", ROWS)
    out = pd.DataFrame(rows).round(3)
    out.to_csv(ROOT / "results/repair.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
