"""Paths, the utterance table, cached clip features and the leave-one-signer-out probe."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
VIDEOS = WORK / "videos"
FEATS = WORK / "features"
MODELS = WORK / "models"
SHUBERT_REPO = WORK / "SHuBERT"
PUBLISHED = WORK / "SL-Models-Analysis/data/results/ASLLRP_WeightedSum"
CORPUS = WORK / "corpus.jsonl"
CACHE = WORK / "clip_features.npz"
UTTERANCES = ROOT / "data/utterances.csv"
ITEMS = ROOT / "data/benchmark_items.csv"

LAYERS = 13
YN_SIGNERS = ["Cory", "Jonathan", "Rachel"]  # the only signers with yes/no questions
TASKS = ["yn", "wh", "neg", "cond", "topic", "fs"]
NULL_FEATS = ["dino_face", "shubert_L0", "shubert_L6", "shubert_L12"]


def utterances():
    return pd.read_csv(UTTERANCES)


def folds(task):
    return YN_SIGNERS if task == "yn" else ["Ben", *YN_SIGNERS]


def setup():
    """Cached clip features aligned with the utterance table."""
    z = np.load(CACHE)
    u = utterances().set_index("dai_id").loc[z["dai_id"]].reset_index()
    feats = {f"shubert_L{k}": z["shubert"][:, k] for k in range(LAYERS)}
    feats |= {k: impute(z[k]) for k in ["dino_face", "dino_hands"]}
    return z, u, u.signer.to_numpy(), feats


def boot_auc(y, s, n=2000, seed=0):
    """AUROC with a 95% bootstrap interval, resampling clips within each class."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats.append(roc_auc_score(y[idx], s[idx]))
    return roc_auc_score(y, s), *np.percentile(stats, [2.5, 97.5])


def loso_scores(X, y, signers, folds=None, make_clf=None):
    """Leave-one-signer-out logistic regression, C chosen by inner leave-one-signer-out AUROC. Scores are z-scored
    within each held-out signer. Signers outside `folds` are scored by the model fitted on all fold signers."""
    folds = np.unique(signers) if folds is None else np.asarray(folds)
    in_folds = np.isin(signers, folds)
    scores = np.zeros(len(y))
    for s in [*folds, None] if (~in_folds).any() else folds:
        tr = np.flatnonzero(in_folds & (signers != s))
        inner = [(np.flatnonzero(signers[tr] != t), np.flatnonzero(signers[tr] == t)) for t in np.unique(signers[tr])]
        clf = make_clf(inner) if make_clf else make_pipeline(StandardScaler(), LogisticRegressionCV(
            Cs=np.logspace(-4, 2, 13), cv=inner, scoring="roc_auc", max_iter=5000))
        clf.fit(X[tr], y[tr])
        for t in ([s] if s is not None else np.unique(signers[~in_folds])):
            te = signers == t
            sc = clf.decision_function(X[te])
            scores[te] = (sc - sc.mean()) / (sc.std() + 1e-12)
    return scores


def rows_for(task, feature, variant, y, sc, sig, folds):
    """Pooled row over the fold signers, then one row per signer."""
    pooled = np.isin(sig, folds)
    out = [dict(task=task, feature=feature, variant=variant, signer="pooled", n=pooled.sum(), n_pos=y[pooled].sum(),
                **dict(zip(["auroc", "lo", "hi"], boot_auc(y[pooled], sc[pooled]))))]
    for s in np.unique(sig):
        m = sig == s
        if 0 < y[m].sum() < m.sum():
            out.append(dict(task=task, feature=feature, variant=variant, signer=s, n=m.sum(), n_pos=y[m].sum(),
                            **dict(zip(["auroc", "lo", "hi"], boot_auc(y[m], sc[m])))))
    return out


def center_by(X, groups):
    X = X.copy()
    for g in np.unique(groups):
        X[groups == g] -= X[groups == g].mean(0)
    return X


def length_matched(y, signers, length, tol=10):
    """Within signer, pair each clip of the smaller class with the unused clip of nearest length (within tol frames)."""
    keep = []
    for s in np.unique(signers):
        a, b = (np.flatnonzero((signers == s) & (y == c)) for c in (1, 0))
        small, big = (a, list(b)) if len(a) <= len(b) else (b, list(a))
        for i in sorted(small, key=lambda i: length[i]):
            j = min(big, key=lambda j: abs(length[j] - length[i]))
            if abs(length[j] - length[i]) <= tol:
                keep += [i, j]
                big.remove(j)
    return np.array(keep)


def zsig(v, sig):
    v = np.where(np.isfinite(v), v, np.nan)
    return pd.Series(v).groupby(sig).transform(lambda x: (x - x.mean()) / x.std()).fillna(0).to_numpy()


def impute(X):
    X = X.copy()
    m = np.isnan(X)
    X[m] = np.take(np.nanmean(X, 0), np.nonzero(m)[1])
    return np.nan_to_num(X)
