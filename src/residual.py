"""Probes on layer features after removing, within each signer, their linear fit on the measured face and head gesture
(MediaPipe pose and 52 blendshapes, plus clip length) -> results/residual_probes.csv."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from common import (LAYERS, NULL_FEATS, ROOT, TASKS, center_by, folds, loso_scores, rows_for, setup,  # noqa: E402
                    shuffle_within, significance)


def within(X, M, sig):
    """X minus its least-squares fit (with intercept) on M, separately within each signer."""
    out = np.zeros_like(X, dtype=float)
    for s in np.unique(sig):
        k = sig == s
        A = np.c_[M[k], np.ones(k.sum())]
        out[k] = X[k] - A @ np.linalg.lstsq(A, X[k], rcond=None)[0]
    return out


def res_probe(task, feature, X, M, y, sig, shuffled):
    if shuffled:
        y = shuffle_within(y, sig)
    variant = "shuffled" if shuffled else "residual"
    rows = rows_for(task, feature, variant, y, loso_scores(within(X, M, sig), y, sig), sig, np.unique(sig))
    return rows[:1] if shuffled else rows


def main(procs):
    z, u, sig0, feats0 = setup()
    feats0 = {k: feats0[k] for k in [f"shubert_L{i}" for i in range(LAYERS)] + ["dino_face"]}
    jobs = []
    for t in TASKS:
        m = np.isin(sig0, folds(t)) & np.isfinite(z["pitch"])
        y, sig = u[t].to_numpy()[m], sig0[m]
        M = center_by(np.c_[z["mp_face"], z["length"]][m], sig)
        feats = {k: X[m] for k, X in feats0.items()}
        jobs += [delayed(res_probe)(t, f, X, M, y, sig, False) for f, X in feats.items()]
        jobs += [delayed(res_probe)(t, f, feats[f], M, y, sig, True) for f in NULL_FEATS]
    print(f"{len(jobs)} jobs", flush=True)
    df = pd.DataFrame([r for x in Parallel(n_jobs=procs, verbose=5)(jobs) for r in x])
    out = ROOT / "results/residual_probes.csv"
    significance(df, "shuffled").round(3).to_csv(out, index=False)
    print(f"{len(df)} rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    main(ap.parse_args().procs)
