"""Per-layer linear probes for six grammatical functions on SHuBERT states.
`cache` builds clip features; `run` writes results/layer_probes.csv."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from common import (CACHE, FEATS, NULL_FEATS, ROOT, TASKS, center_by, folds, length_matched,  # noqa: E402
                    loso_scores, rows_for, setup, utterances, zsig)


def cache():
    u = utterances()
    clip = {k: [] for k in ["shubert", "dino_face", "dino_hands", "mp_face", "pitch", "length"]}
    for r in u.itertuples():
        h = np.load(FEATS / "shubert" / f"{r.dai_id}.npy")
        d = np.load(FEATS / "dino_face" / f"{r.dai_id}_face.npy")[::2]
        z = np.load(FEATS / "head_pose" / f"{r.dai_id}.npz")
        pitch = z["pose"][:, 0]
        assert h.shape[1] == len(d) == len(pitch[::2]), (r.dai_id, h.shape, len(d), len(pitch))
        clip["shubert"].append(h.astype(np.float32).mean(1))
        clip["dino_face"].append(d.mean(0))
        clip["dino_hands"].append(np.concatenate([np.load(FEATS / "dino_hands" / f"{r.dai_id}_hand{k}.npy")[::2].mean(0)
                                                  for k in (1, 2)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clip["mp_face"].append(np.concatenate([np.nanmean(z["pose"], 0), np.nanstd(z["pose"], 0),
                                                   np.nanmean(z["blend"], 0) if z["blend"].size else np.full(52, np.nan)]))
            clip["pitch"].append(np.nanmean(pitch))
        clip["length"].append(len(pitch))
    np.savez(CACHE, dai_id=u.dai_id.to_numpy(), **{k: np.array(v, np.float32) for k, v in clip.items()})
    print(f"-> {CACHE}")


def probe_job(task, feature, variant, X, y, sig, folds, length):
    if variant == "centred":
        return rows_for(task, feature, variant, y, loso_scores(center_by(X, sig), y, sig, folds), sig, folds)
    if variant == "length-matched":
        f = np.isin(sig, folds)
        keep = np.flatnonzero(f)[length_matched(y[f], sig[f], length[f])]
        Xk, yk, sk = center_by(X[keep], sig[keep]), y[keep], sig[keep]
        return rows_for(task, feature, variant, yk, loso_scores(Xk, yk, sk, folds), sk, folds)
    if variant == "shuffled":
        rng = np.random.default_rng(0)
        ys = y.copy()
        for s in np.unique(sig):
            ys[sig == s] = rng.permutation(y[sig == s])
        return rows_for(task, feature, variant, ys, loso_scores(center_by(X, sig), ys, sig, folds), sig, folds)[:1]
    raise ValueError(variant)


def run(procs):
    z, u, sig, feats = setup()
    out = ROOT / "results/layer_probes.csv"
    length = z["length"]
    jobs = []
    for t in TASKS:
        y = u[t].to_numpy()
        for f, X in feats.items():
            for v in ["centred", "length-matched"]:
                jobs.append(delayed(probe_job)(t, f, v, X, y, sig, folds(t), length))
        for f in NULL_FEATS:
            jobs.append(delayed(probe_job)(t, f, "shuffled", feats[f], y, sig, folds(t), length))
        jobs.append(delayed(rows_for)(t, "pitch_mean", "untrained, z within signer", y, zsig(z["pitch"], sig), sig,
                                      folds(t)))
    print(f"{len(jobs)} jobs on {procs} processes", flush=True)
    df = pd.DataFrame([r for x in Parallel(n_jobs=procs, verbose=5)(jobs) for r in x])
    null = df[df.variant == "shuffled"].groupby("task").hi.max()
    df["null_hi"] = df.task.map(null).to_numpy()
    untrained = df.variant.str.startswith("untrained")
    df["significant"] = pd.Series(np.where(untrained, (df.lo > 0.5) | (df.hi < 0.5),
                                           df.lo > np.maximum(0.5, df.null_hi)), dtype=object)
    df.loc[(df.signer != "pooled") | (df.variant == "shuffled"), "significant"] = np.nan
    df.round(3).to_csv(out, index=False)
    print(f"{len(df)} rows -> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "run"])
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    cache() if args.cmd == "cache" else run(args.procs)


if __name__ == "__main__":
    main()
