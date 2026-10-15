"""Per-layer linear probes for six grammatical functions on SHuBERT states.
`cache` builds clip features and head pitch decoded per frame from each layer; `run` writes results/layer_probes.csv.
With `--states DIR` (encoder states with input streams masked) it caches and probes those states only."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402

from common import (CACHE, FEATS, LAYERS, NULL_FEATS, ROOT, TASKS, WORK, center_by, folds, length_matched,  # noqa: E402
                    loso_scores, rows_for, setup, shuffle_within, significance, utterances, zsig)


def cache():
    u = utterances()
    ok = [all((FEATS / p).exists() for p in (f"shubert/{i}.npy", f"head_pose/{i}.npz", f"dino_face/{i}_face.npy"))
          for i in u.dai_id]
    print(f"{sum(ok)}/{len(u)} utterances with features")
    u = u[ok].reset_index(drop=True)
    clip = {k: [] for k in ["shubert", "dino_face", "dino_hands", "mp_face", "pitch", "length"]}
    frames, target, fsig, fclip = [], [], [], []
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
        sel = np.isfinite(pitch[::2])
        frames.append(h[:, sel])
        target.append(pitch[::2][sel])
        fsig += [r.signer] * sel.sum()
        fclip += [r.dai_id] * sel.sum()
    print(f"{len(fclip)} frames with a face", flush=True)
    decoded = decode(frames, np.concatenate(target), np.array(fsig), np.array(fclip), u)
    np.savez(CACHE, dai_id=u.dai_id.to_numpy(), decoded_pitch=decoded,
             **{k: np.array(v, np.float32) for k, v in clip.items()})
    print(f"-> {CACHE}")


def decode(frames, target, sig, clip, u):
    """Per-frame ridge decoder of head pitch from each layer, leave-one-signer-out; clip mean of the prediction."""
    t = center_by(target[:, None], sig)[:, 0]
    out = []
    for k in range(LAYERS):
        F = center_by(np.concatenate([h[k] for h in frames]).astype(np.float32), sig)
        pred = np.zeros_like(t)
        for s in np.unique(sig):
            tr, te = sig != s, sig == s
            pred[te] = RidgeCV(alphas=np.logspace(-1, 5, 7)).fit(F[tr], t[tr]).predict(F[te])
        out.append(pd.Series(pred).groupby(clip).mean().reindex(u.dai_id).to_numpy())
        print(f"L{k} pitch decoded", flush=True)
    return np.array(out, np.float32).T


def cache_states(name):
    ids = np.load(CACHE)["dai_id"]
    X = np.stack([np.load(FEATS / name / f"{i}.npy").astype(np.float32).mean(1) for i in ids])
    np.savez(WORK / f"clip_features_{name}.npz", dai_id=ids, states=X)
    print(f"{len(X)} clips -> {WORK / f'clip_features_{name}.npz'}")


def probe_job(task, feature, variant, X, y, sig, folds, length):
    if variant == "centred":
        return rows_for(task, feature, variant, y, loso_scores(center_by(X, sig), y, sig, folds), sig, folds)
    if variant == "length-matched":
        f = np.isin(sig, folds)
        keep = np.flatnonzero(f)[length_matched(y[f], sig[f], length[f])]
        Xk, yk, sk = center_by(X[keep], sig[keep]), y[keep], sig[keep]
        return rows_for(task, feature, variant, yk, loso_scores(Xk, yk, sk, folds), sk, folds)
    if variant == "shuffled":
        ys = shuffle_within(y, sig)
        return rows_for(task, feature, variant, ys, loso_scores(center_by(X, sig), ys, sig, folds), sig, folds)[:1]
    raise ValueError(variant)


def run(procs, states=None):
    z, u, sig, feats = setup()
    out = ROOT / "results/layer_probes.csv"
    if states:
        s = np.load(WORK / f"clip_features_{states}.npz")
        assert (s["dai_id"] == z["dai_id"]).all()
        feats = {f"shubert_L{k}": s["states"][:, k] for k in range(LAYERS)}
        out = ROOT / f"results/layer_probes_{states.removeprefix('shubert_')}.csv"
    length = z["length"]
    jobs = []
    for t in TASKS:
        y = u[t].to_numpy()
        for f, X in feats.items():
            for v in ["centred", "length-matched"]:
                jobs.append(delayed(probe_job)(t, f, v, X, y, sig, folds(t), length))
        for f in NULL_FEATS:
            if f in feats:
                jobs.append(delayed(probe_job)(t, f, "shuffled", feats[f], y, sig, folds(t), length))
        if not states:
            untrained = {"pitch_mean": z["pitch"]}
            untrained |= {f"decoded_shubert_L{k}:pitch": z["decoded_pitch"][:, k] for k in range(LAYERS)}
            for name, v in untrained.items():
                jobs.append(delayed(rows_for)(t, name, "untrained, z within signer", y, zsig(v, sig), sig, folds(t)))
    print(f"{len(jobs)} jobs on {procs} processes", flush=True)
    df = pd.DataFrame([r for x in Parallel(n_jobs=procs, verbose=5)(jobs) for r in x])
    significance(df, "shuffled").round(3).to_csv(out, index=False)
    print(f"{len(df)} rows -> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "run"])
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--states", help="encoder states folder under the features directory")
    args = ap.parse_args()
    if args.cmd == "cache":
        cache_states(args.states) if args.states else cache()
    else:
        run(args.procs, args.states)


if __name__ == "__main__":
    main()
