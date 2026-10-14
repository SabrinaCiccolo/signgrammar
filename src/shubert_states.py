"""Per-layer SHuBERT states for every clip: [13, T, 768] float16, layer 0 = fused input projection. Frames are
subsampled 2x (29.97 -> about 15 fps, as in pretraining). `--drop` replaces input streams with the learned mask
embedding after their projection."""
import argparse
import sys

import numpy as np
import torch

from common import FEATS, MODELS, SHUBERT_REPO

sys.path.insert(0, str(SHUBERT_REPO / "fairseq"))
from examples.shubert.models.shubert import SHubertConfig, SHubertModel  # noqa: E402

STREAMS = {"face": ("dino_face", "face"), "left_hand": ("dino_hands", "hand1"),
           "right_hand": ("dino_hands", "hand2"), "body_posture": ("body", "pose")}


def load_model(ckpt):
    model = SHubertModel(SHubertConfig())
    res = model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"], strict=False)
    assert not res.missing_keys and not res.unexpected_keys, res
    return model.eval().cuda()


def mask_hooks(model, drop):
    return [getattr(model, f"{k}_proj").register_forward_hook(lambda m, i, o: model.mask_emb[0].expand_as(o))
            for k in drop]


@torch.no_grad()
def layer_states(model, streams):
    T = len(streams["face"])
    sample = {k: torch.from_numpy(v).float().cuda() for k, v in streams.items()}
    sample.update({f"label_{k}": torch.zeros((T, 1)).cuda() for k in streams})
    captured = {}
    hook = model.post_extract_proj.register_forward_hook(lambda m, i, o: captured.update(x0=o))
    res = model.extract_features([sample], padding_mask=None, kmeans_labels=None, mask=False)
    hook.remove()
    # layer_results[i] = (x, attn, ffn_out); x is the layer output
    states = [captured["x0"][0]] + [lr[0][:, 0] for lr in res["layer_results"]]
    return torch.stack(states).cpu().half().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="shubert", help="output folder under the features directory")
    ap.add_argument("--drop", nargs="+", default=[], choices=list(STREAMS), help="streams to mask")
    args = ap.parse_args()
    out = FEATS / args.out
    out.mkdir(exist_ok=True)
    model = load_model(MODELS / "checkpoint_836_400000.pt")
    mask_hooks(model, args.drop)
    ids = sorted(p.name[: -len("_face.npy")] for p in (FEATS / "dino_face").glob("*_face.npy"))
    done = skipped = 0
    for i in ids:
        dst = out / f"{i}.npy"
        paths = {k: FEATS / d / f"{i}_{s}.npy" for k, (d, s) in STREAMS.items()}
        if dst.exists() or not all(p.exists() for p in paths.values()):
            skipped += 1
            continue
        streams = {k: np.load(p)[::2] for k, p in paths.items()}
        assert len({len(v) for v in streams.values()}) == 1, (i, {k: v.shape for k, v in streams.items()})
        np.save(dst, layer_states(model, streams))
        done += 1
    print(f"shubert: {done} clips written, {skipped} skipped (done or missing inputs) -> {out}")


if __name__ == "__main__":
    main()
