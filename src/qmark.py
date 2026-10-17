"""Share of published SHuBERT + ByT5 translations ending in "?" for yes/no questions with and without a manual
question sign (QMwg or YOU) and for declaratives, under input regions hidden at inference -> results/qmark.csv."""
import pandas as pd

from common import PUBLISHED, ROOT, YN_SIGNERS, utterances

CONDS = ["AC", "NF", "NH", "NHB"]  # all cues, no face, no hands, face only


def load(phen, cond, u):
    d = pd.read_json(PUBLISHED / f"asllrp_{phen}_AC_0_{cond}.json", lines=True)
    d["dai_id"] = d.input_clip.str.extract(r"/(\d+)_\d+_\d+\.mp4$")[0].astype(int)
    d = d.join(u, on="dai_id")
    d = d[d.signer.isin(YN_SIGNERS)]
    return d, d.hypothesis.str.strip().str.endswith("?").to_numpy()


def main():
    u = utterances().set_index("dai_id")
    rows = []
    for cond in CONDS:
        d, q = load("yes-no", cond, u)
        _, qd = load("yes-no-reversed", cond, u)
        manual = ((d.qm == 1) | (d.ix2p == 1)).to_numpy()
        rows.append(dict(condition=cond, n=len(d), n_manual=int(manual.sum()), rate_manual=q[manual].mean(),
                         rate_face_only=q[~manual].mean(), n_decl=len(qd), rate_decl=qd.mean()))
    df = pd.DataFrame(rows).round(3)
    df.to_csv(ROOT / "results/qmark.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
