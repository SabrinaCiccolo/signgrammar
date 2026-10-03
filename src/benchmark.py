"""ASL-MTP items (Karabüklü et al.) with their ASLLRP clip id and frame span -> data/benchmark_items.csv."""
import pandas as pd

from common import ITEMS, PUBLISHED

SIGNERS = ("Cory", "Rachel", "Jonathan", "Ben")


def signer(scene):
    found = [s for s in SIGNERS if s in scene]
    assert len(found) == 1, scene
    return found[0]


def main():
    rows = []
    for p in sorted(PUBLISHED.glob("asllrp_*_AC_0_AC.json")):
        df = pd.read_json(p, lines=True)
        clip = df.input_clip.str.extract(r"/(\d+)_(\d+)_(\d+)\.mp4$").astype(int)
        rows.append(pd.DataFrame(dict(phenomenon=df.phenomenon, utterance_id=df.utterance_id, scene=df.video_id,
                                      signer=df.video_id.map(signer), dai_id=clip[0], start=clip[1], end=clip[2])))
    items = pd.concat(rows).sort_values(["phenomenon", "utterance_id"])
    items.to_csv(ITEMS, index=False)
    print(f"{len(items)} items, {items.dai_id.nunique()} clips -> {ITEMS}")


if __name__ == "__main__":
    main()
