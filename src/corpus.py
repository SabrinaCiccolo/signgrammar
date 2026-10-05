"""ASLLRP SignStream 3 corpus on the public DAI site: `fetch` caches every page of the unfiltered utterance search,
`parse` reads glosses and nonmanual tiers, `freeze` writes the labelled utterance table, `download` fetches the front
videos. One request at a time, with a delay; every page is cached."""
import argparse
import csv
import hashlib
import html
import json
import re
import time
import urllib.parse

import requests

from common import CORPUS, ITEMS, UTTERANCES, VIDEOS, WORK

PAGES = WORK / "cache/dai"
BASE = "https://dai.cs.rutgers.edu/dai/s/"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
SIGNERS = ("Ben", "Cory", "Jonathan", "Rachel", "P02", "P03")
ANNOTATED = ("Ben", "Cory", "Jonathan", "Rachel")  # P02 and P03 carry almost no grammatical tiers
GRAMMAR = {"yn": "y/n question", "wh": "wh question", "neg": "negative", "cond": "cond/when",
           "topic": "topic/focus", "rhq": "rhq question"}


def get(url, delay):
    PAGES.mkdir(parents=True, exist_ok=True)
    p = PAGES / (hashlib.sha1(url.encode()).hexdigest() + ".html")
    if p.exists():
        return p.read_text(errors="replace"), False
    r = requests.get(url, headers={"User-Agent": UA}, timeout=120)
    time.sleep(delay)
    r.raise_for_status()
    p.write_bytes(r.content)
    return r.text, True


def search_url(form, page):
    """Default (checked and hidden) inputs of the search form: all data sources, all participants."""
    form = re.sub(r"<script.*?</script>", "", form, flags=re.S)
    pairs = [("page_num", str(page))]
    for s in re.findall(r"<input[^>]*>", form[form.find("<form"):]):
        n, v = re.search(r'name="([^"]+)"', s), re.search(r'value="([^"]*)"', s)
        if n and v and ('type="hidden"' in s or "checked" in s):
            pairs.append((n.group(1), v.group(1)))
    return BASE + "utterancesearchresult?" + urllib.parse.urlencode(pairs)


def pages(delay):
    form, _ = get(BASE + "utterancesearch", delay)
    page = 1
    while True:
        text, new = get(search_url(form, page), delay)
        last = max(map(int, re.findall(r"page_num=(\d+)", text)))
        if new and (page % 10 == 0 or page == last):
            print(f"page {page}/{last}", flush=True)
        yield text
        if page >= last:
            break
        page += 1


def spans(rest):
    """Underline spans [first col, last col, value]."""
    return [[m.start(), m.end() - 1, m.group(0).lstrip("_")] for m in re.finditer(r"\S+", rest)]


def parse_pre(pre):
    text = html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<img[^>]*>", "#", pre)))
    lines = text.split("\n")
    col = next(l.index("dh:") + 2 for l in lines if re.match(r"\s*dh:", l))
    lines = [l for l in lines if len(l) > col and l[col] == ":"]
    start = col + 2
    tiers, gloss = {}, {}
    for l in lines:
        name, rest = l[:start - 2].strip(), l[start:].rstrip()
        if name in ("dh", "ndh", "engl trans"):
            gloss[name] = rest if name != "engl trans" else rest.strip()
        elif name not in ("dh shape", "ndh shape"):
            tiers[name] = spans(rest)
    return gloss, tiers


def parse_page(text):
    rows = []
    marks = [m.start() for m in re.finditer(r'id="ds_\d+"', text)] + [len(text)]
    for a, b in zip(marks, marks[1:]):
        seg = text[a:b]
        n = re.match(r'id="ds_(\d+)"', seg).group(1)
        name = re.sub(r"<[^>]+>", "", seg[seg.find(">") + 1:seg.find("</td>")]).split()[-1]
        full = re.search(rf'id="fg_{n}"[^>]*>\s*<pre>(.*?)</pre>', seg, re.S)
        vid = re.search(rf'id="uv_{n}"[^>]*>.*?ss3front/(\d+)\.mp4', seg, re.S)
        gloss, tiers = parse_pre(full.group(1))
        signer = [s for s in SIGNERS if s in name]
        rows.append(dict(name=name, signer=signer[0] if len(signer) == 1 else None,
                         dai_id=int(vid.group(1)) if vid else None, **gloss, tiers=tiers))
    return rows


def parse(delay):
    rows = [r for text in pages(delay) for r in parse_page(text)]
    assert len({r["name"] for r in rows}) == len(rows), "duplicate utterance names across pages"
    with open(CORPUS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} utterances -> {CORPUS}")


def labels(r):
    t = r["tiers"]
    lab = {k: tier in t for k, tier in GRAMMAR.items()}
    lab["plain"] = not any(lab.values())
    hands = " ".join(v for v in (r.get("dh"), r.get("ndh")) if isinstance(v, str))
    lab["ix2p"] = "IX-2p" in hands  # the sign YOU
    lab["qm"] = "QMwg" in hands  # manual question mark (wiggle)
    lab["fs"] = "fs-" in hands  # fingerspelling
    return lab


def freeze():
    """Utterances of the annotated signers with 0/1 labels; in_benchmark = clip of ASL-MTP."""
    import pandas as pd

    df = pd.read_json(CORPUS, lines=True)
    df = df[df.signer.isin(ANNOTATED)].reset_index(drop=True)
    lab = pd.DataFrame([labels(r) for r in df.to_dict("records")]).astype(int)
    bench = pd.read_csv(ITEMS)
    out = pd.DataFrame(dict(name=df.name, signer=df.signer, scene=df.name.str.replace(r"-U-\d+$", "", regex=True),
                            dai_id=df.dai_id, in_benchmark=df.dai_id.isin(bench.dai_id).astype(int)))
    out = pd.concat([out, lab], axis=1).sort_values(["signer", "name"])
    out.to_csv(UTTERANCES, index=False)
    print(f"{len(out)} utterances ({out.in_benchmark.sum()} in ASL-MTP) -> {UTTERANCES}")


def download(delay):
    import cv2
    import pandas as pd

    VIDEOS.mkdir(parents=True, exist_ok=True)
    log_path = WORK / "download.csv"
    new_log = not log_path.exists()
    log = open(log_path, "a", newline="")
    w = csv.writer(log)
    if new_log:
        w.writerow(["dai_id", "status", "bytes", "frames", "fps", "width", "height"])
    s = requests.Session()
    s.headers["User-Agent"] = UA
    n_get = n_skip = n_fail = 0
    for i in pd.read_csv(UTTERANCES).dai_id:
        path = VIDEOS / f"{i}.mp4"
        if path.exists():
            n_skip += 1
            continue
        r = s.get(f"https://dai.cs.rutgers.edu//ss3front/{i}.mp4", timeout=60)
        time.sleep(delay)
        if r.status_code != 200 or not r.headers.get("Content-Type", "").startswith("video"):
            n_fail += 1
            w.writerow([i, r.status_code, len(r.content), "", "", "", ""])
            log.flush()
            continue
        tmp = path.with_suffix(".part")
        tmp.write_bytes(r.content)
        cap = cv2.VideoCapture(str(tmp))
        meta = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), round(cap.get(cv2.CAP_PROP_FPS), 2),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        cap.release()
        tmp.rename(path)
        n_get += 1
        w.writerow([i, 200, len(r.content), *meta])
        log.flush()
        if n_get % 100 == 0:
            print(f"downloaded {n_get}, skipped {n_skip}, failed {n_fail}", flush=True)
    print(f"done: downloaded {n_get}, skipped {n_skip} (already present), failed {n_fail}; log {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parse", "freeze", "download"])
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    args = ap.parse_args()
    {"parse": lambda: parse(args.delay), "freeze": freeze, "download": lambda: download(args.delay)}[args.cmd]()


if __name__ == "__main__":
    main()
