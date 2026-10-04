"""ASLLRP SignStream 3 corpus on the public DAI site: caches every page of the unfiltered utterance search and parses
glosses and nonmanual tiers. One request at a time, with a delay; every page is cached."""
import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse

import requests

from common import CORPUS, WORK

PAGES = WORK / "cache/dai"
BASE = "https://dai.cs.rutgers.edu/dai/s/"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
SIGNERS = ("Ben", "Cory", "Jonathan", "Rachel", "P02", "P03")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    parse(ap.parse_args().delay)


if __name__ == "__main__":
    main()
