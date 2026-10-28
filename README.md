# Yes/no questions on the face: what a sign language encoder and translator see

In American Sign Language (ASL) a yes/no question is often marked only by the face and head. This project probes
**SHuBERT**, a public pretrained ASL encoder, for six grammatical functions on 2,127 annotated utterances of the ASLLRP
corpus, and tests a published **SHuBERT + ByT5** translator on the yes/no items of the **ASL-MTP** benchmark.
In short, the translator does not read the face, but the encoder keeps the yes/no cue in every layer, mostly as head
pitch (the forward tilt of the head). That one measured number repairs the translator's yes/no choice: balanced
accuracy goes from 0.52 to 0.85.

## Running

Requires Linux, an NVIDIA GPU, [uv](https://docs.astral.sh/uv/) and about 25 GB of disk. Everything is written to
`work/` (or to `$SIGNGRAMMAR_WORK`).

```bash
git clone https://github.com/ShesterG/SHuBERT.git work/SHuBERT
git clone https://github.com/serpilkarabuklu/SL-Models-Analysis.git work/SL-Models-Analysis

uv venv -p 3.10 .venv && uv pip install -p .venv -r requirements.txt
uv venv -p 3.10 .venv-shubert && uv pip install -p .venv-shubert -r requirements-shubert.txt
uv pip install -p .venv-shubert --no-build-isolation -e work/SHuBERT/fairseq

.venv/bin/python -c "from huggingface_hub import hf_hub_download as get; [get('ShesterG/SHuBERT', 'models/' + f, local_dir='work') for f in ['checkpoint_836_400000.pt', 'dinov2face.pth', 'dinov2hand.pth', 'yolov8n.pt', 'face_landmarker_v2_with_blendshapes.task', 'hand_landmarker.task']]"

./run.sh
```

`run.sh` rebuilds the tables in `data/`, downloads the 2,127 videos from the public DAI site (one request at a time),
extracts the features, runs the analyses into `results/` and draws `figures/`. To redraw the figures from the
included results only: `.venv/bin/python src/figures.py`. MediaPipe body landmarks vary slightly between runs, so a
full rerun from the videos matches the included results closely but not to the last digit.

## Data and models

- **ASLLRP SignStream 3 corpus**, served by the Rutgers Data Access Interface (Neidle, Opoku, Metaxas,
  [arXiv 2201.07899](https://arxiv.org/abs/2201.07899)). It is for research and education use only and is
  [not redistributed](http://www.bu.edu/asllrp/dai-terms.html). `data/utterances.csv` holds only utterance ids and
  0/1 labels derived from the public annotations.
- **ASL-MTP** and the published SHuBERT + ByT5 outputs (Karabüklü et al., [arXiv 2604.27232](https://arxiv.org/abs/2604.27232),
  CC BY 4.0). `data/benchmark_items.csv` holds the item ids and clip spans.
- **SHuBERT** (Gueuwou et al., ACL 2025, [arXiv 2411.16765](https://arxiv.org/abs/2411.16765), MIT), used as
  released, without fine-tuning.
