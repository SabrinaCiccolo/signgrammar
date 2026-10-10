# Yes/no questions on the face: what a sign language encoder and translator see

Probing SHuBERT, a public pretrained American Sign Language (ASL) encoder, for the grammar that ASL marks on the face
and head, and testing a published SHuBERT + ByT5 translator on the yes/no items of the ASL-MTP benchmark.

## Running

Requires Linux, an NVIDIA GPU and [uv](https://docs.astral.sh/uv/). Everything is written to `work/`.

```bash
git clone https://github.com/ShesterG/SHuBERT.git work/SHuBERT
git clone https://github.com/serpilkarabuklu/SL-Models-Analysis.git work/SL-Models-Analysis

uv venv -p 3.10 .venv && uv pip install -p .venv -r requirements.txt
uv venv -p 3.10 .venv-shubert && uv pip install -p .venv-shubert -r requirements-shubert.txt
uv pip install -p .venv-shubert --no-build-isolation -e work/SHuBERT/fairseq

.venv/bin/python -c "from huggingface_hub import hf_hub_download as get; [get('ShesterG/SHuBERT', 'models/' + f, local_dir='work') for f in ['checkpoint_836_400000.pt', 'dinov2face.pth', 'dinov2hand.pth', 'yolov8n.pt', 'face_landmarker_v2_with_blendshapes.task', 'hand_landmarker.task']]"
```

Then, from the repository root:

```bash
.venv/bin/python src/benchmark.py
.venv/bin/python src/corpus.py parse && .venv/bin/python src/corpus.py freeze && .venv/bin/python src/corpus.py download
for s in bbox pose face hands body dino_face dino_hands; do .venv/bin/python src/extract.py $s; done
.venv-shubert/bin/python src/shubert_states.py
.venv/bin/python src/head_pose.py
```

## Data and models

- **ASLLRP SignStream 3 corpus**, served by the Rutgers Data Access Interface (Neidle, Opoku, Metaxas,
  [arXiv 2201.07899](https://arxiv.org/abs/2201.07899)). It is for research and education use only and is
  [not redistributed](http://www.bu.edu/asllrp/dai-terms.html). `data/utterances.csv` holds only utterance ids and
  0/1 labels derived from the public annotations.
- **ASL-MTP** and the published SHuBERT + ByT5 outputs (Karabüklü et al., [arXiv 2604.27232](https://arxiv.org/abs/2604.27232),
  CC BY 4.0). `data/benchmark_items.csv` holds the item ids and clip spans.
- **SHuBERT** (Gueuwou et al., ACL 2025, [arXiv 2411.16765](https://arxiv.org/abs/2411.16765), MIT), used as
  released, without fine-tuning.
