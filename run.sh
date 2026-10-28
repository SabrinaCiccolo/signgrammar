#!/usr/bin/env bash
# Full pipeline: data tables, videos, features, analyses, figures. Every step skips outputs already present.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
SPY=${SHUBERT_PY:-.venv-shubert/bin/python}
quiet() { grep -v -E "^(W0000|I0000)" || true; }

$PY src/benchmark.py
$PY src/corpus.py parse
$PY src/corpus.py freeze
$PY src/corpus.py download

for s in bbox pose face hands body dino_face dino_hands; do
  case $s in bbox|dino_*) p=6 ;; *) p=16 ;; esac
  $PY src/extract.py $s --procs $p
done
$SPY src/shubert_states.py
$SPY src/shubert_states.py --out shubert_only_face --drop left_hand right_hand body_posture
$PY src/head_pose.py 2>&1 | quiet
$PY src/head_pose.py --scale 3 --out head_pose_lowres 2>&1 | quiet

$PY src/layer_probes.py cache
$PY src/layer_probes.py run
$PY src/layer_probes.py cache --states shubert_only_face
$PY src/layer_probes.py run --states shubert_only_face
$PY src/residual.py
$PY src/repair.py
$PY src/lowres.py
$PY src/qmark.py
$PY src/clip_pitch.py
$PY src/figures.py
$PY src/animation.py 2>&1 | quiet
