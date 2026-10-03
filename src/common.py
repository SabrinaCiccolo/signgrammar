"""Paths shared by the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
VIDEOS = WORK / "videos"
FEATS = WORK / "features"
MODELS = WORK / "models"
SHUBERT_REPO = WORK / "SHuBERT"
PUBLISHED = WORK / "SL-Models-Analysis/data/results/ASLLRP_WeightedSum"
CORPUS = WORK / "corpus.jsonl"
CACHE = WORK / "clip_features.npz"
UTTERANCES = ROOT / "data/utterances.csv"
ITEMS = ROOT / "data/benchmark_items.csv"
