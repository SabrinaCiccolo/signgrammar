"""Run one SHuBERT preprocessing stage (scripts of the SHuBERT repository) on all videos, in parallel processes."""
import argparse
import gzip
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

from common import FEATS, MODELS, SHUBERT_REPO, VIDEOS

BIG = str(10**8)  # the scripts' time limit in seconds: never reached


def stages():
    r = lambda *p: str(FEATS.joinpath(*p))
    clips = lambda: sorted(FEATS.glob("clips_bbox/*.mp4"))
    return {
        "bbox": ("dataset/clips_bbox.py", lambda: sorted(VIDEOS.glob("*.mp4")), [
            "--output_clips_directory", r("clips_bbox"), "--problem_file_path", r("problems_bbox.txt"),
            "--yolo_model_path", str(MODELS / "yolov8n.pt")]),
        "pose": ("dataset/kpe_mediapipe.py", clips, [
            "--pose_path", r("pose") + "/", "--stats_path", r("stats") + "/", "--time_limit", BIG,
            "--problem_file_path", r("problems_pose.txt"),
            "--face_model_path", str(MODELS / "face_landmarker_v2_with_blendshapes.task"),
            "--hand_model_path", str(MODELS / "hand_landmarker.task")]),
        "face": ("dataset/crop_face.py", clips, [
            "--time_limit", BIG, "--problem_file_path", r("problems_face.txt"), "--pose_path", r("pose"),
            "--face_path", r("face")]),
        "hands": ("dataset/crop_hands.py", clips, [
            "--time_limit", BIG, "--problem_file_path", r("problems_hands.txt"), "--pose_path", r("pose"),
            "--hand_path", r("hands")]),
        "body": ("features/body_features.py", lambda: sorted(FEATS.glob("pose/*_pose.json")), [
            "--time_limit", BIG, "--pose_features_path", r("body")]),
        "dino_face": ("features/dinov2_features.py", lambda: sorted(FEATS.glob("face/*_face.mp4")), [
            "--time_limit", BIG, "--output_folder", r("dino_face"), "--dino_path", str(MODELS / "dinov2face.pth")]),
        "dino_hands": ("features/dinov2_features.py", lambda: sorted(FEATS.glob("hands/*_hand[12].mp4")), [
            "--time_limit", BIG, "--output_folder", r("dino_hands"), "--dino_path", str(MODELS / "dinov2hand.pth")]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(stages()))
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()

    script, files, extra = stages()[args.stage]
    files = files()
    assert files, f"no input files for {args.stage}"
    for d in ("lists", "logs"):
        (FEATS / d).mkdir(parents=True, exist_ok=True)
    list_path = FEATS / "lists" / f"{args.stage}.list"
    with gzip.GzipFile(list_path, "wb") as f:
        f.write(pickle.dumps([str(p) for p in files], protocol=0))

    procs = min(args.procs, len(files))
    batch = math.ceil(len(files) / procs)
    n_jobs = math.ceil(len(files) / batch)
    threads = str(max(1, (os.cpu_count() * 3 // 4) // n_jobs))
    env = {**os.environ, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "OPENCV_FOR_THREADS_NUM": threads}
    print(f"{args.stage}: {len(files)} files, {n_jobs} processes of {batch}", flush=True)
    if "--output_folder" in extra:  # created here: parallel processes would race to create it
        Path(extra[extra.index("--output_folder") + 1]).mkdir(exist_ok=True)
    running = []
    for i in range(n_jobs):
        cmd = [sys.executable, str(SHUBERT_REPO / script), "--index", str(i), "--batch_size", str(batch),
               "--files_list", str(list_path), *extra]
        log = open(FEATS / "logs" / f"{args.stage}_{i}.log", "w")
        running.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=SHUBERT_REPO, env=env))
    codes = [p.wait() for p in running]
    print(f"{args.stage}: exit codes {sorted(set(codes))}; logs in {FEATS / 'logs'}")


if __name__ == "__main__":
    main()
