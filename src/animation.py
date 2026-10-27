"""Animated landmarks (MediaPipe face contours, hands, upper body; no video pixels) of one yes/no clip next to its head
pitch, measured and decoded from SHuBERT layer 12 -> figures/yn_clip.gif."""
import argparse
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from mediapipe.python.solutions import face_mesh_connections as fmc  # noqa: E402
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS  # noqa: E402

from common import FEATS, ROOT  # noqa: E402

INK, MUTED, GRID, SPAN = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"
FACE = [fmc.FACEMESH_FACE_OVAL, fmc.FACEMESH_LIPS, fmc.FACEMESH_LEFT_EYE, fmc.FACEMESH_RIGHT_EYE,
        fmc.FACEMESH_LEFT_EYEBROW, fmc.FACEMESH_RIGHT_EYEBROW, fmc.FACEMESH_NOSE]
FACE = sorted({e for c in FACE for e in c})
BODY = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]  # shoulders, arms


def landmarks(dai_id):
    d = json.load(open(FEATS / "pose" / f"{dai_id}_pose.json"))
    cap = cv2.VideoCapture(str(FEATS / "clips_bbox" / f"{dai_id}.mp4"))
    wh = np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)])
    cap.release()
    xy = lambda pts: np.array(pts)[:, :2] * wh
    return [{k: [xy(p) for p in f[k]] if f and f.get(k) else [] for k in ("face_landmarks", "hand_landmarks",
                                                                          "pose_landmarks")}
            for f in (d[str(i)] for i in range(len(d)))], wh


def segments(points, edges):
    return [[points[a], points[b]] for a, b in edges]


def main(signer):
    c = pd.read_csv(ROOT / "results/clip_pitch.csv")
    c = c[c.signer == signer].reset_index(drop=True)
    dai_id, dur = c.dai_id.iloc[0], c.n.iloc[0] / 29.97
    frames, wh = landmarks(dai_id)

    fig = plt.figure(figsize=(9, 3.3), dpi=100)
    ax = fig.add_axes([0.0, 0.02, 0.26, 0.86])
    ax.set_xlim(0, wh[0])
    ax.set_ylim(wh[1] * 0.85, 0)
    zoom = fig.add_axes([0.26, 0.12, 0.2, 0.7])
    pts = np.concatenate([f["face_landmarks"][0] for f in frames if f["face_landmarks"]])
    centre, half = np.median(pts, 0), 0.75 * (np.percentile(pts, 99, 0) - np.percentile(pts, 1, 0)).max()
    zoom.set_xlim(centre[0] - half, centre[0] + half)
    zoom.set_ylim(centre[1] + half, centre[1] - half)
    zoom.set_title("face, enlarged", color=MUTED, fontsize=8)
    face = matplotlib.collections.LineCollection([], colors=INK, linewidths=0.9)
    face_zoom = matplotlib.collections.LineCollection([], colors=INK, linewidths=1.3)
    hands = matplotlib.collections.LineCollection([], colors="#2a78d6", linewidths=1.2)
    body = matplotlib.collections.LineCollection([], colors=MUTED, linewidths=2.2)
    for lc in (body, hands, face):
        ax.add_collection(lc)
    zoom.add_collection(face_zoom)
    for a in (ax, zoom):
        a.set_aspect("equal")
        a.axis("off")

    p = fig.add_axes([0.54, 0.17, 0.44, 0.65])
    p.axvspan(c.start.iloc[0] * dur, c.end.iloc[0] * dur, color=SPAN, lw=0, zorder=0)
    p.axhline(0, color=GRID, lw=0.8, zorder=0)
    p.plot(c.time, c.measured, color=INK, lw=2, label="measured (MediaPipe)")
    p.plot(c.time, c.L12, color="#2a78d6", lw=1.6, ls=(0, (1, 1.2)), label="decoded from SHuBERT layer 12")
    cursor = p.axvline(0, color="#eb6834", lw=1.5)
    p.set_xlim(0, dur)
    p.set_xlabel("time (s)", color=INK, fontsize=8.5)
    p.set_ylabel("head pitch (z; up = forward)", color=INK, fontsize=8.5)
    for s in ["top", "right"]:
        p.spines[s].set_visible(False)
    p.tick_params(colors=MUTED, labelsize=7.5)
    p.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=INK)
    p.text(c.start.iloc[0] * dur + 0.05, p.get_ylim()[0] + 0.1, "question", color=MUTED, fontsize=7.5, va="bottom")
    fig.text(0.02, 0.94, f"{signer}, yes/no question: the head tilts forward and SHuBERT tracks it", color=INK,
             fontsize=10)

    def draw(k):
        f = frames[min(2 * k, len(frames) - 1)]
        face.set_segments([s for pts in f["face_landmarks"] for s in segments(pts, FACE)])
        face_zoom.set_segments(face.get_segments())
        hands.set_segments([s for pts in f["hand_landmarks"] for s in segments(pts, HAND_CONNECTIONS)])
        body.set_segments([s for pts in f["pose_landmarks"] for s in segments(pts, BODY)])
        cursor.set_xdata([k * 2 / 29.97] * 2)
        return face, face_zoom, hands, body, cursor

    anim = FuncAnimation(fig, draw, frames=(len(frames) + 1) // 2, blit=True)
    out = ROOT / "figures/yn_clip.gif"
    anim.save(out, writer=PillowWriter(fps=15))
    print(f"-> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer", default="Jonathan")
    main(ap.parse_args().signer)
