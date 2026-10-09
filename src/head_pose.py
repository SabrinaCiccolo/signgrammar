"""3D head pose (pitch, yaw, roll from the MediaPipe facial transformation matrix) and the 52 face blendshapes per
frame of each signer crop -> <features>/head_pose/<id>.npz."""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from common import FEATS, MODELS


def euler(m):
    """pitch, yaw, roll in degrees from the 4x4 face transformation matrix (x-y-z intrinsic)."""
    r = m[:3, :3]
    return np.degrees([np.arctan2(r[2, 1], r[2, 2]), np.arcsin(-np.clip(r[2, 0], -1, 1)), np.arctan2(r[1, 0], r[0, 0])])


def one(args):
    video, out = args
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if out.exists():
        return 0
    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker_v2_with_blendshapes.task")),
        num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    cap = cv2.VideoCapture(str(video))
    pose, blend, names = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if r.face_blendshapes:
            names = [c.category_name for c in r.face_blendshapes[0]]
            pose.append(euler(np.array(r.facial_transformation_matrixes[0])))
            blend.append([c.score for c in r.face_blendshapes[0]])
        else:
            pose.append([np.nan] * 3)
            blend.append([np.nan] * 52)
    cap.release()
    fl.close()
    np.savez(out, pose=np.array(pose, np.float32), blend=np.array(blend, np.float32), names=np.array(names or []))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    out = FEATS / "head_pose"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(v, out / f"{v.stem}.npz") for v in sorted((FEATS / "clips_bbox").glob("*.mp4"))]
    os.environ["OMP_NUM_THREADS"] = "1"
    with Pool(args.procs) as pool:
        done = sum(pool.imap_unordered(one, jobs, chunksize=4))
    print(f"head_pose: {done} clips written, {len(jobs) - done} already present -> {out}")


if __name__ == "__main__":
    main()
