import numpy as np
import os

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    R = np.array([
        [1-2*y**2-2*z**2,   2*x*y-2*z*w,     2*x*z+2*y*w],
        [2*x*y+2*z*w,       1-2*x**2-2*z**2, 2*y*z-2*x*w],
        [2*x*z-2*y*w,       2*y*z+2*x*w,     1-2*x**2-2*y**2]
    ])
    return R

images_txt = "clean_frames/sparse_txt/images.txt"
poses = []

with open(images_txt, "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if line.startswith("#") or len(line) == 0:
        continue
    elems = line.split()
    # Only process lines with at least 10 columns (ID, quat, trans, cam, name)
    if len(elems) < 10:
        continue
    try:
        qw, qx, qy, qz = map(float, elems[1:5])
        tx, ty, tz = map(float, elems[5:8])
    except ValueError:
        continue
    R = qvec2rotmat([qw, qx, qy, qz])
    t = np.array([tx, ty, tz])
    near, far = 0.1, 5.0
    poses.append(np.hstack([R.flatten(), t, near, far]))

if len(poses) == 0:
    raise RuntimeError("No valid images found in images.txt. Check your COLMAP sparse output!")

poses_bounds = np.stack(poses)
np.save(os.path.join("clean_frames", "poses_bounds.npy"), poses_bounds)
print("Saved poses_bounds.npy with shape:", poses_bounds.shape)
