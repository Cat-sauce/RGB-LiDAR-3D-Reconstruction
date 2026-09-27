import urllib.request
import numpy as np
import cv2
import json
from pathlib import Path

out_dir = Path("data/raw/sample_frame")
out_dir.mkdir(parents=True, exist_ok=True)

rgb_url = "https://raw.githubusercontent.com/isl-org/Open3D/master/examples/test_data/RGBD/color/00000.jpg"
depth_url = "https://raw.githubusercontent.com/isl-org/Open3D/master/examples/test_data/RGBD/depth/00000.png"

rgb_path = out_dir / "rgb.png"
depth_png_path = out_dir / "depth.png"

print("Downloading standard RGB-D benchmark frame...")
urllib.request.urlretrieve(rgb_url, rgb_path)
urllib.request.urlretrieve(depth_url, depth_png_path)

raw_depth = cv2.imread(str(depth_png_path), cv2.IMREAD_ANYDEPTH)
metric_depth = raw_depth.astype(np.float32) / 1000.0
np.save(str(out_dir / "gt_depth.npy"), metric_depth)

h, w = metric_depth.shape
fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5
intrinsics = {"width": w, "height": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]}

with open(out_dir / "intrinsics.json", "w") as f:
    json.dump(intrinsics, f, indent=4)

valid_pts = metric_depth[metric_depth != 0]
print("Benchmark frame prepared successfully at:", out_dir)
print("Resolution:", w, "x", h)
print("Min valid depth (m):", float(valid_pts.min()))
print("Max depth (m):", float(metric_depth.max()))