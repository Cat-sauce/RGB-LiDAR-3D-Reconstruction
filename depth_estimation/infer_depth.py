import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import pipeline

def run_depth_anything():
    rgb_path = Path("data/raw/sample_frame/rgb.png")
    out_dir = Path("outputs/depth")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Depth Anything V2 pipeline (Small model)...")
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cpu")
    
    print(f"Processing {rgb_path}...")
    image = Image.open(rgb_path).convert("RGB")
    result = pipe(image)
    
    # Depth Anything outputs relative depth / disparity (higher = closer or vice versa depending on head)
    depth_pred = np.array(result["depth"], dtype=np.float32)
    
    # Invert so larger value = greater distance (depth convention)
    # Depth Anything output represents disparity (higher = closer)
    # Standard inversion: relative_depth = 1.0 / (disparity + 1e-4)
    d_min, d_max = depth_pred.min(), depth_pred.max()
    norm_disp = (depth_pred - d_min) / (d_max - d_min + 1e-8)
    rel_depth = 1.0 / (norm_disp + 0.05)
    
    # Save raw relative depth
    np.save(out_dir / "relative_depth.npy", rel_depth)
    
    # Save normalized grayscale visualization
    vis_norm = ((rel_depth - rel_depth.min()) / (rel_depth.max() - rel_depth.min()) * 255.0).astype(np.uint8)
    cv2.imwrite(str(out_dir / "relative_depth_vis.png"), vis_norm)
    
    print(f"Relative depth map saved to {out_dir / 'relative_depth.npy'}")
    print(f"Visualization saved to {out_dir / 'relative_depth_vis.png'}")

if __name__ == "__main__":
    run_depth_anything()
