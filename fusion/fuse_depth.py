import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

def align_scale_shift(rel_depth, sparse_depth):
    mask = sparse_depth > 0
    if np.sum(mask) < 4:
        return rel_depth
    
    x = rel_depth[mask]
    y = sparse_depth[mask]
    
    # Solve linear system: s * x + t = y
    A = np.vstack([x, np.ones_like(x)]).T
    s, t = np.linalg.lstsq(A, y, rcond=None)[0]
    
    aligned = s * rel_depth + t
    # Clamp negative or zero depth values to valid range
    aligned = np.clip(aligned, a_min=0.1, a_max=10.0)
    return aligned

def quality_weighted_fusion(aligned_pred, sparse_depth, ksize=15, sigma=5.0):
    dense_fused = aligned_pred.copy()
    mask = (sparse_depth > 0).astype(np.float32)
    
    # Residual at sparse points
    residual = np.zeros_like(sparse_depth)
    residual[mask > 0] = sparse_depth[mask > 0] - aligned_pred[mask > 0]
    
    # Local quality weighting via Gaussian-smoothed residual propagation
    weights = cv2.GaussianBlur(mask, (ksize, ksize), sigma)
    smooth_residual = cv2.GaussianBlur(residual, (ksize, ksize), sigma)
    
    valid_w = weights > 1e-4
    dense_correction = np.zeros_like(aligned_pred)
    dense_correction[valid_w] = smooth_residual[valid_w] / weights[valid_w]
    
    fused = aligned_pred + dense_correction
    # Preserve exact metric measurements at known sparse LiDAR locations
    fused[mask > 0] = sparse_depth[mask > 0]
    return np.clip(fused, a_min=0.1, a_max=10.0)

def process_all_sparsities():
    rel_depth_path = Path("outputs/depth/relative_depth.npy")
    rel_depth = np.load(rel_depth_path)
    
    out_dir = Path("outputs/depth")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sparsities = [50, 25, 10, 5]
    for pct in sparsities:
        sparse_file = Path(f"data/sparse/{pct}pct/sparse_depth.npy")
        sparse_depth = np.load(sparse_file)
        
        aligned = align_scale_shift(rel_depth, sparse_depth)
        fused = quality_weighted_fusion(aligned, sparse_depth)
        
        np.save(out_dir / f"aligned_{pct}pct.npy", aligned)
        np.save(out_dir / f"fused_{pct}pct.npy", fused)
        
        # Save visualization
        vis = ((fused - fused.min()) / (fused.max() - fused.min()) * 255.0).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"fused_{pct}pct_vis.png"), vis)
        print(f"[{pct}% Sparsity] Fusion complete: min={fused.min():.2f}m, max={fused.max():.2f}m")

if __name__ == "__main__":
    process_all_sparsities()
