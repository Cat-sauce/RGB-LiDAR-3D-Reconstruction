import cv2
import json
import numpy as np
from pathlib import Path
import open3d as o3d

def simulate_sparse_lidar(raw_dir='data/raw/sample_frame', out_base='data/sparse', sparsity_levels=[0.50, 0.25, 0.10, 0.05]):
    raw_path = Path(raw_dir)
    gt_depth = np.load(raw_path / 'gt_depth.npy')
    with open(raw_path / 'intrinsics.json', 'r') as f:
        intrinsics = json.load(f)
    
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # Valid mask where depth > 0
    valid_y, valid_x = np.where(gt_depth > 0)
    num_valid = len(valid_y)
    print(f'Total valid ground-truth depth pixels: {num_valid}')
    
    np.random.seed(42)
    
    for ratio in sparsity_levels:
        percent = int(ratio * 100)
        out_dir = Path(out_base) / f'{percent}pct'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample indices
        sample_count = int(num_valid * ratio)
        sampled_indices = np.random.choice(num_valid, size=sample_count, replace=False)
        
        sparse_depth = np.zeros_like(gt_depth)
        sy = valid_y[sampled_indices]
        sx = valid_x[sampled_indices]
        sparse_depth[sy, sx] = gt_depth[sy, sx]
        
        # Save sparse depth map
        np.save(out_dir / 'sparse_depth.npy', sparse_depth)
        
        # Reproject to 3D point cloud
        z = sparse_depth[sy, sx]
        x = (sx - cx) * z / fx
        y = (sy - cy) * z / fy
        points = np.stack((x, y, z), axis=-1)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(out_dir / 'lidar_points.ply'), pcd)
        
        print(f'[{percent}% Sparsity] Saved {sample_count} points to {out_dir}')

if __name__ == '__main__':
    simulate_sparse_lidar()
