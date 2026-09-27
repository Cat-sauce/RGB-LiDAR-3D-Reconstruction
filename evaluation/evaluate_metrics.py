import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree

def compute_depth_metrics(gt_depth, pred_depth):
    mask = gt_depth > 0.1
    diff = gt_depth[mask] - pred_depth[mask]
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    return rmse, mae

def compute_chamfer_and_fscore(gt_pcd, pred_pcd, threshold=0.02, sample_size=50000):
    gt_pts = np.asarray(gt_pcd.points)
    pred_pts = np.asarray(pred_pcd.points)
    
    # Subsample for faster metric computation if point count is large
    if len(gt_pts) > sample_size:
        idx = np.random.choice(len(gt_pts), sample_size, replace=False)
        gt_pts = gt_pts[idx]
    if len(pred_pts) > sample_size:
        idx = np.random.choice(len(pred_pts), sample_size, replace=False)
        pred_pts = pred_pts[idx]
    
    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    
    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)
    
    chamfer_dist = np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2)
    
    # Precision and Recall at threshold (e.g. 2 cm)
    precision = np.mean(d_pred_to_gt < threshold)
    recall = np.mean(d_gt_to_pred < threshold)
    f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return chamfer_dist, f_score

def run_evaluation():
    gt_depth = np.load("data/raw/sample_frame/gt_depth.npy")
    gt_pcd = o3d.io.read_point_cloud("outputs/pointcloud/gt_pointcloud.ply")
    
    sparsities = [50, 25, 10, 5]
    results = []
    
    np.random.seed(42)
    print("\n--- Quantitative Evaluation --- \n")
    
    for pct in sparsities:
        fused_depth = np.load(f"outputs/depth/fused_{pct}pct.npy")
        pred_pcd = o3d.io.read_point_cloud(f"outputs/pointcloud/fused_{pct}pct_pcd.ply")
        
        rmse, mae = compute_depth_metrics(gt_depth, fused_depth)
        cd, fscore = compute_chamfer_and_fscore(gt_pcd, pred_pcd)
        
        results.append({
            "Sparsity": f"{pct}%",
            "RMSE (m)": round(rmse, 4),
            "MAE (m)": round(mae, 4),
            "Chamfer Distance": round(cd, 6),
            "F-Score @ 2cm": round(fscore, 4)
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    out_csv = Path("outputs/evaluation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

if __name__ == "__main__":
    run_evaluation()
