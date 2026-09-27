import json
import cv2
import numpy as np
from pathlib import Path
import open3d as o3d

def depth_to_pointcloud(depth, rgb, intrinsics):
    h, w = depth.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    
    v, u = np.indices((h, w))
    valid = depth > 0.1
    
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    
    points = np.stack((x, y, z), axis=-1)
    colors = rgb[valid].astype(np.float64) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def reconstruct_mesh(pcd, depth_level=8):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    filtered_pcd = cl
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(filtered_pcd, depth=depth_level)
    
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    return filtered_pcd, mesh

def run_reconstruction():
    raw_dir = Path("data/raw/sample_frame")
    rgb_bgr = cv2.imread(str(raw_dir / "rgb.png"))
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    
    with open(raw_dir / "intrinsics.json", "r") as f:
        intrinsics = json.load(f)
    
    pcd_out = Path("outputs/pointcloud")
    mesh_out = Path("outputs/mesh")
    pcd_out.mkdir(parents=True, exist_ok=True)
    mesh_out.mkdir(parents=True, exist_ok=True)
    
    print("Generating Ground Truth 3D Model...")
    gt_depth = np.load(raw_dir / "gt_depth.npy")
    gt_pcd = depth_to_pointcloud(gt_depth, rgb, intrinsics)
    clean_gt_pcd, gt_mesh = reconstruct_mesh(gt_pcd)
    o3d.io.write_point_cloud(str(pcd_out / "gt_pointcloud.ply"), clean_gt_pcd)
    o3d.io.write_triangle_mesh(str(mesh_out / "gt_mesh.ply"), gt_mesh)
    
    sparsities = [50, 25, 10, 5]
    for pct in sparsities:
        print(f"Reconstructing 3D Model for [{pct}% Sparsity]...")
        fused_depth = np.load(f"outputs/depth/fused_{pct}pct.npy")
        pcd = depth_to_pointcloud(fused_depth, rgb, intrinsics)
        clean_pcd, mesh = reconstruct_mesh(pcd)
        
        o3d.io.write_point_cloud(str(pcd_out / f"fused_{pct}pct_pcd.ply"), clean_pcd)
        o3d.io.write_triangle_mesh(str(mesh_out / f"fused_{pct}pct_mesh.ply"), mesh)
        print(f"Saved point cloud ({len(clean_pcd.points)} pts) and mesh ({len(mesh.triangles)} triangles)")
    
    print("All 3D reconstructions completed successfully!")

if __name__ == "__main__":
    run_reconstruction()
