import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from pathlib import Path
from transformers import pipeline
import io

st.set_page_config(page_title="Multimodal 3D Object Reconstruction", layout="wide")

@st.cache_resource
def load_model():
    return pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cpu")

pipe = load_model()

st.title("Multimodal 3D Object Reconstruction")
st.markdown("Capture or upload any photo to reconstruct a textured, interactive 3D point cloud with Depth Anything V2.")

col_input, col_view = st.columns([1, 2])

with col_input:
    st.subheader("1. Image Acquisition")
    input_mode = st.radio("Input Source:", ["Live Webcam", "Upload Image"])
    img_file = None
    if input_mode == "Live Webcam":
        img_file = st.camera_input("Capture Frame")
    else:
        img_file = st.file_uploader("Choose Image File", type=["jpg", "png", "jpeg"])
    
    pt_size = st.slider("Visual Point Size", min_value=1.0, max_value=6.0, value=2.5, step=0.5)
    depth_cutoff = st.slider("Depth Distance Cutoff (%)", min_value=30, max_value=100, value=85)
    run_btn = st.button("Generate 3D Reconstruction", type="primary")

if run_btn and img_file is not None:
    image = Image.open(img_file).convert("RGB")
    img_np = np.array(image)
    h, w, _ = img_np.shape
    
    with st.spinner("Inferring monocular depth with Depth Anything V2..."):
        res = pipe(image)
        depth_raw = np.array(res["depth"], dtype=np.float32)
        depth_raw = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_CUBIC)
        
        d_min, d_max = depth_raw.min(), depth_raw.max()
        norm_disp = (depth_raw - d_min) / (d_max - d_min + 1e-8)
        rel_depth = 1.0 / (norm_disp + 0.05)
        metric_depth = ((rel_depth - rel_depth.min()) / (rel_depth.max() - rel_depth.min())) * 1.8 + 0.4
    
    with col_input:
        st.subheader("2. Depth Map")
        vis_depth = ((metric_depth - metric_depth.min()) / (metric_depth.max() - metric_depth.min()) * 255).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis_depth, cv2.COLORMAP_INFERNO)
        st.image(cv2.cvtColor(vis_color, cv2.COLOR_BGR2RGB), caption="Metric Depth Map (Inferno colormap)", width=None)
    
    with col_view:
        st.subheader("3. Interactive 3D Model")
        focal_length = max(h, w) * 0.8
        fx, fy = focal_length, focal_length
        cx, cy = w / 2.0, h / 2.0
        
        target_pts = 35000
        step = max(1, int(np.sqrt((h * w) / target_pts)))
        v, u = np.mgrid[0:h:step, 0:w:step]
        v = np.clip(v, 0, h - 1)
        u = np.clip(u, 0, w - 1)
        
        z_grid = metric_depth[v, u]
        max_z_allowed = np.percentile(metric_depth, depth_cutoff)
        mask = z_grid <= max_z_allowed
        
        z = z_grid[mask].flatten()
        x = ((u[mask] - cx) * z / fx).flatten()
        y = -((v[mask] - cy) * z / fy).flatten()
        
        r_vals = img_np[v[mask], u[mask], 0].flatten()
        g_vals = img_np[v[mask], u[mask], 1].flatten()
        b_vals = img_np[v[mask], u[mask], 2].flatten()
        colors_rgb = [f"rgb({r},{g},{b})" for r, g, b in zip(r_vals, g_vals, b_vals)]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=z, z=y,
            mode="markers",
            marker=dict(size=pt_size, color=colors_rgb, opacity=0.95)
        )])
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, visible=False),
                yaxis=dict(showbackground=False, visible=False),
                zaxis=dict(showbackground=False, visible=False),
                aspectmode="data",
                camera=dict(eye=dict(x=0, y=-1.5, z=0.3))
            ),
            paper_bgcolor="#0e1117",
            margin=dict(l=0, r=0, b=0, t=0),
            height=680
        )
        st.plotly_chart(fig, width=None)
        
        ply_header = f"ply\nformat ascii 1.0\nelement vertex {len(x)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        ply_body = "\n".join([f"{px:.4f} {py:.4f} {pz:.4f} {pr} {pg} {pb}" for px, py, pz, pr, pg, pb in zip(x, y, z, r_vals, g_vals, b_vals)])
        st.download_button("Download 3D Model (.ply)", data=(ply_header + ply_body), file_name="reconstruction_3d.ply", mime="application/octet-stream")