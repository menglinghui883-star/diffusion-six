"""
app.py — Interactive Streamlit application supporting both v1 CNN and v2 U-Net architectures.
Course: Data, Technology and Design (Term 7, 2026)
"""

import os
import numpy as np
import streamlit as st
import torch
from PIL import Image

from train import (
    IMAGE_SIZE,
    add_noise,
    compute_mse,
    ensure_clean_image,
    load_trained_models,
    preprocess_image,
    reconstruct_image,
)

try:
    from train_unet_v2 import UNetDenoiserV2
    HAS_UNET_V2 = True
except ImportError:
    HAS_UNET_V2 = False

st.set_page_config(
    page_title="Diffusion Image Denoiser (v1 CNN & v2 U-Net)",
    page_icon="🏔️",
    layout="wide",
)

st.title("🏔️ Diffusion-Based Image Denoising (64×64 RGB)")
st.markdown(
    """
    This application demonstrates the complete MLOps lifecycle (**PoC → Project → Tested → Packaged → Deployed → Observed → Improved**)
    using forward diffusion:
    $$\\mathbf{z}_t = \\sqrt{\\alpha_t}\\mathbf{x} + \\sqrt{1 - \\alpha_t}\\boldsymbol{\\epsilon}_t$$
    """
)

# 1. 左侧边栏：输入设置
st.sidebar.header("1. Input Image & Noise Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload an image (PNG/JPG) or use default snow mountain:",
    type=["png", "jpg", "jpeg"],
)

alpha_t = st.sidebar.slider(
    "Signal Retention Factor (α_t)",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Higher α_t retains more clean signal; lower α_t adds heavier Gaussian noise.",
)

seed = st.sidebar.number_input(
    "Random Seed (for reproducible noise)",
    min_value=0,
    max_value=99999,
    value=2026,
    step=1,
)

# 2. 左侧边栏：核心双模型架构切换！
st.sidebar.header("2. Model Architecture Selection")
has_v2_weights = os.path.exists("model_unet_v2.pt")
model_options = [
    "v1: Single-Image Spatial CNN (PoC)",
    "v2: Multi-Image U-Net (Skip Connections)",
] if has_v2_weights else ["v1: Single-Image Spatial CNN (PoC)"]

model_choice = st.sidebar.radio(
    "Select Model Architecture:",
    options=model_options,
    help="v1 is trained on the single mountain photo; v2 is trained on diverse multi-scale patches with skip connections.",
)

run_btn = st.sidebar.button("Add Noise & Reconstruct", type="primary", use_container_width=True)


@st.cache_resource
def get_v1_models():
    return load_trained_models("model.pt")


@st.cache_resource
def get_v2_model():
    if not os.path.exists("model_unet_v2.pt"):
        return None
    model_v2 = UNetDenoiserV2(in_channels=3, base_ch=32)
    ckpt = torch.load("model_unet_v2.pt", map_location="cpu", weights_only=True)
    model_v2.load_state_dict(ckpt["unet_v2_state_dict"])
    model_v2.eval()
    return model_v2


cnn_v1, baseline_model = get_v1_models()
unet_v2 = get_v2_model()

# 加载干净原图
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    source_label = f"Uploaded Image ({uploaded_file.name})"
else:
    default_path = ensure_clean_image("mountain_64x64.png")
    pil_img = Image.open(default_path)
    source_label = "Default Snow Mountain (mountain_64x64.png)"

x = preprocess_image(pil_img, size=IMAGE_SIZE)

# 前向加噪
torch.manual_seed(int(seed))
z_t, eps_t = add_noise(x, alpha_t=float(alpha_t))

# 逆向重构
with torch.no_grad():
    pred_eps_base = baseline_model(z_t)
    x_hat_base = reconstruct_image(z_t, pred_eps_base, alpha_t=float(alpha_t))

    if "v2" in model_choice and unet_v2 is not None:
        pred_eps_selected = unet_v2(z_t, alpha_t=float(alpha_t))
        active_model_name = "v2 Multi-Image U-Net"
    else:
        pred_eps_selected = cnn_v1(z_t)
        active_model_name = "v1 Spatial CNN"

    x_hat_selected = reconstruct_image(z_t, pred_eps_selected, alpha_t=float(alpha_t))

z_display = torch.clamp(z_t, 0.0, 1.0)
mse_noisy = compute_mse(x, z_display)
mse_base = compute_mse(x, x_hat_base)
mse_selected = compute_mse(x, x_hat_selected)
improvement_pct = (1.0 - mse_selected / max(mse_base, 1e-8)) * 100.0


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    arr = torch.clamp(t.squeeze(0).permute(1, 2, 0), 0.0, 1.0).cpu().numpy()
    return (arr * 255.0).astype(np.uint8)


st.subheader(f"Reconstruction Results — {source_label} ({active_model_name})")

# 评估指标卡片
m1, m2, m3, m4 = st.columns(4)
m1.metric("Noisy Observation MSE", f"{mse_noisy:.4f}")
m2.metric("Baseline (Linear) MSE", f"{mse_base:.4f}")
m3.metric(f"{active_model_name} MSE", f"{mse_selected:.4f}")
m4.metric("Error Reduction vs. Baseline", f"{improvement_pct:.1f}%")

# 四联图像展示
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.image(tensor_to_uint8(x), caption="1. Clean Image (x) [64×64]", use_container_width=True)
with c2:
    st.image(
        tensor_to_uint8(z_display),
        caption=f"2. Noisy Input (z_t, α_t={alpha_t:.2f}) | MSE: {mse_noisy:.4f}",
        use_container_width=True,
    )
with c3:
    st.image(
        tensor_to_uint8(x_hat_base),
        caption=f"3. Baseline Linear Reconstruction | MSE: {mse_base:.4f}",
        use_container_width=True,
    )
with c4:
    st.image(
        tensor_to_uint8(x_hat_selected),
        caption=f"4. {active_model_name} Reconstruction | MSE: {mse_selected:.4f}",
        use_container_width=True,
    )

if "v2" in model_choice:
    st.success(
        "✨ **当前运行模型：v2 Multi-Image U-Net**。采用具备跳跃连接（Skip Connections）的下采样-上采样 U-Net 架构，"
        "并结合了 α_t 噪声强度条件输入与多样化数据集训练，消除了单图记忆偏见，对任意图像均具备强大的去噪泛化能力！"
    )
else:
    st.info(
        "ℹ️ **当前运行模型：v1 Spatial CNN (PoC 原版)**。专门在目标雪山图上完成概念验证训练。"
        "你可以在左侧单选框中随时切换到 v2 U-Net 对比效果！"
    )
