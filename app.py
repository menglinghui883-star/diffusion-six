"""
app.py — Streamlit interface for the 64x64 RGB Diffusion Denoising Application.
Course: Data, Technology and Design (Term 7, 2026)

Workflow:
  Open App -> Upload Image -> Add Noise -> Reconstruct -> View Result & MSE
"""

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

st.set_page_config(
    page_title="Diffusion Image Denoiser (64×64)",
    page_icon="🏔️",
    layout="wide",
)

st.title("🏔️ Diffusion-Based Image Denoising (64×64 RGB)")
st.markdown(
    """
    This application demonstrates the machine learning lifecycle (**PoC → Project → Reproducible → Tested → Packaged → Deployed**)
    using a diffusion-inspired denoising model based on:
    $$\\mathbf{z}_t = \\sqrt{\\alpha_t}\\mathbf{x} + \\sqrt{1 - \\alpha_t}\\boldsymbol{\\epsilon}_t$$
    """
)

# Sidebar controls
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

run_btn = st.sidebar.button("Add Noise & Reconstruct", type="primary", use_container_width=True)


@st.cache_resource
def get_models():
    return load_trained_models("model.pt")


cnn_model, baseline_model = get_models()

# Load either the user-uploaded image or the default 64x64 mountain image
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    source_label = f"Uploaded Image ({uploaded_file.name})"
else:
    default_path = ensure_clean_image("mountain_64x64.png")
    pil_img = Image.open(default_path)
    source_label = "Default Snow Mountain (mountain_64x64.png)"

x = preprocess_image(pil_img, size=IMAGE_SIZE)

# Perform forward diffusion and model reconstruction
torch.manual_seed(int(seed))
z_t, eps_t = add_noise(x, alpha_t=float(alpha_t))

with torch.no_grad():
    pred_eps_base = baseline_model(z_t)
    x_hat_base = reconstruct_image(z_t, pred_eps_base, alpha_t=float(alpha_t))

    pred_eps_cnn = cnn_model(z_t)
    x_hat_cnn = reconstruct_image(z_t, pred_eps_cnn, alpha_t=float(alpha_t))

z_display = torch.clamp(z_t, 0.0, 1.0)
mse_noisy = compute_mse(x, z_display)
mse_base = compute_mse(x, x_hat_base)
mse_cnn = compute_mse(x, x_hat_cnn)
improvement_pct = (1.0 - mse_cnn / max(mse_base, 1e-8)) * 100.0


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    arr = torch.clamp(t.squeeze(0).permute(1, 2, 0), 0.0, 1.0).cpu().numpy()
    return (arr * 255.0).astype(np.uint8)


st.subheader(f"Reconstruction Results — {source_label}")

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Noisy Observation MSE", f"{mse_noisy:.4f}")
m2.metric("Baseline (Linear) MSE", f"{mse_base:.4f}")
m3.metric("Improved (CNN) MSE", f"{mse_cnn:.4f}")
m4.metric("CNN Error Reduction vs. Baseline", f"{improvement_pct:.1f}%")

# Image Comparison Row
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
        tensor_to_uint8(x_hat_cnn),
        caption=f"4. Improved CNN Reconstruction | MSE: {mse_cnn:.4f}",
        use_container_width=True,
    )

st.info(
    "**Why does the CNN outperform Linear Regression?** "
    "The baseline linear regression model operates pixel-wise without spatial context, "
    "shrinking noisy pixels toward the global mean color. The CNN exploits 3×3 local neighborhoods "
    "and multi-scale spatial structure to separate high-frequency Gaussian noise from mountain ridges and clouds."
)
