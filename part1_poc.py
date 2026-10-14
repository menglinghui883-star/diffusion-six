"""
Part 1: Build and Improve the Model (PoC)
Course: Data, Technology and Design (Term 7, 2026)

This script demonstrates the Part 1 workflow on a 64x64 RGB image:
  Baseline (Linear Regression) -> Evaluate -> Identify Limitation -> Improve Model (CNN)

Forward diffusion process:
  z_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * eps_t
"""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt


def load_clean_image(image_path: str = "mountain_64x64.png", size: int = 64) -> torch.Tensor:
    """Load an image, center-crop to square if needed, resize to 64x64 RGB, and normalize to [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w != h:
        side = min(w, h)
        top = int((h - side) * 0.35)
        left = (w - side) // 2
        img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def add_noise(x: torch.Tensor, alpha_t: float = 0.5, noise: torch.Tensor = None):
    """
    Generate noisy observation z_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * eps_t.
    Returns (z_t, eps_t).
    """
    if noise is None:
        noise = torch.randn_like(x)
    sqrt_alpha = float(np.sqrt(alpha_t))
    sqrt_one_minus_alpha = float(np.sqrt(1.0 - alpha_t))
    z_t = sqrt_alpha * x + sqrt_one_minus_alpha * noise
    return z_t, noise


def reconstruct_image(z_t: torch.Tensor, pred_noise: torch.Tensor, alpha_t: float = 0.5) -> torch.Tensor:
    """Reconstruct the clean image x_hat from z_t and predicted noise eps_hat."""
    sqrt_alpha = float(np.sqrt(alpha_t))
    sqrt_one_minus_alpha = float(np.sqrt(1.0 - alpha_t))
    x_hat = (z_t - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
    return torch.clamp(x_hat, 0.0, 1.0)


def compute_mse(x_true: torch.Tensor, x_pred: torch.Tensor) -> float:
    """Compute Mean Squared Error (MSE) between ground truth and reconstructed image."""
    return float(torch.mean((x_true - x_pred) ** 2).item())


class BaselineLinearDenoiser(nn.Module):
    """
    Baseline Linear Regression model for estimating noise.
    Operates pixel-wise (1x1 linear projection across RGB channels) without local spatial receptive field,
    illustrating why a simple linear model struggles to capture 2D spatial structure.
    """
    def __init__(self, channels: int = 3):
        super().__init__()
        self.linear = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.linear(z_t)


class SmallCNNDenoiser(nn.Module):
    """
    Small Convolutional Neural Network (CNN) that exploits local 2D spatial structure
    and multi-scale spatial context to estimate the diffusion noise eps_t.
    """
    def __init__(self, channels: int = 3, hidden_dim: int = 96):
        super().__init__()
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 64),
            torch.linspace(-1.0, 1.0, 64),
            indexing="ij",
        )
        pos_list = []
        for freq in [1.0, 2.0, 4.0, 8.0, 12.0]:
            pos_list.extend([
                torch.sin(np.pi * freq * yy),
                torch.cos(np.pi * freq * yy),
                torch.sin(np.pi * freq * xx),
                torch.cos(np.pi * freq * xx),
            ])
        pos_grid = torch.stack(pos_list, dim=0).unsqueeze(0)  # (1, 20, 64, 64)
        self.register_buffer("pos_grid", pos_grid)

        in_ch = channels + 20
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1, padding_mode="reflect"),
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z_t.shape
        if H == self.pos_grid.shape[2] and W == self.pos_grid.shape[3]:
            pos = self.pos_grid.expand(B, -1, H, W)
        else:
            pos = torch.nn.functional.interpolate(
                self.pos_grid, size=(H, W), mode="bilinear", align_corners=False
            ).expand(B, -1, H, W)
        inp = torch.cat([z_t, pos], dim=1)
        return self.net(inp)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    img_path = "mountain_64x64.png"
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Could not find {img_path}. Please place 'mountain_64x64.png' in the current directory."
        )

    x = load_clean_image(img_path, size=64)
    alpha_t = 0.5

    # 1. Train Baseline Linear Regression Model
    baseline = BaselineLinearDenoiser(channels=3)
    opt_base = torch.optim.Adam(baseline.parameters(), lr=0.02)
    for _ in range(300):
        z_batch, eps_batch = add_noise(x.repeat(8, 1, 1, 1), alpha_t=alpha_t)
        pred_eps = baseline(z_batch)
        loss = torch.mean((pred_eps - eps_batch) ** 2)
        opt_base.zero_grad()
        loss.backward()
        opt_base.step()

    # 2. Train Improved Small CNN Model
    cnn = SmallCNNDenoiser(channels=3, hidden_dim=96)
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=0.005)
    for _ in range(350):
        z_batch, eps_batch = add_noise(x.repeat(4, 1, 1, 1), alpha_t=alpha_t)
        pred_eps = cnn(z_batch)
        loss = torch.mean((pred_eps - eps_batch) ** 2)
        opt_cnn.zero_grad()
        loss.backward()
        opt_cnn.step()

    # 3. Evaluate both models on a held-out noisy observation
    torch.manual_seed(2026)
    z_test, eps_test = add_noise(x, alpha_t=alpha_t)
    with torch.no_grad():
        eps_hat_lin = baseline(z_test)
        x_hat_lin = reconstruct_image(z_test, eps_hat_lin, alpha_t=alpha_t)

        eps_hat_cnn = cnn(z_test)
        x_hat_cnn = reconstruct_image(z_test, eps_hat_cnn, alpha_t=alpha_t)

    mse_noisy = compute_mse(x, torch.clamp(z_test, 0.0, 1.0))
    mse_lin = compute_mse(x, x_hat_lin)
    mse_cnn = compute_mse(x, x_hat_cnn)

    print("=== Part 1: Model Evaluation Results (64x64 RGB, alpha_t = 0.5) ===")
    print(f"Noisy Observation MSE       : {mse_noisy:.5f}")
    print(f"Baseline Linear Model MSE   : {mse_lin:.5f}")
    print(f"Improved Spatial CNN MSE    : {mse_cnn:.5f}")
    print(f"Relative MSE Improvement    : {(1.0 - mse_cnn / mse_lin) * 100:.1f}%")

    # 4. Save side-by-side comparison plot
    def to_numpy_img(t: torch.Tensor) -> np.ndarray:
        return torch.clamp(t.squeeze(0).permute(1, 2, 0), 0.0, 1.0).cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.2), dpi=200)
    panels = [
        (to_numpy_img(x), "Clean Image (x)\n64×64 RGB Ground Truth"),
        (to_numpy_img(z_test), f"Noisy Observation (z_t)\nα_t = {alpha_t:.1f} | MSE = {mse_noisy:.4f}"),
        (to_numpy_img(x_hat_lin), f"Baseline: Linear Regression\nReconstructed | MSE = {mse_lin:.4f}"),
        (to_numpy_img(x_hat_cnn), f"Improved: Spatial CNN\nReconstructed | MSE = {mse_cnn:.4f}"),
    ]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
        ax.axis("off")

    fig.suptitle(
        "Part 1: Diffusion Denoising Comparison — Baseline Linear Regression vs. CNN (64×64 RGB)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig("part1_baseline_vs_cnn.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("Saved comparison figure to 'part1_baseline_vs_cnn.png'.")


if __name__ == "__main__":
    main()
