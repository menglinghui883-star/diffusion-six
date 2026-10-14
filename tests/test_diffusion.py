"""
tests/test_diffusion.py — Automated pytest suite for the Diffusion Image Denoiser.
Course: Data, Technology and Design (Term 7, 2026)

Checks required by Part 4:
  1. Image dimensions are correct (64x64 RGB -> shape (1, 3, 64, 64));
  2. Adding noise returns a 64x64 image;
  3. The CNN produces the expected output shape;
  4. Reconstruction returns valid numerical values (finite and within [0.0, 1.0]).
"""

import os
import sys
import torch

# Ensure the project root directory is on sys.path when running pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import (
    IMAGE_SIZE,
    SmallCNNDenoiser,
    add_noise,
    compute_mse,
    ensure_clean_image,
    preprocess_image,
    reconstruct_image,
)


def test_image_dimensions():
    """Check that loading and preprocessing the clean image returns a (1, 3, 64, 64) tensor."""
    img_path = ensure_clean_image("mountain_64x64.png")
    x = preprocess_image(img_path, size=IMAGE_SIZE)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 3, 64, 64), f"Expected shape (1, 3, 64, 64), got {x.shape}"
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0


def test_add_noise_returns_correct_shape():
    """Check that adding forward diffusion noise returns a 64x64 image tensor of the same shape."""
    img_path = ensure_clean_image("mountain_64x64.png")
    x = preprocess_image(img_path, size=IMAGE_SIZE)
    z_t, eps_t = add_noise(x, alpha_t=0.5)
    assert z_t.shape == (1, 3, 64, 64), f"Expected noisy shape (1, 3, 64, 64), got {z_t.shape}"
    assert eps_t.shape == (1, 3, 64, 64)
    assert not torch.allclose(z_t, x), "Noisy image z_t should differ from clean image x"


def test_cnn_output_shape():
    """Check that the SmallCNNDenoiser produces the expected (1, 3, 64, 64) noise prediction shape."""
    model = SmallCNNDenoiser(channels=3, hidden_dim=96)
    model.eval()
    dummy_noisy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        pred_noise = model(dummy_noisy)
    assert pred_noise.shape == (1, 3, 64, 64), f"Expected CNN output shape (1, 3, 64, 64), got {pred_noise.shape}"


def test_reconstruction_returns_valid_numerical_values():
    """Check that image reconstruction returns finite values strictly within [0.0, 1.0] and valid MSE."""
    img_path = ensure_clean_image("mountain_64x64.png")
    x = preprocess_image(img_path, size=IMAGE_SIZE)
    z_t, _ = add_noise(x, alpha_t=0.5)

    model = SmallCNNDenoiser(channels=3, hidden_dim=96)
    model.eval()
    with torch.no_grad():
        pred_noise = model(z_t)
        x_hat = reconstruct_image(z_t, pred_noise, alpha_t=0.5)

    assert x_hat.shape == (1, 3, 64, 64)
    assert torch.isfinite(x_hat).all(), "Reconstructed image contains NaN or Inf values"
    assert float(x_hat.min()) >= 0.0 and float(x_hat.max()) <= 1.0, "Reconstructed values out of [0, 1] bounds"

    mse_val = compute_mse(x, x_hat)
    assert isinstance(mse_val, float)
    assert mse_val >= 0.0 and not (mse_val != mse_val), "MSE must be a non-negative finite float"
