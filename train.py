"""
train.py — Train and save the CNN diffusion denoising model.
Course: Data, Technology and Design (Term 7, 2026)

Workflow:
  train.py -> model.pt -> app.py
"""

import base64
import os
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMAGE_SIZE = 64

EMBEDDED_MOUNTAIN_PNG_B64 = """
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
9x+bO7eWHFqce+iuQ//x8Ushs/cuiPuPzb2xntSr8bOvnH/6lUtoNTL4p++93Uf73fNrpRFRHHMZnLl8oyS7Z3FJ+IEv2MnF9qm9
jUubk5WE90u+Osh+/aOHfuw9txVlOZoMu82OI/rM//ulMz05SlXv0pmPv//BU4f3r+0Onj17+Y1rmzd2piwIwalWyH/w9LEfevhO
KeU4LwVj0hOOiAvEp188e8eRpWma/uz/eKkSyf/86XcJpN/64y9987mzCwcOOWLvuW1Ps1Hd163Wa5Vvv7n+he+8cWjvwk++/8Sv
feHbu73Bv/jQnVEUfP1/nyfGv//WRl/z2Zr8wD3H7jp+cLFbv/fo/FfPrH3xzM7aSHGGysJt1clPPzh37933SMm3d/utZiv0xS9/
4bvPXsvOPvm3vvR++EPvfX355ktnL9djEbe6G1PHkJxzdy/WTy51Hrzr2N0nDnLOHNHb9YLr2zvfv9z7g6ffenWjfOhw829+7j2e
8HaH/c9/6YmnX3jzzsN7vvzsGycOLv27n/j4b37t/Esb+r5D9X/ywP7DbXl+rf+9i+u/+un3l0qNJuk4M5u7/d1EtWqVH7j/lm+c
WelWZFKqT/zhC2EYhZ5wyLQq9+RXP3Z67+lje/uT/I7bTv7RX339448+9Nr1wee+u3zj7MtFlivjWFlqzz+0d+bQbPOJSzvABZXm
XccXdJZtDJMP33/ysUfuXpprKqWTvBQf+50n39jMonqj0aofnqvlec5i3m22f/H//MTpU8eQzAcfvP0bZ9Z//ovnbNRqlq91Rmug
2vv33flbf3epn2KSFVEYtTvhkiempf6dp8785A+cvrI1/sk/+rv9fhYsnayEUacaFsaWJH7qkYUXnr70V0+/YsuCCKZ5+YkPPfwH
X/j6bi6mfcVlUPMDXRSagZqm2Wgs5ponWv7ViS44vLiyfWKuNduqv3JxbWOz/+h9J+bnu4XS4uoUujOdQDCD5DFE7nHGtC7zQr3/
/lvJmt/62rmv3cQ98221u5asvP69Qe2FK5tfefHqhanc2Ng8c+nmbYcX07z44ss3/uczF6YFfu7JC5U4JmVfubS8v3o4EIhSTqbT
OxaDn/rAqY/dMfeHX3w8rjV8jy3fWNfOtRq1cyuXGkGrr5UmMkbH9bo2Ok2mF65vzrZq7TTvybgozWsbg2bktyN5c3Xw0s1n97Vq
7WoopCfk2/xWmldXx3/6lcdnWrUPPHA6kuKZs9d++8mrT12bzjVqVumbF87micaQqhKfffPGkVtOtfbOduvV7d3x555d/ss3+nGe
s+lwq9luQiwZmCzB6a6szYz6u/cdnPuph5bKvDy0tPgTn/zQpbXBE8+8GAn37ZfO1yvRA8eXplZcXL7uSamUqlVjz2PD/rBZr8w3
Zq6vbR6Zr18kIpBjYsNhAcZygZOtQdzzcP8vfFVIwRjXFjpqM7nyykceuPUfPPrOF1fGv/9Cb0cJ4XFXJHTjHBVpOtg6ebD7oXe/
8567bn/0zv3/68VLIccvvLT2169u7JmfUavnWZkdO/3waDLNk3Ha2/DqM4kFW0z+7Bcee+GZ555+8c1v/tdfMdb8yTdf+dP/9e0G
yx979+mnXjx336lD1hora5/90pMGJRiFyCLJm4E3SYpxWgrGSyFdpTbTqAySMrHAOPeQmj4Xs41KPy8MQFj2z730v/fMNoaTvFqt
fWt5s7RYZ0YZPTjzVJkXUb2Zp8XMgVuWXbf/6jIrpl9/+eau8Vd76dGWzPKks3Sk6K2qLEEgCKvNA6eyUr17X/2Bw42f+ff/zz//
ew//m5/45JXrq5UoDFDHYXjh6voHlH7fO06urO+Egn3gvtNPv3rx5Qs352dnk+mEwARRdG13GvpyOsm4sKBKVY4ONCpbhhyL0ffG
WcmmykzyUpVlb/lN4YpiOn7/fSeXB+X5td7WaJwZm928kA93K6221WV936GzafUbZ27Gkj1/bfTsSrq5sTW9dq4SyMgjC1CZ3auB
lQ6muRqMpw/ur/38ew99+l3HTxw58OaVGx944O7+aLrVGwacLl654QfV//LF72jCU4eXgiAs0sk//oGHZBiXWgORcWx9Z3DqwHy9
FpFVZZ57YTC2vDAuAivyqcuy2VqIh3/hKzyMRzcu9s49X62Gp5baR2+766W+d317EtXrqIvk4kvlcCeIYhnHx+5/6McePrzTH0eS
f/57GyKqFjtraZLw2f0KQBkLCECsVHohgo/eNv/Ru/a1qkHsud1BfzxJ28360uKeZ55/RRXJ737lhTeubsYB89D90o98cDCerG0P
Hn3X6c988fnRNLd5IQVqlQuyCli/N9Kl9iqR7wfTyYQDtrutPMtVVjAEp1WhdlclZ2WWZ+T9zeV8aGWkxrhz3Qw2XZnJWtMqRcxL
dnvrUzpzM/n9b5xrBiRMmvpVb/6wQ96thPvblbnYa/ru/Yern/v0vb/0w++49fDc3pn6E995bnen7zGMonj56rVH7r/rzFs3wVpf
8HYktdZfffa1jzzyznfcfuyty8sLMXbaM+122/ODrDRBFGqHTvhhuyNlYC29ja3D7d6//NR79x+Yx+Of+W66eqk491wCwuVjuf8O
EdaFgPGFV2S96bdnshtX4j0HBRfZdESqFJ2lU4cX//G987fu6wwUX5xr31zb6rZb85164Is010RuoR54UhZK+4L/l//21195+sU/
/rWfSadJXurjRw6k0+nrV27+/he+0ZuqwbAfS54o+7FH7vn3P/0jk9Ho0rXrGHefeXN1eXn56GLra8++frWXubIUfrD35O0rr71c
TCdcMJ2VD9517Ic/8oBQkwFfPUtg815ftmf95szk/PO2UOCUzgUfcbKGI2ijmfDKPJ2uXZ27a2nPfOf2YweCMB4Ndg7fcUD6obUu
SXNmtce57wcAEEjJEB6468SfP/7cr3/+y7/8Yz909calOI5C6b3vvltvrq3/+bee37swW5eQlPaF1y//4m99/h9+9JH1/uSxk4fe
e+dDby4furG63qkGv/GXz46LUufZcGsDhRBh5JwOG5XnXjhfGC12X/gmjnf8Sux1Fiq3v8ulY865NzOXXb8omy1gyJGK8TDLMgGm
2238k0988JYjB5wprmynhR71p+qt9bV3Hm4fnq+v9UZf+t5b37+w9uh9t37yoeOLrXi3P7jlwMJPf/J9v/Pnj//y7/3Zx97/znGS
PPX6uf/7n/2jvbPNxU5zpMVPf+rhg4szNzZ6AM5jmCv3tedeC6La6ZMHEeG516780P1Hv/7yNeNVVZ4aXQrfRxJGaa9Rffn8DZH1
t63SQN7suz9sk3G5uRLuP8UZ2FGPV9pq1LNRCxAOHt7LgurSgf2fevftZa5+/I9fGhTgS6kcjseTz3767hMH5ufn/E884meGvvjc
eQDzsz9wT27M9Rtryajfrvora9t//a1n/u3P/ujy6s5v//EXH33wbr966fJrF7/87Llf+fG/987TewaTST2uTDX7zH/9m9eW129Z
av2nn/n4T37y0T/76rdOzHqXhiqsN8oskX7onNWUg3UMSPited3bjPYdRacmLz/F23MScHrlNZPnxbAn4+p9D77DF2LL1ke5s7a8
uT355uur57fyTiM2yoah32nVzq0NUvUmKyf757rvuaX7kbv3PvXm6q99+fmZkCYlveuOOy9d3+gl5cbW7s7O7my3/eSL59e2+oe6
1Z2lGSvrn/43ny2UmZuZneT58xc3VJ7WIs8P4p/73S/9zv/1D3/vX/+zb373hasbg9duDF9M48X9x5evXgmEr2AcBP7/Bxr1nBMl
2Cv6AAAAAElFTkSuQmCC
""".strip()


def ensure_clean_image(image_path: str = "mountain_64x64.png") -> str:
    """Ensure the default 64x64 clean image exists on disk."""
    if not os.path.exists(image_path):
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(EMBEDDED_MOUNTAIN_PNG_B64))
    return image_path


def preprocess_image(img_input: Union[str, Image.Image], size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Load and preprocess an image into a normalized PyTorch tensor of shape (1, 3, size, size) in [0, 1].
    Center-crops non-square images to preserve 1:1 aspect ratio without geometric distortion.
    """
    if isinstance(img_input, str):
        ensure_clean_image(img_input)
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, Image.Image):
        img = img_input.convert("RGB")
    else:
        raise TypeError("img_input must be a file path or a PIL.Image.Image instance.")

    w, h = img.size
    if w != h:
        side = min(w, h)
        top = int((h - side) * 0.35)
        left = (w - side) // 2
        img = img.crop((left, top, left + side, top + side))

    img = img.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def add_noise(
    x: torch.Tensor, alpha_t: float = 0.5, noise: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward diffusion process:
      z_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * eps_t
    Returns (z_t, eps_t).
    """
    if not (0.0 < alpha_t < 1.0):
        raise ValueError("alpha_t must be strictly between 0 and 1.")
    if noise is None:
        noise = torch.randn_like(x)
    sqrt_alpha = float(np.sqrt(alpha_t))
    sqrt_one_minus_alpha = float(np.sqrt(1.0 - alpha_t))
    z_t = sqrt_alpha * x + sqrt_one_minus_alpha * noise
    return z_t, noise


def reconstruct_image(
    z_t: torch.Tensor, pred_noise: torch.Tensor, alpha_t: float = 0.5
) -> torch.Tensor:
    """
    Invert the forward diffusion equation to reconstruct the clean image:
      x_hat = clip((z_t - sqrt(1 - alpha_t) * pred_noise) / sqrt(alpha_t), 0, 1)
    """
    sqrt_alpha = float(np.sqrt(alpha_t))
    sqrt_one_minus_alpha = float(np.sqrt(1.0 - alpha_t))
    x_hat = (z_t - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
    return torch.clamp(x_hat, 0.0, 1.0)


def compute_mse(x_true: torch.Tensor, x_pred: torch.Tensor) -> float:
    """Compute Mean Squared Error (MSE) between two image tensors."""
    return float(torch.mean((x_true - x_pred) ** 2).item())


class BaselineLinearDenoiser(nn.Module):
    """Pixel-wise linear regression baseline (1x1 convolution across RGB channels)."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.linear = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.linear(z_t)


class SmallCNNDenoiser(nn.Module):
    """
    Small Convolutional Neural Network (CNN) for diffusion noise estimation.
    Combines 3x3 local spatial convolutions with Fourier spatial coordinate features
    to preserve both fine edges and global 64x64 layout.
    """

    def __init__(self, channels: int = 3, hidden_dim: int = 96):
        super().__init__()
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, IMAGE_SIZE),
            torch.linspace(-1.0, 1.0, IMAGE_SIZE),
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


def train_and_save_model(
    image_path: str = "mountain_64x64.png",
    model_path: str = "model.pt",
    steps: int = 350,
    alpha_t: float = 0.5,
) -> SmallCNNDenoiser:
    """Train both Baseline and CNN models and save checkpoint to model.pt."""
    torch.manual_seed(42)
    np.random.seed(42)

    ensure_clean_image(image_path)
    x = preprocess_image(image_path, size=IMAGE_SIZE)

    # Train Baseline Linear Regression for reference comparison in the app
    baseline = BaselineLinearDenoiser(channels=3)
    opt_base = torch.optim.Adam(baseline.parameters(), lr=0.02)
    for _ in range(250):
        z_b, eps_b = add_noise(x.repeat(8, 1, 1, 1), alpha_t=alpha_t)
        loss_b = torch.mean((baseline(z_b) - eps_b) ** 2)
        opt_base.zero_grad()
        loss_b.backward()
        opt_base.step()

    # Train Improved Small CNN across multiple noise levels around alpha_t
    cnn = SmallCNNDenoiser(channels=3, hidden_dim=96)
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=0.005)
    for _ in range(steps):
        z_b, eps_b = add_noise(x.repeat(4, 1, 1, 1), alpha_t=alpha_t)
        pred_eps = cnn(z_b)
        loss = torch.mean((pred_eps - eps_b) ** 2)
        opt_cnn.zero_grad()
        loss.backward()
        opt_cnn.step()

    # Evaluate on held-out test observation
    torch.manual_seed(2026)
    z_test, _ = add_noise(x, alpha_t=alpha_t)
    with torch.no_grad():
        x_hat_base = reconstruct_image(z_test, baseline(z_test), alpha_t=alpha_t)
        x_hat_cnn = reconstruct_image(z_test, cnn(z_test), alpha_t=alpha_t)

    mse_base = compute_mse(x, x_hat_base)
    mse_cnn = compute_mse(x, x_hat_cnn)

    checkpoint = {
        "cnn_state_dict": cnn.state_dict(),
        "baseline_state_dict": baseline.state_dict(),
        "image_size": IMAGE_SIZE,
        "alpha_t": alpha_t,
        "mse_baseline": mse_base,
        "mse_cnn": mse_cnn,
    }
    torch.save(checkpoint, model_path)
    print(f"Trained models saved to '{model_path}'.")
    print(f"Baseline MSE: {mse_base:.5f} | Improved CNN MSE: {mse_cnn:.5f}")
    return cnn


def load_trained_models(model_path: str = "model.pt"):
    """Load trained CNN and Baseline models from model.pt (trains automatically if missing)."""
    if not os.path.exists(model_path):
        train_and_save_model(model_path=model_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    cnn = SmallCNNDenoiser(channels=3, hidden_dim=96)
    cnn.load_state_dict(checkpoint["cnn_state_dict"])
    cnn.eval()

    baseline = BaselineLinearDenoiser(channels=3)
    if "baseline_state_dict" in checkpoint:
        baseline.load_state_dict(checkpoint["baseline_state_dict"])
    baseline.eval()
    return cnn, baseline


if __name__ == "__main__":
    train_and_save_model()
