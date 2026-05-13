"""
tool.py — 실습용 공용 유틸리티
==================================

이 파일은 여러 노트북에서 재사용되는 함수들을 모아둔 곳입니다.
중복을 줄이고 각 셀은 "라이브 코딩"처럼 간결하게 유지하기 위한 목적입니다.

사용법 (Google Colab):
    1. 새 셀에 `!python /tool/upload.py` 또는 files.upload()로 이 파일을 업로드
    2. `from tool import *` 또는 원하는 함수만 import

모듈 구성:
    - seed: 재현성을 위한 시드 고정
    - plot_*: matplotlib 기반 시각화 헬퍼
    - make_*: 합성 데이터 생성
    - train_loop: 공용 학습 루프
    - show_*: 진행 상황 출력/플롯
"""
from __future__ import annotations

import os
import random
import time
from typing import Callable, Iterable

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# 1. 재현성
# ============================================================
def seed_all(seed: int = 42) -> None:
    """numpy / random / torch 시드를 한 번에 고정."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def device() -> "torch.device":
    """GPU 가능하면 cuda / mps, 아니면 cpu 반환."""
    if HAS_TORCH and torch.cuda.is_available():
        return torch.device("cuda")
    if HAS_TORCH and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu") if HAS_TORCH else "cpu"


# ============================================================
# 2. 합성 데이터
# ============================================================
def make_moons_tensor(n: int = 512, noise: float = 0.2, seed: int = 0):
    """이진 분류용 'moons' 데이터셋을 torch tensor로 반환.
    (주의: sklearn.datasets.make_moons 래퍼)
    """
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


def make_spiral(n: int = 300, k: int = 3, seed: int = 0):
    """다중 클래스 'spiral' 데이터 (n 샘플 per 클래스, k 클래스)."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n * k, 2), dtype=np.float32)
    y = np.zeros(n * k, dtype=np.int64)
    for j in range(k):
        ix = range(n * j, n * (j + 1))
        r = np.linspace(0.0, 1, n)
        t = np.linspace(j * 4, (j + 1) * 4, n) + rng.randn(n) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    if HAS_TORCH:
        return torch.tensor(X), torch.tensor(y)
    return X, y


# ============================================================
# 3. 공용 학습 루프
# ============================================================
def train_classifier(
    model,
    X,
    y,
    epochs: int = 200,
    lr: float = 1e-2,
    batch_size: int = 64,
    verbose: bool = True,
    loss_fn=None,
):
    """미니배치 분류 학습 루프. loss 이력을 리스트로 반환."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch가 설치되어 있어야 합니다.")
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    dev = device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history = []
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0)
        ep_loss /= len(ds)
        history.append(ep_loss)
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"ep {ep:4d} | loss {ep_loss:.4f}")
    return history


# ============================================================
# 4. 시각화
# ============================================================
def plot_loss(history, title: str = "Training loss", ax=None):
    """학습 loss 곡선."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history, lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_decision_boundary(model, X, y, title: str = "Decision boundary", ax=None, step: float = 0.02):
    """2D 분류기의 결정 경계 시각화."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch가 설치되어 있어야 합니다.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    X_np = X.detach().cpu().numpy() if hasattr(X, "detach") else X
    y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else y
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Z = model(grid).argmax(dim=1).numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="coolwarm",
               edgecolors="k", s=20)
    ax.set_title(title)
    return ax


def show_images(imgs, titles=None, cols: int = 8, figsize=None, cmap="gray"):
    """이미지 그리드 표시. imgs: (N, H, W) or (N, C, H, W) tensor/ndarray."""
    if hasattr(imgs, "detach"):
        imgs = imgs.detach().cpu().numpy()
    if imgs.ndim == 4 and imgs.shape[1] in (1, 3):
        imgs = imgs.transpose(0, 2, 3, 1).squeeze()
    n = len(imgs)
    rows = (n + cols - 1) // cols
    figsize = figsize or (cols * 1.2, rows * 1.2)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i], cmap=cmap)
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=8)
    plt.tight_layout()
    return fig


def count_params(model) -> int:
    """학습 가능한 파라미터 수 반환."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 5. 간편 출력
# ============================================================
def banner(text: str, ch: str = "=", width: int = 60) -> None:
    print(ch * width)
    print(text.center(width))
    print(ch * width)


# ============================================================
# 6. MNIST / 벤치마크 / 비교 시각화
# ============================================================
def load_mnist_small(n_train: int = 2000, n_test: int = 500, flatten: bool = False, seed: int = 0):
    """Small MNIST subset as torch tensors. Colab-friendly."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required.")
    from torchvision import datasets, transforms
    tfm = transforms.ToTensor()
    root = os.environ.get("MNIST_ROOT", "./data")
    tr = datasets.MNIST(root, train=True, download=True, transform=tfm)
    te = datasets.MNIST(root, train=False, download=True, transform=tfm)
    rng = np.random.RandomState(seed)
    tr_idx = rng.choice(len(tr), n_train, replace=False)
    te_idx = rng.choice(len(te), n_test, replace=False)
    Xtr = torch.stack([tr[i][0] for i in tr_idx])
    ytr = torch.tensor([tr[i][1] for i in tr_idx], dtype=torch.long)
    Xte = torch.stack([te[i][0] for i in te_idx])
    yte = torch.tensor([te[i][1] for i in te_idx], dtype=torch.long)
    if flatten:
        Xtr = Xtr.view(Xtr.size(0), -1)
        Xte = Xte.view(Xte.size(0), -1)
    return Xtr, ytr, Xte, yte


def benchmark(fn: Callable, *args, **kwargs):
    """Run fn, return (result, seconds, peak_mb). Peak memory is CUDA-only; else NaN."""
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mb = float("nan")
    return result, time.perf_counter() - t0, peak_mb


def accuracy(model, X, y, batch_size: int = 256) -> float:
    """Classification accuracy on (X, y)."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required.")
    model.eval()
    dev = next(model.parameters()).device
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(dev)
            yb = y[i:i + batch_size].to(dev)
            correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(X)


def shift_images(X, dx: int = 0, dy: int = 0):
    """Zero-padded shift of image batch (N,C,H,W) by (dy, dx) pixels.
    Pixels shifted out are dropped; new area is filled with 0 (background).
    This is the right 'translation' test — circular roll would wrap digits around.
    """
    if HAS_TORCH and hasattr(X, "roll"):
        out = torch.zeros_like(X)
        H, W = X.shape[-2], X.shape[-1]
        y_src = slice(max(0, -dy), H - max(0, dy))
        x_src = slice(max(0, -dx), W - max(0, dx))
        y_dst = slice(max(0, dy), H - max(0, -dy))
        x_dst = slice(max(0, dx), W - max(0, -dx))
        out[..., y_dst, x_dst] = X[..., y_src, x_src]
        return out
    # numpy fallback
    out = np.zeros_like(X)
    H, W = X.shape[-2], X.shape[-1]
    y_src = slice(max(0, -dy), H - max(0, dy))
    x_src = slice(max(0, -dx), W - max(0, dx))
    y_dst = slice(max(0, dy), H - max(0, -dy))
    x_dst = slice(max(0, dx), W - max(0, -dx))
    out[..., y_dst, x_dst] = X[..., y_src, x_src]
    return out


def compare_bar(results: dict, metric_key: str, title: str = "", ax=None, color="#1a73e8"):
    """Bar chart comparing one metric across named runs. `results` is {name: {metric: value}}."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    names = list(results.keys())
    vals = [results[n][metric_key] for n in names]
    ax.bar(names, vals, color=color)
    ax.set_ylabel(metric_key)
    ax.set_title(title or metric_key)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return ax


def compare_table(results: dict):
    """Print a small comparison table to stdout."""
    names = list(results.keys())
    if not names:
        return
    keys = list(results[names[0]].keys())
    header = "model".ljust(14) + "".join(k.ljust(14) for k in keys)
    print(header)
    print("-" * len(header))
    for n in names:
        row = n.ljust(14) + "".join(
            (f"{results[n][k]:.4g}" if isinstance(results[n][k], (int, float)) else str(results[n][k])).ljust(14)
            for k in keys
        )
        print(row)


# ============================================================
# 7. CNN-specific helpers
# ============================================================
def show_conv_kernels(weight, max_filters: int = 16, title: str = "Conv kernels"):
    """Visualize learned conv kernels. weight: (out_C, in_C, k, k) tensor."""
    if hasattr(weight, "detach"):
        weight = weight.detach().cpu().numpy()
    n = min(max_filters, weight.shape[0])
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            k = weight[i].mean(axis=0) if weight.shape[1] > 1 else weight[i, 0]
            ax.imshow(k, cmap="RdBu_r", vmin=-np.abs(k).max(), vmax=np.abs(k).max())
            ax.set_title(f"f{i}", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def show_feature_maps(maps, n_show: int = 8, title: str = "Feature maps"):
    """Visualize first feature map activation. maps: (1, C, H, W)."""
    if hasattr(maps, "detach"):
        maps = maps.detach().cpu().numpy()
    if maps.ndim == 4:
        maps = maps[0]  # drop batch
    n = min(n_show, maps.shape[0])
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            ax.imshow(maps[i], cmap="viridis")
            ax.set_title(f"ch{i}", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def conv_output_shape(H: int, K: int, S: int = 1, P: int = 0) -> int:
    """Compute conv output spatial size."""
    return (H + 2 * P - K) // S + 1


def model_summary(model, input_shape):
    """Print layer-by-layer output shapes and parameter counts.
    input_shape: tuple without batch dim, e.g. (1, 28, 28).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required.")
    hooks = []
    rows = []

    def hook(name):
        def fn(module, inp, out):
            shape = tuple(out.shape) if hasattr(out, "shape") else None
            n_p = sum(p.numel() for p in module.parameters() if p.requires_grad)
            rows.append((name, type(module).__name__, shape, n_p))
        return fn

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf only
            hooks.append(module.register_forward_hook(hook(name or "root")))

    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, *input_shape)
        model(x)
    for h in hooks:
        h.remove()

    print(f"{'layer':<25}{'type':<14}{'output shape':<24}{'params':>10}")
    print("-" * 73)
    total = 0
    for name, typ, shape, n_p in rows:
        print(f"{name:<25}{typ:<14}{str(shape):<24}{n_p:>10,}")
        total += n_p
    print("-" * 73)
    print(f"{'TOTAL':<63}{total:>10,}")
    return total
