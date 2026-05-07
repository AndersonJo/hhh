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

⚠️  수정 규칙 (IMPORTANT — 반드시 준수)
--------------------------------------
1. 함수 삭제 금지: 모든 노트북이 이 파일을 공유합니다.
   어떤 함수가 어느 노트북에서 쓰이는지 전체 추적이 어려우므로,
   기존 함수는 절대 제거하지 않습니다.

2. 하위 호환성 유지: 기존 함수의 시그니처(파라미터 이름·순서·기본값)를
   변경하지 않습니다. 기능을 확장할 때는 새 파라미터를 기본값과 함께
   뒤에 추가하는 방식만 허용합니다.

3. 추가만 허용: 새 함수는 파일 끝 또는 관련 섹션 끝에 추가합니다.
   기존 코드 블록의 위치를 이동하거나 재구성하지 않습니다.
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
    """GPU 가능하면 cuda, 아니면 cpu 반환."""
    if HAS_TORCH and torch.cuda.is_available():
        return torch.device("cuda")
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


def make_spiral(n: int = 300, k: int = 3, seed: int = 0, noise: float = 0.2):
    """다중 클래스 'spiral' 데이터 (n 샘플 per 클래스, k 클래스)."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n * k, 2), dtype=np.float32)
    y = np.zeros(n * k, dtype=np.int64)
    for j in range(k):
        ix = range(n * j, n * (j + 1))
        r = np.linspace(0.0, 1, n)
        t = np.linspace(j * 4, (j + 1) * 4, n) + rng.randn(n) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    if HAS_TORCH:
        return torch.tensor(X), torch.tensor(y)
    return X, y


def make_simple_backprop_2d_data(n_per_class: int = 80, seed: int = 42):
    """Simple 2D binary classification data for manual backprop demo."""
    rng = np.random.RandomState(seed)
    x0 = rng.randn(n_per_class, 2) * 0.6 + np.array([-1.2, -1.0])
    x1 = rng.randn(n_per_class, 2) * 0.6 + np.array([1.2, 1.0])
    X = np.vstack([x0, x1]).astype(np.float32)
    y = np.vstack([np.zeros((n_per_class, 1)), np.ones((n_per_class, 1))]).astype(np.float32)
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
    device=None,
):
    """미니배치 분류 학습 루프. loss 이력을 리스트로 반환."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch가 설치되어 있어야 합니다.")
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    if device is not None:
        dev = device
    elif HAS_TORCH and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
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
    """학습 loss 곡선. history는 리스트 또는 {label: list} dict."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    if isinstance(history, dict):
        for label, vals in history.items():
            ax.plot(vals, lw=2, label=label)
        ax.legend()
    else:
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
# 6. 텍스트 데이터 / TinyShakespeare
# ============================================================
def load_python_code(n_chars: int = 50000, val_frac: float = 0.1,
                      cache_file: str = "python_code.py"):
    """CPython stdlib 소스 코드 로드 (없으면 자동 다운로드).
    Python 코드는 구조가 명확해 생성 품질을 직관적으로 평가할 수 있음.

    Returns:
        train_data, val_data: torch.LongTensor
        stoi, itos: char <-> index 딕셔너리
        vocab_size: int
    """
    import urllib.request
    url = ("https://raw.githubusercontent.com/python/cpython"
           "/main/Lib/inspect.py")
    if not os.path.exists(cache_file):
        urllib.request.urlretrieve(url, cache_file)
    with open(cache_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()[:n_chars]
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    if HAS_TORCH:
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    else:
        data = [stoi[c] for c in text]
    n = int((1 - val_frac) * len(data))
    return data[:n], data[n:], stoi, itos, vocab_size


def load_tinyshakespeare(n_chars: int = 50000, val_frac: float = 0.1,
                          cache_file: str = "tinyshakespeare.txt"):
    """TinyShakespeare 텍스트 로드 (없으면 자동 다운로드).

    Returns:
        train_data, val_data: torch.LongTensor
        stoi, itos: char <-> index 딕셔너리
        vocab_size: int
    """
    import urllib.request
    url = ("https://raw.githubusercontent.com/karpathy/char-rnn"
           "/master/data/tinyshakespeare/input.txt")
    if not os.path.exists(cache_file):
        urllib.request.urlretrieve(url, cache_file)
    with open(cache_file, "r") as f:
        text = f.read()[:n_chars]
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    if HAS_TORCH:
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    else:
        data = [stoi[c] for c in text]
    n = int((1 - val_frac) * len(data))
    return data[:n], data[n:], stoi, itos, vocab_size


# ============================================================
# 7. 다중 곡선 손실 시각화
# ============================================================
def plot_loss_multi(curves: dict, title: str = "", xlabel: str = "Step",
                    ylabel: str = "Loss", ax=None, semilogy: bool = False):
    """여러 모델의 손실 곡선을 한 축에 비교 시각화.

    Args:
        curves: {label: y_values} 또는 {label: (x_values, y_values)}
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    plot_fn = ax.semilogy if semilogy else ax.plot
    for label, vals in curves.items():
        if isinstance(vals, tuple):
            x, y = vals
        else:
            x, y = range(len(vals)), vals
        plot_fn(x, y, label=label, lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return ax


# ============================================================
# 8. MNIST / 벤치마크 / 비교 시각화
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
    """Run fn, return (result, seconds, peak_mb). Peak memory is CUDA-only; else NaN.
    Pass n=<int> to run fn n times and return average latency in ms (scalar)."""
    n = kwargs.pop('n', None)
    label = kwargs.pop('label', '')
    if n is not None:
        fn(*args, **kwargs)  # warmup
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn(*args, **kwargs)
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1000
        if label:
            print(f"{label}: {ms:.3f} ms")
        return ms
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


def accuracy(model, X, y=None, batch_size: int = 256) -> float:
    """Classification accuracy.
    Logits mode: accuracy(logits_tensor, y_true) — skips inference.
    Model mode:  accuracy(model, X, y) — runs model on X in batches.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required.")
    if isinstance(model, torch.Tensor):
        # logits passed directly; X holds the labels
        return (model.argmax(1) == X).float().mean().item()
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


def compare_bar(results: dict = None, metric_key: str = None, title: str = "", ax=None, color="#1a73e8",
                labels=None, values=None, xlabel: str = "", ylabel: str = ""):
    """Bar chart with two calling modes:
    - Simple: compare_bar(labels=[...], values=[...], title=..., xlabel=..., ylabel=...)
    - Dict:   compare_bar({name: {metric: val}}, metric_key, title=...)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    if labels is not None and values is not None:
        names, vals = list(labels), list(values)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    else:
        names = list(results.keys())
        vals = [results[n][metric_key] for n in names]
        ax.set_ylabel(metric_key)
    ax.bar(names, vals, color=color)
    ax.set_title(title or (metric_key or ""))
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return ax


def compare_table(results: dict = None, columns=None, rows=None, title: str = ''):
    """Print a small comparison table to stdout.
    Simple mode: compare_table(columns=[...], rows=[[...], ...], title=...)
    Dict mode:   compare_table({name: {metric: val}})
    """
    if columns is not None and rows is not None:
        if title:
            print(f"\n{title}")
            print("=" * len(title))
        col_widths = [
            max(len(str(columns[i])), max(len(str(r[i])) for r in rows))
            for i in range(len(columns))
        ]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print(fmt.format(*[str(c) for c in columns]))
        print("-" * (sum(col_widths) + 2 * (len(columns) - 1)))
        for row in rows:
            print(fmt.format(*[str(v) for v in row]))
        return
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
# 9. 활성화 함수 / 미분 헬퍼
# ============================================================
def numerical_deriv(f, x, h: float = 1e-5):
    """중앙 차분으로 f의 수치 미분을 계산."""
    return (f(x + h) - f(x - h)) / (2 * h)


def plot_act(name: str, y, dy, x):
    """활성화 함수 f(x)와 f'(x)를 나란히 시각화."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, vals, label in zip(axes, [y, dy], ["f(x)", "f'(x)"]):
        ax.plot(x, vals, linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"{name}:  f(x)")
    axes[1].set_title(f"{name}:  f'(x)")
    plt.tight_layout()
    plt.show()


# ============================================================
# 10. 오토인코더 / 잠재 공간 시각화
# ============================================================
def plot_image_rows(rows_data, row_labels, img_shape=(28, 28),
                    title: str = "", n_cols: int = None, cmaps=None):
    """이미지 배치를 행 단위로 격자 시각화.

    rows_data: list of tensors/arrays, each (N, *) — flat or shaped
    row_labels: list of strings (행 y축 레이블)
    img_shape: flat 이미지일 때 reshape 할 (H, W)
    cmaps: colormap per row (list) or single string; None → 'gray'
    """
    n_rows = len(rows_data)
    if n_cols is None:
        first = rows_data[0]
        n_cols = first.shape[0] if hasattr(first, "shape") else len(first)
    if cmaps is None:
        cmaps = ["gray"] * n_rows
    elif not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.4, n_rows * 1.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    for row, (label, data, cmap) in enumerate(zip(row_labels, rows_data, cmaps)):
        imgs = data.detach().cpu() if hasattr(data, "detach") else data
        if hasattr(imgs, "numpy"):
            imgs = imgs.numpy()
        for col in range(n_cols):
            img = imgs[col]
            if img.ndim == 1:
                img = img.reshape(img_shape)
            elif img.ndim == 3:
                img = img.squeeze()
            axes[row, col].imshow(img, cmap=cmap)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(label, fontsize=8)
    if title:
        plt.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig


def plot_latent_2d(z, y, title: str = "2D Latent Space", ax=None):
    """2D 잠재 벡터 산포도 (클래스별 색상).

    z: (N, 2) tensor or ndarray
    y: (N,) class labels
    """
    z_np = z.detach().cpu().numpy() if hasattr(z, "detach") else np.array(z)
    y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.array(y)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(z_np[:, 0], z_np[:, 1], c=y_np, cmap="tab10", s=15, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Class")
    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return ax


def plot_interpolation(imgs, alphas, title: str = "",
                       label_start: str = "", label_end: str = ""):
    """잠재 공간 보간 이미지를 한 행으로 시각화.

    imgs: list of (H, W) arrays/tensors
    alphas: 각 이미지의 보간 계수 (축 제목으로 표시)
    """
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.4, 1.8))
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        img_np = img.detach().cpu().numpy() if hasattr(img, "detach") else np.array(img)
        ax.imshow(img_np, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{alphas[i]:.1f}", fontsize=7)
    if label_start:
        axes[0].set_title(label_start, fontsize=7)
    if label_end:
        axes[-1].set_title(label_end, fontsize=7)
    if title:
        plt.suptitle(title, y=1.05)
    plt.tight_layout()
    return fig


# ============================================================
# 11. 기울기 흐름 / 어텐션 시각화
# ============================================================
def plot_grad_norms(grad_dicts, titles, colors=None, suptitle: str = ""):
    """레이어별 기울기 크기를 나란히 바 차트로 시각화.

    grad_dicts: list of {layer_name: grad_norm} dicts
    titles: list of strings (각 subplot 제목)
    """
    n = len(grad_dicts)
    if colors is None:
        colors = ["steelblue"] * n
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, grads, title, color in zip(axes, grad_dicts, titles, colors):
        vals = list(grads.values())
        ax.bar(range(len(vals)), vals, color=color, alpha=0.7)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"L{i}" for i in range(len(vals))], rotation=45, fontsize=7)
        ax.set_ylabel("Gradient Norm")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    if suptitle:
        plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    return fig


def plot_attention(attn_weights, chars=None,
                   title: str = "Causal Attention Pattern", suptitle: str = ""):
    """트랜스포머 어텐션 가중치 히트맵 시각화.

    attn_weights: (B, n_heads, T, T) tensor
    chars: optional list of T string labels for axis ticks
    """
    n_heads = attn_weights.size(1)
    T = attn_weights.size(2)
    fig, axes = plt.subplots(1, n_heads, figsize=(7 * n_heads, 6))
    if n_heads == 1:
        axes = [axes]
    for h in range(n_heads):
        w = attn_weights[0, h].cpu().numpy()
        im = axes[h].imshow(w, cmap="Blues", aspect="auto")
        axes[h].set_title(f"Head {h+1} — {title}")
        axes[h].set_xlabel("Key position")
        axes[h].set_ylabel("Query position")
        if chars is not None:
            step = max(1, T // 10)
            ticks = list(range(0, T, step))
            axes[h].set_xticks(ticks)
            axes[h].set_xticklabels([chars[i] for i in ticks], fontsize=7, rotation=45)
            axes[h].set_yticks(ticks)
            axes[h].set_yticklabels([chars[i] for i in ticks], fontsize=7)
        plt.colorbar(im, ax=axes[h])
    if suptitle:
        plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    return fig


# ============================================================
# 12. 분류 결과 시각화
# ============================================================
def plot_loss_and_confusion(train_losses, y_true, y_pred, labels, ylabel: str = "Loss"):
    """손실 곡선(왼쪽)과 혼동 행렬(오른쪽)을 나란히 표시.
    y_true, y_pred 는 1D numpy array."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, color="steelblue")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True, alpha=0.3)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.show()


# ============================================================
# 13. 가중치 초기화 시각화
# ============================================================
# ============================================================
# 14. GPT 텍스트 생성 / 추론 속도
# ============================================================
def generate_text(model, stoi, itos, context_len,
                  seed_str='To be, or', max_len=150, temperature=0.8):
    """GPT 스타일 자동 회귀 텍스트 생성."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required.")
    dev = next(model.parameters()).device
    model.eval()
    idx = torch.tensor([[stoi.get(c, 0) for c in seed_str]], dtype=torch.long, device=dev)
    generated = seed_str
    with torch.no_grad():
        for _ in range(max_len):
            idx_cond = idx[:, -context_len:]
            logits = model(idx_cond)
            probs = torch.nn.functional.softmax(logits[0, -1] / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            generated += itos[next_tok.item()]
            idx = torch.cat([idx, next_tok.unsqueeze(0)], dim=1)
    return generated


def tokens_per_sec(model, vocab_size, context_len, n_tokens=200, temperature=1.0):
    """GPT 모델의 자동 회귀 추론 속도를 tokens/sec 단위로 측정."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required.")
    dev = next(model.parameters()).device
    model.eval()
    idx = torch.randint(0, vocab_size, (1, 10), device=dev)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_tokens):
            idx_cond = idx[:, -context_len:]
            logits = model(idx_cond)
            probs = torch.nn.functional.softmax(logits[0, -1] / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1).unsqueeze(0)
            idx = torch.cat([idx, next_tok], dim=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return n_tokens / (time.time() - t0)


# ============================================================
# 13. 가중치 초기화 시각화
# ============================================================
def plot_dist(W, title: str):
    """가중치 행렬의 분포를 히스토그램으로 표시."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(W.flatten(), bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_signal_stats(means, stds, title: str):
    """레이어별 신호 평균/표준편차를 시각화."""
    layers = range(len(means))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(layers, means, marker="o")
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set_title(f"{title}: Mean per Layer")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(layers, stds, marker="o", color="orange")
    axes[1].axhline(1, color="gray", linestyle="--")
    axes[1].set_title(f"{title}: Std per Layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Std")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 15. Embedding notebook visualization helpers
# ============================================================
def plot_embedding_recall_curve(embed_dims, recall1_scores, recall5_scores):
    """Plot Recall@1 and Recall@5 against embedding dimensions."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(embed_dims, [recall1_scores[d] for d in embed_dims], "b-o", lw=2, label="Recall@1")
    ax.plot(embed_dims, [recall5_scores[d] for d in embed_dims], "g-s", lw=2, label="Recall@5")
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Recall")
    ax.set_title("Retrieval Performance vs Embedding Dimension")
    ax.legend()
    ax.grid(alpha=0.3)
    for d in embed_dims:
        ax.annotate(
            f"{recall1_scores[d]:.3f}",
            (d, recall1_scores[d]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    plt.tight_layout()
    plt.show()


def plot_tsne_comparison(trained_emb, random_emb, labels_np, perplexity: int = 30, n_iter: int = 500, random_state: int = 42):
    """Run t-SNE and compare trained embeddings with random embeddings."""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, max_iter=n_iter)
    trained_2d = tsne.fit_transform(trained_emb)
    random_2d = tsne.fit_transform(random_emb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, emb_2d, title in zip(
        axes,
        [trained_2d, random_2d],
        ["Trained Embedding (Triplet Loss)", "Random Init Embedding"],
    ):
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels_np, cmap="tab10", s=8, alpha=0.8)
        plt.colorbar(scatter, ax=ax, label="Digit class")
        ax.set_title(title)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

    plt.suptitle("t-SNE: Learned vs Random Embeddings (dim=32)", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_hard_negative_comparison(losses_rand, losses_hard, rand_r1: float, hard_r1: float):
    """Compare random negatives and hard negatives by loss curves and Recall@1."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(losses_rand, label="Random negatives", color="blue", alpha=0.8)
    axes[0].plot(losses_hard, label="Hard negatives", color="red", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Triplet Loss")
    axes[0].set_title("Training Curves: Random vs Hard Negatives")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(["Random\nNegatives", "Hard\nNegatives"], [rand_r1, hard_r1], color=["blue", "red"])
    for i, v in enumerate([rand_r1, hard_r1]):
        axes[1].text(i, v * 0.97, f"{v:.4f}", ha="center", va="top", color="white", fontweight="bold")
    axes[1].set_ylabel("Recall@1")
    axes[1].set_title("Final Recall@1: Random vs Hard Negatives")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Hard Negative Mining Comparison", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_nn_retrieval_grid(
    query_embs,
    gallery_embs,
    X_query,
    y_query,
    X_gallery,
    y_gallery,
    query_indices,
    topk: int = 5,
    figsize=(12, 10),
):
    """Visualize nearest-neighbor retrieval results for selected query indices."""
    n_rows = len(query_indices)
    fig, axes = plt.subplots(n_rows, topk + 1, figsize=figsize)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, q_idx in enumerate(query_indices):
        q_emb = query_embs[q_idx:q_idx + 1]
        dists = torch.cdist(q_emb, gallery_embs)[0]
        topk_idx = dists.topk(topk, largest=False).indices

        axes[row, 0].imshow(X_query[q_idx, 0].numpy(), cmap="gray")
        axes[row, 0].set_title(f"Query\n(digit {y_query[q_idx].item()})", fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 0].patch.set_edgecolor("gold")
        axes[row, 0].patch.set_linewidth(3)

        for col, ridx in enumerate(topk_idx):
            ridx = ridx.item()
            retrieved_label = y_gallery[ridx].item()
            match = retrieved_label == y_query[q_idx].item()
            axes[row, col + 1].imshow(X_gallery[ridx, 0].numpy(), cmap="gray")
            color = "green" if match else "red"
            axes[row, col + 1].set_title(f"#{col + 1} ({retrieved_label})", fontsize=8, color=color)
            axes[row, col + 1].axis("off")

    plt.suptitle("Nearest Neighbor Retrieval: Query \u2192 Top-5 (green=correct, red=wrong)", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()


# ============================================================
# 16. Foundation notebook visualization helpers
# ============================================================
def plot_activation_overview(act_vals: dict, grad_vals: dict, x_np, colors=None):
    """활성화 함수와 도함수 비교 플롯."""
    if colors is None:
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (name, vals), c in zip(act_vals.items(), colors):
        axes[0].plot(x_np, vals, label=name, color=c, lw=2)
    axes[0].set_title("Activation Functions")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axvline(0, color="k", lw=0.5)

    for (name, grads), c in zip(grad_vals.items(), colors):
        axes[1].plot(x_np, grads, label=name, color=c, lw=2)
    axes[1].set_title("Derivatives of Activation Functions")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f'(x)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].axvline(0, color="k", lw=0.5)
    plt.tight_layout()
    plt.show()


def plot_dead_neuron_bars(dead_counts, normal_counts):
    """죽은 뉴런 수 비교 바차트."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x_pos = np.arange(4)
    ax.bar(x_pos - 0.2, dead_counts, 0.4, label="Bias=-5.0 (dying ReLU)", color="red", alpha=0.8)
    ax.bar(x_pos + 0.2, normal_counts, 0.4, label="Bias=0.0 (normal)", color="green", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Layer {i+1}" for i in range(4)])
    ax.set_ylabel("Dead Neuron Count (out of 128)")
    ax.set_title("Dying ReLU: Dead Neurons per Layer")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_activation_training_summary(all_results: dict, colors=None):
    """활성화 함수별 손실 곡선 + 정확도 바차트."""
    if colors is None:
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (act_name, res), c in zip(all_results.items(), colors):
        axes[0].plot(res["loss_hist"], label=act_name, color=c, lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss by Activation Function")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    names = list(all_results.keys())
    accs = [all_results[n]["accuracy"] for n in names]
    bars = axes[1].bar(names, accs, color=colors, alpha=0.85)
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("Final Test Accuracy by Activation Function")
    for bar, acc in zip(bars, accs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_grad_norm_per_activation(all_results: dict, layer_names):
    """활성화 함수별 레이어 그래디언트 노름 비교."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    for ax_i, (act_name, res) in enumerate(all_results.items()):
        for layer_i in range(5):
            if layer_i in res["grad_hist"] and len(res["grad_hist"][layer_i]) > 0:
                axes[ax_i].plot(res["grad_hist"][layer_i], label=layer_names[layer_i], lw=1.5)
        axes[ax_i].set_title(act_name)
        axes[ax_i].set_xlabel("Epoch")
        if ax_i == 0:
            axes[ax_i].set_ylabel("Gradient Norm")
        axes[ax_i].legend(fontsize=7)
        axes[ax_i].grid(alpha=0.3)
    plt.suptitle("Per-Layer Gradient Norm During Training", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_loss_family_and_smoothing(p, ce, fl_g1, fl_g2, fl_g5, K=10, eps_vals=None, true_cls: int = 3):
    """CE/Focal 곡선 + 라벨 스무딩 분포."""
    if eps_vals is None:
        eps_vals = [0.0, 0.05, 0.1, 0.2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(p, ce, label="CE (gamma=0)", lw=2, color="black")
    axes[0].plot(p, fl_g1, label="Focal (gamma=1)", lw=2, color="blue")
    axes[0].plot(p, fl_g2, label="Focal (gamma=2)", lw=2, color="orange")
    axes[0].plot(p, fl_g5, label="Focal (gamma=5)", lw=2, color="red")
    axes[0].set_xlabel("Predicted probability p (for true class)")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-Entropy vs Focal Loss")
    axes[0].set_ylim(0, 5)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    x_pos = np.arange(K)
    width = 0.2
    for i, eps in enumerate(eps_vals):
        target = np.ones(K) * eps / K
        target[true_cls] += (1 - eps)
        axes[1].bar(x_pos + i * width, target, width, label=f"eps={eps}", alpha=0.8)
    axes[1].set_xlabel("Class index")
    axes[1].set_ylabel("Target probability")
    axes[1].set_title("Label Smoothing Effect on Target Distribution (K=10)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_xticks(x_pos + 1.5 * width)
    axes[1].set_xticklabels([f"C{i}" for i in range(K)], fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_per_class_accuracy_grouped(per_class_accs: dict, colors=None):
    """손실 함수별 클래스 정확도 바차트."""
    if colors is None:
        colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(10)
    width = 0.25
    for i, (name, accs) in enumerate(per_class_accs.items()):
        ax.bar(x_pos + i * width, accs, width, label=name, color=colors[i], alpha=0.85)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy on Test Set\n(training: 90% class-0, 10% others)")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f"C{i}" for i in range(10)])
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(0.0, 0.5, "Majority\nclass", ha="center", fontsize=8, color="gray")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_calibration_bars(ece_results: dict, colors=None):
    """모델별 calibration diagram."""
    if colors is None:
        colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    bins_mid = np.linspace(0.05, 0.95, 10)
    for ax, (name, (ece, bin_accs, bin_confs)), c in zip(axes, ece_results.items(), colors):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.bar(bins_mid, bin_accs, 0.08, alpha=0.6, color=c, label="Accuracy")
        ax.step(bins_mid, bin_confs, where="mid", color="navy", lw=1.5, label="Confidence")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{name}\nECE={ece:.4f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Calibration Diagrams", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_xor_training_curve(loss_hist):
    """XOR 학습 손실 곡선."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_hist, lw=2, color="steelblue")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Value-MLP Training on XOR")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_gradient_flow_by_depth(depths, act_fns: dict, build_deep_net, measure_grad_norms, x_dummy, y_dummy):
    """깊이별 tanh/relu 그래디언트 노름 비교."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for ax, n_layers in zip(axes, depths):
        for act_name, act_cls in act_fns.items():
            seed_all(42)
            net = build_deep_net(n_layers, act_cls)
            norms = measure_grad_norms(net, x_dummy, y_dummy)
            layer_idx = list(range(len(norms)))
            ax.plot(layer_idx, norms, marker="o", label=act_name, lw=2)
        ax.set_title(f"{n_layers}-Layer Network")
        ax.set_xlabel("Layer index (0=input side)")
        ax.set_ylabel("Gradient Norm")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_yscale("log")
    plt.suptitle("Gradient Flow: Tanh vs ReLU at Different Depths", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_optimizer_trajectory(W1, W2, Z, trajectories: dict):
    """안장점 손실면에서 옵티마이저 궤적 표시."""
    fig, ax = plt.subplots(figsize=(8, 7))
    contour = ax.contourf(W1, W2, Z, levels=40, cmap="RdYlBu", alpha=0.7)
    plt.colorbar(contour, ax=ax, label="Loss")
    ax.contour(W1, W2, Z, levels=[0], colors="black", linewidths=1.5, linestyles="--")

    traj_colors = {"SGD": "#e41a1c", "Momentum": "#377eb8", "Adam": "#4daf4a"}
    for name, traj in trajectories.items():
        xs, ys = zip(*traj)
        ax.plot(xs, ys, "-o", color=traj_colors[name], ms=3, lw=1.5, label=name, alpha=0.85)
        ax.plot(xs[0], ys[0], "s", color=traj_colors[name], ms=8)

    ax.plot(0, 0, "k*", ms=15, zorder=10, label="Saddle point (0,0)")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_title("Optimizer Trajectories on Saddle-Point Loss\nL(w1,w2) = 0.1·w1² - w2² + 0.5")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_optimizer_grid_results(opt_names, lr_grid, grid_results, best_lr_results, summary, colors=None):
    """옵티마이저-LR 히트맵 + best LR 수렴곡선."""
    if colors is None:
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]
    accs_matrix = np.array([
        [grid_results[opt][lr]["final_acc"] for lr in lr_grid]
        for opt in opt_names
    ])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(accs_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=axes[0], label="Test Accuracy")
    axes[0].set_xticks(range(len(lr_grid)))
    axes[0].set_xticklabels([f"{lr:.0e}" for lr in lr_grid])
    axes[0].set_yticks(range(len(opt_names)))
    axes[0].set_yticklabels(opt_names)
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Optimizer")
    axes[0].set_title("Test Accuracy: Optimizer × LR Grid")
    for i in range(len(opt_names)):
        for j in range(len(lr_grid)):
            axes[0].text(j, i, f"{accs_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color="black")

    for (opt_name, res), c in zip(best_lr_results.items(), colors):
        axes[1].plot(
            res["acc_hist"],
            label=f"{opt_name} (lr={summary[opt_name]['best_lr']:.0e})",
            color=c,
            lw=2,
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train Accuracy")
    axes[1].set_title("Convergence at Best LR per Optimizer")
    axes[1].axhline(0.9, color="gray", linestyle="--", alpha=0.7, label="90% threshold")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_adam_lr_robustness(adam_big_lr, sgd_best_lr, adam_best_lr):
    """Adam LR 강건성 비교 플롯."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(adam_big_lr["loss_hist"], label="Adam (lr=1.0)", color="orange", lw=2)
    axes[0].plot(sgd_best_lr["loss_hist"], label="SGD  (lr=1e-2)", color="blue", lw=2)
    axes[0].plot(adam_best_lr["loss_hist"], label="Adam (lr=1e-3)", color="green", lw=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss: Adam (wrong LR=1.0) vs SGD (right LR)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(adam_big_lr["acc_hist"], label="Adam (lr=1.0)", color="orange", lw=2)
    axes[1].plot(sgd_best_lr["acc_hist"], label="SGD  (lr=1e-2)", color="blue", lw=2)
    axes[1].plot(adam_best_lr["acc_hist"], label="Adam (lr=1e-3)", color="green", lw=2, linestyle="--")
    axes[1].axhline(0.9, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train Accuracy")
    axes[1].set_title("Accuracy: Adam Robustness to Large LR")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.suptitle("Fun Experiment: Is Adam robust to LR=1.0?", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_numdiff_error_curve(h_values, errors, grad_autograd, grad_analytic):
    """수치 미분 오차-스텝 크기 관계."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(h_values, errors, "b-o", ms=4, label="Numerical diff error")
    ax.axhline(
        abs(grad_autograd - grad_analytic) + 1e-16,
        color="r",
        linestyle="--",
        label="Autograd error (~machine eps)",
    )
    ax.set_xlabel("Step size h")
    ax.set_ylabel("|error|")
    ax.set_title("Numerical Differentiation Error vs Step Size")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_chain_rule_comparison(x_vals, f_vals, g_vals, dh_dx_manual, dh_dx_auto):
    """합성함수와 체인룰 미분 비교 플롯."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(x_vals, f_vals, "b")
    axes[0].set_title("f(x) = x^3 - 2x")
    axes[0].set_xlabel("x")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_vals, g_vals, "g")
    axes[1].set_title("h(x) = tanh(f(x))")
    axes[1].set_xlabel("x")
    axes[1].grid(alpha=0.3)

    axes[2].plot(x_vals, dh_dx_manual, "r-", lw=2, label="Manual chain rule")
    axes[2].plot(x_vals, dh_dx_auto, "b--", lw=1.5, label="Autograd")
    axes[2].set_title("dh/dx - Chain Rule")
    axes[2].set_xlabel("x")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_loss_surface_with_gradients(W1, W2, Z, W1q, W2q, G1, G2):
    """2D 손실면 + gradient vector field."""
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(W1, W2, Z, levels=40, cmap="viridis", alpha=0.85)
    plt.colorbar(contour, ax=ax, label="Loss")
    ax.contour(W1, W2, Z, levels=15, colors="white", linewidths=0.5, alpha=0.4)
    ax.quiver(
        W1q,
        W2q,
        -G1,
        -G2,
        color="red",
        alpha=0.6,
        scale=80,
        width=0.003,
        label="Gradient descent direction",
    )
    ax.plot(1.0, 0.5, "w*", ms=15, label="Global minimum (approx)")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_title("Loss Surface: L(w1,w2) with Gradient Vectors")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_jacobian_hessian(J, H_np):
    """Jacobian/Hessian heatmap 시각화."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    j_np = J.detach().cpu().numpy() if hasattr(J, "detach") else np.array(J)

    im0 = axes[0].imshow(j_np, cmap="RdBu", vmin=-3, vmax=3, aspect="auto")
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title("Jacobian (3x3)\nf: R^3->R^3 at x=[1,2,0.5]")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["d/dx1", "d/dx2", "d/dx3"])
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["f1", "f2", "f3"])
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{j_np[i, j]:.2f}", ha="center", va="center", fontsize=10)

    im1 = axes[1].imshow(H_np, cmap="RdBu", aspect="auto")
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title("Hessian (2x2)\nL(w1,w2) at minimum")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["w1", "w2"])
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["w1", "w2"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{H_np[i, j]:.3f}", ha="center", va="center", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_taylor_approximation(x_range, true_f, taylor1, taylor2, x0_val, f0):
    """Taylor 1차/2차 근사 비교 플롯."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_range, true_f, "b-", lw=2, label="f(x) true")
    ax.plot(x_range, taylor1, "r--", lw=1.5, label="1st-order Taylor")
    ax.plot(x_range, taylor2, "g--", lw=1.5, label="2nd-order Taylor")
    ax.axvline(x0_val, color="gray", linestyle=":", alpha=0.7)
    ax.scatter([x0_val], [f0], color="k", zorder=5, s=80, label=f"x0={x0_val}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Taylor Expansion: 1st vs 2nd Order Approximation")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 17. 비교 요약표 시각화
# ============================================================
# ============================================================
# 18. 정규화 비교 시각화 (norm notebooks)
# ============================================================
def plot_preln_postln_comparison(grad_post, grad_pre):
    """Pre-LN vs Post-LN 그래디언트 흐름 비교 바 차트."""
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ['Post-LN\n(Original Transformer)', 'Pre-LN\n(GPT-3, LLaMA)']
    values = [grad_post, grad_pre]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(names, values, color=colors, alpha=0.85, width=0.4)
    y_max = max(values) if max(values) > 0 else 0.1
    ax.set_ylabel('Input Gradient Norm')
    ax.set_title('Gradient Flow: Post-LN vs Pre-LN\n(Higher = Better gradient flow to input)')
    for bar, val in zip(bars, values):
        label = f'{val:.4f}' if val > 1e-6 else f'{val:.2e}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_max * 0.03,
                label, ha='center', va='bottom', fontsize=14, fontweight='bold')
    if grad_post < 1e-6 and grad_pre > 1e-6:
        msg = 'Post-LN: gradient vanished!\nPre-LN: gradient preserved ✓'
    elif grad_post > 0:
        ratio = grad_pre / grad_post
        msg = f'Pre-LN has {ratio:.0f}x better\ngradient flow'
    else:
        msg = 'Pre-LN preserves gradient flow'
    ax.text(0.5, 0.62, msg, transform=ax.transAxes, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#fffde7', alpha=0.9))
    ax.set_ylim(0, y_max * 1.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_adain_transfer(content_means, content_stds, style_means, style_stds, out_means, out_stds):
    """AdaIN 스타일 전달: content/style/output 통계 비교 막대 차트."""
    n = len(content_means)
    x = np.arange(n)
    w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    specs = [
        (content_means, style_means, out_means, 'Mean',          'Output mean ≈ Style mean'),
        (content_stds,  style_stds,  out_stds,  'Std (spread)',  'Output std  ≈ Style std'),
    ]
    for ax, (c_vals, s_vals, o_vals, ylabel, note) in zip(axes, specs):
        ax.bar(x - w, c_vals, w, label='Content', color='steelblue', alpha=0.8)
        ax.bar(x,     s_vals, w, label='Style',   color='#e67e22',   alpha=0.8)
        ax.bar(x + w, o_vals, w, label='Output',  color='#2ecc71',   alpha=0.8)
        ax.set_xlabel('Sample')
        ax.set_ylabel(ylabel)
        ax.set_title(f'AdaIN {ylabel} Transfer\n({note})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Sample {i}' for i in range(n)])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        for i in range(n):
            ax.annotate('', xy=(i + w, o_vals[i]), xytext=(i, s_vals[i]),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.8))
    plt.suptitle('AdaIN: Style Statistics are Transferred to Output', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_rmsnorm_summary(t_ln, t_rms, scale_c_values, scale_diffs):
    """RMSNorm 핵심 특성 요약: 속도 비교 + 스케일 불변성."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    names = ['LayerNorm', 'RMSNorm']
    times_ms = [t_ln, t_rms]
    colors = ['#e74c3c', '#2ecc71']
    bars = axes[0].bar(names, times_ms, color=colors, alpha=0.85, width=0.4)
    speedup = t_ln / t_rms
    axes[0].set_ylabel('Avg Inference Time (ms)')
    axes[0].set_title(f'Speed: LayerNorm vs RMSNorm\nRMSNorm is {speedup:.2f}x faster')
    for bar, val in zip(bars, times_ms):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(times_ms) * 0.01,
                     f'{val:.2f} ms', ha='center', va='bottom', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    x_pos = np.arange(len(scale_c_values))
    axes[1].bar(x_pos, scale_diffs, color='steelblue', alpha=0.85)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'c={c}' for c in scale_c_values])
    axes[1].set_ylabel('Max Difference from Baseline')
    axes[1].set_title('Scale Invariance: RMSNorm(c·x) vs RMSNorm(x)\n(all differences ≈ 0 = perfectly scale invariant)')
    max_diff = max(scale_diffs) if max(scale_diffs) > 0 else 1e-8
    axes[1].set_ylim(0, max_diff * 3)
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    for i, (c, d) in enumerate(zip(scale_c_values, scale_diffs)):
        axes[1].text(i, d + max_diff * 0.1, f'{d:.1e}',
                     ha='center', va='bottom', fontsize=9)
    axes[1].text(0.5, 0.7, 'All ≈ 0\n→ Perfectly Scale Invariant!',
                 transform=axes[1].transAxes, ha='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='#fffde7', alpha=0.9))
    axes[1].grid(axis='y', alpha=0.3)
    plt.suptitle('RMSNorm Key Properties', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_final_norm_comparison(acc_names, accs, speed_names, speeds):
    """정규화 방법 종합 비교: MNIST 정확도 + 추론 속도."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    acc_vals = [accs[n] for n in acc_names]
    best = max(acc_vals)
    bars = axes[0].bar(acc_names, acc_vals, color=palette[:len(acc_names)], alpha=0.85)
    axes[0].axhline(best, color='gold', linestyle='--', linewidth=1.5,
                    label=f'Best: {best:.3f}')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_ylim(0, 1.12)
    axes[0].set_title('MNIST Test Accuracy by Normalization\n(20 epochs, 2,000 training samples)')
    for bar, val in zip(bars, acc_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    spd_vals = [speeds[n] for n in speed_names]
    bars2 = axes[1].bar(speed_names, spd_vals,
                        color=['#3498db', '#2ecc71', '#e74c3c'][:len(speed_names)], alpha=0.85)
    axes[1].set_ylabel('Avg Inference Time (ms)')
    axes[1].set_title('Inference Speed (batch=64, seq=256, d=512)\nLower is faster')
    for bar, val in zip(bars2, spd_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(spd_vals) * 0.01,
                     f'{val:.2f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    plt.suptitle('Normalization Methods — Final Summary', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_bn_experiment_summary(hist_no_bn, hist_bn):
    """BN 학습 실험 핵심 결과 요약: 수렴 속도 + 안정성 비교 바 차트."""
    no_bn = np.array(hist_no_bn)
    bn    = np.array(hist_bn)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metric_names = ['Initial Loss\n(ep 0)', 'Final Loss\n(ep last)', 'Best Loss']
    no_bn_vals   = [no_bn[0],   no_bn[-1],  no_bn.min()]
    bn_vals      = [bn[0],      bn[-1],     bn.min()]
    x = np.arange(len(metric_names))
    w = 0.35
    bars1 = axes[0].bar(x - w / 2, no_bn_vals, w, label='No BN',   color='#e74c3c', alpha=0.85)
    bars2 = axes[0].bar(x + w / 2, bn_vals,    w, label='With BN', color='#2ecc71', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_names)
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison\n(Spiral dataset, depth=8, lr=0.05)')
    axes[0].legend()
    all_bars = list(bars1) + list(bars2)
    all_vals = no_bn_vals + bn_vals
    for bar, val in zip(all_bars, all_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(all_vals) * 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    stab_vals = [no_bn[50:].std(), bn[50:].std()]
    colors    = ['#e74c3c', '#2ecc71']
    bars3 = axes[1].bar(['No BN', 'With BN'], stab_vals, color=colors, alpha=0.85, width=0.4)
    axes[1].set_ylabel('Loss Std   (lower = more stable)')
    axes[1].set_title('Training Stability\n(Std of loss after ep 50 — lower is better)')
    y_top = max(stab_vals)
    for bar, val in zip(bars3, stab_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + y_top * 0.02,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    if stab_vals[1] < stab_vals[0] and stab_vals[1] > 0:
        ratio = stab_vals[0] / stab_vals[1]
        msg = f'With BN is {ratio:.1f}x\nmore stable'
    elif stab_vals[1] == 0:
        msg = 'With BN: perfectly stable!\n(std = 0)'
    else:
        msg = 'Similar stability'
    axes[1].text(0.5, 0.65, msg, transform=axes[1].transAxes, ha='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='#fffde7', alpha=0.9))
    axes[1].set_ylim(0, y_top * 1.5)
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle('BatchNorm Effect on Deep Network Training', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_summary_table(columns, rows, title: str = '', header_color: str = '#4472C4'):
    """비교 요약표를 matplotlib 테이블로 시각화.

    columns: 헤더 리스트
    rows: [[값, ...], ...] 형태의 데이터 리스트
    """
    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.55 + 1))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=columns, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for j in range(len(columns)):
        table[0, j].set_facecolor(header_color)
        table[0, j].set_text_props(color='white', fontweight='bold')
    if title:
        ax.set_title(title, fontsize=13, pad=10)
    plt.tight_layout()
    plt.show()


# ============================================================
# 19. RAG / 벡터 검색 유틸리티
# ============================================================

# --- 검색 평가 유틸리티 ---
def compute_mrr(retrieved_lists, relevant_sets, k=10):
    """Mean Reciprocal Rank@k over a list of queries."""
    mrrs = []
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        rr = 0.0
        for rank, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                rr = 1.0 / (rank + 1)
                break
        mrrs.append(rr)
    return float(np.mean(mrrs))


def compute_precision_at_k(retrieved_lists, relevant_sets, k=5):
    """Precision@k averaged over queries."""
    ps = []
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        hits = sum(1 for d in retrieved[:k] if d in relevant)
        ps.append(hits / k)
    return float(np.mean(ps))


def compute_recall_at_k(pred_indices, gt_indices, k=10):
    """Mean Recall@k for ANN index comparison (pred and gt are lists of index arrays)."""
    recalls = []
    for pred, gt in zip(pred_indices, gt_indices):
        pred_set = set(pred[:k])
        gt_set = set(gt[:k])
        recalls.append(len(pred_set & gt_set) / len(gt_set))
    return float(np.mean(recalls))


def measure_query_latency(search_fn, queries, n_warmup=10):
    """Return avg query latency in ms. search_fn accepts a single (1, D) float32 array."""
    import time as _time
    for q in queries[:n_warmup]:
        search_fn(q[None])
    times = []
    for q in queries:
        t0 = _time.perf_counter()
        search_fn(q[None])
        times.append((_time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def token_diversity(samples):
    """Fraction of unique token types across all text samples."""
    all_toks = []
    for s in samples:
        all_toks.extend(s.lower().split())
    if not all_toks:
        return 0.0
    return len(set(all_toks)) / len(all_toks)


# --- RAG 코퍼스 ---
CORPUS = [
    # 0
    "Backpropagation is the core algorithm for training neural networks. "
    "It computes the gradient of the loss function with respect to each weight using the chain rule. "
    "These gradients are then used by an optimizer such as SGD or Adam to update the weights. "
    "Without backpropagation, training deep networks would be computationally infeasible.",

    # 1
    "The attention mechanism allows a model to focus on relevant parts of the input sequence. "
    "It computes a weighted sum of values, where weights are determined by the compatibility "
    "between a query and a set of keys. Self-attention lets each position attend to all other positions. "
    "The Transformer architecture is built entirely on attention without recurrence.",

    # 2
    "Gradient descent is an optimization algorithm that iteratively moves parameters "
    "in the direction of the negative gradient. Mini-batch gradient descent uses a subset "
    "of training data at each step, balancing speed and stability. "
    "The learning rate controls the step size, and choosing it well is critical for convergence.",

    # 3
    "Convolutional Neural Networks (CNNs) are designed for processing grid-like data such as images. "
    "They use shared-weight filters that slide over the input to detect local patterns. "
    "Pooling layers reduce spatial dimensions, making the network translation-invariant. "
    "Architectures like ResNet and VGG have achieved state-of-the-art results on image classification.",

    # 4
    "Recurrent Neural Networks (RNNs) process sequential data by maintaining a hidden state. "
    "The hidden state is updated at each timestep and passed to the next step, enabling the model "
    "to capture temporal dependencies. However, vanilla RNNs suffer from the vanishing gradient problem "
    "for long sequences, which LSTMs and GRUs address with gating mechanisms.",

    # 5
    "Batch normalization normalizes the activations within a mini-batch to have zero mean and unit variance. "
    "It introduces learnable scale and shift parameters, allowing the network to undo the normalization "
    "if needed. Batch norm significantly accelerates training and acts as a mild regularizer. "
    "It is applied between the linear transformation and the activation function.",

    # 6
    "Dropout is a regularization technique that randomly zeros out activations during training "
    "with a probability p. This forces the network to learn redundant representations and "
    "prevents co-adaptation of neurons. At inference time, all neurons are used but their "
    "outputs are scaled by (1-p) to maintain the expected value.",

    # 7
    "The Transformer model was introduced in 'Attention is All You Need' (Vaswani et al., 2017). "
    "It uses multi-head self-attention and feed-forward layers stacked in an encoder-decoder structure. "
    "Positional encodings are added to give the model a sense of token order. "
    "Transformers have become the dominant architecture in NLP and are now used in vision too.",

    # 8
    "Transfer learning involves taking a model pretrained on a large dataset and fine-tuning it "
    "on a specific downstream task. This approach is highly effective when labeled data is scarce. "
    "Models like BERT, GPT, and ResNet serve as powerful feature extractors. "
    "Fine-tuning updates only part of the model or adds task-specific heads.",

    # 9
    "The cross-entropy loss is commonly used for classification tasks. "
    "It measures the distance between the predicted probability distribution and the true distribution. "
    "For binary classification, it reduces to the binary cross-entropy. "
    "Lower cross-entropy means the model's predicted probabilities are closer to the true labels.",

    # 10
    "The Adam optimizer combines momentum and adaptive learning rates. "
    "It maintains first and second moment estimates of the gradients and uses them to scale updates. "
    "Adam is robust to hyperparameter choices and works well across many tasks. "
    "Weight decay can be added to Adam (AdamW) to improve generalization.",

    # 11
    "Overfitting occurs when a model memorizes training data and fails to generalize. "
    "It is indicated by low training loss but high validation loss. "
    "Techniques to combat overfitting include dropout, weight decay, data augmentation, "
    "early stopping, and reducing model complexity.",

    # 12
    "Embeddings are dense vector representations of discrete objects such as words or sentences. "
    "Word embeddings like Word2Vec and GloVe map semantically similar words to nearby vectors. "
    "Sentence embeddings encode the meaning of an entire sentence into a fixed-size vector. "
    "These representations are learned from large corpora and capture rich semantic relationships.",

    # 13
    "BERT (Bidirectional Encoder Representations from Transformers) is pretrained using "
    "masked language modeling and next sentence prediction. Its bidirectional context makes it "
    "particularly strong for understanding tasks. Fine-tuned BERT has achieved top results on "
    "question answering, named entity recognition, and text classification.",

    # 14
    "GPT models are autoregressive language models that predict the next token given previous tokens. "
    "They are trained using next-token prediction on massive text corpora. "
    "GPT-3 and GPT-4 demonstrate few-shot and zero-shot generalization across diverse tasks. "
    "The scaling of data, model size, and compute has proven critical to GPT's capabilities.",

    # 15
    "Activation functions introduce nonlinearity into neural networks, enabling them to model "
    "complex functions. ReLU (Rectified Linear Unit) outputs max(0, x) and avoids the vanishing "
    "gradient problem. Variants like Leaky ReLU, GELU, and Swish address ReLU's dying neuron issue "
    "and are widely used in modern architectures.",

    # 16
    "Data augmentation artificially expands the training set by applying transformations "
    "such as random cropping, flipping, rotation, and color jitter. "
    "It improves generalization, especially when labeled data is limited. "
    "For NLP, augmentation includes synonym replacement, back-translation, and random deletion.",

    # 17
    "The vanishing gradient problem arises in deep networks when gradients become exponentially small "
    "as they propagate backward. This makes early layers learn very slowly or not at all. "
    "Residual connections (skip connections) in ResNet provide gradient highways to alleviate this. "
    "Careful weight initialization (e.g., He initialization) also helps.",

    # 18
    "Reinforcement Learning from Human Feedback (RLHF) is used to align language models with "
    "human preferences. A reward model is trained on human comparisons of model outputs. "
    "The language model is then fine-tuned using PPO to maximize reward. "
    "RLHF was central to training InstructGPT and ChatGPT.",

    # 19
    "Mixture of Experts (MoE) is a technique that activates only a subset of model parameters "
    "per token, using a learned gating network to route each token to the best experts. "
    "MoE allows a very large total parameter count while keeping compute per token manageable. "
    "Mixtral 8x7B and GPT-4 are believed to use MoE architectures.",
]


# --- RAG 파이프라인 평가 ---
def plot_rag_pipeline_eval(rag_labels, norag_labels, results, acc_rag, acc_norag):
    """RAG vs No-RAG: accuracy bars, per-query correctness scatter, retrieval score histogram."""
    all_scores = [s for r in results for s in r['top_scores']]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(['RAG', 'No-RAG'], [acc_rag, acc_norag], color=['steelblue', 'tomato'])
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Keyword Accuracy (10 Queries)')
    for i, v in enumerate([acc_rag, acc_norag]):
        axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12)

    x = range(len(results))
    axes[1].scatter(x, rag_labels, label='RAG', color='steelblue', s=80, zorder=3)
    axes[1].scatter(x, norag_labels, label='No-RAG', color='tomato', s=80, marker='x', zorder=3)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['Incorrect', 'Correct'])
    axes[1].set_xlabel('Query index')
    axes[1].set_title('Per-Query Correctness')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].hist(all_scores, bins=15, color='mediumpurple', edgecolor='white')
    axes[2].set_xlabel('Cosine Similarity')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Top-3 Retrieval Score Distribution')
    axes[2].axvline(np.mean(all_scores), color='red', linestyle='--',
                    label=f'Mean={np.mean(all_scores):.2f}')
    axes[2].legend()

    plt.suptitle('RAG Pipeline Evaluation', fontsize=14)
    plt.tight_layout()
    plt.show()
    print(f"Mean retrieval score: {np.mean(all_scores):.4f}")


# --- 벡터 검색 비교 ---
def plot_vector_search_comparison(methods, recalls5, recalls10, latencies_ms, build_times_ms,
                                   colors=None, title='Retrieval Method Comparison'):
    """Three subplots: Recall@5 vs @10 grouped bar, latency, build time."""
    if colors is None:
        colors = ['#4e79a7', '#f28e2b', '#59a14f']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.arange(len(methods))
    w = 0.35
    axes[0].bar(x - w / 2, recalls5, w, label='Recall@5', color=colors)
    axes[0].bar(x + w / 2, recalls10, w, label='Recall@10', color=colors, alpha=0.5, edgecolor='k')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel('Recall')
    axes[0].set_title('Recall@5 vs Recall@10')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(methods, latencies_ms, color=colors)
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title('Query Latency')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(latencies_ms):
        axes[1].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    axes[2].bar(methods, build_times_ms, color=colors)
    axes[2].set_ylabel('Build Time (ms)')
    axes[2].set_title('Index Build Time')
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(build_times_ms):
        axes[2].text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_recall_vs_dim(dims, recall_by_dim, ylabel='Recall@10',
                       title='Recall vs Embedding Dimension (PCA)'):
    """Line chart of recall at each embedding dimension with value annotations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dims, [recall_by_dim[d] for d in dims], 'o-', color='steelblue',
            linewidth=2, markersize=8)
    for d in dims:
        ax.annotate(f'{recall_by_dim[d]:.3f}', xy=(d, recall_by_dim[d]),
                    xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(dims)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- ANN 인덱스 비교 (21_02) ---
def plot_pareto_curve(results, title='Pareto Curve: Recall@10 vs Latency'):
    """Scatter plot of latency vs Recall@10 for ANN index configs.
    results: list of dicts with keys 'index_type', 'latency_ms', 'recall', 'label'.
    """
    marker_map = {'Flat': '*', 'IVF': 's', 'HNSW': '^'}
    color_map = {'Flat': '#e15759', 'IVF': '#4e79a7', 'HNSW': '#59a14f'}

    fig, ax = plt.subplots(figsize=(9, 6))
    for itype in ['Flat', 'IVF', 'HNSW']:
        pts = [r for r in results if r['index_type'] == itype]
        xs = [p['latency_ms'] for p in pts]
        ys = [p['recall'] for p in pts]
        ax.scatter(xs, ys, marker=marker_map[itype], color=color_map[itype],
                   s=120, label=itype, zorder=3, edgecolors='k', linewidths=0.5)
        for p in pts:
            short = (p['label'].replace('nlist=', 'nl=').replace('nprobe=', 'np=')
                     .replace('HNSW ', '').replace('IVF ', ''))
            ax.annotate(short, (p['latency_ms'], p['recall']),
                        fontsize=6, xytext=(3, 3), textcoords='offset points', alpha=0.75)
    ax.set_xlabel('Avg Query Latency (ms)', fontsize=12)
    ax.set_ylabel('Recall@10', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pq_comparison(results_sec2):
    """PQ compression: 3 subplots — index size, latency, compression ratio vs Recall@10."""
    labels = [r['label'] for r in results_sec2]
    recalls = [r['recall'] for r in results_sec2]
    sizes_mb = [r['total_bytes'] / 1e6 for r in results_sec2]
    lats = [r['latency_ms'] for r in results_sec2]
    comps = [r['compression'] for r in results_sec2]
    colors = ['#e15759', '#4e79a7', '#f28e2b', '#59a14f', '#b07aa1']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, xs, xlabel in zip(
            axes,
            [sizes_mb, lats, comps],
            ['Total Index Size (MB)', 'Avg Query Latency (ms)', 'Compression Ratio (x)']):
        ax.scatter(xs, recalls, c=colors[:len(recalls)], s=140, zorder=3,
                   edgecolors='k', linewidths=0.6)
        for x, y, lbl in zip(xs, recalls, labels):
            weight = 'bold' if '48 bytes' in lbl else 'normal'
            ax.annotate(lbl, (x, y), fontsize=7.5, xytext=(4, 4),
                        textcoords='offset points', fontweight=weight)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Recall@10', fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.grid(alpha=0.3)

    axes[0].set_title('Index Size vs Recall@10', fontsize=12)
    axes[1].set_title('Latency vs Recall@10', fontsize=12)
    axes[2].set_title('Compression Ratio vs Recall@10', fontsize=12)

    idx_48 = next(i for i, r in enumerate(results_sec2) if '48 bytes' in r['label'])
    for ax, xs in zip(axes, [sizes_mb, lats, comps]):
        ax.scatter([xs[idx_48]], [recalls[idx_48]], s=300, facecolors='none',
                   edgecolors='red', linewidths=2, zorder=5, label='48 bytes (32x)')
        ax.legend(fontsize=9)

    plt.suptitle('Product Quantization: Compression vs Quality\n(50k corpus, 384-dim)', fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_scaling_loglog(results_sec3):
    """Log-log plot of query latency vs corpus size for Flat, IVF, HNSW.
    results_sec3: {method: {'sizes': [...], 'latencies': [...]}}
    """
    color_map3 = {'Flat': '#e15759', 'IVF': '#4e79a7', 'HNSW': '#59a14f'}
    marker_map3 = {'Flat': 'o', 'IVF': 's', 'HNSW': '^'}

    fig, ax = plt.subplots(figsize=(9, 6))
    for method in ['Flat', 'IVF', 'HNSW']:
        xs = results_sec3[method]['sizes']
        ys = results_sec3[method]['latencies']
        ax.loglog(xs, ys, marker=marker_map3[method], color=color_map3[method],
                  linewidth=2, markersize=9, label=method)

    flat_lats = results_sec3['Flat']['latencies']
    flat_sizes = results_sec3['Flat']['sizes']
    for sz, lt in zip(flat_sizes, flat_lats):
        if lt >= 100:
            ax.axhline(100, color='gray', linestyle='--', alpha=0.6)
            ax.text(flat_sizes[0] * 1.2, 110, 'Flat > 100 ms threshold',
                    fontsize=9, color='gray')
            ax.scatter([sz], [lt], s=200, color='red', zorder=5)
            ax.annotate(f'Flat breaks\n100ms @ N={sz:,}',
                        (sz, lt), xytext=(-80, 15), textcoords='offset points',
                        fontsize=9, arrowprops=dict(arrowstyle='->', color='red'), color='red')
            break

    ax.set_xlabel('Corpus Size (N)', fontsize=12)
    ax.set_ylabel('Avg Query Latency (ms)', fontsize=12)
    ax.set_title('Query Latency vs Corpus Size (log-log)\n128-dim, 100 queries, nprobe=32', fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()


def plot_model_comparison_sec4(sec4_results, model_short):
    """Embedding model comparison: Recall@1/5/10, corpus embed time, efficiency score.
    sec4_results: list of dicts with recall@1, recall@5, recall@10, embed_time_s, avg_q_latency_ms.
    """
    r1_vals = [r['recall@1'] for r in sec4_results]
    r5_vals = [r['recall@5'] for r in sec4_results]
    r10_vals = [r['recall@10'] for r in sec4_results]
    emb_times = [r['embed_time_s'] for r in sec4_results]
    q_lats = [r['avg_q_latency_ms'] for r in sec4_results]
    efficiency = [r10 / lat for r10, lat in zip(r10_vals, q_lats)]
    best_eff_idx = int(np.argmax(efficiency))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(model_short))
    w = 0.25
    c = ['#4e79a7', '#f28e2b', '#59a14f']

    axes[0].bar(x - w, r1_vals, w, label='Recall@1', color=c[0])
    axes[0].bar(x, r5_vals, w, label='Recall@5', color=c[1])
    axes[0].bar(x + w, r10_vals, w, label='Recall@10', color=c[2])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_short)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel('Recall')
    axes[0].set_title('Recall by Model and K')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    bars = axes[1].bar(model_short, emb_times, color=c)
    for bar, v in zip(bars, emb_times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v, f'{v:.2f}s',
                     ha='center', va='bottom', fontsize=9)
    axes[1].set_ylabel('Embedding Time (s)')
    axes[1].set_title('Corpus Embedding Time (500 docs)')
    axes[1].grid(axis='y', alpha=0.3)

    eff_bars = axes[2].bar(model_short, efficiency, color=c)
    for i, (bar, v) in enumerate(zip(eff_bars, efficiency)):
        label_txt = f'{v:.2f}' + (' ★ BEST' if i == best_eff_idx else '')
        if i == best_eff_idx:
            bar.set_edgecolor('red')
            bar.set_linewidth(2)
        axes[2].text(bar.get_x() + bar.get_width() / 2, v, label_txt,
                     ha='center', va='bottom', fontsize=9)
    axes[2].set_ylabel('Recall@10 / Query Latency (ms)')
    axes[2].set_title('Efficiency: Recall@10 per ms')
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle('Embedding Model Comparison (500 docs, 20 queries)', fontsize=13)
    plt.tight_layout()
    plt.show()
    print(f"\nBest efficiency: {model_short[best_eff_idx]} "
          f"(Recall@10={r10_vals[best_eff_idx]:.3f}, "
          f"latency={q_lats[best_eff_idx]:.2f} ms, "
          f"efficiency={efficiency[best_eff_idx]:.2f})")


def plot_cosine_dim_histograms(cos_samples, cos_stats, dims):
    """Grid of pairwise cosine similarity histograms as dimension increases.
    cos_samples: {dim: np.array of similarities}
    cos_stats: list of dicts with 'dim' and 'std' keys
    """
    nrows, ncols = 2, (len(dims) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, 7))
    axes = axes.flatten()

    for ax, d in zip(axes, dims):
        sims = cos_samples[d]
        if len(sims) > 200_000:
            sims = sims[np.random.choice(len(sims), 200_000, replace=False)]
        ax.hist(sims, bins=80, color='steelblue', alpha=0.8, edgecolor='none')
        stat = next(s for s in cos_stats if s['dim'] == d)
        ax.set_title(f'd={d}  std={stat["std"]:.4f}', fontsize=10)
        ax.set_xlabel('Cosine Similarity', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.set_xlim(-1.05, 1.05)
        ax.tick_params(labelsize=7)

    for ax in axes[len(dims):]:
        ax.axis('off')

    plt.suptitle('Pairwise Cosine Similarity Distribution vs Dimension\n'
                 '(5000 random unit vectors — distribution collapses to spike near 0)',
                 fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_cosine_std_vs_dim(cos_stats):
    """Log-log plot of std(cosine similarity) vs dimension with O(1/√d) theory curve."""
    dims_plot = [s['dim'] for s in cos_stats]
    stds_plot = [s['std'] for s in cos_stats]
    d_ref = np.array(dims_plot, dtype=float)
    theory = stds_plot[0] * np.sqrt(dims_plot[0]) / np.sqrt(d_ref)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(dims_plot, stds_plot, 'o-', color='steelblue', linewidth=2,
              markersize=8, label='Observed std(cosine)')
    ax.loglog(d_ref, theory, '--', color='gray', linewidth=1.5, label='O(1/√d) theory')
    for d, s in zip(dims_plot, stds_plot):
        ax.annotate(f'd={d}', (d, s), fontsize=8, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Embedding Dimension', fontsize=12)
    ax.set_ylabel('Std of Pairwise Cosine Similarity', fontsize=12)
    ax.set_title('Curse of Dimensionality: Std(cosine) vs Dimension\n'
                 'Smaller std → harder to distinguish nearest neighbors', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()


def plot_nn_vs_random_cosine(nn_list, rand_list, dims):
    """Compare NN cosine similarity vs random neighbor at given dimensions.
    nn_list: list of arrays (one per dim), rand_list: same shape.
    """
    fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 5))
    if len(dims) == 1:
        axes = [axes]
    for ax, nn_s, rand_s, d in zip(axes, nn_list, rand_list, dims):
        ax.hist(nn_s, bins=40, alpha=0.7, color='#59a14f', label='Nearest Neighbor')
        ax.hist(rand_s, bins=40, alpha=0.7, color='#e15759', label='Random Neighbor')
        gap = float(np.mean(nn_s) - np.mean(rand_s))
        ax.set_title(f'd={d}: NN vs Random cosine sim\nmean gap = {gap:.4f}', fontsize=11)
        ax.set_xlabel('Cosine Similarity', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle('d=2: NN clearly separable — d=768: NN barely better than random', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_matryoshka_tradeoff(results_s6, using_matryoshka=False):
    """Three subplots: recall@10, size+latency, efficiency vs truncated dimension.
    results_s6: list of dicts with 'dim', 'recall@10', 'latency_ms', 'idx_bytes', 'efficiency'.
    """
    dims_s6 = [r['dim'] for r in results_s6]
    r10_s6 = [r['recall@10'] for r in results_s6]
    lats_s6 = [r['latency_ms'] for r in results_s6]
    sizes_s6 = [r['idx_bytes'] / 1024 for r in results_s6]
    effs_s6 = [r['efficiency'] for r in results_s6]
    sweet_idx = int(np.argmax(effs_s6))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(dims_s6, r10_s6, 'o-', color='steelblue', linewidth=2, markersize=8)
    for d, r in zip(dims_s6, r10_s6):
        axes[0].annotate(f'{r:.3f}', (d, r), fontsize=8, xytext=(0, 7),
                         textcoords='offset points', ha='center')
    axes[0].set_xlabel('Embedding Dimension', fontsize=11)
    axes[0].set_ylabel('Recall@10', fontsize=11)
    axes[0].set_title('Recall@10 vs Truncated Dimension\n(graceful degradation curve)', fontsize=11)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_xticks(dims_s6)
    axes[0].grid(alpha=0.3)

    ax2a = axes[1]
    ax2b = ax2a.twinx()
    ax2a.bar(np.arange(len(dims_s6)) - 0.2, sizes_s6, 0.4, color='#4e79a7', alpha=0.8,
             label='Index Size (KB)')
    ax2b.plot(np.arange(len(dims_s6)) + 0.2, lats_s6, 'o--', color='#e15759',
              linewidth=2, markersize=8, label='Latency (ms)')
    ax2a.set_xticks(np.arange(len(dims_s6)))
    ax2a.set_xticklabels([str(d) for d in dims_s6])
    ax2a.set_xlabel('Embedding Dimension', fontsize=11)
    ax2a.set_ylabel('Index Size (KB)', color='#4e79a7', fontsize=10)
    ax2b.set_ylabel('Query Latency (ms)', color='#e15759', fontsize=10)
    axes[1].set_title('Index Size & Latency vs Dimension', fontsize=11)
    lines1, labs1 = ax2a.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2a.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper left')
    ax2a.grid(alpha=0.3)

    bars = axes[2].bar(np.arange(len(dims_s6)), effs_s6, color='#59a14f', alpha=0.85)
    bars[sweet_idx].set_color('#f28e2b')
    bars[sweet_idx].set_edgecolor('red')
    bars[sweet_idx].set_linewidth(2)
    axes[2].set_xticks(np.arange(len(dims_s6)))
    axes[2].set_xticklabels([str(d) for d in dims_s6])
    axes[2].set_xlabel('Embedding Dimension', fontsize=11)
    axes[2].set_ylabel('Recall@10 / sqrt(dim)', fontsize=11)
    axes[2].set_title('Efficiency Score = Recall@10 / sqrt(dim)\nOrange bar = sweet spot', fontsize=11)
    axes[2].annotate(f'Sweet spot\ndim={dims_s6[sweet_idx]}',
                     (sweet_idx, effs_s6[sweet_idx]),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9, color='red',
                     arrowprops=dict(arrowstyle='->', color='red'))
    axes[2].grid(axis='y', alpha=0.3)

    mode = 'Matryoshka native' if using_matryoshka else 'PCA proxy'
    plt.suptitle(f'Matryoshka Embeddings: Dim vs Quality/Cost\n({mode})', fontsize=13)
    plt.tight_layout()
    plt.show()
    print(f"\nSweet spot: dim={dims_s6[sweet_idx]} "
          f"(Recall@10={r10_s6[sweet_idx]:.3f}, "
          f"efficiency={effs_s6[sweet_idx]:.4f})")


# --- 청킹 비교 ---
def plot_chunk_size_histogram(sizes, title='Sentence Chunk Size Distribution'):
    """Histogram of chunk word counts with mean line."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(sizes, bins=20, color='steelblue', edgecolor='white')
    ax.axvline(np.mean(sizes), color='red', linestyle='--',
               label=f'Mean={np.mean(sizes):.1f}')
    ax.set_xlabel('Chunk size (words)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_chunking_comparison(results, colors=None):
    """Three subplots: Recall@3 bar, chunk count bar, recall vs avg size scatter.
    results: {name: {'recall@3', 'n_chunks', 'avg_size', 'redundancy'}}
    """
    if colors is None:
        colors = ['#4e79a7', '#4e79a7', '#4e79a7', '#f28e2b', '#f28e2b', '#59a14f']
    names = list(results.keys())
    recalls = [results[n]['recall@3'] for n in names]
    n_chunks = [results[n]['n_chunks'] for n in names]
    avg_sizes = [results[n]['avg_size'] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(names, recalls, color=colors[:len(names)])
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylim(0, 1.2)
    axes[0].set_ylabel('Recall@3')
    axes[0].set_title('Recall@3 by Chunking Strategy')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(recalls):
        axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)

    axes[1].bar(names, n_chunks, color=colors[:len(names)])
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('Number of Chunks')
    axes[1].set_title('Chunk Count')
    axes[1].grid(axis='y', alpha=0.3)

    axes[2].scatter(avg_sizes, recalls, c=colors[:len(names)], s=100, zorder=3)
    for i, n in enumerate(names):
        axes[2].annotate(n, (avg_sizes[i], recalls[i]),
                         textcoords='offset points', xytext=(5, 3), fontsize=7)
    axes[2].set_xlabel('Avg Chunk Size (words)')
    axes[2].set_ylabel('Recall@3')
    axes[2].set_title('Recall@3 vs Chunk Size')
    axes[2].grid(alpha=0.3)

    plt.suptitle('Chunking Strategy Comparison', fontsize=13)
    plt.tight_layout()
    plt.show()


# --- 리랭킹 비교 ---
def plot_reranking_comparison(methods, mrr_vals, p5_vals, lat_vals,
                               title='Retrieval System Comparison', colors=None):
    """Three subplots: MRR@10, Precision@5, query latency bars."""
    if colors is None:
        colors = ['#4e79a7', '#f28e2b', '#e15759']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(methods, mrr_vals, color=colors)
    axes[0].set_ylabel('MRR@10')
    axes[0].set_title('Mean Reciprocal Rank @10')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(mrr_vals):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    axes[1].bar(methods, p5_vals, color=colors)
    axes[1].set_ylabel('P@5')
    axes[1].set_title('Precision@5')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(p5_vals):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    axes[2].bar(methods, lat_vals, color=colors)
    axes[2].set_ylabel('Avg Latency (ms)')
    axes[2].set_title('Query Latency')
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(lat_vals):
        axes[2].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


# --- 프롬프팅 비교 ---
def plot_prompting_styles(eval_results, styles=None, title='Prompt Style Comparison', colors=None):
    """Two subplots: accuracy bars and avg prompt token count bars.
    eval_results: {style: {'correct': int, 'total': int, 'token_counts': list}}
    """
    if styles is None:
        styles = list(eval_results.keys())
    if colors is None:
        colors = ['#4e79a7', '#f28e2b', '#59a14f']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    accs = [eval_results[s]['correct'] / eval_results[s]['total'] for s in styles]
    axes[0].bar(styles, accs, color=colors[:len(styles)])
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel(f"Accuracy ({eval_results[styles[0]]['total']} questions)")
    axes[0].set_title('Arithmetic Accuracy by Prompt Style')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=11)

    avg_tokens = [np.mean(eval_results[s]['token_counts']) for s in styles]
    axes[1].bar(styles, avg_tokens, color=colors[:len(styles)])
    axes[1].set_ylabel('Avg Prompt Tokens')
    axes[1].set_title('Prompt Length by Style')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(avg_tokens):
        axes[1].text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=11)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_temperature_diversity(temperatures, diversities):
    """Line chart of output token diversity vs temperature with value annotations."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(temperatures, [diversities[T] for T in temperatures],
            'o-', color='steelblue', linewidth=2, markersize=8)
    for T in temperatures:
        ax.annotate(f'{diversities[T]:.2f}', (T, diversities[T]),
                    textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Token Diversity (unique / total)')
    ax.set_title('Output Diversity vs Temperature')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_prompting_sensitivity(temperatures, diversities, rephrasing_results,
                                title='distilgpt2 Prompting Experiments'):
    """Two subplots: temperature diversity bar chart and rephrasing correctness bars.
    rephrasing_results: list of dicts with 'rephrasing' (int) and 'correct' (bool).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    divs = [diversities[T] for T in temperatures]
    axes[0].bar([str(T) for T in temperatures], divs, color='mediumpurple')
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Token Diversity')
    axes[0].set_title('Diversity vs Temperature')
    axes[0].grid(axis='y', alpha=0.3)

    correct_flags = [int(r['correct']) for r in rephrasing_results]
    colors_r = ['#59a14f' if c else '#e15759' for c in correct_flags]
    axes[1].bar([f"Phrasing {r['rephrasing']}" for r in rephrasing_results],
                correct_flags, color=colors_r)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['Wrong', 'Correct'])
    axes[1].set_title('Prompt Sensitivity (15+27=42)')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=15)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


# --- 고급 RAG 비교 ---
def plot_advanced_rag_comparison(all_results, title='Advanced RAG Method Comparison'):
    """Three subplots: recall@3, answer quality, latency bars.
    all_results: {name: {'recalls': [...], 'quality': [...], 'latencies': [...]}}
    """
    names = list(all_results.keys())
    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']
    recalls = [np.mean(all_results[n]['recalls']) for n in names]
    quality = [np.mean(all_results[n]['quality']) for n in names]
    latency = [np.mean(all_results[n]['latencies']) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(names, recalls, color=colors[:len(names)])
    axes[0].set_ylim(0, 1.2)
    axes[0].set_ylabel('Recall@3')
    axes[0].set_title('Retrieval Recall@3')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(recalls):
        axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    axes[0].tick_params(axis='x', rotation=15)

    axes[1].bar(names, quality, color=colors[:len(names)])
    axes[1].set_ylim(0, 1.2)
    axes[1].set_ylabel('Answer Quality (binary)')
    axes[1].set_title('Answer Quality (keyword match)')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(quality):
        axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    axes[1].tick_params(axis='x', rotation=15)

    axes[2].bar(names, latency, color=colors[:len(names)])
    axes[2].set_ylabel('Avg Latency (ms)')
    axes[2].set_title('Query Latency')
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(latency):
        axes[2].text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=10)
    axes[2].tick_params(axis='x', rotation=15)

    plt.suptitle(f'{title} ({len(all_results[names[0]]["recalls"])} queries)', fontsize=13)
    plt.tight_layout()
    plt.show()


# ============================================================
# 20. Simple backpropagation demo helpers
# ============================================================
def plot_simple_backprop_training_curves(losses, grad_norms, accuracies):
    """Simple backprop demo: loss, gradient norm, and accuracy curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(losses, color="tab:blue", lw=2)
    axes[0].set_title("BCE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(grad_norms, color="tab:orange", lw=2)
    axes[1].set_title("Gradient Norm")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("||grad||")
    axes[1].grid(alpha=0.3)

    axes[2].plot(accuracies, color="tab:green", lw=2)
    axes[2].set_title("Train Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_simple_backprop_decision_snapshots(X, y, snapshots, grid_x, grid_y):
    """Simple backprop demo: decision boundary snapshots across epochs."""
    n_cols = len(snapshots)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharex=True, sharey=True)

    if n_cols == 1:
        axes = [axes]

    for ax, s in zip(axes, snapshots):
        ax.contourf(grid_x, grid_y, s["grid_prob"], levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.35)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=18)

        w0, w1 = s["w"]
        b = s["b"]
        x_line = np.linspace(X[:, 0].min() - 0.8, X[:, 0].max() + 0.8, 100)
        if abs(w1) > 1e-8:
            y_line = -(w0 * x_line + b) / w1
            ax.plot(x_line, y_line, color="black", lw=2)

        ax.set_title(f"ep={s['epoch']}\nloss={s['loss']:.3f}, acc={s['acc']:.2f}")
        ax.set_xlabel("x1")

    axes[0].set_ylabel("x2")
    plt.suptitle("Decision Boundary Evolution", y=1.02)
    plt.tight_layout()
    plt.show()


# ============================================================
# 21. Data exploration notebook visualization helpers
# ============================================================
def plot_class_distribution(n_malignant: int, n_benign: int):
    """클래스 분포 바 차트."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(
        ["malignant", "benign"],
        [n_malignant, n_benign],
        color=["#e07b7b", "#7bb8e0"],
        width=0.5,
    )
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    plt.tight_layout()
    plt.show()


def plot_mean_feature_histograms(df_malignant, df_benign, mean_features):
    """악성/양성 mean 피처 분포 히스토그램."""
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    for i, feat in enumerate(mean_features):
        axes[i].hist(
            df_malignant[feat],
            bins=25,
            alpha=0.6,
            color="#e07b7b",
            label="malignant",
            density=True,
        )
        axes[i].hist(
            df_benign[feat],
            bins=25,
            alpha=0.6,
            color="#7bb8e0",
            label="benign",
            density=True,
        )
        axes[i].set_title(feat, fontsize=9)
        axes[i].set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle("Feature Distribution: Malignant vs Benign", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_feature_separability(separability):
    """피처 분리도 수평 바 차트."""
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.barh(separability.index, separability.values, color="#7bb8e0")
    ax.axvline(separability.median(), color="gray", linestyle="--", linewidth=1, label="median")
    ax.set_xlabel("Separability (normalized mean difference)")
    ax.set_title("Feature Separability")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(corr):
    """상관행렬 히트맵."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Matrix (mean features + label)")
    plt.tight_layout()
    plt.show()


def plot_standardization_boxplots(X_raw, X_scaled, x_labels=None):
    """정규화 전/후 스케일 비교 박스플롯."""
    if x_labels is None:
        x_labels = [
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave pts",
            "symmetry",
            "fractal dim",
        ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].boxplot(X_raw[:, :10], patch_artist=True)
    axes[0].set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Before Standardization")
    axes[0].set_ylabel("Value")

    axes[1].boxplot(X_scaled[:, :10], patch_artist=True)
    axes[1].set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("After Standardization")
    axes[1].set_ylabel("Value")

    plt.suptitle("Feature Scale: Before vs After Standardization", fontsize=12)
    plt.tight_layout()
    plt.show()
