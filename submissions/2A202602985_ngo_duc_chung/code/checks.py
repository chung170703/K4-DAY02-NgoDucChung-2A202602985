"""checks.py - kiểm tra pipeline trước khi chạy thí nghiệm thật (GUIDE.md mục 1.3, slide trang 59).

    initial_loss(cfg)          loss ban đầu của head mới, kỳ vọng ≈ -ln(1/9) = 2.197
    overfit_one_batch(cfg)     một batch nhỏ phải overfit tới loss gần 0
    show_augmented(...)        vẽ ảnh sau augmentation (đã giải chuẩn hoá) cùng nhãn

Chỉ dùng TRAIN, không đụng val/test.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import dataset as D
import losses as L
import model as M
from train import Config, set_seed


def _device():
    return M.get_device()


def _first_batch(cfg: Config, n: int, train_tf: bool):
    train_df, _, _ = D.load_split(cfg.labels_dir, cfg.fold)
    tf = D.build_transforms(train_tf, cfg.img_size, cfg.aug)
    ds = D.DeepWeedsDataset(train_df.sample(n, random_state=cfg.seed), cfg.images_dir, tf)
    xs, ys, fs = zip(*[ds[i] for i in range(n)])
    return torch.stack(xs), torch.as_tensor(ys), list(fs)


@torch.no_grad()
def initial_loss(cfg: Config, n: int = 64) -> dict:
    """Loss CE của model vừa tạo (head ngẫu nhiên) trên n ảnh train, ở eval. Kỳ vọng ≈ 2.197."""
    set_seed(cfg.seed)
    dev = _device()
    model = M.build_model(cfg.backbone, init=cfg.init, drop_rate=cfg.drop_rate).to(dev).eval()
    x, y, _ = _first_batch(cfg, n, train_tf=False)
    loss = torch.nn.functional.cross_entropy(model(x.to(dev)).float(), y.to(dev)).item()
    out = {"loss": loss, "expected": math.log(D.NUM_CLASSES)}
    print(f"loss ban đầu = {loss:.3f} (kỳ vọng ≈ {out['expected']:.3f})")
    return out


def overfit_one_batch(cfg: Config, n: int = 16, steps: int = 60, lr: float = 1e-3) -> list[float]:
    """Huấn luyện lặp trên đúng một batch n ảnh (không augmentation). Loss cuối phải gần 0."""
    set_seed(cfg.seed)
    dev = _device()
    model = M.build_model(cfg.backbone, init=cfg.init, drop_rate=0.0).to(dev)
    x, y, _ = _first_batch(cfg, n, train_tf=False)
    x, y = x.to(dev), y.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    trace = []
    for _ in range(steps):
        M.set_train_mode(model)
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x).float(), y)
        loss.backward()
        opt.step()
        trace.append(loss.item())
    print(f"overfit {n} ảnh: loss {trace[0]:.3f} -> {trace[-1]:.4f}" + ("  OK" if trace[-1] < 0.1 else "  CHƯA ĐẠT"))
    return trace


def show_augmented(cfg: Config, n: int = 8, mix: str | None = None):
    """Vẽ n ảnh train sau augmentation (và sau Mixup/CutMix nếu mix) kèm nhãn để kiểm tra ảnh-nhãn khớp nhau."""
    import matplotlib.pyplot as plt

    np.random.seed(cfg.seed)
    x, y, _ = _first_batch(cfg, n, train_tf=True)
    titles = [D.CLASS_NAMES[i] for i in y.tolist()]
    if mix:
        x, (ya, yb, lam) = L.mix_batch(x, y, cfg.mix_alpha, mix)
        titles = [f"{D.CLASS_NAMES[a]}\n{lam:.2f} | {D.CLASS_NAMES[b]}" for a, b in zip(ya.tolist(), yb.tolist())]
    mean = torch.tensor(D.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(D.IMAGENET_STD).view(1, 3, 1, 1)
    img = (x * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    fig, ax = plt.subplots(1, n, figsize=(2.4 * n, 3))
    for a, im, t in zip(np.atleast_1d(ax), img, titles):
        a.imshow(im)
        a.set_title(t, fontsize=8)
        a.axis("off")
    fig.suptitle(f"aug={cfg.aug}" + (f", mix={mix}" if mix else ""))
    return fig
