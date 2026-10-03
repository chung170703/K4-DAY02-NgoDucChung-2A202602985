"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    kw: smoothing=0.1 (ls), gamma=2.0 và alpha=None|tensor (focal), weight=tensor (ce_weighted).
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted cần weight=tensor (xem class_weights)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(kw["weight"], dtype=torch.float32))
    raise ValueError(f"loss không biết: {kind!r}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Tự cài đặt (không dùng label_smoothing của PyTorch); eps = 0 cho đúng CE (xem test_code.py).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        k = logits.shape[-1]
        q = torch.full_like(logp, self.smoothing / k)
        q.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / k)
        return -(q * logp).sum(dim=-1).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    alpha: None hoặc vector trọng số theo lớp. Trung bình theo batch.
    gamma = 0 và alpha = None cho đúng cross-entropy (xem test_code.py).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logp.exp()
        loss = -((1.0 - pt) ** self.gamma) * logp
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN (tuyệt đối không dùng val/test).

    - beta = 0: w_c ∝ 1 / n_c, chuẩn hoá về trung bình 1
    - beta > 0: class-balanced, w_c = (1 - beta) / (1 - beta ** n_c) (Cui et al. 1901.05555),
      chuẩn hoá để tổng trọng số bằng số lớp
    """
    n = torch.as_tensor(np.asarray(counts, dtype=np.float64))
    if beta == 0:
        w = 1.0 / n
    else:
        w = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), n))
    w = w * len(n) / w.sum()
    return w.float()


def _rand_bbox(h: int, w: int, lam: float):
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn. lam ~ Beta(alpha, alpha).

    mixup : x_mix = lam * x + (1 - lam) * x[perm]
    cutmix: dán một hộp từ x[perm] vào x; lam được tính lại theo DIỆN TÍCH THỰC của hộp
            sau khi hộp bị cắt ở biên (slide trang 48).
    Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]. Dùng RNG toàn cục của numpy/torch.
    """
    if mode not in ("mixup", "cutmix"):
        raise ValueError(f"mode phải là mixup|cutmix, nhận {mode!r}")
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    else:
        x_mix = x.clone()
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = _rand_bbox(h, w, lam)
        x_mix[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
