"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - đo ở đây KHÔNG tính tiền xử lý/đọc ảnh: chỉ forward của model trên tensor đã nằm sẵn trên thiết bị
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    """
    if warmup < 10 or iters < 50:
        raise ValueError("cần warmup >= 10 và iters >= 50 (GUIDE.md mục 4.1)")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean()), "n": iters}


def _prepare(model, dtype: str, device: str):
    if dtype not in ("fp32", "amp", "fp16"):
        raise ValueError("dtype phải là fp32|amp|fp16")
    m = copy.deepcopy(model).to(device).eval()  # bản sao: .half() không làm hỏng model gốc
    if dtype == "fp16":
        m = m.half()
    return m


def _make_forward(m, x, dtype: str, device: str, k: int = 1):
    use_amp = dtype == "amp" and device.startswith("cuda")

    def fn():
        with torch.inference_mode(), torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                                    dtype=torch.float16, enabled=use_amp):
            for _ in range(k):
                m(x)
    return fn


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict ghi thẳng được vào sheet `Latency`:
        {"gpu", "dtype", "batch", "img_size", "p50", "p95", "p99", "images_per_s", "torch"}
    Ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định.
    """
    m = _prepare(model, dtype, device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    sync = torch.cuda.synchronize if device.startswith("cuda") else None
    r = bench(_make_forward(m, x, dtype, device), warmup, iters, sync)
    return {"gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
            "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"], "n": r["n"],
            "images_per_s": batch_size / (r["p50"] / 1000.0), "torch": torch.__version__}


def tta_latency(model, k_views: int, batch_size: int = 1, img_size: int = 224, dtype: str = "fp32",
                device: str = "cuda", warmup: int = 10, iters: int = 100) -> dict:
    """Độ trễ của TTA K view, đo thật bằng K lần forward liên tiếp, kèm so với K * p50 của 1 view."""
    m = _prepare(model, dtype, device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    sync = torch.cuda.synchronize if device.startswith("cuda") else None
    one = bench(_make_forward(m, x, dtype, device, 1), warmup, iters, sync)
    many = bench(_make_forward(m, x, dtype, device, k_views), warmup, iters, sync)
    return {"k_views": k_views, "p50_1view": one["p50"], "p50": many["p50"], "p95": many["p95"],
            "p99": many["p99"], "k_x_p50_1view": k_views * one["p50"],
            "ratio_vs_k_x_single": many["p50"] / (k_views * one["p50"]),
            "dtype": dtype, "batch": batch_size, "img_size": img_size, "torch": torch.__version__}
