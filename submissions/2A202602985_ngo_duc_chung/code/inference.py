"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None)  -> (filenames, y_true, logits[N, 9])
    predict_multiview(model, loader, device, views)   -> (filenames, y_true, [logits[N, 9], ...])
    aggregate_views(list_of_logits, space)            -> probs[N, 9]
    fit_temperature(val_logits, val_labels)           -> float T
    apply_temperature(logits, T)                      -> probs
    ensemble_probs(list_of_probs)                     -> probs
    fuse_conv_bn(model)                               -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _softmax(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def view_vflip(x):
    return torch.flip(x, dims=[-2])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop`; flip=True thêm bản lật của từng crop (10 view).

    Đầu vào phải lớn hơn `crop` (ví dụ ảnh 256 với crop 224: dùng loader có img_size=256,
    xem dataset.build_transforms). Trả về list các batch.
    """
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError(f"crop {crop} lớn hơn ảnh {h}x{w}")
    oy, ox = (h - crop) // 2, (w - crop) // 2
    offsets = [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), (oy, ox)]
    out = [x[..., y:y + crop, xx:xx + crop] for y, xx in offsets]
    if flip:
        out += [torch.flip(v, dims=[-1]) for v in out]
    return out


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch.

    Model phải chấp nhận ảnh khác kích thước lúc train: CNN có global pooling thì được.
    ViT/DeiT/Swin (timm) cố định kích thước đầu vào nên sẽ báo lỗi; ghi rõ giới hạn này trong báo cáo.
    """
    return [x if s == x.shape[-1] else
            F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=s < x.shape[-1])
            for s in sizes]


def predict_multiview(model, loader, device, views, amp: bool = False):
    """Chạy loader MỘT lần, áp mọi view lên từng batch. Trả về (filenames, y_true, [logits_k]).

    `views` là: list các hàm (batch -> batch), hoặc một hàm (batch -> list batch) như
    functools.partial(views_multicrop, crop=224). Thứ tự logit theo thứ tự view.
    """
    model.eval()
    names, ys, outs = [], [], None
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            vs = views(x) if callable(views) else [v(x) for v in views]
            if outs is None:
                outs = [[] for _ in vs]
            for k, v in enumerate(vs):
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=amp and device.type == "cuda"):
                    outs[k].append(model(v).float().cpu().numpy())
            names += list(f)
            ys.append(y.numpy())
    return names, np.concatenate(ys), [np.concatenate(o) for o in outs]


def predict_logits(model, loader, device, view=None, amp: bool = False):
    """Một view (hoặc không view): trả về (filenames, y_true, logits[N, 9]) theo đúng thứ tự file."""
    names, y, outs = predict_multiview(model, loader, device, [view or view_identity], amp)
    return names, y, outs[0]


def aggregate_views(logits_per_view, space: str = "prob", T: float = 1.0):
    """Gộp K lượt chạy của TTA thành xác suất (N, 9) đã chuẩn hoá (slide trang 62).

      space="prob":  trung bình softmax(logit / T) của từng view
      space="logit": trung bình logit rồi softmax(. / T)
    T = 1 là không hiệu chuẩn. Với một view, hai cách trùng nhau.
    """
    arr = np.stack([np.asarray(z, dtype=np.float64) for z in logits_per_view]) / T
    if space == "prob":
        return _softmax(arr).mean(0)
    if space == "logit":
        return _softmax(arr.mean(0))
    raise ValueError(f"space phải là prob|logit, nhận {space!r}")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình. Chỉ ghép trên CÙNG tập ảnh và cùng thứ tự file.

    Chi phí suy luận = số mô hình.
    """
    shapes = {np.asarray(p).shape for p in list_of_probs}
    if len(shapes) != 1:
        raise ValueError(f"các ma trận xác suất khác dạng: {shapes}")
    return np.mean([np.asarray(p, dtype=np.float64) for p in list_of_probs], axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """Nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T) (slide trang 69).

    Tối ưu log T bằng LBFGS (T = exp(logT) luôn dương). Accuracy không đổi. KHÔNG khớp T trên test.
    """
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def fit_temperature_views(val_logits_per_view, val_labels, space: str = "prob") -> float:
    """Như fit_temperature nhưng cho TTA: T cực tiểu NLL của aggregate_views(..., space, T) trên VAL."""
    z = torch.as_tensor(np.stack([np.asarray(v) for v in val_logits_per_view]), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        zt = z / log_t.exp()
        p = F.softmax(zt, -1).mean(0) if space == "prob" else F.softmax(zt.mean(0), -1)
        loss = F.nll_loss(torch.log(p.clamp_min(1e-12)), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, T: float):
    """softmax(logits / T)."""
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma * w / sqrt(var + eps); b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps) if bn.affine else \
        1.0 / torch.sqrt(bn.running_var + bn.eps)
    shift = bn.bias if bn.affine else torch.zeros_like(bn.running_mean)
    b0 = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        fused.bias.copy_(shift + (b0 - bn.running_mean) * scale)
    return fused.to(conv.weight.device, conv.weight.dtype)


def _is_plain_bn(m) -> bool:
    return type(m) is nn.BatchNorm2d


def _is_bn_act(m) -> bool:
    return type(m).__name__ == "BatchNormAct2d"


def fuse_conv_bn(model, check_input=None, tol: float = 1e-4, inplace: bool = False):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75).

    Cách ghép: trong cùng một module cha, Conv2d đứng ngay trước BatchNorm2d (hoặc BatchNormAct2d của
    timm: phần BN được gộp, activation giữ lại) và số kênh khớp nhau. Vì ghép theo thứ tự đăng ký
    module, hàm KIỂM TRA bằng số: so đầu ra trước/sau trên `check_input` (mặc định ảnh ngẫu nhiên
    224x224) và báo lỗi nếu lệch quá `tol`. Sai số lớn nhất lưu ở `model.fuse_max_abs_err`,
    số cặp đã gộp ở `model.fuse_pairs`.
    Mặc định làm trên BẢN SAO (inplace=False). Kiến trúc không có BN2d (ViT, Swin, ConvNeXt dùng
    LayerNorm) trả về bản sao không đổi với fuse_pairs = 0.
    """
    ref = model if inplace else copy.deepcopy(model)
    ref.eval()
    orig = copy.deepcopy(ref)
    device = next(ref.parameters()).device
    pairs = 0

    def walk(parent: nn.Module) -> int:
        n = 0
        names = list(parent._modules.keys())
        for i, name in enumerate(names):
            child = parent._modules[name]
            if child is None:
                continue
            n += walk(child)
            if i + 1 >= len(names):
                continue
            nxt = parent._modules[names[i + 1]]
            if type(child) is nn.Conv2d and nxt is not None and (_is_plain_bn(nxt) or _is_bn_act(nxt)) \
                    and nxt.num_features == child.out_channels and nxt.track_running_stats:
                parent._modules[name] = _fold(child, nxt)
                if _is_bn_act(nxt):
                    parent._modules[names[i + 1]] = nn.Sequential(nxt.drop, nxt.act)
                else:
                    parent._modules[names[i + 1]] = nn.Identity()
                n += 1
        return n

    pairs = walk(ref)
    if check_input is None:
        size = getattr(ref, "pretrained_cfg", {}).get("input_size", (3, 224, 224))
        check_input = torch.randn(2, *size, device=device)
    with torch.inference_mode():
        err = (ref(check_input) - orig(check_input)).abs().max().item()
    print(f"fuse_conv_bn: gộp {pairs} cặp Conv+BN, sai số lớn nhất {err:.2e}")
    if err > tol:
        raise RuntimeError(f"gộp BN lệch {err:.2e} > {tol:.0e}: ghép sai cặp, kiểm tra kiến trúc")
    ref.fuse_max_abs_err, ref.fuse_pairs = err, pairs
    return ref
