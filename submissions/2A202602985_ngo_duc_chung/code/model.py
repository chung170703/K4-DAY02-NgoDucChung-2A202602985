"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    set_train_mode(model)                                         -> None (thay cho model.train())
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import timm
import torch

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    init: "scratch" (không tải trọng số), "frozen" (tải trọng số, chỉ train head),
          "finetune" (tải trọng số, train toàn bộ).
    Gắn thêm vào model: `.init_mode` và `.pretrained_tag` (tag trọng số timm thực sự được tải,
    "none" nếu huấn luyện từ đầu) để ghi vào results.xlsx.
    """
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init phải là scratch|frozen|finetune, nhận {init!r}")
    use_weights = pretrained and init != "scratch"
    model = timm.create_model(name, pretrained=use_weights, num_classes=num_classes, drop_rate=drop_rate)
    cfg = getattr(model, "pretrained_cfg", None) or {}
    model.init_mode = init
    model.pretrained_tag = (cfg.get("tag") or cfg.get("hf_hub_id") or cfg.get("url") or "unknown") \
        if use_weights else "none"
    if init == "frozen":
        freeze_backbone(model)
    return model


def _head_param_ids(model) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """requires_grad=False cho mọi tham số trừ head (model.get_classifier()).

    BatchNorm của backbone đóng băng cũng phải ở eval; train loop gọi set_train_mode(model)
    thay cho model.train() để điều đó luôn đúng sau mỗi lần chuyển chế độ.
    """
    head = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.init_mode = "frozen"


def set_train_mode(model) -> None:
    """model.train(), trừ khi backbone đóng băng: khi đó chỉ head ở train, phần còn lại ở eval."""
    model.train()
    if getattr(model, "init_mode", None) == "frozen":
        model.eval()
        model.get_classifier().train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone ndim > 1:           lr = lr_backbone, weight_decay = weight_decay
    - norm/bias backbone (ndim<=1): lr = lr_backbone, weight_decay = 0
    - head mới:                    lr = lr_head,     weight_decay = weight_decay
    Ngoài quy tắc ndim, các tham số mà timm khai báo trong model.no_weight_decay()
    (pos_embed, cls_token, relative_position_bias_table...) cũng không bị weight decay.
    Bỏ qua tham số requires_grad=False.
    """
    head = _head_param_ids(model)
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay, no_decay, head_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head:
            head_params.append(p)
        elif p.ndim <= 1 or name in skip:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_norm_bias"})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay, "name": "head"})
    return groups


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Công cụ: torch.utils.flop_counter.FlopCounterMode (đếm FLOPs của conv/matmul/attention,
    1 MAC = 2 FLOPs nên GMAC = FLOPs / 2 / 1e9). Chỉ tính các phép nhân-cộng này; bỏ qua
    BN, activation, softmax. Có thể lệch vài phần trăm so với fvcore/ptflops.
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    model.train(was_training)
    return fc.get_total_flops() / 2 / 1e9
