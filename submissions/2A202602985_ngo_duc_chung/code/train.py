"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Một hàm `run(cfg)` dùng chung cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc, cùng định nghĩa lúc chấm.

Chống ngắt phiên (Colab): cuối mỗi epoch ghi last.pt; chạy lại cùng cấu hình sẽ tiếp tục từ epoch dở.
Chạy xong thì last.pt bị xoá, còn lại best.pt, history.csv, config.json, summary.json, logit val (và test).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _import_eval():
    """Tìm eval.py của repo (đi ngược lên các thư mục cha) rồi import."""
    for p in Path(__file__).resolve().parents:
        if (p / "eval.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            break
    import eval as ev
    return ev


ev = _import_eval()
import dataset as D  # noqa: E402
import losses as L  # noqa: E402
import model as M  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | geo | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    clip_grad: float = 1.0
    amp: bool = True
    num_workers: int = 2
    cache_images: bool = False        # nạp trước ảnh vào RAM (~2 GB cho train)
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"        # ảnh đường cong training
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    # --- tiện ích ---
    resume: bool = True               # tiếp tục từ last.pt nếu có
    skip_if_done: bool = True         # đã có summary.json thì trả luôn kết quả cũ
    debug_batches: int | None = None  # CHỈ để gỡ lỗi nhanh: giới hạn số batch mỗi epoch (train và val)


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    suffix = "" if cfg.seed == 0 else f"_seed{cfg.seed}"
    return Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}{suffix}.png"


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU + CUDA). cudnn.deterministic=True, benchmark=False.

    Mức tái lập: cùng seed cho kết quả rất gần nhau nhưng không bảo đảm trùng từng bit
    (một số phép CUDA và AMP không tất định). Ghi điều này vào báo cáo.
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    groups = M.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính (từ 0 lên LR đặt) rồi cosine về 0, cập nhật THEO BƯỚC (iteration)."""
    total = max(1, cfg.epochs * steps_per_epoch)
    warm = int(cfg.warmup_epochs * steps_per_epoch)

    def factor(step: int) -> float:
        if warm > 0 and step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    Giữ một bản sao `self.module` (ở eval) để đánh giá. Decay có warmup: d_eff = min(d, (1+n)/(10+n))
    để EMA không bị kéo về trọng số khởi tạo ở những bước đầu. Buffer của BatchNorm (running_mean/var)
    được sao chép thẳng từ model (không lấy trung bình); tham số số thực mới được lấy trung bình.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad = False
        self._pnames = {k for k, _ in model.named_parameters()}
        self.n = 0

    @torch.no_grad()
    def update(self, model) -> None:
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        msd = model.state_dict()
        pnames = self._pnames
        for k, v in self.module.state_dict().items():
            if k in pnames and v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


def _to_device(x, device):
    x = x.to(device, non_blocking=True)
    if device.type == "cuda" and x.ndim == 4:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc", "lr", "lr_trace"}.

    train_acc là NaN khi dùng Mixup/CutMix (nhãn bị trộn nên accuracy trên batch không còn nghĩa).
    """
    M.set_train_mode(model)
    use_amp = cfg.amp and device.type == "cuda"
    tot_loss = tot_correct = tot_n = 0.0
    lr_trace = []
    for i, (x, y, _) in enumerate(loader):
        if cfg.debug_batches is not None and i >= cfg.debug_batches:
            break
        x, y = _to_device(x, device), y.to(device, non_blocking=True)
        targets = y
        if cfg.mix:
            x, targets = L.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        loss = L.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss không hữu hạn ở batch {i}: {loss.item()}")
        scaler.scale(loss).backward()
        if cfg.clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        lr_trace.append(optimizer.param_groups[-1]["lr"])
        n = y.size(0)
        tot_loss += loss.item() * n
        tot_n += n
        if not cfg.mix:
            tot_correct += (logits.argmax(1) == y).sum().item()
    return {"train_loss": tot_loss / max(1, tot_n),
            "train_acc": float("nan") if cfg.mix else tot_correct / max(1, tot_n),
            "lr": lr_trace[-1] if lr_trace else float("nan"), "lr_trace": lr_trace}


def evaluate(model, loader, criterion, device, amp: bool = False, max_batches: int | None = None):
    """Chạy model ở chế độ eval, không gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9] float32, loss: float).
    Giữ đúng thứ tự của loader. max_batches chỉ dùng để gỡ lỗi.
    """
    model.eval()
    names, ys, outs = [], [], []
    tot_loss = tot_n = 0.0
    with torch.inference_mode():
        for i, (x, y, f) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = _to_device(x, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp and device.type == "cuda"):
                logits = model(x)
            logits = logits.float().cpu()
            tot_loss += criterion(logits, y).item() * y.size(0)
            tot_n += y.size(0)
            names += list(f)
            ys.append(y.numpy())
            outs.append(logits.numpy())
    return names, np.concatenate(ys), np.concatenate(outs), tot_loss / max(1, tot_n)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps=None) -> None:
    """Vẽ loss train/val, macro-F1 val (và top-1 val), LR theo bước -> ảnh .png (GUIDE.md mục 6.2)."""
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules:  # trong notebook pyplot đã nạp: đừng đổi backend inline
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    ep = h["epoch"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, h["train_loss"], "o-", label="train loss")
    ax[0].plot(ep, h["val_loss"], "s-", label="val loss")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[1].plot(ep, h["val_macro_f1"], "s-", color="C2", label="val macro-F1")
    ax[1].plot(ep, h["val_top1"], "^--", color="C3", label="val top-1")
    if h["train_acc"].notna().any():
        ax[1].plot(ep, h["train_acc"], "o:", color="C0", label="train acc")
    best = h["val_macro_f1"].idxmax()
    ax[1].axvline(h.loc[best, "epoch"], color="gray", ls=":", label=f"best epoch {int(h.loc[best, 'epoch'])}")
    ax[1].set(xlabel="epoch", ylabel="score", title="Metric")
    if lr_steps is not None and len(lr_steps):
        ax[2].plot(np.arange(len(lr_steps)), lr_steps)
        ax[2].set(xlabel="step", ylabel="LR (nhóm head)", title="LR theo bước")
    else:
        ax[2].plot(ep, h["lr"], "o-")
        ax[2].set(xlabel="epoch", ylabel="LR (nhóm head)", title="LR cuối epoch")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend()
    ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_ckpt(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)  # ghi nguyên tử: phiên bị ngắt giữa chừng không để lại file hỏng


def load_best(cfg: Config, device=None):
    """Dựng lại model và nạp best.pt của một lần chạy (dùng cho Bước 3: suy luận, ensemble)."""
    device = device or M.get_device()
    model = M.build_model(cfg.backbone, pretrained=False, drop_rate=cfg.drop_rate, init="finetune")
    ck = torch.load(run_dir(cfg) / "best.pt", map_location="cpu")
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    Test chỉ được đụng tới ở cuối và chỉ khi cfg.save_test_predictions (S4).
    """
    rd = run_dir(cfg)
    summary_file = rd / "summary.json"
    if cfg.skip_if_done and summary_file.exists() and \
            (not cfg.save_test_predictions or pred_path(cfg, "test").exists()):
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã xong, dùng lại {summary_file}")
        return json.loads(summary_file.read_text())

    set_seed(cfg.seed)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False))
    device = M.get_device()
    print(f"[{cfg.exp_id} seed{cfg.seed}] thiết bị: {device} | AMP: {cfg.amp and device.type == 'cuda'}")

    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_loader = D.make_loader(train_df, cfg.images_dir, D.build_transforms(True, cfg.img_size, cfg.aug),
                                 cfg.batch_size, True, cfg.sampler, cfg.num_workers, cfg.seed, cfg.cache_images)
    val_loader = D.make_loader(val_df, cfg.images_dir, D.build_transforms(False, cfg.img_size),
                               cfg.batch_size, False, None, cfg.num_workers, cfg.seed, cfg.cache_images)

    model = M.build_model(cfg.backbone, pretrained=True, drop_rate=cfg.drop_rate, init=cfg.init)
    model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    params_m, gmacs = M.count_params(model), M.count_gmacs(model, cfg.img_size)

    # loss: trọng số lớp chỉ lấy từ TRAIN
    kw = {}
    if cfg.loss == "ls":
        kw["smoothing"] = cfg.label_smoothing
    elif cfg.loss in ("focal", "ce_weighted"):
        counts = train_df["Label"].value_counts().sort_index().reindex(range(D.NUM_CLASSES)).to_numpy()
        w = L.class_weights(counts, cfg.class_weight_beta) if cfg.class_weight_beta is not None else None
        if cfg.loss == "focal":
            kw.update(gamma=cfg.focal_gamma, alpha=w)
        else:
            kw["weight"] = w if w is not None else L.class_weights(counts, 0.0)
    criterion = L.build_criterion(cfg.loss, **kw).to(device)
    eval_criterion = torch.nn.CrossEntropyLoss()  # val_loss luôn là CE thường để so sánh được giữa các loss

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    eval_model = ema.module if ema else model

    history, lr_steps = [], []
    best = {"f1": -1.0, "epoch": -1}
    start = 0
    last = rd / "last.pt"
    if cfg.resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        if ema:
            ema.module.load_state_dict(ck["ema"])
            ema.n = ck["ema_n"]
        history, lr_steps, best, start = ck["history"], ck["lr_steps"], ck["best"], ck["epoch"] + 1
        print(f"[{cfg.exp_id} seed{cfg.seed}] tiếp tục từ epoch {start}")

    t_train = 0.0
    for epoch in range(start, cfg.epochs):
        set_seed(cfg.seed * 1000 + epoch)  # mỗi epoch seed riêng -> chạy tiếp sau khi ngắt vẫn nhất quán
        train_loader.generator.manual_seed(cfg.seed * 1000 + epoch)
        t0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        M.sync_device(device)
        sec = time.perf_counter() - t0
        t_train += sec
        names, y, logits, vloss = evaluate(eval_model, val_loader, eval_criterion, device, use_amp, cfg.debug_batches)
        probs = _softmax(logits)
        m = ev.compute_metrics(y, probs.argmax(1), probs)
        lr_steps += tr.pop("lr_trace")
        row = {"epoch": epoch + 1, **tr, "val_loss": vloss, "val_top1": m["top1"],
               "val_macro_f1": m["macro_f1"], "val_ece": m["ece"], "sec_train": sec}
        history.append(row)
        print(f"[{cfg.exp_id} seed{cfg.seed}] ep {epoch + 1}/{cfg.epochs} "
              f"loss {row['train_loss']:.4f} | val loss {vloss:.4f} F1 {m['macro_f1']:.4f} "
              f"top1 {m['top1']:.4f} | {sec:.0f}s")
        if m["macro_f1"] > best["f1"]:  # '>' nghiêm ngặt: hòa thì giữ epoch sớm hơn
            best = {"f1": m["macro_f1"], "epoch": epoch + 1}
            _save_ckpt(rd / "best.pt", {"model": eval_model.state_dict(), "epoch": epoch + 1,
                                        "val_macro_f1": m["macro_f1"]})
        _save_ckpt(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                          "ema": ema.module.state_dict() if ema else None, "ema_n": ema.n if ema else 0,
                          "history": history, "lr_steps": lr_steps, "best": best, "epoch": epoch})
        pd.DataFrame(history).to_csv(rd / "history.csv", index=False)

    # nạp checkpoint tốt nhất, lưu logit/dự đoán val
    eval_model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu")["model"])
    names, y, logits, vloss = evaluate(eval_model, val_loader, eval_criterion, device, use_amp, cfg.debug_batches)
    probs = _softmax(logits)
    mval = ev.compute_metrics(y, probs.argmax(1), probs)
    np.save(rd / "val_logits.npy", logits)
    ev.save_predictions(pred_path(cfg, "val"), names, y, probs)

    summary = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "init": cfg.init,
               "pretrained_tag": model.pretrained_tag, "best_epoch": best["epoch"],
               "val_macro_f1": mval["macro_f1"], "val_top1": mval["top1"], "val_loss": vloss,
               "val_ece": mval["ece"], "sec_per_epoch": float(np.mean([r["sec_train"] for r in history])),
               "params_m": params_m, "gmacs": gmacs, "epochs": cfg.epochs}

    if cfg.save_test_predictions:  # Bước 4: đúng MỘT lần, toàn bộ tập test
        test_loader = D.make_loader(test_df, cfg.images_dir, D.build_transforms(False, cfg.img_size),
                                    cfg.batch_size, False, None, cfg.num_workers, cfg.seed, cfg.cache_images)
        tn, ty, tl, _ = evaluate(eval_model, test_loader, eval_criterion, device, use_amp)
        tp = _softmax(tl)
        np.save(rd / "test_logits.npy", tl)
        ev.save_predictions(pred_path(cfg, "test"), tn, ty, tp)
        mt = ev.compute_metrics(ty, tp.argmax(1), tp)
        summary.update(test_macro_f1=mt["macro_f1"], test_top1=mt["top1"], test_ece=mt["ece"])

    plot_curves(history, curve_path(cfg), f"{cfg.exp_id} {cfg.backbone} seed{cfg.seed}", lr_steps)
    plot_curves(history, rd / "curves.png", f"{cfg.exp_id} {cfg.backbone} seed{cfg.seed}", lr_steps)
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    last.unlink(missing_ok=True)
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    fields = {f.name: f for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"'{pair}' phải có dạng KEY=VALUE")
        key, raw = pair.split("=", 1)
        if key not in fields:
            raise ValueError(f"'{key}' không có trong Config. Các key hợp lệ: {sorted(fields)}")
        t = str(fields[key].type)
        if "None" in t and raw.lower() in ("none", "null", ""):
            out[key] = None
        elif t.startswith("bool"):
            if raw.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{key} cần true/false, nhận {raw!r}")
            out[key] = raw.lower() in ("true", "1")
        elif t.startswith("int"):
            out[key] = int(raw)
        elif t.startswith("float"):
            out[key] = float(raw)
        else:
            out[key] = raw
    return out


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè field của Config")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
