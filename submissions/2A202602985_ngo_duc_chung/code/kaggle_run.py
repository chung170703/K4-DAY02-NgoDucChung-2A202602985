"""kaggle_run.py - điều phối chạy một mạch trên Kaggle (`lab_day2_kaggle.ipynb` gọi các hàm ở đây).

Khác notebook Colab: không có bước "bạn chọn tay" nào. Mọi lựa chọn đều là một quy tắc ghi rõ trong code
và chỉ đọc số liệu VAL (S2, S4):
  - backbone đi tiếp: macro-F1 val cao nhất ở Bước 1;
  - yếu tố thắng của mỗi trục: Delta macro-F1 val so với T00 vượt nhiễu (std của T00 qua 3 seed, sàn 0.002);
  - cấu hình chung kết: tốt nhất theo val trong {T00, tổ hợp các yếu tố thắng, yếu tố thắng đơn tốt nhất};
  - cách suy luận chung kết: macro-F1 val cao nhất, hòa (trong 0.001) thì lấy cách rẻ hơn.
Test chỉ được chạm ở Bước 4, một lượt mỗi seed.
"""
from __future__ import annotations

import contextlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dataset as D
import train as T

ev = T.ev

BACKBONES = [("B01", "resnet50"), ("B02", "resnext50_32x4d"), ("B03", "convnext_tiny"),
             ("B04", "deit_small_patch16_224"), ("B05", "swin_tiny_patch4_window7_224"),
             ("B06", "efficientnet_b0"), ("B07", "mobilenetv3_large_100")]

# (exp_id, trục A-G của GUIDE, mô tả khác T00, override)
AXES = [  # thứ tự = thứ tự chạy: yếu tố nhiều khả năng thắng trước (nếu hết ngân sách giờ thì bỏ các mục cuối)
    ("T14", "E", "LR backbone 3e-4",           dict(lr_backbone=3e-4, lr_head=3e-3)),
    ("T09", "C", "loss=label smoothing 0.1",   dict(loss="ls", label_smoothing=0.1)),
    ("T03", "B", "aug=geo (lật dọc + xoay 90)", dict(aug="geo")),
    ("T04", "B", "aug=color",                  dict(aug="color")),
    ("T08", "B", "mix=cutmix",                 dict(mix="cutmix", mix_alpha=1.0)),
    ("T16", "F", "EMA 0.99",                   dict(ema_decay=0.99)),
    ("T17", "G", "epochs=20",                  dict(epochs=20)),
    ("T10", "C", "loss=focal gamma=2",         dict(loss="focal", focal_gamma=2.0)),
    ("T11", "C", "loss=CE trọng số lớp (1/n)", dict(loss="ce_weighted")),
    ("T12", "D", "sampler=balanced",           dict(sampler="balanced")),
    ("T02", "A", "init=frozen (linear probe)", dict(init="frozen")),
]
# Không chạy trong ablation (tiết kiệm giờ GPU, ghi trong báo cáo): T01 từ đầu (có trong ma trận LAB2_RUNS, đủ 3 seed),
# T05 TrivialAugment, T06 RandAugment, T07 Mixup, T13 cùng LR, T15 không warmup.

# Lab #2 gốc của slide (trang 78): ba cách khởi tạo (từ đầu, đóng băng, tinh chỉnh) x có/không CutMix, mean ± std qua >= 3 seed.
# T00/T01/T02/T08 là các lần chạy đã có ở trên (thêm seed 1, 2); T19/T20 là hai ô còn thiếu của ma trận 3x2.
LAB2_RUNS = [
    ("T00", "tinh chỉnh, không CutMix", {}),
    ("T01", "từ đầu, không CutMix", dict(init="scratch")),
    ("T02", "đóng băng, không CutMix", dict(init="frozen")),
    ("T08", "tinh chỉnh + CutMix", dict(mix="cutmix", mix_alpha=1.0)),
    ("T19", "từ đầu + CutMix", dict(init="scratch", mix="cutmix", mix_alpha=1.0)),
    ("T20", "đóng băng + CutMix", dict(init="frozen", mix="cutmix", mix_alpha=1.0)),
]

# Nhóm "cùng một núm chỉnh": trong mỗi nhóm chỉ lấy yếu tố thắng tốt nhất khi ghép. None = không đưa vào tổ hợp
# (khởi tạo: tinh chỉnh toàn bộ là nền). Số epoch (T17) được xét như mọi yếu tố khác: thắng thì vào tổ hợp T18 và F01.
GROUP = {"T01": None, "T02": None, "T03": "aug", "T04": "aug", "T05": "aug",
         "T07": "mix", "T08": "mix", "T09": "loss", "T10": "loss", "T11": "loss", "T12": "sampler",
         "T14": "lr", "T15": "warmup", "T16": "ema", "T17": "epochs"}

# preset suy luận -> (mã I, số view K, tta, không gian gộp)
TTA_CANDIDATES = [("I00", 1, "none", "prob"), ("I01", 2, "hflip", "prob"), ("I02a", 5, "crop5", "prob"),
                  ("I02b", 10, "crop5flip", "prob"), ("I03b", 5, "crop5", "logit")]


# --------------------------------------------------------------------------- #
# Nhật ký và chạy từng giai đoạn không làm sập cả phiên
# --------------------------------------------------------------------------- #
LOG_FILE: Path | None = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


STRICT = False   # True (chạy thử): lỗi dừng ngay để thấy; False (chạy thật): ghi log rồi chạy tiếp các bước sau


@contextlib.contextmanager
def guard(name: str):
    """with guard("Bước 2"): ...  - lỗi trong khối được ghi vào nhật ký, không làm cả notebook dừng (trừ khi STRICT)."""
    t0 = time.perf_counter()
    log(f"=== BẮT ĐẦU {name}")
    try:
        yield
        log(f"=== XONG {name} ({(time.perf_counter() - t0) / 60:.1f} phút)")
    except Exception:
        log(f"=== LỖI {name} sau {(time.perf_counter() - t0) / 60:.1f} phút\n{traceback.format_exc()}")
        if STRICT:
            raise


def stage(name: str, fn, *args, **kw):
    """Chạy fn; lỗi thì in traceback và trả (False, None) để các giai đoạn độc lập vẫn chạy tiếp."""
    t0 = time.perf_counter()
    log(f"=== BẮT ĐẦU {name}")
    try:
        out = fn(*args, **kw)
        log(f"=== XONG {name} ({(time.perf_counter() - t0) / 60:.1f} phút)")
        return True, out
    except Exception:
        log(f"=== LỖI {name} sau {(time.perf_counter() - t0) / 60:.1f} phút\n{traceback.format_exc()}")
        if STRICT:
            raise
        return False, None


# --------------------------------------------------------------------------- #
# Quy tắc chọn (chỉ dùng số liệu VAL)
# --------------------------------------------------------------------------- #
def pick_backbone(bb: pd.DataFrame) -> str:
    """Backbone có macro-F1 val cao nhất; hòa thì lấy mạng ít tham số hơn."""
    d = bb.sort_values(["val_macro_f1", "params_m"], ascending=[False, True])
    return str(d.iloc[0]["backbone"])


def top_backbones(bb: pd.DataFrame, k: int = 3) -> list[tuple[str, str]]:
    """k backbone tốt nhất theo val (cho ensemble I05): [(exp_id, backbone), ...]."""
    d = bb.sort_values(["val_macro_f1", "params_m"], ascending=[False, True]).head(k)
    return [(str(r.exp_id), str(r.backbone)) for r in d.itertuples()]


def noise_threshold(t00_f1, floor: float = 0.002) -> float:
    """std mẫu (ddof=1) của macro-F1 val của T00 qua các seed; có sàn để std quá nhỏ không làm ngưỡng vô nghĩa."""
    s = float(np.std(np.asarray(t00_f1, dtype=float), ddof=1)) if len(t00_f1) > 1 else 0.0
    return max(s, floor)


def pick_winners(training_tbl: pd.DataFrame, thr: float):
    """Chọn yếu tố thắng cho tổ hợp T18.

    Trả về (combo, chosen, best_single):
      combo: dict override ghép các yếu tố thắng (mỗi nhóm một yếu tố có Delta cao nhất và > thr);
      chosen: list[exp_id] tạo nên combo;
      best_single: (exp_id, override) của yếu tố có Delta dương lớn nhất (kể cả chưa vượt nhiễu), hoặc None.
    """
    ov = {e: kw for e, _, _, kw in AXES}
    best_in_group: dict[str, tuple[str, float]] = {}
    best_single = None
    for r in training_tbl.itertuples():
        d = float(r.delta_vs_T00)
        g = GROUP.get(r.exp_id)
        if g is None or not np.isfinite(d):
            continue
        if d > 0 and (best_single is None or d > best_single[2]):
            best_single = (r.exp_id, ov[r.exp_id], d)
        if d > thr and (g not in best_in_group or d > best_in_group[g][1]):
            best_in_group[g] = (r.exp_id, d)
    # sampler cân bằng và CE có trọng số cùng bù mất cân bằng lớp -> không ghép cả hai (giữ cái Delta cao hơn)
    if "sampler" in best_in_group and "loss" in best_in_group and best_in_group["loss"][0] == "T11":
        drop = "sampler" if best_in_group["sampler"][1] < best_in_group["loss"][1] else "loss"
        best_in_group.pop(drop)
    combo, chosen = {}, []
    for g, (e, _) in sorted(best_in_group.items(), key=lambda kv: -kv[1][1]):
        combo.update(ov[e])
        chosen.append(e)
    return combo, chosen, (best_single[:2] if best_single else None)


def pick_final_recipe(runs: pd.DataFrame, combo: dict, best_single, base_f1: float):
    """Chọn công thức chung kết bằng val (một seed): tốt nhất trong T00 (nền), T18 (tổ hợp), yếu tố thắng đơn.

    Trả về (exp_id_nguồn, override). Hòa thì ưu tiên cấu hình đơn giản hơn (nền < đơn < tổ hợp).
    """
    cands = [("T00", {}, base_f1)]
    if best_single is not None:
        r = runs[(runs.exp_id == best_single[0]) & (runs.seed == 0)]
        if len(r):
            cands.append((best_single[0], best_single[1], float(r.iloc[0].val_macro_f1)))
    r = runs[(runs.exp_id == "T18") & (runs.seed == 0)]
    if len(r) and combo:
        cands.append(("T18", combo, float(r.iloc[0].val_macro_f1)))
    best = cands[0]
    for c in cands[1:]:
        if c[2] > best[2]:
            best = c
    return best[0], dict(best[1])


def choose_tta(inference_tbl: pd.DataFrame, p95_by_id: dict | None = None, budget_ms: float = 100.0,
               tol: float = 0.001):
    """Cách suy luận chung kết theo macro-F1 val, có ràng buộc thời gian thực. Trả về (tta, space, exp_id).

    p95_by_id: {mã I: p95 batch-1 (ms) đo được}. Nếu có, chỉ nhận cách có p95 <= budget_ms (I00 luôn được nhận
    để còn phương án dự phòng). Hòa trong tol thì lấy cách ít view hơn.
    """
    rows = []
    for eid, k, tta, space in TTA_CANDIDATES:
        r = inference_tbl[inference_tbl.exp_id == eid]
        if not (len(r) and np.isfinite(r.iloc[0]["macro_f1"])):
            continue
        if p95_by_id is not None and eid != "I00" and p95_by_id.get(eid, np.inf) > budget_ms:
            continue
        rows.append((float(r.iloc[0]["macro_f1"]), k, tta, space, eid))
    if not rows:
        return "none", "prob", "I00"
    top = max(r[0] for r in rows)
    near = sorted([r for r in rows if top - r[0] <= tol], key=lambda r: (r[1], r[4]))
    _, _, tta, space, eid = near[0]
    return tta, space, eid


# --------------------------------------------------------------------------- #
# Mốc T00: test một lượt từ checkpoint đã chọn trên val (không huấn luyện lại)
# --------------------------------------------------------------------------- #
def test_once_from_ckpt(cfg: T.Config, device=None) -> dict:
    """Ghi predictions/<exp_id>_seed<k>_test.csv bằng đúng best.pt đã được chọn trên val.

    Chạy toàn bộ tập test đúng một lượt (S4). Từ chối chạy nếu file test của seed này đã tồn tại.
    Cập nhật summary.json với test_macro_f1, test_top1, test_ece. Nếu run chưa tồn tại thì huấn luyện bằng
    train.run(save_test_predictions=True).
    """
    device = device or T.M.get_device()
    rd = T.run_dir(cfg)
    test_file = T.pred_path(cfg, "test")
    if test_file.exists():
        raise FileExistsError(f"{test_file} đã tồn tại: test đã chạy một lần cho seed này (S4).")
    if not (rd / "summary.json").exists() or not (rd / "best.pt").exists():
        return T.run(T.Config(**{**cfg.__dict__, "save_test_predictions": True}))
    model = T.load_best(cfg, device)
    _, _, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    loader = D.make_loader(test_df, cfg.images_dir, D.build_transforms(False, cfg.img_size), cfg.batch_size,
                           False, None, cfg.num_workers, cfg.seed, cfg.cache_images)
    use_amp = cfg.amp and device.type == "cuda"
    tn, ty, tl, _ = T.evaluate(model, loader, torch.nn.CrossEntropyLoss(), device, use_amp)
    tp = T._softmax(tl)
    np.save(rd / "test_logits.npy", tl)
    ev.save_predictions(test_file, tn, ty, tp)
    mt = ev.compute_metrics(ty, tp.argmax(1), tp)
    summary = json.loads((rd / "summary.json").read_text())
    summary.update(test_macro_f1=mt["macro_f1"], test_top1=mt["top1"], test_ece=mt["ece"])
    (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# --------------------------------------------------------------------------- #
# Bảng Final và dọn dẹp
# --------------------------------------------------------------------------- #
def final_table(pred_dir: str | Path, tags: dict[str, str], summaries: pd.DataFrame) -> pd.DataFrame:
    """Bảng Final: mỗi seed một dòng + dòng tổng hợp mean ± std, tính LẠI từ file dự đoán (đúng như giảng viên)."""
    rows = []
    for exp_id, desc in tags.items():
        files = sorted(Path(pred_dir).glob(f"{exp_id}_seed*_test.csv"))
        ms = []
        for f in files:
            p = ev.read_pred(str(f))
            m = ev.compute_metrics(p.y_true, p.y_pred, p.probs)
            ms.append(m)
            vf = Path(pred_dir) / f"{exp_id}_seed{p.seed}_val.csv"
            val_f1 = np.nan
            if vf.exists():
                pv = ev.read_pred(str(vf))
                val_f1 = ev.compute_metrics(pv.y_true, pv.y_pred, pv.probs)["macro_f1"]
            elif len(summaries):
                s = summaries[(summaries.exp_id == exp_id) & (summaries.seed == p.seed)]
                val_f1 = float(s.iloc[0].val_macro_f1) if len(s) else np.nan
            rows.append({"exp_id": exp_id, "cấu hình": desc, "seed": p.seed, "macro_f1_val": val_f1,
                         "macro_f1_test": m["macro_f1"], "top1_test": m["top1"], "ece_test": m["ece"],
                         "mean ± std": ""})
        if ms:
            def agg(key):
                a = np.array([m[key] for m in ms])
                return a.mean(), (a.std(ddof=1) if len(a) > 1 else float("nan"))
            f1m, f1s = agg("macro_f1")
            t1m, t1s = agg("top1")
            em, es = agg("ece")
            vals = [r["macro_f1_val"] for r in rows if r["exp_id"] == exp_id]
            rows.append({"exp_id": exp_id, "cấu hình": desc, "seed": f"mean ± std (n={len(ms)})",
                         "macro_f1_val": float(np.nanmean(vals)), "macro_f1_test": f1m, "top1_test": t1m,
                         "ece_test": em, "mean ± std":
                         f"F1 {f1m:.4f} ± {f1s:.4f} | top1 {t1m:.4f} ± {t1s:.4f} | ECE {em:.4f} ± {es:.4f}"})
    return pd.DataFrame(rows)


def prune_checkpoints(out_dir: str | Path) -> float:
    """Xoá *.pt (best/last) sau khi mọi bước đã dùng xong để thư mục kết quả nhỏ gọn. Trả về MB đã giải phóng."""
    freed = 0
    for f in Path(out_dir).rglob("*.pt"):
        freed += f.stat().st_size
        f.unlink()
    return freed / 1e6


def backup(sub_dir: str | Path, dest: str | Path | None) -> None:
    """Chép các file KẾT QUẢ NHỎ (không .pt) sang nơi lưu bền (vd Google Drive) bằng rsync. Lỗi chỉ ghi log."""
    if not dest or not Path(str(dest)).parent.exists():
        return
    import subprocess
    try:
        Path(dest).mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--exclude", "*.pt", "--exclude", "__pycache__", str(sub_dir).rstrip("/") + "/", str(dest)],
                       check=True, capture_output=True, timeout=300)
    except Exception as e:  # noqa: BLE001
        log(f"backup lỗi (bỏ qua): {e}")
