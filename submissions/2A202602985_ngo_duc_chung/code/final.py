"""final.py - suy luận chung kết (Bước 4): TTA + temperature scaling, chạm TEST đúng một lần mỗi seed.

`train.run(cfg)` (với save_test_predictions=False) huấn luyện và chọn checkpoint trên val.
`final_predict(cfg, ...)` sau đó:
  1. chạy model trên VAL với mọi view, khớp nhiệt độ T trên VAL (không dùng test);
  2. chạy model trên TEST một lượt duy nhất với mọi view, rồi tính từ cùng các logit đó cả bản
     chưa hiệu chuẩn (T = 1) lẫn bản đã hiệu chuẩn;
  3. ghi:  <exp_id>_seed<k>_val.csv, <exp_id>_seed<k>_test.csv (đã hiệu chuẩn),
           <exp_id>uncal_seed<k>_test.csv (chưa hiệu chuẩn, để chấm I4a).
Từ chối chạy nếu file test của seed này đã tồn tại (chống chạy test lần hai).
"""
from __future__ import annotations

import json
from functools import partial

import numpy as np
import torch

import dataset as D
import inference as I
import train as T

ev = T.ev

# preset TTA -> (cạnh ảnh đầu vào của loader, views). crop5 cần ảnh 256 để cắt 224.
def tta_preset(name: str, img_size: int):
    if name == "none":
        return img_size, [I.view_identity]
    if name == "hflip":
        return img_size, [I.view_identity, I.view_hflip]
    if name == "flips":
        return img_size, [I.view_identity, I.view_hflip, I.view_vflip, lambda x: I.view_vflip(I.view_hflip(x))]
    if name == "crop5":
        return 256, partial(I.views_multicrop, crop=img_size)
    if name == "crop5flip":
        return 256, partial(I.views_multicrop, crop=img_size, flip=True)
    raise ValueError(f"tta phải thuộc none|hflip|flips|crop5|crop5flip, nhận {name!r}")


def final_predict(cfg: T.Config, tta: str = "none", space: str = "prob", calibrate: bool = True,
                  device=None, amp: bool = False) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_file = T.pred_path(cfg, "test")
    uncal_cfg = T.Config(**{**cfg.__dict__, "exp_id": cfg.exp_id + "uncal"})
    if test_file.exists():
        raise FileExistsError(f"{test_file} đã tồn tại: test đã chạy một lần cho seed này (S4). "
                              "Không chạy lại; nếu phát hiện lỗi, ghi vào báo cáo.")
    size, views = tta_preset(tta, cfg.img_size)
    model = T.load_best(cfg, device)
    _, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)

    def loader(df):
        return D.make_loader(df, cfg.images_dir, D.build_transforms(False, size), cfg.batch_size, False,
                             None, cfg.num_workers, cfg.seed)

    # 1) VAL: khớp T
    vn, vy, vz = I.predict_multiview(model, loader(val_df), device, views, amp)
    temp = I.fit_temperature_views(vz, vy, space) if calibrate else 1.0
    val_probs = I.aggregate_views(vz, space, temp)
    ev.save_predictions(T.pred_path(cfg, "val"), vn, vy, val_probs)
    ev.save_predictions(T.pred_path(uncal_cfg, "val"), vn, vy, I.aggregate_views(vz, space, 1.0))

    # 2) TEST: một lượt duy nhất
    tn, ty, tz = I.predict_multiview(model, loader(test_df), device, views, amp)
    ev.save_predictions(T.pred_path(uncal_cfg, "test"), tn, ty, I.aggregate_views(tz, space, 1.0))
    ev.save_predictions(test_file, tn, ty, I.aggregate_views(tz, space, temp))

    out = {"exp_id": cfg.exp_id, "seed": cfg.seed, "tta": tta, "space": space, "T": temp,
           "n_views": len(tz), "val_T_fit_on": "val"}
    for split, names, y, z in (("val", vn, vy, vz), ("test", tn, ty, tz)):
        for tag, t in (("uncal", 1.0), ("cal", temp)):
            p = I.aggregate_views(z, space, t)
            m = ev.compute_metrics(y, p.argmax(1), p)
            out[f"{split}_{tag}_macro_f1"], out[f"{split}_{tag}_ece"] = m["macro_f1"], m["ece"]
    (T.run_dir(cfg) / "final_predict.json").write_text(json.dumps(out, indent=2))
    return out
