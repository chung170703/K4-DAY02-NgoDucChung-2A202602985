"""assemble_results.py - ghép kết quả của hai lượt chạy thành results.xlsx (7 sheet bắt buộc + Lab2_Slide).

Lượt 1 (Colab T4, `lab_day2_kaggle.ipynb`): Bước 0, Bước 1 (7 backbone), T00 x3 seed, T03, T09, T14. Máy ảo bị thu hồi (hết hạn mức GPU)
nên checkpoint và log các lần chạy sau đó mất; thư mục `run_logs_colab_run1/`, `predictions/colab_run1/`, `logs/progress_colab_run1.log`
là phần đã sao lưu kịp.
Lượt 2 (Kaggle T4, `lab_day2_recovery.ipynb`): F01 (chung kết) x3 seed, T00 x3 seed (mốc, có test), Bước 3, ma trận Lab #2, các ablation
T04, T08, T10, T11, T12, T17, T02 và ensemble.

Mọi số trong results.xlsx đọc từ run_logs*/**/summary.json, tables/*.csv và predictions/*.csv (tính lại bằng eval.py); không nhập tay.
Chạy: python assemble_results.py   (từ thư mục code/ của bài nộp)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUB.parents[1]))      # eval.py ở gốc repo
sys.path.insert(0, str(SUB / "code"))
import eval as ev          # noqa: E402
import tables as TB        # noqa: E402

names = ev.load_names(None) if False else ev.CLASS_NAMES
HARD = {"Chinee apple": 0, "Snake weed": 7}   # chỉ số lớp theo cột Label của labels.csv


def load_runs(d: Path) -> pd.DataFrame:
    return TB.collect_summaries(d)


def val_hard_f1(pred_dir: Path, exp_id: str, seed: int) -> dict:
    f = pred_dir / f"{exp_id}_seed{seed}_val.csv"
    if not f.exists():
        return {}
    p = ev.read_pred(str(f))
    m = ev.compute_metrics(p.y_true, p.y_pred, p.probs)
    return {f"F1 val {k}": float(m["f1"][i]) for k, i in HARD.items()}


def verdict(d: float, sigma: float) -> str:
    a = abs(d)
    return "vượt nhiễu (>2σ)" if a > 2 * sigma else ("gần ngưỡng (1-2σ)" if a > sigma else "không phân biệt được với nhiễu")


def main():
    colab = load_runs(SUB / "run_logs_colab_run1")
    kag = load_runs(SUB / "run_logs")
    PRED, PRED_C = SUB / "predictions", SUB / "predictions" / "colab_run1"
    TAB = SUB / "tables"
    PRED_C_MAIN = SUB / "predictions"          # B01-B07, T03, T09, T14 (Colab) nằm ở predictions/
    colab = colab.assign(lượt="Colab T4 (lượt 1)")
    kag = kag.assign(lượt="Kaggle T4 (lượt 2)")

    # ---------------- Backbones ----------------
    lat = pd.read_csv(TAB / "latency_backbones.csv")
    b1 = lat[(lat.batch == 1) & (lat.dtype == "fp32")].set_index("exp_id")["p50"]
    bb = colab[colab.exp_id.str.startswith("B")].copy()
    bb["lat_b1_fp32_ms"] = bb.exp_id.map(b1)
    bb["ghi chú"] = "1 seed; công thức nền T00; lượt 1 (Colab)"
    bk = kag[kag.exp_id.isin(["B03", "B04", "B05"])].copy()
    bk["lat_b1_fp32_ms"] = np.nan
    bk["ghi chú"] = "huấn luyện lại ở lượt 2 (Kaggle) để làm ensemble I05; so với lượt 1 cho thấy nhiễu giữa hai lần chạy"
    bbt = pd.concat([bb, bk], ignore_index=True)[["exp_id", "lượt", "backbone", "pretrained_tag", "params_m", "gmacs", "img_size", "epochs",
                                                  "seed", "val_macro_f1", "val_top1", "sec_per_epoch", "lat_b1_fp32_ms", "ghi chú"]]

    # ---------------- Training ----------------
    rows = []

    def block(runs: pd.DataFrame, pred_dir: Path, label: str, ablations: list[tuple[str, str, str, int]], extra: list[dict] | None = None):
        t00 = runs[runs.exp_id == "T00"].sort_values("seed")
        sigma = float(t00.val_macro_f1.std(ddof=1))
        ref0 = float(t00[t00.seed == 0].val_macro_f1.iloc[0]); mean = float(t00.val_macro_f1.mean())
        for r in t00.itertuples():
            rows.append({"exp_id": "T00", "lượt": label, "backbone": "convnext_tiny", "trục": "nền", "khác T00": "(công thức nền)", "seed": r.seed,
                         "val_macro_f1": r.val_macro_f1, "val_top1": r.val_top1, "delta_vs_T00_seed0": r.val_macro_f1 - ref0, "delta/σ": np.nan,
                         "delta_vs_T00_mean": r.val_macro_f1 - mean, **val_hard_f1(pred_dir if label.startswith("Kaggle") else PRED_C, "T00", r.seed),
                         "ghi chú": f"mốc; σ (std 3 seed, ddof=1) = {sigma:.4f}; mean = {mean:.4f}"})
        for e, ax, desc, seed in ablations:
            r = runs[(runs.exp_id == e) & (runs.seed == seed)]
            if r.empty:
                continue
            r = r.iloc[0]; d = float(r.val_macro_f1 - ref0)
            rows.append({"exp_id": e, "lượt": label, "backbone": "convnext_tiny", "trục": ax, "khác T00": desc, "seed": seed,
                         "val_macro_f1": r.val_macro_f1, "val_top1": r.val_top1, "delta_vs_T00_seed0": d, "delta/σ": d / sigma,
                         "delta_vs_T00_mean": float(r.val_macro_f1 - mean), **val_hard_f1(pred_dir, e, seed),
                         "ghi chú": f"1 seed; {verdict(d, sigma)} (σ T00 cùng lượt = {sigma:.4f})"})
        for x in extra or []:
            d = x["val_macro_f1"] - ref0
            rows.append({**x, "lượt": label, "backbone": "convnext_tiny", "delta_vs_T00_seed0": d, "delta/σ": d / sigma,
                         "delta_vs_T00_mean": x["val_macro_f1"] - mean,
                         "ghi chú": x.get("ghi chú", "") + f"; {verdict(d, sigma)} (σ T00 cùng lượt = {sigma:.4f})"})
        return sigma, ref0, mean

    sig_c, ref_c, mean_c = block(colab, PRED_C_MAIN, "Colab T4 (lượt 1)",
                                 [("T14", "E", "LR backbone 3e-4, head 3e-3", 0), ("T09", "C", "loss = label smoothing 0.1", 0),
                                  ("T03", "B", "aug = geo (lật dọc + xoay 90°)", 0)])
    f01 = kag[(kag.exp_id == "F01") & (kag.seed == 0)].iloc[0]
    sig_k, ref_k, mean_k = block(kag, PRED, "Kaggle T4 (lượt 2)",
                                 [("T04", "B", "aug = color", 0), ("T08", "B", "mix = CutMix (alpha 1.0)", 0), ("T10", "C", "loss = focal γ=2", 0),
                                  ("T11", "C", "loss = CE trọng số lớp (1/n)", 0), ("T12", "D", "sampler = cân bằng lớp", 0),
                                  ("T17", "G", "20 epoch (thay vì 10)", 0), ("T02", "A", "đóng băng backbone (linear probe)", 0),
                                  ("T01", "A", "khởi tạo từ đầu", 0)],
                                 [{"exp_id": "F01 (seed 0)", "trục": "F", "khác T00": "EMA 0.99 (đánh giá bằng trọng số EMA; cùng seed, cùng quá trình huấn luyện)",
                                   "seed": 0, "val_macro_f1": float(f01.val_macro_f1), "val_top1": float(f01.val_top1), **val_hard_f1(PRED, "F01", 0),
                                   "ghi chú": "F01 seed 0 chính là T00 seed 0 + EMA (EMA không đổi quá trình huấn luyện, chỉ đổi trọng số đánh giá); tương ứng ablation T16 của lượt 1"}])
    # T21: tổ hợp CE trọng số lớp + CutMix + EMA (lượt 3, Kaggle, chỉ val), so cặp theo seed với T00 cùng lượt
    t21 = load_runs(SUB / "run_logs")
    t21 = t21[t21.exp_id == "T21"].sort_values("seed")
    if len(t21):
        t00k = kag[kag.exp_id == "T00"].set_index("seed")
        for r in t21.itertuples():
            d_pair = r.val_macro_f1 - float(t00k.loc[r.seed, "val_macro_f1"])
            rows.append({"exp_id": "T21", "lượt": "Kaggle T4 (lượt 3, chỉ T21)", "backbone": "convnext_tiny", "trục": "kết hợp (C+B+F)",
                         "khác T00": "CE trọng số lớp + CutMix + EMA 0.99", "seed": r.seed, "val_macro_f1": r.val_macro_f1, "val_top1": r.val_top1,
                         "delta_vs_T00_seed0": r.val_macro_f1 - ref_k, "delta/σ": (r.val_macro_f1 - ref_k) / sig_k,
                         "delta_vs_T00_mean": r.val_macro_f1 - mean_k, **val_hard_f1(PRED, "T21", r.seed),
                         "ghi chú": f"tổ hợp; Δ so với T00 cùng seed = {d_pair:+.4f}"})
        m, sd = float(t21.val_macro_f1.mean()), float(t21.val_macro_f1.std(ddof=1))
        rows.append({"exp_id": "T21 (mean ± std, 3 seed)", "lượt": "Kaggle T4 (lượt 3, chỉ T21)", "backbone": "convnext_tiny", "trục": "kết hợp (C+B+F)",
                     "khác T00": "CE trọng số lớp + CutMix + EMA 0.99", "seed": "0,1,2", "val_macro_f1": m, "val_top1": float(t21.val_top1.mean()),
                     "delta_vs_T00_seed0": np.nan, "delta/σ": (m - mean_k) / sig_k, "delta_vs_T00_mean": m - mean_k,
                     "ghi chú": f"mean macro-F1 val {m:.4f} ± {sd:.4f} so với T00 {mean_k:.4f} ± {sig_k:.4f}: {verdict(m - mean_k, max(sig_k, sd))}"})
    train_t = pd.DataFrame(rows)

    # ---------------- Inference ----------------
    inf = pd.read_csv(TAB / "inference.csv")
    i05 = pd.read_csv(TAB / "inference_I05.csv")
    t00k = kag[(kag.exp_id == "T00") & (kag.seed == 0)].iloc[0]
    i00 = inf[inf.exp_id == "I00"].iloc[0]
    i06 = {"exp_id": "I06", "phương pháp": "Trọng số EMA 0.99 (F01 seed 0 so với T00 seed 0)", "mô hình": "F01 seed 0 vs T00 seed 0", "K": 1,
           "macro_f1": float(f01.val_macro_f1), "top1": float(f01.val_top1), "ece": float(f01.val_ece),
           "ghi chú": f"không tốn thêm khi suy luận; T00 seed 0: macro-F1 {t00k.val_macro_f1:.4f}, top-1 {t00k.val_top1:.4f}, ECE {t00k.val_ece:.4f}",
           "p50_ms_b1": i00.p50_ms_b1, "p95_ms_b1": i00.p95_ms_b1, "p99_ms_b1": i00.p99_ms_b1, "chi phí tương đối vs I00": 1.0}
    inf_t = pd.concat([inf, i05, pd.DataFrame([i06])], ignore_index=True)
    inf_t = inf_t.rename(columns={"macro_f1": "macro_f1_val", "top1": "top1_val", "ece": "ece_val", "p50_ms_b1": "p50_ms_batch1",
                                  "p95_ms_b1": "p95_ms_batch1", "p99_ms_b1": "p99_ms_batch1", "images_per_s": "thông lượng (ảnh/s, batch 32 hoặc batch 1)"})

    # ---------------- Final / PerClass (từ results_recovery.xlsx của lượt 2) ----------------
    rec = pd.ExcelFile(TAB / "results_recovery_kaggle.xlsx")
    final_t, perclass_t = pd.read_excel(rec, "Final"), pd.read_excel(rec, "PerClass")

    # ---------------- Latency ----------------
    lat_c = lat.assign(**{"cấu hình": lat.exp_id, "lượt": "Colab T4 (lượt 1)"})
    lat_k = pd.read_csv(TAB / "latency_inference.csv").assign(lượt="Kaggle T4 (lượt 2)")
    lat_t = pd.concat([lat_c, lat_k], ignore_index=True)[["cấu hình", "lượt", "gpu", "dtype", "batch", "bn_fused", "img_size", "p50", "p95", "p99", "images_per_s", "torch"]]

    # ---------------- Summary: top 10 theo macro-F1 val (mọi lần chạy huấn luyện) ----------------
    allr = pd.concat([colab, kag], ignore_index=True)
    allr = allr[~allr.exp_id.isin(["F01uncal"])]
    mean_by = {"Colab T4 (lượt 1)": mean_c, "Kaggle T4 (lượt 2)": mean_k}
    top = allr.sort_values("val_macro_f1", ascending=False).head(10)[["exp_id", "lượt", "backbone", "seed", "val_macro_f1", "val_top1", "params_m", "gmacs", "sec_per_epoch"]].copy()
    lat_bb = lat[(lat.batch == 1) & (lat.dtype == "fp32")].groupby("backbone")["p50"].first()
    top["lat_b1_fp32_p50_ms"] = top.backbone.map(lat_bb)
    top["delta_vs_T00_mean_cùng_lượt"] = [v - mean_by[l] for v, l in zip(top.val_macro_f1, top.lượt)]
    top["ghi chú"] = np.where(top.exp_id == "T00", "mốc", np.where(top.exp_id == "F01", "chung kết (EMA); số test ở sheet Final", ""))

    lab2 = pd.read_csv(TAB / "lab2_slide_matrix.csv")
    # bài làm thêm (điểm thưởng, lượt 4, chỉ val): DINOv2 linear probe, lệch phân phối, ONNX
    bonus = {}
    if (TAB / "bonus_dino_runs.csv").exists():
        bonus["Bonus_DINOv2"] = pd.read_csv(TAB / "bonus_dino_runs.csv")[["exp_id", "backbone", "pretrained_tag", "init", "params_m", "val_macro_f1", "val_top1", "val_ece", "best_epoch", "sec_per_epoch"]]
        bonus["Bonus_Shift"] = pd.read_csv(TAB / "bonus_shift.csv")
        bonus["Bonus_ONNX"] = pd.read_csv(TAB / "bonus_onnx.csv")

    sheets = {"Backbones": bbt, "Training": train_t, "Inference": inf_t, "Final": final_t, "PerClass": perclass_t, "Latency": lat_t,
              "Summary": top, "Lab2_Slide": lab2, **bonus}
    TB.write_results_xlsx(SUB / "results.xlsx", sheets, {"Backbones": "val_macro_f1", "Summary": "val_macro_f1"})
    print("đã ghi", SUB / "results.xlsx", {k: len(v) for k, v in sheets.items()})
    (TAB / "noise_summary.json").write_text(json.dumps({"colab_T00_sigma": sig_c, "colab_T00_mean": mean_c, "colab_T00_seed0": ref_c,
                                                         "kaggle_T00_sigma": sig_k, "kaggle_T00_mean": mean_k, "kaggle_T00_seed0": ref_k}, indent=1))


if __name__ == "__main__":
    main()
