"""report_figs.py - hình và bảng cho báo cáo từ file dự đoán (Bước 4-5): ma trận nhầm lẫn, ảnh bị đoán sai,
biểu đồ ablation, sheet PerClass. Mọi số đều tính từ predictions/*.csv bằng eval.py của repo.

Chỉ dùng SAU Bước 4 với dự đoán test: nhìn test để phân tích lỗi là được, nhưng KHÔNG quay lại sửa
cấu hình rồi chạy test lần nữa (GUIDE mục 5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import train as T

ev = T.ev


def _plt():
    import sys
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules:  # trong notebook pyplot đã nạp: đừng đổi backend inline
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def top_confusions(cm: np.ndarray, names: list[str], n: int = 6) -> pd.DataFrame:
    """Các cặp (thật -> đoán) sai nhiều nhất: số ảnh và tỉ lệ % của lớp thật."""
    rows = []
    support = cm.sum(1)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i, j] > 0:
                rows.append({"thật": names[i], "đoán": names[j], "số ảnh": float(cm[i, j]),
                             "% lớp thật": 100.0 * cm[i, j] / max(1, support[i])})
    return pd.DataFrame(rows).sort_values("số ảnh", ascending=False).head(n).reset_index(drop=True)


def confusion_figure(pattern, test_csv: str, labels_csv: str, title: str, path=None):
    """Ma trận nhầm lẫn của một nhóm seed (cộng số ảnh các seed rồi chia số seed).

    Trả về (fig, cm_trung_bình, bảng các nhầm lẫn lớn nhất). Trái: số ảnh; phải: % theo hàng (= recall ở đường chéo).
    """
    g = ev.load_group(pattern, test_csv, ref_what="test")
    names = ev.load_names(labels_csv)
    cm = np.mean([m["confusion"] for m in g.metrics], axis=0)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for a, mat, sub, fmt_ in ((ax[0], cm, "số ảnh (trung bình các seed)", "{:.0f}"), (ax[1], norm, "% theo hàng (recall)", "{:.0%}")):
        im = a.imshow(mat, cmap="Blues")
        a.set(xticks=range(len(names)), yticks=range(len(names)), xlabel="dự đoán", ylabel="nhãn thật", title=sub)
        a.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        a.set_yticklabels(names, fontsize=8)
        for i in range(len(names)):
            for j in range(len(names)):
                v = mat[i, j]
                if v > 0:
                    a.text(j, i, fmt_.format(v), ha="center", va="center", fontsize=7,
                           color="white" if v > mat.max() * 0.55 else "black")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"{title} (n seed = {len(g.preds)})")
    fig.tight_layout()
    return _save(fig, path), cm, top_confusions(cm, names)


def misclassified_grid(pred_file: str, images_dir: str, names: list[str], n: int = 12,
                       pair: tuple[int, int] | None = None, path=None):
    """Lưới ảnh bị đoán sai của MỘT file dự đoán, sắp theo độ tự tin giảm dần (sai mà chắc chắn là đáng xem nhất).

    pair=(a, b): chỉ lấy ảnh thật a đoán b hoặc thật b đoán a (ví dụ Chinee Apple và Snake Weed).
    """
    from PIL import Image
    p = ev.read_pred(pred_file)
    wrong = np.where(p.y_true != p.y_pred)[0]
    if pair is not None:
        a, b = pair
        wrong = np.array([i for i in wrong if {int(p.y_true[i]), int(p.y_pred[i])} == {a, b}], dtype=int)
    conf = p.probs[wrong, p.y_pred[wrong]] if len(wrong) else np.array([])
    wrong = wrong[np.argsort(-conf)][:n]
    plt = _plt()
    cols = 4
    rows = max(1, int(np.ceil(max(1, len(wrong)) / cols)))
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.3 * rows), squeeze=False)
    for a in axs.ravel():
        a.axis("off")
    for a, i in zip(axs.ravel(), wrong):
        a.imshow(Image.open(Path(images_dir) / str(p.filenames[i])).convert("RGB"))
        a.set_title(f"thật: {names[p.y_true[i]]}\nđoán: {names[p.y_pred[i]]} ({p.probs[i, p.y_pred[i]]:.2f})", fontsize=8)
    total = int((p.y_true != p.y_pred).sum())
    fig.suptitle(f"Ảnh bị đoán sai ({len(wrong)} / {total} lỗi" + (f", cặp {names[pair[0]]} - {names[pair[1]]}" if pair else "") + ")")
    fig.tight_layout()
    return _save(fig, path), [str(p.filenames[i]) for i in wrong]


def ablation_figure(training_df: pd.DataFrame, noise_std: float, path=None, ref_name: str = "T00"):
    """Cột Δ macro-F1 val so với T00 của từng thí nghiệm, kèm dải ±1 và ±2 std của T00 (nhiễu do seed)."""
    plt = _plt()
    d = training_df.sort_values("delta_vs_T00").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.34 * len(d) + 1)))
    ax.axvspan(-2 * noise_std, 2 * noise_std, color="0.9", label="±2 std của T00")
    ax.axvspan(-noise_std, noise_std, color="0.8", label="±1 std của T00")
    cmap = plt.get_cmap("tab10")
    axes = sorted(d["trục"].unique())
    colors = {a: cmap(i % 10) for i, a in enumerate(axes)}
    ax.barh(range(len(d)), d["delta_vs_T00"], color=[colors[a] for a in d["trục"]])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{e}  {s}" for e, s in zip(d["exp_id"], d["khác T00"])], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel=f"Δ macro-F1 val so với {ref_name}", title="Ablation công thức huấn luyện (màu = trục A-G)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[a]) for a in axes]
    ax.legend(handles + [plt.Rectangle((0, 0), 1, 1, color="0.8"), plt.Rectangle((0, 0), 1, 1, color="0.9")],
              [f"trục {a}" for a in axes] + ["±1 std", "±2 std"], fontsize=7, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def per_class_sheet(groups: dict[str, str], test_csv: str, labels_csv: str) -> pd.DataFrame:
    """Sheet PerClass: với mỗi cấu hình (nhóm seed) và mỗi lớp: số ảnh test, precision, recall, F1 (mean qua seed), std recall."""
    names = ev.load_names(labels_csv)
    rows = []
    for label, pattern in groups.items():
        g = ev.load_group(pattern, test_csv, ref_what="test")
        (p_m, _), (r_m, r_s), (f_m, f_s) = (g.summary["precision"], g.summary["recall"], g.summary["f1"])
        support = g.metrics[0]["support"]
        for k, nm in enumerate(names):
            rows.append({"cấu hình": label, "lớp": nm, "số ảnh test": int(support[k]), "precision": p_m[k],
                         "recall": r_m[k], "recall_std": r_s[k], "F1": f_m[k], "F1_std": f_s[k], "n_seed": len(g.preds)})
    return pd.DataFrame(rows)
