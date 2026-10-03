# Lab Day 2 — DeepWeeds (bài làm của Ngô Đức Chung — 2A202602985)

> Điền các mục `<...>` (link Colab, phiên bản thư viện, seed) sau khi chạy xong.

## Chạy lại
1. Colab: mở `code/lab_day2.ipynb` (link: <link Colab/Kaggle>), bật GPU T4, kiểm tra ô đầu (`GH_USER`, `REPO`, `MSSV`, `SLUG`), chạy lần lượt từ trên xuống.
2. Dữ liệu: DeepWeeds `images.zip` (Zenodo 7939060, MD5 `b7b30f96d466fba86016aa5a26606e0f`) và nhãn fold 0 từ GitHub của tác giả; notebook tự tải và kiểm tra.
3. Test CPU của code (không cần GPU/dữ liệu thật): `cd code && python -m unittest test_code`.

## Cấu trúc code
| File | Nội dung |
|---|---|
| `dataset.py` | đọc fold 0, kiểm tra chia dữ liệu (S1–S4), transform/augmentation, DataLoader |
| `model.py` | backbone timm, đóng băng, 3 nhóm tham số, đếm params/GMAC |
| `losses.py` | label smoothing, focal, trọng số lớp, Mixup/CutMix |
| `train.py` | `run(cfg)`: AMP, warmup+cosine, EMA, chọn checkpoint theo macro-F1 val, resume, vẽ đường cong |
| `inference.py` | TTA, gộp prob/logit, ensemble, temperature scaling, gộp BN |
| `benchmark.py` | độ trễ p50/p95/p99 (warmup, synchronize) |
| `final.py` | suy luận chung kết: test một lượt/seed, ghi `F01`, `F01uncal` |
| `checks.py`, `tables.py` | kiểm tra pipeline; gom bảng và ghi `results.xlsx` |

## Phiên bản và seed
- Python <...>, PyTorch <...>, timm <...>, GPU <...> (notebook in ra khi chạy).
- Seed: ablation `<...>`; chung kết `0, 1, 2`.

## Mức tái lập
Cùng seed cho kết quả rất gần nhau, không bảo đảm trùng từng bit (AMP và một số phép CUDA không tất định).
