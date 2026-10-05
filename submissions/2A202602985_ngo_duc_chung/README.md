# Lab Day 2 — DeepWeeds (bài làm của Ngô Đức Chung — 2A202602985)

Kết quả chính (test fold 0, mean ± std qua 3 seed, cấu hình chọn bằng val): ConvNeXt-tiny + EMA + TTA 5 crop + temperature scaling cho **top-1 97,91 ± 0,29%, macro-F1 0,9738 ± 0,0035**; mốc T00 + 1 view: top-1 97,66 ± 0,38%, macro-F1 0,9710 ± 0,0048 (chênh nhỏ hơn std, không phân biệt được). Chi tiết ở [`report.md`](report.md) và [`results.xlsx`](results.xlsx).

## Link notebook chạy lại
- **Notebook đầy đủ, chạy một mạch từ đầu** (tải dữ liệu → Bước 0–5 → Lab #2), tự chứa mã nguồn, dùng được trên Colab hoặc Kaggle. Đây là notebook của lượt 1: đã chạy thật tới giữa Bước 2 thì máy ảo bị thu hồi, và đã chạy hết ở chế độ thử (`LAB_SMOKE=1`); phần còn lại do notebook phục hồi bên dưới chạy (xem mục "Các lượt chạy"):
  [`code/lab_day2_kaggle.ipynb`](code/lab_day2_kaggle.ipynb) · mở trên Colab: <https://colab.research.google.com/github/chung170703/K4-DAY02-NgoDucChung-2A202602985/blob/main/submissions/2A202602985_ngo_duc_chung/code/lab_day2_kaggle.ipynb>
- **Notebook phục hồi** (phần chạy thành công trên Kaggle cho chung kết, mốc, Bước 3, Lab #2 — xem mục "Các lượt chạy"): [`code/lab_day2_recovery.ipynb`](code/lab_day2_recovery.ipynb) · Colab: <https://colab.research.google.com/github/chung170703/K4-DAY02-NgoDucChung-2A202602985/blob/main/submissions/2A202602985_ngo_duc_chung/code/lab_day2_recovery.ipynb> · bản đã chạy trên Kaggle: <https://www.kaggle.com/code/chung140204/notebook119554a435> (công khai; bản nộp chính thức là file `.ipynb` trong repo).
- **Notebook thí nghiệm tổ hợp T21** (CE trọng số lớp + CutMix + EMA, 3 seed, chỉ val): [`code/lab_day2_combo.ipynb`](code/lab_day2_combo.ipynb) · Colab: <https://colab.research.google.com/github/chung170703/K4-DAY02-NgoDucChung-2A202602985/blob/main/submissions/2A202602985_ngo_duc_chung/code/lab_day2_combo.ipynb> · bản đã chạy trên Kaggle (công khai): <https://www.kaggle.com/code/chung140204/notebook0d9be9c453>.
- **Notebook bài làm thêm (điểm thưởng)**: DINOv2 linear probe, lệch phân phối, Grad-CAM, ONNX (chỉ val): [`code/lab_day2_bonus.ipynb`](code/lab_day2_bonus.ipynb) · Colab: <https://colab.research.google.com/github/chung170703/K4-DAY02-NgoDucChung-2A202602985/blob/main/submissions/2A202602985_ngo_duc_chung/code/lab_day2_bonus.ipynb> · bản đã chạy trên Kaggle (công khai): <https://www.kaggle.com/code/chung140204/notebook108982fed1>.
- **Notebook nhiều fold (điểm thưởng)**: cấu hình cuối trên fold 1 và fold 2: [`code/lab_day2_folds.ipynb`](code/lab_day2_folds.ipynb) · Colab: <https://colab.research.google.com/github/chung170703/K4-DAY02-NgoDucChung-2A202602985/blob/main/submissions/2A202602985_ngo_duc_chung/code/lab_day2_folds.ipynb> · bản đã chạy trên Kaggle: <https://www.kaggle.com/code/chung140204/notebook621de3668e> (công khai).
- `code/lab_day2.ipynb` là bản Colab + Drive ban đầu (không dùng để sinh kết quả nộp).

## Thứ tự chạy lại
1. Bật GPU (T4 trở lên) và Internet. Mở `lab_day2_kaggle.ipynb` (hoặc `lab_day2_recovery.ipynb`) rồi **Run all**. Ô 1 giải nén mã nguồn (các `.py` trong `code/`) và `eval.py` gốc; ô 2 tải DeepWeeds (Zenodo 7939060, kiểm tra MD5 `b7b30f96d466fba86016aa5a26606e0f`) và nhãn fold 0 từ GitHub của tác giả; các ô sau chạy Bước 0 → 5. Biến môi trường `LAB_SMOKE=1` chạy thử 1 epoch/vài batch (số liệu đó không dùng để nộp).
2. Mỗi lần chạy huấn luyện đều qua `train.run(Config(...))`; chạy lại được sau khi ngắt (`resume`, `skip_if_done`). Các lựa chọn (backbone, yếu tố ablation thắng, công thức chung kết, TTA) được chọn tự động chỉ từ số liệu val, bằng các hàm trong `kaggle_run.py`.
3. Tính lại chỉ số từ file dự đoán (từ thư mục gốc của repo):
```bash
python eval.py score --pred "submissions/2A202602985_ngo_duc_chung/predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out
python eval.py grade --final "submissions/2A202602985_ngo_duc_chung/predictions/F01_seed*_test.csv" \
    --baseline "submissions/2A202602985_ngo_duc_chung/predictions/T00_seed*_test.csv" \
    --uncal "submissions/2A202602985_ngo_duc_chung/predictions/F01uncal_seed*_test.csv" \
    --final-val "submissions/2A202602985_ngo_duc_chung/predictions/F01_seed*_val.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --latency-p95-ms 31.8
```
4. Ghép `results.xlsx` từ hai lượt chạy: `cd code && python assemble_results.py`. Test CPU của code (34 test, không cần GPU): `cd code && python -m unittest test_code`.

## Phiên bản và seed
- Python 3.13 (Colab), PyTorch 2.11 (`2.11.0+cu130` Colab, `2.11.0+cu128` Kaggle), timm 1.0.29, GPU Tesla T4 (Colab và Kaggle); torch, timm và tên GPU được in ở đầu mỗi lần chạy (`logs/`).
- Seed: mốc `T00` và chung kết `F01` 0, 1, 2; backbone và hầu hết ablation seed 0; ma trận Lab #2 (khởi tạo × CutMix) 0, 1, 2. Seed chỉ đổi khởi tạo head, thứ tự batch và augmentation, không đổi cách chia dữ liệu (fold 0 nguyên bản).
- 10 epoch cho công thức nền (GUIDE: 10–15, giảm vì ngân sách GPU).

## Các lượt chạy (đọc trước khi chấm)
Lượt 1 chạy trên Colab T4; sau khoảng 6 giờ, hạn mức GPU free cạn và máy ảo bị thu hồi giữa Bước 2, nên mất checkpoint và log của một số lần chạy. Phần kịp sao lưu: `run_logs_colab_run1/`, `predictions/` (B01–B07, T03, T09, T14), `predictions/colab_run1/` (T00), `logs/progress_colab_run1.log`, `tables/` (EDA, kiểm tra pipeline, độ trễ backbone). Lượt 5 (Kaggle T4, `lab_day2_folds.ipynb`) chạy cấu hình cuối trên fold 1 và 2 (mục 9.5 của báo cáo). Lượt 4 (Kaggle T4, `lab_day2_bonus.ipynb`) chỉ chạy các bài làm thêm trên val (mục 9 của báo cáo). Lượt 3 (Kaggle T4, `lab_day2_combo.ipynb`) chỉ chạy thí nghiệm tổ hợp T21 trên val sau khi chung kết đã xong. Lượt 2 chạy trên Kaggle T4 bằng `lab_day2_recovery.ipynb` với đúng các quyết định đã chốt bằng val ở lượt 1: chung kết `F01`, mốc `T00`, Bước 3, ma trận Lab #2 và các ablation bị mất. Số liệu `T00` của hai lượt có cột `lượt` riêng; Δ của mỗi ablation chỉ so với `T00` cùng lượt. Số liệu lượt 1 đã mất log được nêu ở Phụ lục B của báo cáo và không dùng để kết luận. Trước lượt 1 tôi có chạy thử cả pipeline 1 epoch trên Colab (gồm cả bước test, với mô hình chưa huấn luyện, kết quả bị bỏ).

## Cấu trúc thư mục
| Đường dẫn | Nội dung |
|---|---|
| `report.md` | Báo cáo kết luận |
| `results.xlsx` | 7 sheet bắt buộc (`Backbones`, `Training`, `Inference`, `Final`, `PerClass`, `Latency`, `Summary`) + `Lab2_Slide` + `Bonus_DINOv2`, `Bonus_Shift`, `Bonus_ONNX`, `Bonus_Folds` |
| `curves/` | Một ảnh đường cong cho mỗi lần huấn luyện, tên `<exp_id>_<backbone>[_seed<k>].png` (`_colab_run1` và `_kaggle_run2` đánh dấu bản chạy lại cùng `exp_id`) |
| `predictions/` | `F01`/`F01uncal`/`T00` × seed 0–2: `*_test.csv` và `*_val.csv`; val của các lần huấn luyện khác; `colab_run1/`, `kaggle_run2/` cho các bản trùng `exp_id` |
| `eval_results/` | Đầu ra của `eval.py score` và `grade` (tên `eval_out/` bị `.gitignore` chặn) |
| `tables/` | Biểu đồ EDA, kiểm tra pipeline, ảnh đoán sai, ma trận nhầm lẫn, bảng suy luận và độ trễ |
| `run_logs/`, `run_logs_colab_run1/` | `config.json`, `history.csv`, `summary.json`, logit val của từng lần chạy (không có checkpoint) |
| `logs/` | Nhật ký chạy hai lượt, stdout Kaggle, kết quả unit test |
| `code/` | Toàn bộ mã nguồn và notebook; `eval.py` dùng bản gốc ở gốc repo (không sửa) |

Mức tái lập: cùng seed trên cùng một máy cho kết quả trùng (trên Kaggle, `B03` và `T00` seed 0 trùng khít); giữa Colab và Kaggle hoặc giữa hai lần chạy khác tiến trình, macro-F1 val chênh khoảng 0,002–0,012 (AMP và một số phép CUDA không tất định). Đây là lý do báo cáo chỉ kết luận khi chênh lệch lớn hơn nhiễu.
