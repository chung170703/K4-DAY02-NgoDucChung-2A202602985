"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
TOTAL_IMAGES = 17509
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # trọng số timm mặc định của các backbone dùng trong lab
IMAGENET_STD = (0.229, 0.224, 0.225)

AUG_CHOICES = ("basic", "geo", "color", "trivial", "randaug")


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train/val/test_subset{fold}.csv (S1). Không sửa, không lọc, không chia lại."""
    labels_dir = Path(labels_dir)
    dfs = tuple(pd.read_csv(labels_dir / f"{s}_subset{fold}.csv") for s in ("train", "val", "test"))
    for s, df in zip(("train", "val", "test"), dfs):
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{s}_subset{fold}.csv thiếu cột {missing}")
    return dfs


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    Dừng ngay (AssertionError) nếu: giao các tập khác rỗng, hợp khác 17.509 ảnh, tỉ lệ lệch
    khỏi 60/20/20 quá 1 điểm phần trăm, hoặc thiếu file ảnh.
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: int(len(v)) for k, v in splits.items()}
    total = sum(n.values())
    per_class = {k: {int(c): int(m) for c, m in v["Label"].value_counts().sort_index().items()}
                 for k, v in splits.items()}
    names = {k: set(v["Filename"]) for k, v in splits.items()}
    for k, v in splits.items():
        assert len(names[k]) == len(v), f"{k}: có Filename trùng lặp trong cùng một tập"
    overlap = {"train∩val": len(names["train"] & names["val"]),
               "train∩test": len(names["train"] & names["test"]),
               "val∩test": len(names["val"] & names["test"])}
    union = len(names["train"] | names["val"] | names["test"])
    ratio = {k: n[k] / total for k in n}

    images_dir = Path(images_dir)
    missing = [f for f in sorted(names["train"] | names["val"] | names["test"])
               if not (images_dir / f).exists()]

    print(f"Số ảnh: {n} | tổng {total} | hợp (unique) {union}")
    print("Tỉ lệ:", {k: f"{r:.2%}" for k, r in ratio.items()})
    print("Giao:", overlap)
    table = pd.DataFrame(per_class).reindex(range(NUM_CLASSES)).fillna(0).astype(int)
    table.index = [f"{i} {CLASS_NAMES[i]}" for i in range(NUM_CLASSES)]
    print(table)
    print(f"File thiếu trong {images_dir}: {len(missing)}")

    assert all(v == 0 for v in overlap.values()), f"Các tập giao nhau: {overlap}"
    assert union == total == TOTAL_IMAGES, f"Hợp ba tập = {union}, tổng = {total}, kỳ vọng {TOTAL_IMAGES}"
    for k, target in zip(("train", "val", "test"), (0.6, 0.2, 0.2)):
        assert abs(ratio[k] - target) <= 0.01, f"Tỉ lệ {k} = {ratio[k]:.2%} lệch khỏi {target:.0%} quá 1 điểm %"
    assert not missing, f"Thiếu {len(missing)} file ảnh, ví dụ {missing[:3]}"
    return {"n": n, "per_class": per_class, "overlap": overlap, "union": union, "ratio": ratio}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform.

    aug (trục B của GUIDE.md mục 3), chỉ áp dụng cho train:
      basic   : RandomResizedCrop + lật ngang (công thức nền)
      geo     : basic + lật dọc + xoay bội 90° (ảnh cỏ chụp từ trên xuống, không có hướng "lên")
      color   : basic + ColorJitter
      trivial : basic + TrivialAugmentWide
      randaug : basic + RandAugment(2, 9)
    Mixup/CutMix trộn theo batch nên nằm ở losses.py.

    Val/test (không ngẫu nhiên): ảnh gốc 256x256.
      img_size <= 256: CenterCrop(img_size)    (mặc định: 256 -> crop 224)
      img_size  > 256: Resize(img_size)        (thí nghiệm độ phân giải kiểm tra, FixRes)
    """
    if aug not in AUG_CHOICES:
        raise ValueError(f"aug phải thuộc {AUG_CHOICES}, nhận {aug!r}")
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        if img_size <= 256:
            return T.Compose([T.CenterCrop(img_size), T.ToTensor(), normalize])
        return T.Compose([T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
                          T.ToTensor(), normalize])

    ops = [T.RandomResizedCrop(img_size, interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip()]
    if aug == "geo":
        ops += [T.RandomVerticalFlip(), T.RandomApply([T.RandomRotation((90, 90))], p=0.5)]
    elif aug == "color":
        ops += [T.ColorJitter(0.3, 0.3, 0.3, 0.05)]
    elif aug == "trivial":
        ops += [T.TrivialAugmentWide()]
    elif aug == "randaug":
        ops += [T.RandAugment(num_ops=2, magnitude=9)]
    return T.Compose(ops + [T.ToTensor(), normalize])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform, nhãn int, tên file str). Tên file cần để ghi
    predictions/*.csv đúng định dạng eval.py. Giữ đúng thứ tự dòng của `df`.
    cache=True nạp trước toàn bộ ảnh vào RAM dạng uint8 (~190 KB/ảnh) để khỏi đọc đĩa/giải mã JPEG.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, cache: bool = False):
        self.filenames = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform
        self._cache = None
        if cache:
            self._cache = np.stack([np.asarray(self._open(f)) for f in self.filenames])

    def _open(self, filename: str) -> Image.Image:
        with Image.open(self.images_dir / filename) as im:
            return im.convert("RGB")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        img = Image.fromarray(self._cache[i]) if self._cache is not None else self._open(self.filenames[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.filenames[i]


def _worker_init_fn(worker_id: int) -> None:
    # torch tự seed RNG của torch trong worker; ở đây seed thêm random và numpy từ cùng nguồn.
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, cache: bool = False):
    """Tạo DataLoader.

    train=True : shuffle (hoặc sampler "balanced"), drop_last=True để BatchNorm không gặp batch 1-2 ảnh.
    train=False: không shuffle, giữ đúng thứ tự df (để ghép logit với Filename).
    sampler="balanced": WeightedRandomSampler, trọng số 1/(số ảnh của lớp), lấy len(df) mẫu mỗi epoch.
    Bộ sinh ngẫu nhiên nằm ở `loader.generator`; train.py seed lại nó mỗi epoch để chạy tiếp được sau khi ngắt.
    """
    ds = DeepWeedsDataset(df, images_dir, transform, cache=cache)
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              worker_init_fn=_worker_init_fn, generator=g,
              persistent_workers=num_workers > 0)
    if not train:
        return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, **kw)
    if sampler is None:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    if sampler == "balanced":
        counts = df["Label"].value_counts()
        w = df["Label"].map(lambda c: 1.0 / counts[c]).to_numpy()
        ws = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(df),
                                   replacement=True, generator=g)
        return DataLoader(ds, batch_size=batch_size, sampler=ws, drop_last=True, **kw)
    raise ValueError(f"sampler phải là None hoặc 'balanced', nhận {sampler!r}")
