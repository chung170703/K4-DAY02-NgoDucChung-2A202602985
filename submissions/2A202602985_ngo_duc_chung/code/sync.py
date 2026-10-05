"""sync.py - gom kết quả từ Drive vào thư mục nộp bài rồi commit + push lên GitHub.

Trên Colab, `curves/`, `predictions/`, `ckpt/`, `logs/`, `eval_out/` trong thư mục nộp là SYMLINK sang Drive
(để kết quả không mất khi phiên bị ngắt). `git add` một symlink chỉ commit đường dẫn của nó, không phải ảnh,
nên hàm này tạm thay symlink bằng bản sao thật, commit, rồi khôi phục symlink.

Chỉ đẩy file nhỏ cần nộp: curves/*.png, predictions của mốc `T00` và chung kết `F*` (kèm *_val.csv, *uncal*),
tables/*, results.xlsx, report.md, README.md, code/. KHÔNG đẩy ckpt/, logs/, checkpoint, dữ liệu.
Lỗi mạng hoặc quyền push chỉ in cảnh báo, không làm hỏng lần huấn luyện đang chạy.
"""
from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
from pathlib import Path

PRED_GLOBS = ("T00_*", "F*")
SYNC_DIRS = ("curves", "predictions")


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def _copy_filtered(src: Path, dst: Path, name: str) -> int:
    """Chép src -> dst; với predictions chỉ chép các file khớp PRED_GLOBS."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("*")):
        if not f.is_file():
            continue
        if name == "predictions" and not any(fnmatch.fnmatch(f.name, g) for g in PRED_GLOBS):
            continue
        shutil.copy2(f, dst / f.name)
        n += 1
    return n


def sync_to_repo(repo_dir: str | Path, sub: str | Path, message: str = "Update results",
                 push: bool = True, branch: str = "main") -> dict:
    """Chép kết quả từ nơi symlink trỏ tới vào thư mục nộp, commit, (pull --rebase) và push.

    Trả về {"copied": {...}, "committed": bool, "pushed": bool, "error": str | None}.
    Luôn khôi phục symlink dù có lỗi.
    """
    repo, sub = Path(repo_dir), Path(sub)
    out = {"copied": {}, "committed": False, "pushed": False, "error": None}
    restored = []
    try:
        for name in SYNC_DIRS:
            target = sub / name
            if target.is_symlink():
                real = Path(os.path.realpath(target))
                target.unlink()                      # gỡ symlink, thay bằng thư mục thật
                restored.append((target, real))
                out["copied"][name] = _copy_filtered(real, target, name)
            elif target.is_dir():
                out["copied"][name] = len(list(target.glob("*")))
        paths = [str(sub / p) for p in (*SYNC_DIRS, "tables", "results.xlsx", "report.md", "README.md", "code")
                 if (sub / p).exists()]
        _git(repo, "add", "--", *paths)
        if _git(repo, "diff", "--cached", "--quiet", check=False).returncode != 0:
            _git(repo, "commit", "-m", message)
            out["committed"] = True
        if push:
            # --autostash: các thay đổi chưa stage (ví dụ .gitignore sửa trên Colab) không chặn pull
            pull = _git(repo, "pull", "--rebase", "--autostash", "origin", branch, check=False)
            if pull.returncode != 0:
                _git(repo, "rebase", "--abort", check=False)
                raise RuntimeError(f"pull --rebase lỗi: {pull.stderr.strip()[:300]}")
            # đẩy cả các commit còn tồn từ lần trước (push lỗi) dù lần này không có file mới
            ahead = _git(repo, "rev-list", "--count", f"origin/{branch}..HEAD", check=False).stdout.strip()
            if ahead and int(ahead) > 0:
                p = _git(repo, "push", "origin", f"HEAD:{branch}", check=False)
                if p.returncode != 0:
                    raise RuntimeError(f"push lỗi (token hết hạn/không đủ quyền?): {p.stderr.strip()[:300]}")
                out["pushed"] = True
    except Exception as e:  # không để lỗi git làm hỏng thí nghiệm
        out["error"] = str(e)
        print(f"[sync] CẢNH BÁO: {e}")
    finally:
        for target, real in restored:                # khôi phục symlink sang Drive
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            if not target.exists():
                os.symlink(real, target)
    msg = f"[sync] chép {out['copied']} | commit={out['committed']} push={out['pushed']}"
    print(msg)
    return out
