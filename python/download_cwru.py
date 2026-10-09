"""
下载 CWRU 数据集（在你自己的电脑上运行）
======================================
用法：python download_cwru.py
下载 40 个 .mat 文件到 data/cwru/，约 100 MB。
如果网络下载失败，也可以手动从官网下载，放到 data/cwru/ 目录下，文件名保持 “编号.mat”。
"""
import sys
import time
import urllib.request

from config import CWRU_DIR, CWRU_FILES, CWRU_URL


def main():
    CWRU_DIR.mkdir(parents=True, exist_ok=True)
    nums = sorted({n for loads in CWRU_FILES.values() for n in loads.values()})
    for i, num in enumerate(nums, 1):
        dst = CWRU_DIR / f"{num}.mat"
        if dst.exists() and dst.stat().st_size > 10_000:
            print(f"[{i}/{len(nums)}] 已存在 {dst.name}")
            continue
        url = CWRU_URL.format(num=num)
        for attempt in range(3):
            try:
                print(f"[{i}/{len(nums)}] 下载 {url}")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
                    f.write(r.read())
                break
            except Exception as e:  # noqa: BLE001
                print(f"  失败（{e}），重试…")
                time.sleep(2)
        else:
            print(f"  放弃 {num}.mat，请手动下载。")
    print("完成。")


if __name__ == "__main__":
    sys.exit(main())
