"""download_ml20m.py -- fetch the MovieLens-20M dataset into data/ml-20m/ (project rule: all data under data/).

Idempotent: skips the download if ratings.csv already present; skips the unzip if the extracted files
already exist. ~190 MB zip from grouplens.org. The zip extracts to a top-level `ml-20m/` folder; we
flatten it so ratings.csv lands at data/ml-20m/ratings.csv (what liang_split.py expects).

Usage:  python scripts/baselines/download_ml20m.py
"""
import os
import sys
import shutil
import zipfile
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data", "ml-20m")
URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
ZIP_PATH = os.path.join(DATA_DIR, "ml-20m.zip")
RATINGS = os.path.join(DATA_DIR, "ratings.csv")


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(100.0, count * block_size * 100.0 / total_size)
    sys.stdout.write(f"\r[download] {pct:5.1f}%  ({count*block_size/1e6:6.1f} / {total_size/1e6:.1f} MB)")
    sys.stdout.flush()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RATINGS):
        print(f"[download] ratings.csv already present at {RATINGS}; nothing to do.")
        return
    if not os.path.exists(ZIP_PATH):
        print(f"[download] fetching {URL}")
        urllib.request.urlretrieve(URL, ZIP_PATH, _progress)
        print()
    else:
        print(f"[download] zip already present: {ZIP_PATH}")

    print("[download] extracting ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DATA_DIR)

    # zip extracts to data/ml-20m/ml-20m/*  -> flatten into data/ml-20m/
    nested = os.path.join(DATA_DIR, "ml-20m")
    if os.path.isdir(nested):
        for fn in os.listdir(nested):
            src = os.path.join(nested, fn)
            dst = os.path.join(DATA_DIR, fn)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        try:
            os.rmdir(nested)
        except OSError:
            pass

    if os.path.exists(RATINGS):
        n_bytes = os.path.getsize(RATINGS)
        print(f"[download] DONE. ratings.csv = {n_bytes/1e6:.1f} MB at {RATINGS}")
    else:
        raise SystemExit("[download] ERROR: ratings.csv not found after extraction")


if __name__ == "__main__":
    main()
