"""Download the public He et al. Reddit-Movie raw corpus from Hugging Face.

Dataset: https://huggingface.co/datasets/ZhankuiHe/reddit_movie_raw  (public, ungated)

HARD CONSTRAINT: the Reddit API is never touched. This script only fetches the
third-party-released dataset from the Hugging Face CDN. No rehydration, ever.

Usage (from C:/dev/phd/casper):
    poetry run python scripts/reddit/download_he.py --list
    poetry run python scripts/reddit/download_he.py            # download everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOS = {
    # raw thread trees (5 subreddits) + the IMDb gazetteer used by their matcher
    "raw": "ZhankuiHe/reddit_movie_raw",
    # their own "V1 processed" release: this is where the machine-matched movie
    # mentions actually live (the raw release ships NO entity annotations).
    "v1_large": "ZhankuiHe/reddit_movie_large_v1",
    "v1_small": "ZhankuiHe/reddit_movie_small_v1",
}
ROOT = Path(__file__).resolve().parents[2] / "data" / "reddit_he"
SUBDIR = {"raw": ".", "v1_large": "large_v1", "v1_small": "small_v1"}


def list_files(repo_id: str) -> list[tuple[str, int | None]]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    return sorted((s.rfilename, s.size) for s in info.siblings)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="only list repo files + sizes")
    ap.add_argument("--which", default="raw", choices=sorted(REPOS))
    ap.add_argument("--dest", default=None)
    args = ap.parse_args()

    repo_id = REPOS[args.which]
    DEST = ROOT if SUBDIR[args.which] == "." else ROOT / SUBDIR[args.which]
    if args.dest is None:
        args.dest = str(DEST)

    files = list_files(repo_id)
    total = sum(s or 0 for _, s in files)
    for name, size in files:
        print(f"{(size or 0)/1e6:12.2f} MB  {name}")
    print(f"{'-'*40}\n{total/1e9:.3f} GB across {len(files)} files")
    if args.list:
        return 0

    from huggingface_hub import snapshot_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        max_workers=4,
    )
    print(f"downloaded to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
