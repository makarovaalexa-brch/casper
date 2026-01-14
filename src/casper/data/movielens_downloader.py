"""
MovieLens Dataset Downloader

Simple utility to download MovieLens 25M dataset.
Can be used from notebooks or scripts.
"""

import urllib.request
import zipfile
from pathlib import Path


def download_movielens_data(data_dir: str = "data/movielens") -> bool:
    """
    Download MovieLens 25M dataset.

    Args:
        data_dir: Directory to save the dataset (default: data/movielens)

    Returns:
        True if successful, False otherwise
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    zip_path = data_path / "ml-25m.zip"

    # Check if already downloaded
    if (data_path / "ratings.csv").exists() and (data_path / "movies.csv").exists():
        print("MovieLens 25M dataset already exists")
        return True

    try:
        if not zip_path.exists():
            print("Downloading MovieLens 25M dataset (~250MB)...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete")

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)

        print("MovieLens 25M dataset ready!")
        return True

    except Exception as e:
        print(f"Error downloading MovieLens data: {e}")
        return False
