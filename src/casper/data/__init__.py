"""
Data loading modules for CASPER.
"""

from casper.data.movielens_loader import MovieLensLoader
from casper.data.movielens_downloader import download_movielens_data

__all__ = ['MovieLensLoader', 'download_movielens_data']
