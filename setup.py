#!/usr/bin/env python3
"""
Project setup script for CASPER

This script helps initialize the project after cloning:
1. Creates necessary directories
2. Sets up API keys
3. Installs dependencies
4. Validates environment
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path


def check_poetry_installed():
    """Check if Poetry is installed"""
    try:
        result = subprocess.run(['poetry', '--version'],
                              capture_output=True, text=True)
        print(f"Poetry found: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("Poetry not found. Please install Poetry first:")
        print("   https://python-poetry.org/docs/#installation")
        return False


def install_dependencies():
    """Install project dependencies using Poetry"""
    print("Installing dependencies...")
    try:
        subprocess.run(['poetry', 'install'], check=True)
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies")
        return False


def setup_environment():
    """Set up environment variables"""
    env_file = Path('.env')
    env_example = Path('.env.example')

    if not env_file.exists():
        if env_example.exists():
            # Copy example file
            with open(env_example, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print("Created .env file from template")
            print("Please edit .env with your API keys")
        else:
            print("Error: .env.example not found")
            return False
    else:
        print(".env file already exists")

    return True


def validate_project_structure():
    """Ensure all necessary directories exist"""
    directories = [
        'data/movielens',
        'data/reddit',
        'data/processed',
        'experiments/checkpoints',
        'experiments/logs',
        'notebooks',
        'tests'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("Project structure validated")


def setup_git_hooks():
    """Set up pre-commit hooks"""
    try:
        subprocess.run(['poetry', 'run', 'pre-commit', 'install'],
                      check=True)
        print("Git hooks installed")
    except subprocess.CalledProcessError:
        print("Could not install git hooks (optional)")


def download_movielens_data(data_dir: str = "data/movielens"):
    """
    Download MovieLens 25M dataset.

    Args:
        data_dir: Directory to save the dataset (default: data/movielens)

    Returns:
        True if successful, False otherwise
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # MovieLens 25M dataset URL
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


def main():
    """Main setup function"""
    print("Setting up CASPER project...\n")

    # Check prerequisites
    if not check_poetry_installed():
        return False

    # Validate structure
    validate_project_structure()

    # Install dependencies
    if not install_dependencies():
        return False

    # Setup environment
    if not setup_environment():
        return False

    # Download MovieLens dataset
    print("\nDownloading MovieLens dataset...")
    if not download_movielens_data():
        print("Warning: MovieLens dataset not downloaded")
        print("You can download it manually later")

    # Optional: setup git hooks
    setup_git_hooks()

    print("\nProject setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: poetry shell")
    print("3. Run: poetry run jupyter notebook")
    print("4. Start development!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
