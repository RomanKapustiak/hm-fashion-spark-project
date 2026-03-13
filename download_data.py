"""
Download H&M dataset files from Kaggle competition.

Downloads only:
    - articles.csv
    - customers.csv
    - transactions_train.csv

Images are NOT downloaded.

Prerequisites:
    1. pip install kaggle
    2. Place your Kaggle API token at ~/.kaggle/kaggle.json
       (generate it at: https://www.kaggle.com/settings → API → Create New Token)
    3. Accept the competition rules at:
       https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules

Usage:
    python download_data.py
"""

import zipfile
from pathlib import Path

from kaggle import KaggleApi

COMPETITION: str = "h-and-m-personalized-fashion-recommendations"
FILES: list[str] = [
    "articles.csv",
    "customers.csv",
    "transactions_train.csv",
]
DEST: Path = Path("data/raw")


def download_file(api: KaggleApi, file_name: str, dest: Path) -> None:
    zip_path = dest / f"{file_name}.zip"

    print(f"[1/3] Downloading {file_name} ...")
    api.competition_download_file(
        competition=COMPETITION,
        file_name=file_name,
        path=str(dest),
        quiet=False,
        force=False,
    )

    # The API saves the file as <file_name>.zip
    if zip_path.exists():
        print(f"[2/3] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink()
        print(f"[3/3] Done → {dest / file_name}\n")
    elif (dest / file_name).exists():
        # Some API versions save the file directly without zipping
        print(f"[3/3] Done (no zip) → {dest / file_name}\n")
    else:
        raise FileNotFoundError(
            f"Expected '{zip_path}' or '{dest / file_name}' after download, but neither exists."
        )


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    print(f"Authenticated with Kaggle API.\n")

    for file_name in FILES:
        download_file(api, file_name, DEST)

    print("All files downloaded successfully:")
    for file_name in FILES:
        csv_path = DEST / file_name
        size_mb = csv_path.stat().st_size / (1024 ** 2)
        print(f"  {csv_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
