# Copyright (c) 2026, Chun-Shih Chang <austin20463@gmail.com>,
# Chun-Hsu Lai <as2266317@gmail.com>
# BSD 3-Clause License, see COPYING

"""
Download, extract and preprocess nasa flight dataset.
"""

import json
import ssl
import urllib.request
import zipfile

from pathlib import Path


class NasaDataset:
    """
    Helper for downloading, extracting NASA files.
    """

    def __init__(self):
        """
        Initialize download/load configuration.
        """

        self.url = "https://techport.nasa.gov/api/file/presignedUrl/380503"
        self.download_dir = Path.cwd() / ".cache" / "download"
        self.filename = "DDL-F1_Dataset-20201013.zip"

    def download(self):
        """
        Download zipped dataset from NASA presigned URL.

        :return: ``None``.
        :rtype: None
        """
        file_path = Path(self.download_dir / self.filename)
        if file_path.exists():
            print(f"{file_path} exists,skip download.")
            return

        response = urllib.request.urlopen(self.url)
        presigned_url = json.loads(response.read())["presignedUrl"]
        self.download_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            presigned_url,
            f"{self.download_dir}/{self.filename}",
            reporthook=self._download_hook,
        )

    def extract(self):
        """
        Extract downloaded dataset zip into ``download_dir``.

        :return: ``None``.
        :rtype: None
        """
        with zipfile.ZipFile(
            f"{self.download_dir}/{self.filename}",
            "r",
        ) as file:
            file.extractall(self.download_dir)

    def _download_hook(self, block_num, block_size, total_size):
        """
        Progress callback for ``urllib.request.urlretrieve``.

        :param block_num: Downloaded block count.
        :type block_num: int
        :param block_size: Block size in bytes.
        :type block_size: int
        :param total_size: Total file size in bytes.
        :type total_size: int
        :return: ``None``.
        :rtype: None
        """
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        ratio = min(downloaded / total_size, 1.0)
        width = 30
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\rDownloading [{bar}] {ratio:6.2%}", end="", flush=True)


def main():
    ssl._create_default_https_context = ssl._create_stdlib_context
    dataset = NasaDataset()
    dataset.download()
    dataset.extract()


if __name__ == "__main__":
    main()
