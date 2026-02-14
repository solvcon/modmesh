# Copyright (c) 2026, Chun-Shih Chang <austin20463@gmail.com>
# BSD 3-Clause License, see COPYING

"""
Download, extract and preprocess nasa flight dataset.
"""

import json
import numpy as np
import os
import ssl
import urllib.request
import zipfile


class Base:
    """
    Shared helpers for dataset CSV loading and interpolation.
    """

    @staticmethod
    def load_text(path):
        """
        Load a CSV file as a float64 NumPy matrix.

        :param path: CSV file path.
        :type path: str
        :return: Matrix with shape ``(N, M)``.
        :rtype: numpy.ndarray
        """
        raw = np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding=None,
        )
        return raw.view((np.float64, len(raw.dtype.names)))

    @staticmethod
    def interp_linear(t_tgt, t_src, y_src):
        """
        Linearly interpolate source samples to target timestamps.

        :param t_tgt: Target timestamps.
        :type t_tgt: numpy.ndarray
        :param t_src: Source timestamps.
        :type t_src: numpy.ndarray
        :param y_src: Source samples with shape ``(N, C)``.
        :type y_src: numpy.ndarray
        :return: Interpolated samples with shape ``(len(t_tgt), C)``.
        :rtype: numpy.ndarray
        """
        out = np.empty((len(t_tgt), y_src.shape[1]), dtype=np.float64)
        for i in range(y_src.shape[1]):
            out[:, i] = np.interp(t_tgt, t_src, y_src[:, i])
        return out


class DLC(Base):
    """
    DLC IMU dataset loader and timestamp aligner.
    """

    def __init__(self, root):
        """
        Initialize DLC dataset reader.

        :param root: Root directory that contains extracted NASA files.
        :type root: str
        """
        self.path = "Flight1_Catered_Dataset-20201013/Data/dlc.csv"
        self.root = root
        self.imu_pos_in_con = np.array([-0.08035, 0.28390, -1.42333])
        self.dcm_con_imu = np.array([
            [-0.2477, -0.1673,  0.9543],
            [-0.0478,  0.9859,  0.1604],
            [-0.9677, -0.0059, -0.2522],
        ])

    def load(self):
        """
        Load DLC IMU delta-velocity and delta-angle samples from CSV.

        :return: ``None``.
        :rtype: None
        """
        raw_data = Base.load_text(f"{self.root}/{self.path}")
        self.timestamp = raw_data[:, 0]
        self.imu_dvel_in_imu = raw_data[:, 1:4]
        self.imu_dangle_in_imu = raw_data[:, 4:7]

    def alignment(self, target_timestamp):
        """
        Align IMU increments to target timestamps by interpolation.

        The method integrates raw increments to cumulative signals,
        interpolates those cumulative signals to target timestamps, and
        differences them back
        to per-step increments.

        :param target_timestamp: Target timestamps.
        :type target_timestamp: numpy.ndarray
        :return: ``None``.
        :rtype: None
        """
        imu_velocity = np.cumsum(self.imu_dvel_in_imu.copy(), axis=0)
        imu_angle = np.cumsum(self.imu_dangle_in_imu.copy(), axis=0)
        imu_velocity = Base.interp_linear(
            target_timestamp,
            self.timestamp,
            imu_velocity,
        )
        imu_angle = Base.interp_linear(
            target_timestamp,
            self.timestamp,
            imu_angle,
        )

        aligned_diff_velocity = np.empty(
            (len(target_timestamp), 3),
            dtype=np.float64,
        )
        aligned_diff_angle = np.empty(
            (len(target_timestamp), 3),
            dtype=np.float64,
        )

        aligned_diff_velocity[0, :] = 0.0
        aligned_diff_angle[0, :] = 0.0
        aligned_diff_velocity[1:, :] = (
            imu_velocity[1:, :] - imu_velocity[:-1, :]
        )
        aligned_diff_angle[1:, :] = imu_angle[1:, :] - imu_angle[:-1, :]

        self.timestamp = target_timestamp
        self.imu_dvel_in_imu = aligned_diff_velocity
        self.imu_dangle_in_imu = aligned_diff_angle


class GroundTruth(Base):
    """
    Ground-truth state dataset loader.
    """

    def __init__(self, root):
        """
        Initialize ground-truth dataset reader.

        :param root: Root directory that contains extracted NASA files.
        :type root: str
        """
        self.path = "Flight1_Catered_Dataset-20201013/Data/truth.csv"
        self.root = root

    def load(self):
        """
        Load ground-truth position, velocity, and quaternion from CSV.

        :return: ``None``.
        :rtype: None
        """
        raw_data = Base.load_text(f"{self.root}/{self.path}")
        self.timestamp = raw_data[:, 0]
        self.gt_con_pos_in_ecef = raw_data[:, 1:4]
        self.gt_con_vel_in_ecef = raw_data[:, 4:7]
        self.gt_quat_con_ecef = raw_data[:, 7:11]


class NasaDataset:
    """
    Helper for downloading, extracting, and loading NASA files.
    """

    def __init__(self):
        """
        Initialize download/load configuration.
        """

        self.url = "https://techport.nasa.gov/api/file/presignedUrl/380503"
        self.download_dir = "./download"
        self.filename = "DDL-F1_Dataset-20201013.zip"

    def download(self):
        """
        Download zipped dataset from NASA presigned URL.

        :return: ``None``.
        :rtype: None
        """
        response = urllib.request.urlopen(self.url)
        presigned_url = json.loads(response.read())["presignedUrl"]
        os.makedirs(self.download_dir, exist_ok=True)
        urllib.request.urlretrieve(
            presigned_url,
            f"{self.download_dir}/{self.filename}",
            reporthook=self._download_hook,
        )
        print()

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

    def load(self):
        """
        Load DLC and ground-truth datasets, then align timestamps.

        :return: ``None``.
        :rtype: None
        """
        self.dlc = DLC(self.download_dir)
        self.dlc.load()
        self.ground_truth = GroundTruth(self.download_dir)
        self.ground_truth.load()
        self.alignment()

    def alignment(self):
        """
        Align DLC timestamps to ground-truth timestamps.

        :return: ``None``.
        :rtype: None
        """
        self.dlc.alignment(self.ground_truth.timestamp)

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
    """
    Download and extract the NASA flight dataset.

    :return: ``None``.
    :rtype: None
    """
    ssl._create_default_https_context = ssl._create_stdlib_context
    dataset = NasaDataset()
    dataset.download()
    dataset.extract()


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
