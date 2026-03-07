# Copyright (c) 2026, Chun-Shih Chang <austin20463@gmail.com>
# BSD 3-Clause License, see COPYING


"""
Track nasa flight dataset with Kalman Filter.
"""

import argparse
import numpy as np

from .npkalmanfilter import KalmanFilter
from .dataset import NasaDataset
from .earth import Earth
from .attitude import attitude


def propagate_attitude(dcm_ecef_imu, imu_dangle_in_imu, dcm_con_imu):
    """
    Propagate IMU attitude and convert to container quaternion.

    :param dcm_ecef_imu: Current IMU DCM in ECEF frame.
    :type dcm_ecef_imu: numpy.ndarray
    :param imu_dangle_in_imu: IMU delta-angle vector in IMU frame.
    :type imu_dangle_in_imu: numpy.ndarray
    :param dcm_con_imu: Fixed DCM from container frame to IMU frame.
    :type dcm_con_imu: numpy.ndarray
    :return: Updated ``(dcm_ecef_imu, quat_con_ecef)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    dcm_delta_imu = attitude.dangle_to_dcm(imu_dangle_in_imu)
    dcm_ecef_imu = dcm_ecef_imu @ dcm_delta_imu
    dcm_ecef_con = dcm_ecef_imu @ dcm_con_imu.T
    quat_con_ecef = attitude.dcm_to_quat(dcm_ecef_con.T)
    return dcm_ecef_imu, quat_con_ecef


def compute_accel_ecef(
    dcm_ecef_imu,
    imu_dvel_in_imu,
    dt,
    con_pos_in_ecef,
    con_vel_in_ecef,
    use_earth_rotation,
):
    """
    Compute ECEF acceleration from IMU delta-velocity and compensation terms.

    :param dcm_ecef_imu: IMU DCM in ECEF frame.
    :type dcm_ecef_imu: numpy.ndarray
    :param imu_dvel_in_imu: IMU delta-velocity in IMU frame.
    :type imu_dvel_in_imu: numpy.ndarray
    :param dt: Time step in seconds.
    :type dt: float
    :param con_pos_in_ecef: Container position in ECEF.
    :type con_pos_in_ecef: numpy.ndarray
    :param con_vel_in_ecef: Container velocity in ECEF.
    :type con_vel_in_ecef: numpy.ndarray
    :param use_earth_rotation: Enable Coriolis/centrifugal compensation.
    :type use_earth_rotation: bool
    :return: Estimated ECEF acceleration.
    :rtype: numpy.ndarray
    """
    dv_ecef = dcm_ecef_imu @ imu_dvel_in_imu
    accel_ecef = dv_ecef / dt
    accel_ecef = accel_ecef + Earth.gravity_ecef(con_pos_in_ecef)
    if use_earth_rotation:
        accel_ecef = Earth.apply_earth_rotation_compensation(
            accel_ecef, con_vel_in_ecef, con_pos_in_ecef
        )
    return accel_ecef


def predict_translation(kf, accel_ecef, dt):
    """
    Predict translational states with constant-acceleration discrete model.

    :param kf: Kalman filter instance.
    :type kf: KalmanFilter
    :param accel_ecef: Input acceleration in ECEF frame.
    :type accel_ecef: numpy.ndarray
    :param dt: Time step in seconds.
    :type dt: float
    :return: ``None``.
    :rtype: None
    """
    kf.F = np.eye(10)
    kf.F[0:3, 3:6] = np.eye(3) * dt
    control_input_matrix = np.zeros((10, 3))
    control_input_matrix[0:3, 0:3] = 0.5 * dt ** 2 * np.eye(3)
    control_input_matrix[3:6, 0:3] = dt * np.eye(3)
    kf.predict(u=accel_ecef, B=control_input_matrix)


def quat_error_angle_deg(predict, ground_truth):
    """
    Compute absolute quaternion attitude error in degrees.

    :param predict: Predicted quaternion in ``[x, y, z, w]``.
    :type predict: numpy.ndarray
    :param ground_truth: Ground-truth quaternion in ``[x, y, z, w]``.
    :type ground_truth: numpy.ndarray
    :return: Attitude error angle in degrees.
    :rtype: float
    """
    dot = np.dot(
        predict / np.linalg.norm(predict),
        ground_truth / np.linalg.norm(ground_truth),
    )
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def run_kalman_filter(data, use_earth_rotation=True):
    """
    Run prediction-only Kalman propagation on the NASA flight dataset.

    :param data: Loaded dataset object.
    :type data: NasaDataset
    :param use_earth_rotation: Enable Earth rotation compensation terms.
    :type use_earth_rotation: bool
    :return: Final-state summary dictionary.
    :rtype: dict
    """
    time_ns = data.dlc.timestamp
    imu_dvel_in_imu = data.dlc.imu_dvel_in_imu
    imu_dangle_in_imu = data.dlc.imu_dangle_in_imu
    dcm_con_imu = data.dlc.dcm_con_imu
    gt_con_pos_in_ecef = data.ground_truth.gt_con_pos_in_ecef
    gt_con_vel_in_ecef = data.ground_truth.gt_con_vel_in_ecef
    gt_quat_con_ecef = data.ground_truth.gt_quat_con_ecef
    dcm_ecef_con = attitude.quat_to_dcm(gt_quat_con_ecef[0]).T
    dcm_ecef_imu = dcm_ecef_con @ dcm_con_imu

    kf = KalmanFilter(dim_x=10, dim_z=1)
    con_pos_in_ecef_0 = gt_con_pos_in_ecef[0]
    con_vel_in_ecef_0 = gt_con_vel_in_ecef[0]
    gt_quat_con_ecef_0 = gt_quat_con_ecef[0]
    quat_con_ecef = gt_quat_con_ecef_0.copy()
    kf.x = np.hstack(
        [con_pos_in_ecef_0, con_vel_in_ecef_0, gt_quat_con_ecef_0]
    )

    for i in range(1, len(time_ns)):
        dt = (time_ns[i] - time_ns[i - 1]) / 1e9
        dcm_ecef_imu, quat_con_ecef = propagate_attitude(
            dcm_ecef_imu, imu_dangle_in_imu[i], dcm_con_imu
        )

        con_pos_in_ecef = kf.x[0:3]
        con_vel_in_ecef = kf.x[3:6]
        accel_ecef = compute_accel_ecef(
            dcm_ecef_imu,
            imu_dvel_in_imu[i],
            dt,
            con_pos_in_ecef,
            con_vel_in_ecef,
            use_earth_rotation
        )
        predict_translation(kf, accel_ecef, dt)
        kf.x[6:10] = quat_con_ecef

    con_pos_in_ecef_n1 = kf.x[0:3]
    gt_con_pos_in_ecef_n1 = gt_con_pos_in_ecef[-1]
    quat_con_ecef_n1 = kf.x[6:10].copy()
    gt_quat_con_ecef_n1 = gt_quat_con_ecef[-1]
    attitude_error_deg_n1 = quat_error_angle_deg(
        quat_con_ecef_n1,
        gt_quat_con_ecef_n1,
    )

    return {
        "final_con_pos_in_ecef": con_pos_in_ecef_n1,
        "final_gt_con_pos_in_ecef": gt_con_pos_in_ecef_n1,
        "final_quat_con_ecef": quat_con_ecef_n1,
        "final_gt_quat_con_ecef": gt_quat_con_ecef_n1,
        "final_attitude_error_deg": attitude_error_deg_n1,
    }


def main():
    """
    Entry point for running the track flight prediction.

    :return: ``None``.
    :rtype: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-earth-rotation", action="store_true")
    args = parser.parse_args()

    data = NasaDataset()
    data.load()
    result = run_kalman_filter(
        data, use_earth_rotation=not args.no_earth_rotation
    )

    print("Kalman Filter Prediction Result:")
    print(f"final_con_pos_in_ecef = {result['final_con_pos_in_ecef']}")
    print(f"final_gt_con_pos_in_ecef = {result['final_gt_con_pos_in_ecef']}")
    final_pos_error_ecef = (
        result["final_con_pos_in_ecef"] - result["final_gt_con_pos_in_ecef"]
    )
    print(f"final_pos_error_ecef = {final_pos_error_ecef}")
    print(f"final_error_m = {np.linalg.norm(final_pos_error_ecef)}")

    print(f"final_quat_con_ecef = {result['final_quat_con_ecef']}")
    print(f"final_gt_quat_con_ecef = {result['final_gt_quat_con_ecef']}")
    print(f"final_attitude_error_deg = {result['final_attitude_error_deg']}")


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
