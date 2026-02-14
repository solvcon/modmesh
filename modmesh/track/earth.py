# Copyright (c) 2026, Chun-Shih Chang <austin20463@gmail.com>
# BSD 3-Clause License, see COPYING

"""
Earth-related calculations, such as gravity, Coriolis force and
centrifugal force.
"""

import numpy as np


class Earth:
    """
    Static utilities for Earth constants and inertial-frame compensation.
    Reference1: https://earth-info.nga.mil/?action=wgs84&dir=wgs84
    Reference2:
    https://physics.stackexchange.com/questions/835050/equations-of-motion-in-earth-centered-earth-fixed-coordinates
    """

    OMEGA_EARTH = 7.292115e-5
    MU_EARTH = 3.986004418e14

    @staticmethod
    def earth_rotation_rate_ecef():
        """
        Return Earth rotation rate vector in ECEF frame.

        :return: Angular velocity vector ``[0, 0, omega]`` in rad/s.
        :rtype: numpy.ndarray
        """
        return np.array([0.0, 0.0, Earth.OMEGA_EARTH], dtype=np.float64)

    @staticmethod
    def gravity_ecef(pos_ecef):
        """
        Compute central gravity acceleration in ECEF frame.

        :param pos_ecef: Position in ECEF coordinates, shape ``(3,)``.
        :type pos_ecef: numpy.ndarray
        :return: Gravity acceleration in m/s^2, shape ``(3,)``.
        :rtype: numpy.ndarray
        """
        r = np.linalg.norm(pos_ecef)
        return -Earth.MU_EARTH / (r ** 3) * pos_ecef

    @staticmethod
    def coriolis_force_ecef(vel_ecef):
        """
        Compute Coriolis acceleration term in ECEF frame.

        :param vel_ecef: Velocity in ECEF coordinates, shape ``(3,)``.
        :type vel_ecef: numpy.ndarray
        :return: Coriolis acceleration in m/s^2.
        :rtype: numpy.ndarray
        """
        omega = Earth.earth_rotation_rate_ecef()
        return -2.0 * np.cross(omega, vel_ecef)

    @staticmethod
    def centrifugal_force_accel(pos_ecef):
        """
        Compute centrifugal acceleration term in ECEF frame.

        :param pos_ecef: Position in ECEF coordinates, shape ``(3,)``.
        :type pos_ecef: numpy.ndarray
        :return: Centrifugal acceleration in m/s^2.
        :rtype: numpy.ndarray
        """
        omega = Earth.earth_rotation_rate_ecef()
        return -np.cross(omega, np.cross(omega, pos_ecef))

    @staticmethod
    def apply_earth_rotation_compensation(accel_ecef, vel_ecef, pos_ecef):
        """
        Add Coriolis and centrifugal terms to acceleration.

        :param accel_ecef: Input acceleration in ECEF frame.
        :type accel_ecef: numpy.ndarray
        :param vel_ecef: Velocity in ECEF frame.
        :type vel_ecef: numpy.ndarray
        :param pos_ecef: Position in ECEF frame.
        :type pos_ecef: numpy.ndarray
        :return: Compensated acceleration in ECEF frame.
        :rtype: numpy.ndarray
        """
        return (
            accel_ecef
            + Earth.coriolis_force_ecef(vel_ecef)
            + Earth.centrifugal_force_accel(pos_ecef)
        )

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
