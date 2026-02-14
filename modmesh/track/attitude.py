# Copyright (c) 2026, Chun-Shih Chang <austin20463@gmail.com>
# BSD 3-Clause License, see COPYING

"""
Attitude representation and conversion.
"""

import numpy as np


class attitude:
    """
    Static utilities for attitude conversions.
    """

    @staticmethod
    def skew(v):
        """
        Return the skew-symmetric matrix of a 3D vector.
        For a vector ``v``, this function returns matrix ``[v]_x`` such that
        ``[v]_x @ a == np.cross(v, a)``.

        :param v: 3D vector.
        :type v: numpy.ndarray
        :return: 3x3 skew-symmetric matrix.
        :rtype: numpy.ndarray
        """
        return np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ]
        )

    @staticmethod
    def dangle_to_dcm(dtheta):
        """
        Convert delta-angle vector to direction cosine matrix (DCM).
        Uses Rodrigues' rotation formula. For very small angles, a first-order
        approximation is used to avoid division by zero.
        Reference1:
        https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
        Reference2:
        reference: https://arxiv.org/abs/1312.0788

        :param dtheta: Rotation vector in radians (axis * angle),
            shape ``(3,)``.
        :type dtheta: numpy.ndarray
        :return: Rotation matrix, shape ``(3, 3)``.
        :rtype: numpy.ndarray
        """
        angle = np.linalg.norm(dtheta)
        if angle < 1e-12:
            return np.eye(3) + attitude.skew(dtheta)

        k = dtheta / angle
        kx = attitude.skew(k)
        c = np.cos(angle)
        s = np.sin(angle)
        return c * np.eye(3) + (1.0 - c) * np.outer(k, k) + s * kx

    @staticmethod
    def quat_to_dcm(q):
        """
        Convert quaternion to direction cosine matrix (DCM).
        Reference:
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/rotation.html#Formation%20of%20a%20rotation%20matrix%20from%20a%20quaternion

        :param q: Quaternion in ``[x, y, z, w]`` order.
        :type q: numpy.ndarray
        :return: Rotation matrix, shape ``(3, 3)``.
        :rtype: numpy.ndarray
        """
        x, y, z, w = q
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ]
        )

    @staticmethod
    def dcm_to_quat(dcm):
        """
        Convert direction cosine matrix (DCM) to quaternion.
        Reference:
        https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        The returned quaternion is normalized and follows ``[x, y, z, w]``
        order (scalar-last).

        :param dcm: Rotation matrix, shape ``(3, 3)``.
        :type dcm: numpy.ndarray
        :return: Unit quaternion in ``[x, y, z, w]`` order.
        :rtype: numpy.ndarray
        """
        m = np.asarray(dcm, dtype=np.float64)
        trace = np.trace(m)

        if trace > 0.0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

        q = np.array([x, y, z, w], dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / q_norm

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
