# Copyright (c) 2026, Chun-Shih Chang <austin20463@gmail.com>
# BSD 3-Clause License, see COPYING


"""
Kalman Filter implementation in numpy.
"""


import numpy as np


class KalmanFilter:
    """
    Minimal linear Kalman filter implementation using NumPy.
    """

    def __init__(self, dim_x, dim_z):
        """
        Initialize filter dimensions and default matrices.

        :param dim_x: State dimension.
        :type dim_x: int
        :param dim_z: Measurement dimension.
        :type dim_z: int
        :raises ValueError: If any dimension is non-positive.
        """
        if dim_x <= 0 or dim_z <= 0:
            raise ValueError("dim_x and dim_z must be positive")

        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros(dim_x, dtype=np.float64)
        self.F = np.eye(dim_x, dtype=np.float64)
        self.P = np.eye(dim_x, dtype=np.float64)
        self.Q = np.eye(dim_x, dtype=np.float64)
        self.H = np.zeros((dim_z, dim_x), dtype=np.float64)
        self.R = np.eye(dim_z, dtype=np.float64)

    def predict(self, u=None, B=None):
        """
        Perform linear prediction step.

        Implements:
        ``x = F x + B u`` and ``P = F P F^T + Q``.

        :param u: Optional control input.
        :type u: numpy.ndarray or None
        :param B: Optional control-input matrix.
        :type B: numpy.ndarray or None
        :return: Predicted state and covariance.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        :raises ValueError: If ``u`` is provided but ``B`` is ``None``.
        """
        self.x = self.F @ self.x
        if u is not None:
            if B is None:
                raise ValueError("B is required when u is provided")
            self.x = self.x + B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z, H=None, R=None):
        """
        Perform linear measurement update step.

        :param z: Measurement vector. If ``None``, update is skipped.
        :type z: numpy.ndarray or None
        :param H: Optional measurement matrix overriding ``self.H``.
        :type H: numpy.ndarray or None
        :param R: Optional measurement covariance overriding ``self.R``.
        :type R: numpy.ndarray or None
        :return: Updated state and covariance.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        if z is None:
            return self.x, self.P

        z = np.asarray(z, dtype=np.float64).reshape(-1)
        h = self.H if H is None else np.asarray(H, dtype=np.float64)
        r = self.R if R is None else np.asarray(R, dtype=np.float64)

        y = z - h @ self.x
        s = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.inv(s)

        self.x = self.x + k @ y

        identity = np.eye(self.dim_x, dtype=np.float64)
        i_kh = identity - k @ h
        self.P = i_kh @ self.P @ i_kh.T + k @ r @ k.T

        return self.x, self.P

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
