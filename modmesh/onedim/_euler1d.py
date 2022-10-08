# Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
One-dimensional Solver for the Euler Equations
"""

import numpy as np

try:
    from _modmesh import onedim as _impl  # noqa: F401
except ImportError:
    from .._modmesh import onedim as _impl  # noqa: F401


__all__ = [
    'Euler1DSolver',
]


class Euler1DSolver:
    def __init__(self, xmin, xmax, ncoord, time_increment=0.05):
        self._svr = self.init_solver(xmin, xmax, ncoord, time_increment)

    def __getattr__(self, name):
        """
        If something is not available in this Python object, fall back to the
        C++ implementation.
        """
        return getattr(self._svr, name)

    @staticmethod
    def init_solver(xmin, xmax, ncoord, time_increment):
        # Create the solver object.
        svr = _impl.Euler1DCore(ncoord=ncoord, time_increment=time_increment)

        # Initialize spatial grid.
        svr.coord[...] = np.linspace(xmin, xmax, num=ncoord)

        # Initialize field.
        svr.cfl.fill(0)
        svr.gamma.fill(1.4)  # Air.
        svr.so0[...] = 0.0
        svr.so1[...] = 0.0

        return svr

    @staticmethod
    def calc_u2(gamma, density, velocity, pressure):
        ie = 1. / (gamma - 1) * pressure / density
        ke = velocity * velocity / 2
        return density * (ie + ke)

    def init_sods_problem(self, density0, pressure0, density1, pressure1,
                          xdiaphragm=0.0, gamma=1.4):
        # Fill gamma.
        self._svr.gamma.fill(gamma)
        # Determine u0 and u2 value at left and right.
        u0_left = density0
        u0_right = density1
        u2_left = self.calc_u2(gamma, density0, 0.0, pressure0)
        u2_right = self.calc_u2(gamma, density1, 0.0, pressure1)
        # Create Boolean selection arrays for left and right.
        slct_left = self._svr.coord < xdiaphragm
        slct_right = np.logical_not(slct_left)
        # u0
        self._svr.so0[slct_left, 0] = u0_left
        self._svr.so0[slct_right, 0] = u0_right
        # u1
        self._svr.so0[:, 1] = 0.0
        # u2
        self._svr.so0[slct_left, 2] = u2_left
        self._svr.so0[slct_right, 2] = u2_right
        # Initialize derivative to zero.
        self._svr.so1.fill(0)
        # Setup the rest in the solver for time-marching.
        self._svr.setup_march()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
