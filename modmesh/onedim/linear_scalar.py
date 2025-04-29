# Copyright (c) 2025, Ting-Yu Chuang <tychuang.cs10@nycu.edu.tw>
# BSD 3-Clause License, see COPYING


import numpy as np

try:
    from _modmesh import spacetime as _impl  # noqa: F401
except ImportError:
    from .._modmesh import spacetime as _impl  # noqa: F401

__all__ = [
    'LinearScalarSolver',
]


class LinearScalarSolver:
    """
    """

    def __init__(self, xmin, xmax, ncelm, cfl=1):
        self._core = self.init_solver(xmin, xmax, ncelm, cfl)

    def __getattr__(self, name):
        return getattr(self._core, name)

    @staticmethod
    def init_solver(xmin, xmax, ncelm, cfl=1):
        grid = _impl.Grid(xmin, xmax, ncelm)

        dx = (grid.xmax - grid.xmin) / grid.ncelm
        dt = dx * cfl

        # Create the solver object.
        svr = _impl.LinearScalarSolver(grid=grid,
                                       time_increment=dt)

        # Initialize
        for e in svr.selms(odd_plane=False):
            if e.xctr < 2 * np.pi or e.xctr > 2 * 2 * np.pi:
                v = 0
                dv = 0
            else:
                v = np.sin(e.xctr)
                dv = np.cos(e.xctr)
            e.set_so0(0, v)
            e.set_so1(0, dv)

        return svr

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
