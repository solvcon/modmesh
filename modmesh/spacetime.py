# Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
One-dimensional space-time CESE method implementation.
"""


try:
    from _modmesh import spacetime as _impl  # noqa: F401
except ImportError:
    from ._modmesh import spacetime as _impl  # noqa: F401

_toload = [
    'Grid',
    'Celm',
    'Selm',
    'Kernel',
    'Solver',
    'InviscidBurgersSolver',
    'LinearScalarSolver',
]


def _load():
    for name in _toload:  # noqa: F821
        globals()[name] = getattr(_impl, name)


__all__ = _toload + [
    'SolverProxy',
]


_load()
del _load
del _toload


class SolverProxy():

    def __init__(self, *args, **kw):

        self.svr = Solver(*args, **kw)  # noqa: F821
        self.svr.kernel.xp_calc = self._xp_calc
        self.svr.kernel.xn_calc = self._xn_calc
        self.svr.kernel.tp_calc = self._tp_calc
        self.svr.kernel.tn_calc = self._tn_calc
        self.svr.kernel.so0p_calc = self._so0p_calc
        self.svr.kernel.cfl_updater = self._cfl_updater

    def _xp_calc(self, se, iv):
        return 1.0

    def _xn_calc(self, se, iv):
        return 1.0

    def _tp_calc(self, se, iv):
        return 1.0

    def _tn_calc(self, se, iv):
        return 1.0

    def _so0p_calc(self, se, iv):
        return se.get_so(iv)

    def _cfl_updater(self, se):
        se.set_cfl(1.0)

    def __getattr__(self, name):
        return getattr(self.svr, name)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
