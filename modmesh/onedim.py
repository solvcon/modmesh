# Copyright (c) 2022, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
One-dimensional space-time CESE method implementation.
"""


try:
    from _modmesh import onedim as _impl  # noqa: F401
except ImportError:
    from ._modmesh import onedim as _impl  # noqa: F401

_toload = [
    'Euler1DSolver',
]


def _load():
    for name in _toload:  # noqa: F821
        globals()[name] = getattr(_impl, name)


__all__ = _toload + [
]


_load()
del _load
del _toload

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
