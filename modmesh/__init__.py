"""
General mesh data definition and manipulation in one, two, and
three-dimensional space.
"""

from . import _modmesh


__all__ = [
    'GridD1',
    'GridD2',
    'GridD3',
]


GridD1 = _modmesh.GridD1
GridD2 = _modmesh.GridD2
GridD3 = _modmesh.GridD3

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
