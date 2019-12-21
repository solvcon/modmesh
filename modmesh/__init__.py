"""
General mesh data definition and manipulation in one, two, and
three-dimensional space.
"""

from . import _modmesh


__all__ = [
    'StaticGrid1d',
    'StaticGrid2d',
    'StaticGrid3d',
]


StaticGrid1d = _modmesh.StaticGrid1d
StaticGrid2d = _modmesh.StaticGrid2d
StaticGrid3d = _modmesh.StaticGrid3d

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
