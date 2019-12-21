"""
General mesh data definition and manipulation in one, two, and
three-dimensional space.
"""

from . import _modmesh


__all__ = [
    'Grid1d',
    'Grid2d',
    'Grid3d',
]


Grid1d = _modmesh.Grid1d
Grid2d = _modmesh.Grid2d
Grid3d = _modmesh.Grid3d

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
