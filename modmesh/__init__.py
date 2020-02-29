# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
# BSD-style license; see COPYING

"""
General mesh data definition and manipulation in one, two, and
three-dimensional space.
"""


__all__ = [
    'StopWatch',
    'stop_watch',
    'TimeRegistry',
    'time_registry',
    'ConcreteBuffer',
    'SimpleArrayInt8',
    'SimpleArrayInt16',
    'SimpleArrayInt32',
    'SimpleArrayInt64',
    'SimpleArrayUint8',
    'SimpleArrayUint16',
    'SimpleArrayUint32',
    'SimpleArrayUint64',
    'SimpleArrayFloat32',
    'SimpleArrayFloat64',
    'StaticGrid1d',
    'StaticGrid2d',
    'StaticGrid3d',
]


# A hidden loophole to impolementation; it should only be used for testing
# during development.
from . import _modmesh as _impl

from ._modmesh import (
    StopWatch,
    stop_watch,
    TimeRegistry,
    time_registry,
    ConcreteBuffer,
    SimpleArrayInt8,
    SimpleArrayInt16,
    SimpleArrayInt32,
    SimpleArrayInt64,
    SimpleArrayUint8,
    SimpleArrayUint16,
    SimpleArrayUint32,
    SimpleArrayUint64,
    SimpleArrayFloat32,
    SimpleArrayFloat64,
    StaticGrid1d,
    StaticGrid2d,
    StaticGrid3d,
)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
