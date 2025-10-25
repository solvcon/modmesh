# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


"""
General mesh data definition and manipulation in one, two, and
three-dimensional space.
"""


# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html

import os

from . import pylibmgr

# buffer directory symbols
list_of_buffer = [
    'ConcreteBuffer',
    'BufferExpander',
    'SimpleArray',
    'SimpleArrayBool',
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
    'SimpleArrayComplex64',
    'SimpleArrayComplex128',
    'SimpleCollectorBool',
    'SimpleCollectorInt8',
    'SimpleCollectorInt16',
    'SimpleCollectorInt32',
    'SimpleCollectorInt64',
    'SimpleCollectorUint8',
    'SimpleCollectorUint16',
    'SimpleCollectorUint32',
    'SimpleCollectorUint64',
    'SimpleCollectorFloat32',
    'SimpleCollectorFloat64',
]

# inout directory symbols
list_of_inout = [
    'Gmsh',
    'Plot3d',
]

# math directory symbols
list_of_math = [
    'complex64',
    'complex128',
]

# mesh directory symbols
list_of_mesh = [
    'StaticGrid1d',
    'StaticGrid2d',
    'StaticGrid3d',
    'StaticMesh',
]

# multidim directory symbols
list_of_multidim = [
    'EulerCore',
]

# python directory symbols
list_of_python = [
    'CommandLineInfo',
    'ProcessInfo',
    'HAS_PILOT',
]

# testhelper directory symbols
list_of_testhelper = [
    'testhelper',
]

# toggle directory symbols
list_of_toggle = [
    'WrapperProfilerStatus',
    'wrapper_profiler_status',
    'StopWatch',
    'stop_watch',
    'TimeRegistry',
    'time_registry',
    'CallProfiler',
    'call_profiler',
    'CallProfilerProbe',
    'HierarchicalToggleAccess',
    'Toggle',
    'METAL_BUILT',
    'metal_running',
]

# transform directory symbols
list_of_transform = [
    'FourierTransform',
]

# linalg directory symbols
list_of_linalg = [
    'llt_factorization',
    'llt_solve',
]

# universe directory symbols
list_of_universe = [
    'calc_bernstein_polynomial',
    'interpolate_bernstein',
    'Point3dFp32',
    'Point3dFp64',
    'Segment3dFp32',
    'Segment3dFp64',
    'Bezier3dFp32',
    'Bezier3dFp64',
    'PointPadFp32',
    'PointPadFp64',
    'SegmentPadFp32',
    'SegmentPadFp64',
    'CurvePadFp32',
    'CurvePadFp64',
    'WorldFp32',
    'WorldFp64',
]

list_of_svg = [
    'svg',
]

__all__ = (  # noqa: F822
    list_of_buffer +
    list_of_inout +
    list_of_math +
    list_of_mesh +
    list_of_multidim +
    list_of_python +
    list_of_testhelper +
    list_of_toggle +
    list_of_transform +
    list_of_linalg +
    list_of_universe +
    list_of_svg
)


# A hidden loophole to impolementation; it should only be used for testing
# during development.
try:
    import _modmesh as _impl  # noqa: F401
except ImportError:
    from . import _modmesh as _impl  # noqa: F401


def _load(symbol_list):
    for name in symbol_list:
        globals()[name] = getattr(_impl, name)


_load(list_of_buffer)
_load(list_of_inout)
_load(list_of_math)
_load(list_of_mesh)
_load(list_of_multidim)
_load(list_of_python)
_load(list_of_testhelper)
_load(list_of_toggle)
_load(list_of_transform)
_load(list_of_linalg)
_load(list_of_universe)
_load(list_of_svg)

# Walk through the thirdparty folder and register all library
# into a dictionary.
pylibmgr.search_library_root(os.getcwd(), 'thirdparty')

del _load

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
