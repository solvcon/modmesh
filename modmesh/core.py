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

__all__ = [  # noqa: F822
    'WrapperProfilerStatus',
    'wrapper_profiler_status',
    'StopWatch',
    'stop_watch',
    'TimeRegistry',
    'time_registry',
    'CallProfiler',
    'call_profiler',
    'CallProfilerProbe',
    'ConcreteBuffer',
    'BufferExpander',
    'Gmsh',
    'Plot3d',
    'complex64',
    'complex128',
    'FourierTransform',
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
    'StaticGrid1d',
    'StaticGrid2d',
    'StaticGrid3d',
    'StaticMesh',
    'EulerCore',
    'HierarchicalToggleAccess',
    'Toggle',
    'CommandLineInfo',
    'ProcessInfo',
    'METAL_BUILT',
    'metal_running',
    'HAS_PILOT',
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
    'testhelper'
]


# A hidden loophole to impolementation; it should only be used for testing
# during development.
try:
    import _modmesh as _impl  # noqa: F401
except ImportError:
    from . import _modmesh as _impl  # noqa: F401


def _load():
    for name in __all__:
        globals()[name] = getattr(_impl, name)

    # Walk through the thirdparty folder and register all library
    # into a dictionary.
    pylibmgr.search_library_root(os.getcwd(), 'thirdparty')


_load()
del _load

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
