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


__all__ = [
    'WrapperProfilerStatus',
    'wrapper_profiler_status',
    'StopWatch',
    'stop_watch',
    'TimeRegistry',
    'time_registry',
    'ConcreteBuffer',
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
    'StaticGrid1d',
    'StaticGrid2d',
    'StaticGrid3d',
    'StaticMesh',
]


# A hidden loophole to impolementation; it should only be used for testing
# during development.
from . import _modmesh as _impl  # noqa: F401


from ._modmesh import (
    WrapperProfilerStatus,
    wrapper_profiler_status,
    StopWatch,
    stop_watch,
    TimeRegistry,
    time_registry,
    ConcreteBuffer,
    SimpleArrayBool,
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
    StaticMesh,
)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
