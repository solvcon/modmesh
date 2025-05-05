# Copyright (c) 2025, Ting-Yu Chuang <tychuang.cs10@nycu.edu.tw>
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


import numpy as np

try:
    from _modmesh import spacetime as _impl  # noqa: F401
except ImportError:
    from .._modmesh import spacetime as _impl  # noqa: F401

__all__ = [
    'LinearScalarSolver',
]


class LinearScalarSolver:

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
