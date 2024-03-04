# Copyright (c) 2021, Yung-Yu Chen <yyc@solvcon.net>
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
Show sample mesh
"""

from .. import core
from .. import view
from .. import apputil


def help_3dmix(set_command=False):
    cmd = """
# Open a sub window for triangles and quadrilaterals:
w_3dmix = add3DWidget()
mh_3dmix = make_3dmix()
w_3dmix.updateMesh(mh_3dmix)
w_3dmix.showMark()
print("3dmix nedge:", mh_3dmix.nedge)
"""
    view.mgr.pycon.writeToHistory(cmd)
    if set_command:
        view.mgr.pycon.command = cmd.strip()


def make_3dmix():
    HEX = core.StaticMesh.HEXAHEDRON
    TET = core.StaticMesh.TETRAHEDRON
    PSM = core.StaticMesh.PRISM
    PYR = core.StaticMesh.PYRAMID

    mh = core.StaticMesh(ndim=3, nnode=11, nface=0, ncell=4)
    mh.ndcrd.ndarray[:, :] = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        (0.5, 1.5, 0.5),
        (1.5, 1, 0.5), (1.5, 0, 0.5),
    ]
    mh.cltpn.ndarray[:] = [
        HEX, PYR, TET, PSM,
    ]
    mh.clnds.ndarray[:, :9] = [
        (8, 0, 1, 2, 3, 4, 5, 6, 7), (5, 2, 3, 7, 6, 8, -1, -1, -1),
        (4, 2, 6, 9, 8, -1, -1, -1, -1), (6, 2, 6, 9, 1, 5, 10, -1, -1),
    ]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def load_app():
    aenv = apputil.get_current_appenv()
    symbols = (
        'help_3dmix',
        'make_3dmix',
    )
    for k in symbols:
        if isinstance(k, tuple):
            k, o = k
        else:
            o = globals().get(k, None)
            if o is None:
                o = locals().get(k, None)
        view.mgr.pycon.writeToHistory(f"Adding symbol {k}\n")
        aenv.globals[k] = o
    # Open a sub window for triangles and quadrilaterals:
    w_3dmix = view.mgr.add3DWidget()
    mh_3dmix = make_3dmix()
    w_3dmix.updateMesh(mh_3dmix)
    w_3dmix.showMark()
    print("3dmix nedge:", mh_3dmix.nedge)   

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: