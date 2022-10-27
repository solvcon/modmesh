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


def load_app():
    view.app().pytext.code = """import modmesh as mm

#mm.view.app().viewer.up_vector = (0, 1, 0)
#mm.view.app().viewer.position = (-10, -10, -20)
#mm.view.app().viewer.view_center = (0, 0, 0)

mh = mm.app.sample_mesh.make_triangle()
#mh = mm.app.sample_mesh.make_tetrahedron()
mm.view.show_mark()
mm.view.show(mh)

print("nedge:", mh.nedge)
print("position:", mm.view.app().viewer.position)
print("up_vector:", mm.view.app().viewer.up_vector)
print("view_center:", mm.view.app().viewer.view_center)

# line = mm.view.RLine(-1, -1, -1, -2, -2, -2, 0, 128, 128)
# print(line)
"""


def make_triangle():
    mh = core.StaticMesh(ndim=2, nnode=4, nface=0, ncell=3)
    mh.ndcrd.ndarray[:, :] = (0, 0), (-1, -1), (1, -1), (0, 1)
    mh.cltpn.ndarray[:] = core.StaticMesh.TRIANGLE
    mh.clnds.ndarray[:, :4] = (3, 0, 1, 2), (3, 0, 2, 3), (3, 0, 3, 1)
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def make_tetrahedron():
    mh = core.StaticMesh(ndim=3, nnode=4, nface=4, ncell=1)
    mh.ndcrd.ndarray[:, :] = (0, 0, 0), (0, 1, 0), (-1, 1, 0), (0, 1, 1)
    mh.cltpn.ndarray[:] = core.StaticMesh.TETRAHEDRON
    mh.clnds.ndarray[:, :5] = [(4, 0, 1, 2, 3)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
