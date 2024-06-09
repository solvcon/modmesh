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
Show example meshes.
"""

from .. import core
from .. import view


def mesh_triangle():
    mh = core.StaticMesh(ndim=2, nnode=4, nface=0, ncell=3)
    mh.ndcrd.ndarray[:, :] = (0, 0), (-1, -1), (1, -1), (0, 1)
    mh.cltpn.ndarray[:] = core.StaticMesh.TRIANGLE
    mh.clnds.ndarray[:, :4] = (3, 0, 1, 2), (3, 0, 2, 3), (3, 0, 3, 1)
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    w_tri = view.mgr.add3DWidget()
    w_tri.updateMesh(mh)
    w_tri.showMark()
    print("tri nedge:", mh.nedge)


def mesh_tetrahedron():
    mh = core.StaticMesh(ndim=3, nnode=4, nface=4, ncell=1)
    mh.ndcrd.ndarray[:, :] = (0, 0, 0), (0, 1, 0), (-1, 1, 0), (0, 1, 1)
    mh.cltpn.ndarray[:] = core.StaticMesh.TETRAHEDRON
    mh.clnds.ndarray[:, :5] = [(4, 0, 1, 2, 3)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    w_tet = view.mgr.add3DWidget()
    w_tet.updateMesh(mh)
    w_tet.showMark()
    print("tet nedge:", mh.nedge)


def mesh_solvcon_2dtext():
    Q = core.StaticMesh.QUADRILATERAL
    mh = core.StaticMesh(ndim=2, nnode=140, nface=0, ncell=65)
    mh.ndcrd.ndarray[:, :] = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
        (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0),
        (14, 0), (15, 0), (16, 0), (18, 0), (19, 0), (20, 0), (21, 0),
        (22, 0), (23, 0), (24, 0), (25, 0), (26, 0), (27, 0), (28, 0),
        (29, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
        (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1),
        (14, 1), (15, 1), (16, 1), (18, 1), (19, 1), (20, 1), (21, 1),
        (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1),
        (29, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
        (7, 2), (8, 2), (9, 2), (12, 2), (13, 2), (15, 2), (16, 2),
        (18, 2), (19, 2), (22, 2), (23, 2), (24, 2), (25, 2), (26, 2),
        (27, 2), (28, 2), (29, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
        (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (11, 3), (12, 3), (13, 3),
        (15, 3), (16, 3), (17, 3), (18, 3), (19, 3), (20, 3), (21, 3),
        (22, 3), (23, 3), (24, 3), (25, 3), (26, 3), (27, 3), (28, 3),
        (29, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
        (7, 4), (8, 4), (9, 4), (11, 4), (12, 4), (13, 4), (15, 4),
        (16, 4), (17, 4), (18, 4), (19, 4), (20, 4), (21, 4), (22, 4),
        (23, 4), (24, 4), (25, 4), (26, 4), (27, 4), (0, 5), (1, 5),
        (2, 5), (3, 5)
    ]
    mh.cltpn.ndarray[:] = [
        Q, Q, Q, Q, Q, Q, Q, Q, Q, Q,
        Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q,  # 0-20
        Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q,  # 21-31
        Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q,  # 32-45
        Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q,  # 46-61
        Q, Q, Q  # 62-64
    ]
    mh.clnds.ndarray[:, :5] = [
        (4, 0, 1, 30, 29),
        (4, 1, 2, 31, 30),
        (4, 2, 3, 32, 31),
        (4, 4, 5, 34, 33),
        (4, 5, 6, 35, 34),
        (4, 6, 7, 36, 35),
        (4, 8, 9, 38, 37),
        (4, 9, 10, 39, 38),
        (4, 10, 11, 40, 39),
        (4, 12, 13, 42, 41),
        (4, 13, 14, 43, 42),
        (4, 14, 15, 44, 43),
        (4, 15, 16, 45, 44),
        (4, 17, 18, 47, 46),
        (4, 18, 19, 48, 47),
        (4, 19, 20, 49, 48),
        (4, 21, 22, 51, 50),
        (4, 22, 23, 52, 51),
        (4, 23, 24, 53, 52),
        (4, 25, 26, 55, 54),
        (4, 27, 28, 57, 56),
        (4, 31, 32, 61, 60),
        (4, 33, 34, 63, 62),
        (4, 35, 36, 65, 64),
        (4, 37, 38, 67, 66),
        (4, 41, 42, 69, 68),
        (4, 44, 45, 71, 70),
        (4, 46, 47, 73, 72),
        (4, 50, 51, 75, 74),
        (4, 52, 53, 77, 76),
        (4, 54, 55, 79, 78),
        (4, 56, 57, 81, 80),
        (4, 58, 59, 83, 82),
        (4, 59, 60, 84, 83),
        (4, 60, 61, 85, 84),
        (4, 62, 63, 87, 86),
        (4, 64, 65, 89, 88),
        (4, 66, 67, 91, 90),
        (4, 68, 69, 94, 93),
        (4, 70, 71, 96, 95),
        (4, 72, 73, 99, 98),
        (4, 74, 75, 103, 102),
        (4, 76, 77, 105, 104),
        (4, 78, 79, 107, 106),
        (4, 79, 80, 108, 107),
        (4, 80, 81, 109, 108),
        (4, 82, 83, 111, 110),
        (4, 86, 87, 115, 114),
        (4, 87, 88, 116, 115),
        (4, 88, 89, 117, 116),
        (4, 90, 91, 119, 118),
        (4, 92, 93, 121, 120),
        (4, 93, 94, 122, 121),
        (4, 95, 96, 124, 123),
        (4, 96, 97, 125, 124),
        (4, 98, 99, 127, 126),
        (4, 99, 100, 128, 127),
        (4, 100, 101, 129, 128),
        (4, 102, 103, 131, 130),
        (4, 103, 104, 132, 131),
        (4, 104, 105, 133, 132),
        (4, 106, 107, 135, 134),
        (4, 110, 111, 137, 136),
        (4, 111, 112, 138, 137),
        (4, 112, 113, 139, 138)
    ]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    # Open a sub window for solvcon icon:
    w_solvcon = view.mgr.add3DWidget()
    w_solvcon.updateMesh(mh)
    w_solvcon.showMark()
    print("solvcon nedge:", mh.nedge)


def mesh_3dmix():
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

    # Open a sub window for triangles and quadrilaterals:
    w_3dmix = view.mgr.add3DWidget()
    w_3dmix.updateMesh(mh)
    w_3dmix.showMark()
    view.mgr.pycon.writeToHistory(f"3dmix nedge: {mh.nedge}\n")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
