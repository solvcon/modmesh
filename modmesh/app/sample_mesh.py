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


def help_tri(set_command=False):
    cmd = """
# Open a sub window for a triangle:
w_tri = add3DWidget()
mh_tri = make_triangle()
w_tri.updateMesh(mh_tri)
w_tri.showMark()
print("tri nedge:", mh_tri.nedge)
"""
    view.mgr.pycon.writeToHistory(cmd)
    if set_command:
        view.mgr.pycon.command = cmd.strip()


def help_tet(set_command=False):
    cmd = """
# Open a sub window for a tetrahedron:
w_tet = add3DWidget()
mh_tet = make_tetrahedron()
w_tet.updateMesh(mh_tet)
w_tet.showMark()
print("tet nedge:", mh_tet.nedge)
"""
    view.mgr.pycon.writeToHistory(cmd)
    if set_command:
        view.mgr.pycon.command = cmd.strip()


def help_2dmix(set_command=False):
    cmd = """
# Open a sub window for triangles and quadrilaterals:
w_2dmix = add3DWidget()
mh_2dmix = make_2dmix(do_small=False)
w_2dmix.updateMesh(mh_2dmix)
w_2dmix.showMark()
print("2dmix nedge:", mh_2dmix.nedge)
"""
    view.mgr.pycon.writeToHistory(cmd)
    if set_command:
        view.mgr.pycon.command = cmd.strip()


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

def help_sciwork(set_command=False):
    cmd = """
# Open a sub window for sciwork icon:
w_sciwork = add3DWidget()
mh_sciwork = make_sciwork()
w_sciwork.updateMesh(mh_sciwork)
w_sciwork.showMark()
print("sciwork nedge:", mh_sciwork.nedge)
"""
    view.mgr.pycon.writeToHistory(cmd)
    if set_command:
        view.mgr.pycon.command = cmd.strip()

def help_other(set_command=False):
    cmd = """
# Show triangle information:
print("position:", w_tri.position)
print("up_vector:", w_tri.up_vector)
print("view_center:", w_tri.view_center)

# Set view:
w_tri.up_vector = (0, 1, 0)
w_tri.position = (-10, -10, -20)
w_tri.view_center = (0, 0, 0)

# Outdated and may not work:
# line = mm.view.RLine(-1, -1, -1, -2, -2, -2, 0, 128, 128)
# print(line)
"""
    view.mgr.pycon.writeToHistory(cmd)
    if set_command:
        view.mgr.pycon.command = cmd.strip()


def help_mesh_viewer(path, set_command=False):
    cmd = f"""
# Open a sub window for the mesh viewer:
w_mh_viewer = add3DWidget()
mh_viewer = make_mesh_viewer("{path}")
w_mh_viewer.updateMesh(mh_viewer)
w_mh_viewer.showMark()
"""
    view.mgr.pycon.writeToHistory(cmd)
    if set_command:
        view.mgr.pycon.command = cmd.strip()


def help_bezier(set_command=False):
    cmd = """
# Open a sub window for some bezier curves:
w = make_bezier()
"""
    view.mgr.pycon.writeToHistory(cmd)
    if set_command:
        view.mgr.pycon.command = cmd.strip()


def make_mesh_viewer(path):
    data = open(path, 'rb').read()
    gm = core.Gmsh(data)
    mh = gm.to_block()
    return mh


def make_triangle():
    mh = core.StaticMesh(ndim=2, nnode=4, nface=0, ncell=3)
    mh.ndcrd.ndarray[:, :] = (0, 0), (-1, -1), (1, -1), (0, 1)
    mh.cltpn.ndarray[:] = core.StaticMesh.TRIANGLE
    mh.clnds.ndarray[:, :4] = (3, 0, 1, 2), (3, 0, 2, 3), (3, 0, 3, 1)
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def make_2dmix(do_small=False):
    T = core.StaticMesh.TRIANGLE
    Q = core.StaticMesh.QUADRILATERAL

    if do_small:
        mh = core.StaticMesh(ndim=2, nnode=6, nface=0, ncell=3)
        mh.ndcrd.ndarray[:, :] = [
            (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)
        ]
        mh.cltpn.ndarray[:] = [
            T, T, Q,
        ]
        mh.clnds.ndarray[:, :5] = [
            (3, 0, 3, 2, -1), (3, 0, 1, 3, -1), (4, 1, 4, 5, 3),
        ]
    else:
        mh = core.StaticMesh(ndim=2, nnode=16, nface=0, ncell=14)
        mh.ndcrd.ndarray[:, :] = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2), (3, 2),
            (0, 3), (1, 3), (2, 3), (3, 3),
        ]
        mh.cltpn.ndarray[:] = [
            T, T, T, T, T, T,  # 0-5,
            Q, Q,  # 6-7
            T, T, T, T,  # 8-11
            Q, Q,  # 12-13
        ]
        mh.clnds.ndarray[:, :5] = [
            (3, 0, 5, 4, -1), (3, 0, 1, 5, -1),  # 0-1 triangles
            (3, 1, 2, 5, -1), (3, 2, 6, 5, -1),  # 2-3 triangles
            (3, 2, 7, 6, -1), (3, 2, 3, 7, -1),  # 4-5 triangles
            (4, 4, 5, 9, 8), (4, 5, 6, 10, 9),  # 6-7 quadrilaterals
            (3, 6, 7, 10, -1), (3, 7, 11, 10, -1),  # 8-9 triangles
            (3, 8, 9, 12, -1), (3, 9, 13, 12, -1),  # 10-11 triangles
            (4, 9, 10, 14, 13), (4, 10, 11, 15, 14),  # 12-13 quadrilaterals
        ]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


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

def make_sciwork():
    Q = core.StaticMesh.QUADRILATERAL
    mh = core.StaticMesh(ndim=2, nnode=179, nface=0, ncell=84)
    mh.ndcrd.ndarray[:, :] = [
            (0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),(10,0),
            (11,0),(13,0),(14,0),(15,0),(16,0),(17,0),(18,0),(20,0),(21,0),(22,0),
            (23,0),(24,0),(25,0),(26,0),(27,0),(28,0),(29,0),(30,0),(31,0),(32,0),

            (0,1),(1,1),(2,1),(3,1),(4,1),(5,1),(6,1),(7,1),(8,1),(9,1),(10,1),(11,1),
            (13,1),(14,1),(15,1),(16,1),(17,1),(18,1),(20,1),(21,1),(22,1),(23,1),(24,1),
            (25,1),(26,1),(27,1),(28,1),(29,1),(30,1),(31,1),(32,1),

            (0,2),(1,2),(2,2),(3,2),(4,2),(5,2),(9,2),(10,2),(13,2),(14,2),(15,2),(16,2),
            (17,2),(18,2),(20,2),(21,2),(23,2),(24,2),(25,2),(26,2),(27,2),(29,2),(30,2),
            (31,2),(32,2),
            
            (0,3),(1,3),(2,3),(3,3),(4,3),(5,3),(9,3),(10,3),(13,3),(14,3),(15,3),(16,3),
            (17,3),(18,3),(20,3),(21,3),(23,3),(24,3),(25,3),(26,3),(27,3),(28,3),(29,3),
            (30,3),(31,3),(32,3),
            
            (0,4),(1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(8,4),(9,4),(10,4),(11,4),(12,4),
            (13,4),(14,4),(15,4),(16,4),(17,4),(18,4),(19,4),(20,4),(21,4),(22,4),(23,4),(24,4),
            (25,4),(26,4),(27,4),(28,4),(29,4),(30,4),(31,4),(32,4),
            
            (0,5),(1,5),(2,5),(3,5),(4,5),(5,5),(6,5),(7,5),(8,5),(9,5),(10,5),(11,5),(12,5),(13,5),
            (14,5),(15,5),(16,5),(17,5),(18,5),(19,5),(20,5),(21,5),(22,5),(23,5),(24,5),(25,5),(26,5),
            (27,5),(28,5),(29,5),(30,5),(31,5),(32,5)
        ]
    mh.cltpn.ndarray[:] = [
           Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q, #0-21
           Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q, #22-34
           Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q, #35-46
           Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q, #47-59
           Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q #60-82
        ]
    mh.clnds.ndarray[:, :5] = [
            (4,0,1,32,31),
            (4,1,2,33,32),
            (4,2,3,34,33),
            (4,4,5,36,35),
            (4,5,6,37,36),
            (4,6,7,38,37),
            (4,8,9,40,39),
            (4,9,10,41,40),
            (4,10,11,42,41),
            (4,12,13,44,43),
            (4,13,14,45,44),
            (4,14,15,46,45),
            (4,15,16,47,46),
            (4,16,17,48,47),
            (4,18,19,50,49),
            (4,19,20,51,50),
            (4,20,21,52,51),
            (4,21,22,53,52),
            (4,23,24,55,54),
            (4,25,26,57,56),
            (4,27,28,59,58),
            (4,29,30,61,60),
 
            (4,33,34,65,64),
            (4,35,36,67,66),
            (4,40,41,69,68),
            (4,43,44,71,70),
            (4,45,46,73,72),
            (4,47,48,75,74),
            (4,49,50,77,76),
            (4,52,53,79,78),
            (4,54,55,81,80),
            (4,55,56,82,81),
            (4,58,59,84,83),
            (4,59,60,85,84),
            (4,60,61,86,85),

            (4,62,63,88,87),
            (4,63,64,89,88),
            (4,64,65,90,89),
            (4,66,67,92,91),
            (4,68,69,94,93),
            (4,70,71,96,95),
            (4,72,73,98,97),
            (4,74,75,100,99),
            (4,76,77,102,101),
            (4,78,79,104,103),
            (4,80,81,106,105),
            (4,83,84,110,109),

            (4,87,88,114,113),
            (4,91,92,118,117),
            (4,93,94,123,122),
            (4,95,96,127,126),
            (4,99,100,131,130),
            (4,101,102,134,133),
            (4,103,104,137,136),
            (4,105,106,139,138),
            (4,106,107,140,139),
            (4,107,108,141,140),
            (4,109,110,143,142),
            (4,110,111,144,143),
            (4,111,112,145,144),

            (4,113,114,147,146),
            (4,114,115,148,147),
            (4,115,116,149,148),
            (4,117,118,151,150),
            (4,118,119,152,151),
            (4,119,120,153,152),
            (4,121,122,155,154),
            (4,122,123,156,155),
            (4,123,124,157,156),
            (4,125,126,159,158),
            (4,126,127,160,159),
            (4,127,128,161,160),
            (4,129,130,163,162),
            (4,130,131,164,163),
            (4,131,132,165,164),
            (4,133,134,167,166),
            (4,134,135,168,167),
            (4,135,136,169,168),
            (4,136,137,170,169),
            (4,138,139,172,171),
            (4,139,140,173,172),
            (4,140,141,174,173),
            (4,142,143,176,175),
            (4,144,145,178,177)
        ]
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


def make_bezier():
    """
    A simple example for drawing a couple of cubic Bezier curves.
    """
    Vector = core.Vector3dFp64
    World = core.WorldFp64
    w = World()
    b1 = w.add_bezier(
        [Vector(0, 0, 0), Vector(1, 1, 0), Vector(3, 1, 0),
         Vector(4, 0, 0)])
    b1.sample(7)
    b2 = w.add_bezier(
        [Vector(4, 0, 0), Vector(3, -1, 0), Vector(1, -1, 0),
         Vector(0, 0, 0)])
    b2.sample(25)
    b3 = w.add_bezier(
        [Vector(0, 2, 0), Vector(1, 3, 0), Vector(3, 3, 0),
         Vector(4, 2, 0)])
    b3.sample(9)
    wid = view.mgr.add3DWidget()
    wid.updateWorld(w)
    wid.showMark()
    return wid


def load_app():
    aenv = apputil.get_current_appenv()
    symbols = (
        'help_tri',
        'help_tet',
        'help_2dmix',
        'help_3dmix',
        'help_mesh_viewer',
        'help_sciwork',
        'help_other',
        'help_bezier',
        'make_triangle',
        'make_tetrahedron',
        'make_2dmix',
        'make_3dmix',
        'make_sciwork',
        'make_mesh_viewer',
        'make_bezier',
        ('add3DWidget', view.mgr.add3DWidget),
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
    view.mgr.pycon.writeToHistory("""
# Use the functions for more examples:
help_tri(set_command=False)  # or True
help_tet(set_command=False)  # or True
help_2dmix(set_command=False)  # or True
help_3dmix(set_command=False)  # or True
help_sciwork(set_command=False)  # or True                                 
help_other(set_command=False)  # or True
help_mesh_viewer(path, set_command=False)  # or True
help_bezier(set_command=False)  # or True
""")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
