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


def make_mesh_viewer(path):
    gm = core.Gmsh(path)
    mh = gm.toblock()
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


def make_tetrahedron():
    mh = core.StaticMesh(ndim=3, nnode=4, nface=4, ncell=1)
    mh.ndcrd.ndarray[:, :] = (0, 0, 0), (0, 1, 0), (-1, 1, 0), (0, 1, 1)
    mh.cltpn.ndarray[:] = core.StaticMesh.TETRAHEDRON
    mh.clnds.ndarray[:, :5] = [(4, 0, 1, 2, 3)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def load_app():
    aenv = apputil.get_current_appenv()
    symbols = (
        'help_tri',
        'help_tet',
        'help_mesh_viewer',
        'help_other',
        'make_triangle',
        'make_tetrahedron',
        'make_mesh_viewer',
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
help_other(set_command=False)  # or True
help_mesh_viewer(path, set_command=False)  # or True
""")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
