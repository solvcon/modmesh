# Copyright (c) 2026, Yung-Yu Chen <yyc@solvcon.net>
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
Example pilot app drawing the oblique-shock reflection mesh.

The mesh construction and boundary tagging live in
:mod:`modmesh.multidim.euler.oblique`; this feature only draws the mesh in a 3D
widget and reports the boundary classification (inlet / slip wall / outflow)
to the console.
"""

from ..multidim.euler import oblique
from . import _gui_common

__all__ = [  # noqa: F822
    'ObliqueShockMesh',
]


class ObliqueShockMesh(_gui_common.PilotFeature):
    """
    Draw the oblique-shock reflection mesh and tag its boundary.
    """

    def populate_menu(self):
        self._add_menu_item(
            menu=self._mgr.meshMenu,
            text="Sample: oblique-shock reflection mesh (2D quad)",
            tip="Draw the quad wedge mesh for the oblique-shock reflection",
            func=self.draw_quad_mesh,
        )
        self._add_menu_item(
            menu=self._mgr.meshMenu,
            text="Sample: oblique-shock reflection mesh (2D triangle)",
            tip="Draw the triangle wedge mesh for the oblique-shock "
                "reflection",
            func=self.draw_triangle_mesh,
        )
        self._add_menu_item(
            menu=self._mgr.meshMenu,
            text="Sample: oblique-shock reflection mesh (2D unstructured)",
            tip="Draw the unstructured (Delaunay) triangle wedge mesh for "
                "the oblique-shock reflection",
            func=self.draw_unstructured_mesh,
        )

    def draw_quad_mesh(self):
        self._draw_mesh('quad')

    def draw_triangle_mesh(self):
        self._draw_mesh('triangle')

    def draw_unstructured_mesh(self):
        self._draw_mesh('unstructured')

    def _draw_mesh(self, cell_type):
        mesher = oblique.ObliqueShockMesher()
        mh = mesher.make_mesh(cell_type=cell_type)
        inlet, walls, outflow = mesher.classify_boundary(mh)
        w = self._mgr.add3DWidget()
        w.updateMesh(mh)
        w.showMark()
        self._pycon.writeToHistory(
            f"oblique-shock {cell_type} mesh: {mh.ncell} cells, "
            f"{mh.nedge} edges\n"
            f"boundary faces: {len(inlet)} inlet, {len(walls)} slip wall, "
            f"{len(outflow)} outflow\n")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
