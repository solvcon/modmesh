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
Example pilot apps for the oblique-shock reflection.

The mesh construction, boundary tagging, and solver driver live in
:mod:`modmesh.multidim.euler.oblique`.  :class:`ObliqueShockMesh` draws the
mesh in a 3D widget and reports the boundary classification (inlet / slip wall
/ outflow) to the console; :class:`ObliqueShockSolver` runs the Euler driver
and animates the evolving density as a flat 2D color field, drawn with the
native ``R3DWidget.updateColorField`` per-cell-colored triangles.
"""

import numpy as np

from PySide6 import QtCore

from .. import core
from ..multidim.euler import oblique
from . import _gui_common

__all__ = [  # noqa: F822
    'ObliqueShockMesh',
    'ObliqueShockSolver',
]


def _colormap(t):
    """Map ``t`` in [0, 1] to a jet-like RGB array (``..., 3``) in [0, 1].

    A four-stop blue-cyan-yellow-red ramp; the triangle-wave channels are the
    standard compact "jet" approximation, enough to read a scalar field
    without pulling in a plotting dependency.
    """
    t = np.clip(np.asarray(t, dtype='float64'), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _cell_triangulation(mh):
    """
    Per-cell unshared triangle vertices and indices for a flat color field.
    """
    verts, tris, nnds = [], [], []
    base = 0
    for icl in range(mh.ncell):
        nnd = mh.clnds[icl, 0]
        nnds.append(nnd)
        for it in range(nnd):
            ind = mh.clnds[icl, 1 + it]
            verts.append((mh.ndcrd[ind, 0], mh.ndcrd[ind, 1], 0.0))
        for it in range(1, nnd - 1):
            tris.append((base, base + it, base + it + 1))
        base += nnd
    return (np.array(verts, dtype='float32'),
            np.array(tris, dtype='uint32'),
            np.array(nnds, dtype='int64'))


def _field_colors(field, nnds, vmin, vmax):
    span = vmax - vmin
    t = (field - vmin) / span if span > 0 else np.zeros_like(field)
    return np.repeat(_colormap(t), nnds, axis=0).astype('float32')


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


class ObliqueShockSolver(_gui_common.PilotFeature):
    """
    Run the oblique-shock Euler driver and animate the density field.
    """

    #: Solver steps marched per timer frame.
    STEPS_PER_FRAME = 5
    #: Stop the animation after this many steps.
    MAX_STEPS = 2000
    #: Qt timer interval in milliseconds.
    INTERVAL_MS = 50

    def __init__(self, *args, **kw):
        # Keep every running session (3D widget, timer, driver) referenced so
        # Qt and the driver are not garbage-collected mid-run.
        self._sessions = []
        super(ObliqueShockSolver, self).__init__(*args, **kw)

    def populate_menu(self):
        self._add_menu_item(
            menu=self._mgr.meshMenu,
            text="Sample: oblique-shock solution (density)",
            tip="March the oblique-shock Euler solver and draw the density "
                "as a 2D color field",
            func=self._run,
        )

    def _run(self):
        shock = oblique.ObliqueShock()
        shock.build_constant()
        shock.build_numerical(cell_type='quad')

        # The cell geometry is fixed across the run; only the colors change,
        # so triangulate once and cache the vertex/index arrays.
        verts, indices, nnds = _cell_triangulation(shock.mesh)
        widget = self._mgr.add3DWidget()
        widget.showMark()
        timer = QtCore.QTimer()
        session = dict(shock=shock, widget=widget, timer=timer, nnds=nnds,
                       verts=core.SimpleArrayFloat32(array=verts),
                       indices=core.SimpleArrayUint32(array=indices), step=0)
        self._draw_frame(session)
        timer.timeout.connect(lambda: self._advance(session))
        timer.start(self.INTERVAL_MS)
        self._sessions.append(session)

    def _advance(self, session):
        if session['step'] >= self.MAX_STEPS:
            session['timer'].stop()
            return
        session['shock'].march(self.STEPS_PER_FRAME)
        session['step'] += self.STEPS_PER_FRAME
        self._draw_frame(session)

    def _draw_frame(self, session):
        svr = session['shock'].svr
        # Density is the first conserved variable; the raw .ndarray view
        # prepends the ghost rows, so colour only the body cells.
        field = svr.so0n.ndarray[svr.ngstcell:, 0]
        colors = _field_colors(field, session['nnds'],
                               field.min(), field.max())
        session['widget'].updateColorField(
            session['verts'], core.SimpleArrayFloat32(array=colors),
            session['indices'])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
