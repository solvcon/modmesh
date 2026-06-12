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


"""Phase 1 of porting the 2/3D Euler solver: the oblique-shock
reflection mesh and its boundary tagging, implemented in
``modmesh.multidim.euler.oblique`` and shared with the pilot GUI.

A uniform supersonic stream enters from the left over a slip wall whose bottom
turns into a wedge inclined by a fixed angle.  The wedge deflects the flow, an
oblique shock forms at the wedge tip and reflects off the flat top slip wall,
and the flow leaves through the non-reflective outflow on the right.  The mesh
comes in three element flavors: one quadrilateral per grid box, each box cut
into two triangles, or a Delaunay triangulation of jittered interior points.
"""

import unittest

from numpy.testing import assert_almost_equal

import modmesh
from modmesh.multidim.euler.oblique import ObliqueShockMesher


class _ObliqueMeshBase:
    """The oblique-shock mesh builds and its boundary classifies cleanly.

    The boundary checks are identical for all three element flavors because
    every flavor shares the same boundary node layout (interior-only changes
    in cell shape never touch the boundary faces); subclasses set the flavor,
    and the box-based flavors also set the cell count per grid box.
    """

    NX = 24
    NY = 8
    LL = (0.0, 0.0)
    UR = (3.0, 1.0)
    CELL_TYPE = None
    CELLS_PER_BOX = None
    CLTPN = None

    @classmethod
    def setUpClass(cls):
        cls.mesher = ObliqueShockMesher(nx=cls.NX, ny=cls.NY, ll=cls.LL,
                                        ur=cls.UR, x_ramp=0.5,
                                        wedge_angle=15.0)
        cls.mesh = cls.mesher.make_mesh(cell_type=cls.CELL_TYPE)

    def _boundary_faces(self):
        return {ifc for ifc in range(self.mesh.nface)
                if self.mesh.fccls[ifc, 1] < 0}

    def _signed_area2(self, icl):
        # Twice the signed shoelace area of cell icl, straight from
        # ndcrd/clnds via element access (the .ndarray views carry
        # prepended ghost rows).
        mh = self.mesh
        nnd = mh.clnds[icl, 0]
        xy = [(mh.ndcrd[mh.clnds[icl, 1 + it], 0],
               mh.ndcrd[mh.clnds[icl, 1 + it], 1])
              for it in range(nnd)]
        return sum(xy[it][0] * xy[(it + 1) % nnd][1]
                   - xy[(it + 1) % nnd][0] * xy[it][1]
                   for it in range(nnd))

    def test_mesh_shape(self):
        mh = self.mesh
        nbox = self.NX * self.NY
        self.assertEqual(2, mh.ndim)
        self.assertEqual(self.CELLS_PER_BOX * nbox, mh.ncell)
        # Element access reads the body cells; the .ndarray views also
        # carry the ghost cells that build_ghost prepends.
        self.assertTrue(all(mh.cltpn[icl] == self.CLTPN
                            for icl in range(mh.ncell)))
        # A structured quad grid has 2*nx*ny + nx + ny faces; each box split
        # adds one cell and one (diagonal, interior) face.
        self.assertEqual((self.CELLS_PER_BOX + 1) * nbox + self.NX + self.NY,
                         mh.nface)
        # The boundary is the four edges of the logical grid either way.
        self.assertEqual(2 * (self.NX + self.NY), len(self._boundary_faces()))

    def test_classification_partitions_boundary(self):
        inlet, walls, outflow = self.mesher.classify_boundary(self.mesh)
        # Every role is present.
        self.assertTrue(inlet)
        self.assertTrue(walls)
        self.assertTrue(outflow)
        si, sw, so = set(inlet), set(walls), set(outflow)
        # The three roles are pairwise disjoint.
        self.assertEqual(set(), si & sw)
        self.assertEqual(set(), si & so)
        self.assertEqual(set(), sw & so)
        # Together they cover every boundary face and nothing else.
        self.assertEqual(self._boundary_faces(), si | sw | so)

    def test_classification_counts(self):
        inlet, walls, outflow = self.mesher.classify_boundary(self.mesh)
        # The left and right edges carry one face per cell row; the top and
        # bottom edges carry one per cell column.
        self.assertEqual(self.NY, len(inlet))
        self.assertEqual(self.NY, len(outflow))
        self.assertEqual(2 * self.NX, len(walls))

    def test_inlet_outflow_geometry(self):
        mh = self.mesh
        inlet, walls, outflow = self.mesher.classify_boundary(mh)
        # The real domain spans [LL[0], UR[0]]; ndcrd would include ghost-node
        # coordinates that overshoot the edges, so use the construction
        # bounds.
        xmin, xmax = self.LL[0], self.UR[0]
        # Inlet faces sit on x == xmin with the outward normal pointing in -x.
        for ifc in inlet:
            assert_almost_equal(mh.fccnd[ifc, 0], xmin, decimal=12)
            assert_almost_equal(
                [mh.fcnml[ifc, 0], mh.fcnml[ifc, 1]], [-1.0, 0.0], decimal=12)
        # Outflow faces sit on x == xmax with the normal pointing in +x.
        for ifc in outflow:
            assert_almost_equal(mh.fccnd[ifc, 0], xmax, decimal=12)
            assert_almost_equal(
                [mh.fcnml[ifc, 0], mh.fcnml[ifc, 1]], [1.0, 0.0], decimal=12)
        # Wall faces are the top and bottom edges: their centres are strictly
        # interior in x and their outward normals are vertical-dominant.
        for ifc in walls:
            xc = mh.fccnd[ifc, 0]
            self.assertGreater(xc, xmin)
            self.assertLess(xc, xmax)
            self.assertGreater(abs(mh.fcnml[ifc, 1]), abs(mh.fcnml[ifc, 0]))

    def test_wedge_is_present(self):
        # The defining feature versus a plain channel: some slip-wall faces
        # are inclined, so their outward normal has a non-zero x component.
        mh = self.mesh
        _, walls, _ = self.mesher.classify_boundary(mh)
        inclined = [ifc for ifc in walls if abs(mh.fcnml[ifc, 0]) > 1e-6]
        self.assertTrue(inclined)

    def test_cell_winding_and_volume(self):
        # clvol alone cannot pin the winding: build_interior repairs
        # inverted faces and accumulates absolute values, so it stays
        # positive even for clockwise connectivity.  Check the signed
        # shoelace area instead, and keep the clvol check for degeneracy
        # (zero or NaN volume).
        mh = self.mesh
        for icl in range(mh.ncell):
            self.assertGreater(self._signed_area2(icl), 0.0)
            self.assertGreater(mh.clvol[icl], 0.0)

    def test_cells_tile_the_domain(self):
        # The cells tile the domain exactly (no overlap, no gap): the
        # signed cell areas must sum to the area enclosed by the boundary
        # faces, computed independently through the divergence theorem
        # (area = 1/2 * sum of fcara * (fccnd . fcnml) over the boundary).
        mh = self.mesh
        total = sum(self._signed_area2(icl) for icl in range(mh.ncell)) / 2.0
        enclosed = sum(mh.fcara[ifc] * (mh.fccnd[ifc, 0] * mh.fcnml[ifc, 0]
                                        + mh.fccnd[ifc, 1] * mh.fcnml[ifc, 1])
                       for ifc in self._boundary_faces()) / 2.0
        assert_almost_equal(total, enclosed, decimal=10)


class _SingleBoundaryFaceTB:
    """No cell touches the domain boundary with more than one face.

    Mixed into the triangular flavors only: a corner ``'quad'`` cell always
    owns its two adjacent boundary edges, so quads are exempt.
    """

    def test_at_most_one_boundary_face_per_cell(self):
        # Each cell keeps at least two interior neighbours for the CESE
        # solver.  fccls(ifc, 0) is the interior cell of boundary face ifc,
        # so tallying it counts the boundary faces each cell owns.
        mh = self.mesh
        per_cell = {}
        for ifc in self._boundary_faces():
            icl = mh.fccls[ifc, 0]
            per_cell[icl] = per_cell.get(icl, 0) + 1
        self.assertTrue(per_cell)
        self.assertLessEqual(max(per_cell.values()), 1)


class ObliqueShockQuadMeshTC(_ObliqueMeshBase, unittest.TestCase):
    """One quadrilateral per grid box."""

    CELL_TYPE = 'quad'
    CELLS_PER_BOX = 1
    CLTPN = modmesh.StaticMesh.QUADRILATERAL


class ObliqueShockTriangleMeshTC(_ObliqueMeshBase, _SingleBoundaryFaceTB,
                                 unittest.TestCase):
    """Each grid box cut along its diagonal into two triangles."""

    CELL_TYPE = 'triangle'
    CELLS_PER_BOX = 2
    CLTPN = modmesh.StaticMesh.TRIANGLE


class ObliqueShockUnstructuredMeshTC(_ObliqueMeshBase, _SingleBoundaryFaceTB,
                                     unittest.TestCase):
    """Delaunay triangulation of the wedge: the structured boundary layout
    with jittered interior points.  The boundary tests of the base apply
    unchanged because all flavors share the boundary node layout.
    """

    CELL_TYPE = 'unstructured'
    CELLS_PER_BOX = None  # not box-based; test_mesh_shape is overridden
    CLTPN = modmesh.StaticMesh.TRIANGLE

    def _body_connectivity(self, mh):
        # The body cells as (type, node-id tuple) pairs, plus the node
        # coordinates, via element access -- the canonical identity of a
        # mesh, free of the ghost rows and uninitialised trailing clnds
        # columns that the raw .ndarray views carry.
        cells = [(mh.cltpn[icl],
                  tuple(mh.clnds[icl, 1 + it]
                        for it in range(mh.clnds[icl, 0])))
                 for icl in range(mh.ncell)]
        nodes = [(mh.ndcrd[ind, 0], mh.ndcrd[ind, 1])
                 for ind in range(mh.nnode)]
        return cells, nodes

    def test_mesh_shape(self):
        # The counts are not box-based, but any triangulation using every
        # vertex of a convex region obeys exact combinatorics: with nb
        # boundary and ni interior vertices it has 2*ni + nb - 2 cells,
        # and the Euler characteristic of a disk (V - E + F = 1) pins the
        # face count.
        mh = self.mesh
        nbnd = 2 * (self.NX + self.NY)
        self.assertEqual(2, mh.ndim)
        self.assertTrue(all(mh.cltpn[icl] == self.CLTPN
                            for icl in range(mh.ncell)))
        self.assertEqual(nbnd, len(self._boundary_faces()))
        self.assertEqual(2 * (mh.nnode - nbnd) + nbnd - 2, mh.ncell)
        self.assertEqual(1, mh.nnode - mh.nface + mh.ncell)

    def test_triangulation_is_deterministic(self):
        # The flavor advertises a reproducible mesh (RNG-free jitter): a
        # second build of the same parameters must reproduce the node
        # coordinates and the triangle connectivity exactly.  Guards against
        # a future switch to an unseeded RNG, which the count/Euler/tiling
        # invariants would not catch.
        again = self.mesher.make_mesh(cell_type=self.CELL_TYPE)
        self.assertEqual(self._body_connectivity(self.mesh),
                         self._body_connectivity(again))


class ObliqueShockMesherTC(unittest.TestCase):

    def test_unknown_cell_type(self):
        with self.assertRaises(ValueError):
            ObliqueShockMesher(nx=2, ny=2).make_mesh(cell_type='hexagon')

    def test_wedge_top_must_stay_below_ly(self):
        # Past atan(height / (ur[0] - x_ramp)) the grid columns invert and
        # the mesh would silently self-overlap; the constructor refuses
        # instead.  Here the limit is atan(1.0 / 2.5) ~ 21.8 degrees, so 30
        # degrees overflows.
        with self.assertRaises(ValueError):
            ObliqueShockMesher(nx=2, ny=2, ll=(0.0, 0.0), ur=(3.0, 1.0),
                               x_ramp=0.5, wedge_angle=30.0)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
