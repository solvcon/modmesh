import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import modmesh


class StaticMesh2dQuadSingleTC(unittest.TestCase):
    """Single quad (unit square): face count, normals, areas, clvol,
    clcnd, boundary edges, ghost cells."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=2, nnode=4, nface=0, ncell=1)
        mh.ndcrd[:, :] = [(0, 0), (1, 0), (1, 1), (0, 1)]
        mh.cltpn.fill(modmesh.StaticMesh.QUADRILATERAL)
        mh.clnds[:, :5] = [(4, 0, 1, 2, 3)]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh

    def test_face_count(self):
        self.assertEqual(4, self.mesh.nface)

    def test_normals_unit_length(self):
        for i in range(self.mesh.nface):
            nml = [self.mesh.fcnml[i, d] for d in range(self.mesh.ndim)]
            assert_almost_equal(np.linalg.norm(nml), 1.0, decimal=10)

    def test_normals_values(self):
        mh = self.mesh
        normals = [[mh.fcnml[i, d] for d in range(mh.ndim)]
                   for i in range(mh.nface)]
        assert_almost_equal(
            normals, [[0, -1], [1, 0], [0, 1], [-1, 0]], decimal=10)

    def test_areas(self):
        assert_almost_equal(list(self.mesh.fcara), [1.0] * 4, decimal=10)

    def test_clvol(self):
        assert_almost_equal(self.mesh.clvol[0], 1.0, decimal=10)

    def test_clcnd(self):
        mh = self.mesh
        assert_almost_equal(
            [mh.clcnd[0, 0], mh.clcnd[0, 1]], [0.5, 0.5], decimal=10)

    def test_boundary_edges(self):
        self.assertEqual(4, self.mesh.nbound)

    def test_ghost_cells(self):
        self.assertEqual(4, self.mesh.ngstcell)


class StaticMesh2dQuadGridTC(unittest.TestCase):
    """2x2 quad grid (9 nodes, 4 cells): interior/boundary face split,
    fccls, all clvol = 0.25."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=2, nnode=9, nface=0, ncell=4)
        mh.ndcrd[:, :] = [
            (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
            (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
            (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
        ]
        mh.cltpn.fill(modmesh.StaticMesh.QUADRILATERAL)
        mh.clnds[:, :5] = [
            (4, 0, 1, 4, 3), (4, 1, 2, 5, 4),
            (4, 3, 4, 7, 6), (4, 4, 5, 8, 7),
        ]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh

    def test_interior_boundary_face_split(self):
        mh = self.mesh
        n_boundary = sum(1 for i in range(mh.nface) if mh.fccls[i, 1] < 0)
        n_interior = mh.nface - n_boundary
        self.assertEqual(4, n_interior)
        self.assertEqual(8, n_boundary)

    def test_fccls_interior_faces_have_two_cells(self):
        mh = self.mesh
        for i in range(mh.nface):
            if mh.fccls[i, 1] >= 0:
                self.assertGreaterEqual(mh.fccls[i, 0], 0)
                self.assertGreaterEqual(mh.fccls[i, 1], 0)

    def test_all_clvol_quarter(self):
        assert_almost_equal(list(self.mesh.clvol), [0.25] * 4, decimal=10)


class StaticMesh2dMixedTC(unittest.TestCase):
    """Mixed 2 tri + 1 quad (6 nodes): cltpn mix, face extraction,
    clvol sum."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=2, nnode=6, nface=0, ncell=3)
        mh.ndcrd[:, :] = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
        ]
        mh.cltpn[:] = [
            modmesh.StaticMesh.QUADRILATERAL,
            modmesh.StaticMesh.TRIANGLE,
            modmesh.StaticMesh.TRIANGLE,
        ]
        mh.clnds[:, :5] = (
            (4, 0, 1, 4, 3), (3, 1, 2, 4, 0), (3, 2, 5, 4, 0))
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh

    def test_cltpn_mix(self):
        self.assertEqual(
            list(self.mesh.cltpn),
            [modmesh.StaticMesh.QUADRILATERAL,
             modmesh.StaticMesh.TRIANGLE,
             modmesh.StaticMesh.TRIANGLE])

    def test_face_count(self):
        self.assertEqual(8, self.mesh.nface)

    def test_clvol_sum(self):
        assert_almost_equal(sum(self.mesh.clvol), 2.0, decimal=10)

    def test_individual_clvol(self):
        assert_almost_equal(list(self.mesh.clvol), [1.0, 0.5, 0.5], decimal=10)

    def test_clfcs_varying_counts(self):
        mh = self.mesh
        self.assertEqual(4, mh.clfcs[0, 0])
        self.assertEqual(3, mh.clfcs[1, 0])
        self.assertEqual(3, mh.clfcs[2, 0])


class GmshMixedTQTC(unittest.TestCase):
    """Gmsh import of mixed_tq.msh: counts, spot-check geometry."""

    TESTDIR = os.path.abspath(os.path.dirname(__file__))
    DATADIR = os.path.join(TESTDIR, "data")

    @classmethod
    def setUpClass(cls):
        path = os.path.join(cls.DATADIR, "mixed_tq.msh")
        with open(path, 'rb') as f:
            data = f.read()
        gmsh = modmesh.Gmsh(data)
        cls.blk = gmsh.to_block()

    def test_node_count(self):
        self.assertEqual(8, self.blk.nnode)

    def test_cell_count(self):
        self.assertEqual(4, self.blk.ncell)

    def test_mixed_cell_types(self):
        types = set(self.blk.cltpn)
        self.assertIn(modmesh.StaticMesh.TRIANGLE, types)
        self.assertIn(modmesh.StaticMesh.QUADRILATERAL, types)

    def test_volume_sum(self):
        assert_almost_equal(sum(self.blk.clvol), 3.0, decimal=10)

    def test_quad_clvol(self):
        assert_almost_equal(self.blk.clvol[0], 1.0, decimal=10)
        assert_almost_equal(self.blk.clvol[1], 1.0, decimal=10)

    def test_triangle_clvol(self):
        assert_almost_equal(self.blk.clvol[2], 0.5, decimal=10)
        assert_almost_equal(self.blk.clvol[3], 0.5, decimal=10)


class StaticMesh2dSingleTriEdgeTC(unittest.TestCase):
    """Single-cell edge cases for a triangle."""

    @classmethod
    def setUpClass(cls):
        mh = modmesh.StaticMesh(ndim=2, nnode=3, nface=0, ncell=1)
        mh.ndcrd[:, :] = [(0, 0), (1, 0), (0, 1)]
        mh.cltpn.fill(modmesh.StaticMesh.TRIANGLE)
        mh.clnds[:, :4] = [(3, 0, 1, 2)]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh

    def test_face_count(self):
        self.assertEqual(3, self.mesh.nface)

    def test_all_faces_are_boundary(self):
        self.assertEqual(3, self.mesh.nbound)

    def test_ghost_cells(self):
        self.assertEqual(3, self.mesh.ngstcell)

    def test_clvol(self):
        assert_almost_equal(self.mesh.clvol[0], 0.5, decimal=10)

    def test_clcnd(self):
        mh = self.mesh
        assert_almost_equal(
            [mh.clcnd[0, 0], mh.clcnd[0, 1]],
            [1.0 / 3.0, 1.0 / 3.0], decimal=10)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
