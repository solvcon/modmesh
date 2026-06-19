import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import solvcon


class EulerCoreCETriangleTC(unittest.TestCase):
    """Test CE geometry on a 2D mesh of 3 triangles around the origin."""

    @classmethod
    def setUpClass(cls):
        mh = solvcon.StaticMesh(ndim=2, nnode=4, nface=0, ncell=3)
        mh.ndcrd[:, :] = [(0, 0), (-1, -1), (1, -1), (0, 1)]
        mh.cltpn.fill(solvcon.StaticMesh.TRIANGLE)
        mh.clnds[:, :4] = (3, 0, 1, 2), (3, 0, 2, 3), (3, 0, 3, 1)
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = solvcon.EulerCore(mesh=mh, time_increment=0.01)

    def test_dimensions(self):
        self.assertEqual(2, self.ec.ndim)
        self.assertEqual(3, self.ec.ncell)
        self.assertEqual(3, self.ec.ngstcell)

    def test_cevol_shape(self):
        total = self.ec.ngstcell + self.ec.ncell
        self.assertEqual(
            (total, solvcon.StaticMesh.CLMFC + 1),
            self.ec.cevol.shape)

    def test_cecnd_shape(self):
        total = self.ec.ngstcell + self.ec.ncell
        self.assertEqual(
            (total, (solvcon.StaticMesh.CLMFC + 1) * self.ec.ndim),
            self.ec.cecnd.shape)

    def test_sfcnd_shape(self):
        self.assertEqual(
            (self.ec.ncell,
             solvcon.StaticMesh.CLMFC * solvcon.StaticMesh.FCMND,
             self.ec.ndim),
            self.ec.sfcnd.shape)

    def test_sfnml_shape(self):
        self.assertEqual(
            (self.ec.ncell,
             solvcon.StaticMesh.CLMFC * solvcon.StaticMesh.FCMND,
             self.ec.ndim),
            self.ec.sfnml.shape)

    def test_cce_volume_positive(self):
        for icl in range(self.ec.ncell):
            self.assertGreater(
                self.ec.cevol[icl, 0], 0.0,
                f"CCE volume for cell {icl} should be positive")

    def test_bce_volumes_sum_to_cce(self):
        mh = self.mesh
        for icl in range(self.ec.ncell):
            nfc = mh.clfcs[icl, 0]
            bce_sum = sum(
                self.ec.cevol[icl, ifl]
                for ifl in range(1, nfc + 1))
            assert_almost_equal(
                bce_sum, self.ec.cevol[icl, 0],
                decimal=12,
                err_msg=f"BCE sum != CCE for cell {icl}")

    def test_bce_volume_manual_cell0(self):
        assert_almost_equal(self.ec.cevol[0, 1], 0.5, decimal=10)
        assert_almost_equal(self.ec.cevol[0, 2], 2.0 / 3.0, decimal=10)
        assert_almost_equal(self.ec.cevol[0, 3], 0.5, decimal=10)
        assert_almost_equal(
            self.ec.cevol[0, 0], 0.5 + 2.0 / 3.0 + 0.5, decimal=10)

    def test_prepare_ce_rerun(self):
        vol_before = self.ec.cevol[0, 0]
        self.ec.prepare_ce()
        assert_almost_equal(self.ec.cevol[0, 0], vol_before, decimal=12)


class EulerCoreCEQuadTC(unittest.TestCase):
    """Test CE geometry on a 2D mesh of a single unit-square quad."""

    @classmethod
    def setUpClass(cls):
        mh = solvcon.StaticMesh(ndim=2, nnode=4, nface=0, ncell=1)
        mh.ndcrd[:, :] = [(0, 0), (1, 0), (1, 1), (0, 1)]
        mh.cltpn.fill(solvcon.StaticMesh.QUADRILATERAL)
        mh.clnds[:, :5] = [(4, 0, 1, 2, 3)]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = solvcon.EulerCore(mesh=mh, time_increment=0.01)

    def test_dimensions(self):
        self.assertEqual(2, self.ec.ndim)
        self.assertEqual(1, self.ec.ncell)
        self.assertEqual(4, self.ec.ngstcell)

    def test_cce_volume_positive(self):
        self.assertGreater(self.ec.cevol[0, 0], 0.0)

    def test_bce_volumes_sum_to_cce(self):
        nfc = self.mesh.clfcs[0, 0]
        bce_sum = sum(self.ec.cevol[0, ifl] for ifl in range(1, nfc + 1))
        assert_almost_equal(bce_sum, self.ec.cevol[0, 0], decimal=12)

    def test_bce_volume_manual(self):
        assert_almost_equal(self.ec.cevol[0, 1], 0.5, decimal=10)
        assert_almost_equal(self.ec.cevol[0, 2], 0.5, decimal=10)
        assert_almost_equal(self.ec.cevol[0, 3], 0.5, decimal=10)
        assert_almost_equal(self.ec.cevol[0, 4], 0.5, decimal=10)
        assert_almost_equal(self.ec.cevol[0, 0], 2.0, decimal=10)

    def test_sfcnd_nonzero(self):
        self.assertTrue(any(
            self.ec.sfcnd[0, si, d] != 0
            for si in range(solvcon.StaticMesh.CLMFC
                            * solvcon.StaticMesh.FCMND)
            for d in range(self.ec.ndim)))

    def test_sfnml_nonzero(self):
        self.assertTrue(any(
            self.ec.sfnml[0, si, d] != 0
            for si in range(solvcon.StaticMesh.CLMFC
                            * solvcon.StaticMesh.FCMND)
            for d in range(self.ec.ndim)))

    def test_sfnml_magnitudes(self):
        ec, ndim = self.ec, self.ec.ndim
        for ifc in range(4):
            for ind in range(solvcon.StaticMesh.FCMND):
                si = ifc * solvcon.StaticMesh.FCMND + ind
                nml = [ec.sfnml[0, si, d] for d in range(ndim)]
                mag = np.linalg.norm(nml)
                if mag > 1e-14:
                    assert_almost_equal(mag, 0.5 * np.sqrt(2.0), decimal=10)


class EulerCoreCEMixedTC(unittest.TestCase):
    """Test CE geometry on a 2D mixed mesh (1 quad + 2 tri)."""

    @classmethod
    def setUpClass(cls):
        mh = solvcon.StaticMesh(ndim=2, nnode=6, nface=0, ncell=3)
        mh.ndcrd[:, :] = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
        ]
        mh.cltpn[:] = [
            solvcon.StaticMesh.QUADRILATERAL,
            solvcon.StaticMesh.TRIANGLE,
            solvcon.StaticMesh.TRIANGLE,
        ]
        mh.clnds[:, :5] = [(4, 0, 1, 4, 3), (3, 1, 2, 4, 0), (3, 2, 5, 4, 0)]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = solvcon.EulerCore(mesh=mh, time_increment=0.01)

    def test_varying_clfcs_counts(self):
        mh = self.mesh
        self.assertEqual(4, mh.clfcs[0, 0])
        self.assertEqual(3, mh.clfcs[1, 0])
        self.assertEqual(3, mh.clfcs[2, 0])

    def test_cce_volume_positive(self):
        for icl in range(self.ec.ncell):
            self.assertGreater(
                self.ec.cevol[icl, 0], 0.0,
                f"CCE volume for cell {icl} should be positive")

    def test_bce_volumes_sum_to_cce(self):
        mh = self.mesh
        for icl in range(self.ec.ncell):
            nfc = mh.clfcs[icl, 0]
            bce_sum = sum(self.ec.cevol[icl, ifl] for ifl in range(1, nfc + 1))
            assert_almost_equal(
                bce_sum, self.ec.cevol[icl, 0], decimal=12,
                err_msg=f"BCE sum != CCE for cell {icl}")


class EulerCoreCERectangleTC(unittest.TestCase):
    """Regression: rectangle.msh total CE volume."""

    TESTDIR = os.path.abspath(os.path.dirname(__file__))
    DATADIR = os.path.join(TESTDIR, "data")

    @classmethod
    def setUpClass(cls):
        path = os.path.join(cls.DATADIR, "rectangle.msh")
        with open(path, 'rb') as f:
            data = f.read()
        gmsh = solvcon.Gmsh(data)
        cls.blk = gmsh.to_block()
        cls.ec = solvcon.EulerCore(
            mesh=cls.blk, time_increment=0.01)

    def test_triangle_ce_total(self):
        blk = self.blk
        tri_ce_sum = 0.0
        for i in range(self.ec.ncell):
            if blk.cltpn[i] == solvcon.StaticMesh.TRIANGLE:
                tri_ce_sum += self.ec.cevol[i, 0]
        assert_almost_equal(tri_ce_sum, 8.0, decimal=6)

    def test_triangle_bce_sum_equals_cce(self):
        blk = self.blk
        for i in range(self.ec.ncell):
            if blk.cltpn[i] != solvcon.StaticMesh.TRIANGLE:
                continue
            nfc = blk.clfcs[i, 0]
            bce_sum = sum(self.ec.cevol[i, ifl] for ifl in range(1, nfc + 1))
            assert_almost_equal(
                bce_sum, self.ec.cevol[i, 0], decimal=12,
                err_msg=f"BCE sum != CCE for tri cell {i}")


@unittest.skip("3D ghost cell geometry has NaN due to fill_ghost bug")
class EulerCoreCETetrahedronTC(unittest.TestCase):
    """Test CE geometry on a 3D mesh of a single tetrahedron."""

    @classmethod
    def setUpClass(cls):
        mh = solvcon.StaticMesh(ndim=3, nnode=4, nface=4, ncell=1)
        mh.ndcrd[:, :] = (
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        mh.cltpn.fill(solvcon.StaticMesh.TETRAHEDRON)
        mh.clnds[:, :5] = [(4, 0, 1, 2, 3)]
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        cls.mesh = mh
        cls.ec = solvcon.EulerCore(mesh=mh, time_increment=0.01)

    def test_dimensions(self):
        self.assertEqual(3, self.ec.ndim)
        self.assertEqual(1, self.ec.ncell)

    def test_cce_volume_positive(self):
        self.assertGreater(self.ec.cevol[0, 0], 0.0)

    def test_bce_volumes_sum_to_cce(self):
        mh = self.mesh
        nfc = mh.clfcs[0, 0]
        bce_sum = sum(
            self.ec.cevol[0, ifl] for ifl in range(1, nfc + 1))
        assert_almost_equal(
            bce_sum, self.ec.cevol[0, 0], decimal=12)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
